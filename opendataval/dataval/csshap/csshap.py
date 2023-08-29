import math
import warnings
from functools import partial
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.margcontrib import Sampler, TMCSampler


class ClassWiseShapley(DataEvaluator, ModelMixin):
    """Class-wise shapley data valuation implementation

    NOTE only categorical labels is a valid input to Class-Wise Shapley.

    References
    ----------
    .. [1] S. Schoch, H. Xu, and Y. Ji,
        CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification
        arXiv.org, 2022. https://arxiv.org/abs/2211.06800.

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. NOTE the sampler may not use
        a cache and cache_name should explicitly be passes None. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.TMCSampler` but removes
        cache.
    """

    def __init__(self, sampler: Sampler = None, *args, **kwargs):
        self.sampler = sampler
        if getattr(self.sampler, "cache_name", None) is not None:
            warnings.warn("Samplers passed into CS Shap should disable caching!")

        if self.sampler is None:
            self.sampler = TMCSampler(*args, **kwargs, cache_name=None)

    def input_data(
        self, x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor
    ):
        """Store and transform input data for CS-Shapley.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.num_valid = len(x_valid)

        # Undoes one hot encoding if applied
        if (torch.sum(y_train, dim=1) - torch.ones(len(x_train))).count_nonzero() == 0:
            self.train_classes = torch.argmax(self.y_train, dim=1)
            self.valid_classes = torch.argmax(self.y_valid, dim=1)
        else:
            self.train_classes, self.valid_classes = self.y_train, self.y_valid

        self.classes = torch.unique(self.train_classes)
        self.data_values = np.zeros((len(x_train),))
        return self

    def train_data_values(self, *args, **kwargs):
        """Uses sampler to trains model to find marginal contribs and data values.

        For each class, we separate the training and validation data into in-class
        and out-class. Then we will compute the class-wise shapley values using the
        sampler. Finally, we record the shapley value in self.data_values.
        """
        sampler = self.sampler

        for label in self.classes:
            train_in, train_out, valid_in, valid_out = self._get_class_indices(label)

            # Separates training and valid data to in-class and out-class
            x_train_in_class = Subset(self.x_train, train_in)
            y_train_in_class = Subset(self.y_train, train_in)
            x_train_out_class = Subset(self.x_train, train_out)
            y_train_out_class = Subset(self.y_train, train_out)

            x_valid_in_class = Subset(self.x_valid, valid_in)
            y_valid_in_class = self.y_valid[valid_in]  # Required to be a tensor
            x_valid_out_class = Subset(self.x_valid, valid_out)
            y_valid_out_class = self.y_valid[valid_out]  # Required to be a tensor

            given_utility_func = partial(
                self._compute_class_wise_utility,
                x_train_in_class=x_train_in_class,
                y_train_in_class=y_train_in_class,
                x_train_out_class=x_train_out_class,
                y_train_out_class=y_train_out_class,
                x_valid_in_class=x_valid_in_class,
                y_valid_in_class=y_valid_in_class,
                x_valid_out_class=x_valid_out_class,
                y_valid_out_class=y_valid_out_class,
            )
            sampler.set_coalition(x_train_in_class)  # Marg contrib for in class only
            sampler.set_evaluator(given_utility_func)
            marg_contrib = sampler.compute_marginal_contribution(*args, **kwargs)

            # The sampler only computes data values for in-class with the indices
            # index i of marg contrib corresponds to index train_in[i] of data value.
            self.data_values[train_in] += np.sum(marg_contrib / len(train_in), axis=1)

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values for CS-Shapley"""
        return self.data_values

    def _get_class_indices(self, label: int) -> tuple[Sequence[int], ...]:
        """Gets indices of train and valid data with and without the specified label."""
        return (
            (self.train_classes == label).nonzero(as_tuple=True)[0],
            (self.train_classes != label).nonzero(as_tuple=True)[0],
            (self.valid_classes == label).nonzero(as_tuple=True)[0],
            (self.train_classes != label).nonzero(as_tuple=True)[0],
        )

    def _compute_class_wise_utility(
        self,
        subset: list[int],
        *args,
        x_train_in_class: Tensor,
        y_train_in_class: Tensor,
        x_train_out_class: Tensor,
        y_train_out_class: Tensor,
        x_valid_in_class: Tensor,
        y_valid_in_class: Tensor,
        x_valid_out_class: Tensor,
        y_valid_out_class: Tensor,
        **kwargs
    ) -> float:
        """Computes the utility given a subset of the in-class training data.

        References
        ----------
        .. [1] S. Schoch, H. Xu, and Y. Ji,
            CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification
            arXiv.org, 2022. https://arxiv.org/abs/2211.06800.

        Parameters
        ----------
        subset : list[int]
            List of indices of the in-class training data
        x_train_in_class : Tensor
            In-class data covariates
        y_train_in_class : Tensor
            In-class data labels
        x_train_out_class : Tensor
            Out-class data covariates
        y_train_out_class : Tensor
            Out-class data labels
        x_valid_in_class : Tensor
            In-class Test+Held-out covariates
        y_valid_in_class : Tensor
            In-class Test+Held-out labels
        x_valid_out_class : Tensor
            Out-class Test+Held-out covariates
        y_valid_out_class : Tensor
            Out-class est+Held-out labels

        Returns
        -------
        float
            Utility of the given subset
        """
        x_train = ConcatDataset([Subset(x_train_in_class, subset), x_train_out_class])
        y_train = ConcatDataset([Subset(y_train_in_class, subset), y_train_out_class])

        curr_model = self.pred_model.clone()
        curr_model.fit(x_train, y_train, *args, **kwargs)

        y_hat_in_class = curr_model.predict(x_valid_in_class)
        y_hat_out_class = curr_model.predict(x_valid_out_class)

        in_class_perf = self.evaluate(y_valid_in_class, y_hat_in_class)
        out_class_perf = self.evaluate(y_valid_out_class, y_hat_out_class)
        return in_class_perf * math.exp(out_class_perf) / self.num_valid
