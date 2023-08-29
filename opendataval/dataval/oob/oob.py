from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin


class DataOob(DataEvaluator, ModelMixin):
    """Data Out-of-Bag data valuation implementation.

    Input evaluation metrics are valid if we compare one data point across several
    predictions. Examples include: `accuracy` and `L2 distance`

    References
    ----------
    .. [1] Y. Kwon and J. Zou,
        Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value
        arXiv.org, 2023. Available: https://arxiv.org/abs/2304.07718.

    Parameters
    ----------
    num_models : int, optional
        Number of models to bag/aggregate, by default 1000
    proportion : float, optional
        Proportion of data points in the in-bag sample.
        sample_size = len(dataset) * proportion, by default 1.0
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: int = 1.0,
        random_state: Optional[RandomState] = None,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Data Out-Of-Bag Evaluator.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates, unused by DataOob
        y_valid : torch.Tensor
            Test+Held-out labels, unused by DataOob
        """
        self.x_train = x_train
        self.y_train = y_train
        _ = x_valid, y_valid  # Unused parameters

        self.num_points = len(x_train)
        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape
        self.max_samples = round(self.proportion * self.num_points)

        self.oob_pred = torch.zeros((0, *self.label_dim), requires_grad=False)
        self.oob_indices = GroupingIndex()
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains Data Out-of-Bag model by bagging a model and collecting all out-of-bag
        predictions. We then evaluate each data point to their out-of-bag predictions.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        sample_dim = (self.num_models, self.max_samples)
        subsets = self.random_state.randint(0, self.num_points, size=sample_dim)

        for i in tqdm.tqdm(range(self.num_models)):
            in_bag = subsets[i]

            # out_bag is the indices where the bincount is zero.
            out_bag = (np.bincount(in_bag, minlength=self.num_points) == 0).nonzero()[0]
            if not out_bag.any():
                continue

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=in_bag),
                Subset(self.y_train, indices=in_bag),
                *args,
                **kwargs,
            )

            y_hat = curr_model.predict(Subset(self.x_train, indices=out_bag))
            self.oob_pred = torch.cat((self.oob_pred, y_hat.detach().cpu()), dim=0)
            self.oob_indices.add_indices(out_bag)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values by evaluating how the OOB labels compare to training labels.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        self.data_values = np.zeros(self.num_points)

        for i, indices in self.oob_indices.items():
            # Expands the label to the desired size, squeezes for regression
            oob_labels = self.y_train[i].expand((len(indices), *self.label_dim))
            self.data_values[i] = self.evaluate(oob_labels, self.oob_pred[indices])

        return self.data_values


class GroupingIndex(defaultdict[int, list[int]]):
    """Stores value and position of insertion in a stack.

    Parameters
    ----------
    start : int, optional
        Starting insertion position, increments after each insertion, by default 0
    """

    def __init__(self, start: int = 0):
        super().__init__(list)
        self.position = start

    def add_indices(self, values: list[int]):
        """Add values to defaultdict and record position in stack in-order."""
        for i in values:
            self.__getitem__(i).append(self.position)
            self.position += 1
