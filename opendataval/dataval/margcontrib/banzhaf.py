from itertools import accumulate
from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.margcontrib.shap import ShapEvaluator


class DataBanzhaf(DataEvaluator, ModelMixin):
    """Data Banzhaf implementation.

    References
    ----------
    .. [1] J. T. Wang and R. Jia,
        Data Banzhaf: A Robust Data Valuation Framework for Machine Learning,
        arXiv.org, 2022. Available: https://arxiv.org/abs/2205.15466.

    Parameters
    ----------
    num_models : int, optional
        Number of models to take to compute Banzhaf values, by default 1000
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self, num_models: int = 1000, random_state: Optional[RandomState] = None
    ):
        self.num_models = num_models
        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Data Banzhaf.

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

        self.num_points = len(x_train)
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.sample_utility = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))

        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the Data Banzhaf value by sampling from the powerset. We compute
        average performance of all subsets including and not including a data point.

        References
        ----------
        .. [1] J. T. Wang and R. Jia,
            Data Banzhaf: A Robust Data Valuation Framework for Machine Learning,
            arXiv.org, 2022. Available: https://arxiv.org/abs/2205.15466.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        sample_dim = (self.num_models, self.num_points)
        subsets = self.random_state.binomial(1, 0.5, size=sample_dim)

        for i in tqdm.tqdm(range(self.num_models)):
            subset = subsets[i].nonzero()[0]
            if not subset.any():
                continue

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                *args,
                **kwargs,
            )
            y_valid_hat = curr_model.predict(self.x_valid)

            curr_perf = self.evaluate(self.y_valid, y_valid_hat)
            self.sample_utility[range(self.num_points), subsets[i]] += curr_perf
            self.sample_counts[range(self.num_points), subsets[i]] += 1

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the Data Banzhaf data valuator. Finds difference
        of average performance of all sets including data point minus not-including.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = np.divide(
            self.sample_utility,
            self.sample_counts,
            out=np.zeros_like(self.sample_utility),
            where=self.sample_counts != 0,
        )
        return msr[:, 1] - msr[:, 0]  # Diff of subsets including/excluding i data point


class DataBanzhafMargContrib(ShapEvaluator):
    """Data Banzhaf implementation using the marginal contributions.

    Data Banzhaf implementation using the ShapEvaluator, which already computes the
    marginal contributions for other evaluators. This approach may not be as efficient
    as the previous approach, but is recommended to minimize compute time if
    you cache a previous computation.

    References
    ----------
    .. [1] J. T. Wang and R. Jia,
        Data Banzhaf: A Robust Data Valuation Framework for Machine Learning,
        arXiv.org, 2022. Available: https://arxiv.org/abs/2205.15466.

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.GrTMCSampler`.
    """

    def compute_weight(self) -> float:
        """Compute weights for each cardinality of training set.

        Banzhaf weights each data point according to the number of combinations of
        :math:`j` cardinality to number of data points

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """

        def pascals(prev: int, position: int):  # Get level of pascal's triangle
            return (prev * (self.num_points - position + 1)) // position

        weights = np.fromiter(
            accumulate(range(1, self.num_points), func=pascals, initial=1),
            dtype=float,
        )
        return weights / weights.sum()
