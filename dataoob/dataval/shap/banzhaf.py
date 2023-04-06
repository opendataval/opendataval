from itertools import accumulate

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from dataoob.dataval.api import DataEvaluator
from dataoob.dataval.shap.shap import ShapEvaluator


class DataBanzhaf(DataEvaluator):
    """Data Banzhaf implementation.

    References
    ----------
    .. [1] J. T. Wang and R. Jia,
        Data Banzhaf: A Robust Data Valuation Framework for Machine Learning,
        arXiv.org, 2022. Available: https://arxiv.org/abs/2205.15466.

    Parameters
    ----------
    samples : int, optional
        Number of samples to take to compute Banzhaf values, by default 1000
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, samples: int = 1000, random_state: RandomState = None):
        self.samples = samples
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
        sample_dim = (self.samples, self.num_points)
        subsets = self.random_state.binomial(1, 0.5, size=sample_dim)

        for i in tqdm.tqdm(range(self.samples)):
            subset = subsets[i].nonzero()[0]

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                *args,
                **kwargs
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
    as the previous approach, but is reccomended to minimize compute time if
    you cache a previous computation.

    References
    ----------
    .. [1] J. T. Wang and R. Jia,
        Data Banzhaf: A Robust Data Valuation Framework for Machine Learning,
        arXiv.org, 2022. Available: https://arxiv.org/abs/2205.15466.

    Parameters
    ----------
    gr_threshold : float, optional
        Convergence threshold for the Gelman-Rubin statistic.
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.01
    max_iterations : int, optional
        Max number of outer iterations of MCMC sampling, by default 100
    min_samples : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    model_name : str, optional
        Unique name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
        random_state: RandomState = None,
    ):
        super().__init__(
            gr_threshold, max_iterations, min_samples, model_name, random_state
        )

    def compute_weight(self) -> float:
        """Compute weights for each cardinality of training set.

        Banzhaf weights each data point according to the number of combinations of
        :math:`j` cardinality to number of data points

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """

        def pascals(prev: int, position: int):  # Get level of pascal's traingle
            return (prev * (self.num_points - position + 1)) // position

        weights = np.fromiter(
            accumulate(range(2, self.num_points + 1), pascals, initial=self.num_points),
            dtype=float,
        )
        return weights / weights.sum()

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Multiplies the marginal contribution with their respective weights to get
        Data Banzhaf

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)
