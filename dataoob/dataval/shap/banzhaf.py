import copy

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from dataoob.dataval import DataEvaluator


class DataBanzhaf(DataEvaluator):
    """Data Banzhaf implementation

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
        """Stores and transforms input data for Data Banzhaf

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

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Trains the Data Banzhaf value by sampling from the powerset. We compute
        average perfromance of all subsets including and not including a data point.

        References
        ----------
        .. [1] J. T. Wang and R. Jia,
            Data Banzhaf: A Robust Data Valuation Framework for Machine Learning,
            arXiv.org, 2022. Available: https://arxiv.org/abs/2205.15466.

        Parameters
        ----------
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        """
        num_subsets = self.random_state.binomial(
            1, 0.5, size=(self.samples, self.num_points)
        )

        for i in tqdm.tqdm(range(self.samples)):
            subset = num_subsets[i].nonzero()[0]

            curr_model = copy.deepcopy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                batch_size=batch_size,
                epochs=epochs,
            )
            y_valid_hat = curr_model.predict(self.x_valid)

            curr_perf = self.evaluate(self.y_valid, y_valid_hat)
            self.sample_utility[range(self.num_points), num_subsets[i]] += curr_perf
            self.sample_counts[range(self.num_points), num_subsets[i]] += 1

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values using the Data Banzhaf data valuator. Finds difference
        between average performance of all sets including data point minus not-including

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = np.divide(
            self.sample_utility, self.sample_counts, where=self.sample_counts != 0.0
        )
        return msr[:, 1] - msr[:, 0]  # Diff of subsets including/excluding i data point
