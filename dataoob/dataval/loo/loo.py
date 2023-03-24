import copy

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, Subset

from dataoob.dataval import DataEvaluator


class LeaveOneOut(DataEvaluator):
    """Leave One Out data valuation implementation.

    References
    ----------
    .. [1] Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def input_data(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        x_valid: torch.Tensor | Dataset,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for Leave-One-Out data valuation

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor | Dataset
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        # Additional parameters
        self.num_points = len(x_train)

        return self

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes the data values using the Leave-One-Out data valuation.

        Equivalently, LOO can be computed from the marginal contributions as it's a
        semivalue

        Parameters
        ----------
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        """
        self.data_values = np.zeros((self.num_points,))
        indices = self.random_state.permutation(self.num_points)

        curr_model = copy.deepcopy(self.pred_model)

        curr_model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        y_valid_hat = self.pred_model.predict(self.x_valid)
        baseline_score = self.evaluate(self.y_valid, y_valid_hat)

        for i in tqdm.tqdm(range(self.num_points)):
            loo_coalition = np.delete(indices, i)  # Deletes random point

            curr_model = copy.deepcopy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=loo_coalition),
                Subset(self.y_train, indices=loo_coalition),
                batch_size=batch_size,
                epochs=epochs,
            )

            y_hat = curr_model.predict(self.x_valid)
            loo_score = self.evaluate(self.y_valid, y_hat)
            self.data_values[indices[i]] = baseline_score - loo_score

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values using Leave One Out data valuation

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values
