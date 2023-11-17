import numpy as np
import torch
import tqdm
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin


class LeaveOneOut(DataEvaluator, ModelMixin):
    """Leave One Out data valuation implementation.

    References
    ----------
    .. [1] R. Cook,
        Detection of Influential Observation in Linear Regression,
        Technometrics, Vol. 19, No. 1 (Feb., 1977), pp. 15-18 (4 pages).

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Leave-One-Out data valuation.

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

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Compute the data values using the Leave-One-Out data valuation.
        Equivalently, LOO can be computed from the marginal contributions as it's a
        semivalue.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        self.data_values = np.zeros((self.num_points,))
        indices = self.random_state.permutation(self.num_points)

        curr_model = self.pred_model.clone()

        curr_model.fit(self.x_train, self.y_train, *args, **kwargs)
        y_valid_hat = curr_model.predict(self.x_valid)
        baseline_score = self.evaluate(self.y_valid, y_valid_hat)

        for i in tqdm.tqdm(range(self.num_points)):
            loo_coalition = np.delete(indices, i)  # Deletes random point

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=loo_coalition),
                Subset(self.y_train, indices=loo_coalition),
                *args,
                **kwargs,
            )

            y_hat = curr_model.predict(self.x_valid)
            loo_score = self.evaluate(self.y_valid, y_hat)
            self.data_values[indices[i]] = baseline_score - loo_score

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Compute data values using Leave One Out data valuation.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values
