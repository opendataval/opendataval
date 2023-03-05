from dataoob.model import Model
import copy
import numpy as np
from dataoob.dataval import DataEvaluator
from torch.utils.data import Subset, Dataset
import torch
import tqdm

class LeaveOneOut(DataEvaluator):
    """Leave One Out data valuation.
    Ref. https://arxiv.org/pdf/2110.14049

    :param Model pred_model: Prediction model
    :param callable (torch.Tensor, torch.Tensor -> float) metric: Evaluation function
    to determine model performance
    """
    def __init__(
        self,
        pred_model: Model,
        metric: callable,
    ):
        self.pred_model = copy.copy(pred_model)
        self.metric = metric

    def input_data(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        x_valid: torch.Tensor | Dataset,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for Leave-One-Out data valuation

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        # Additional parameters
        self.n_points = len(x_train)

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes the data values using the Leave-One-Out data valuation.
        Equivalently, LOO can be computed from the marginal contributions as it's a
        semivalue, however the current implementation of ShapEvaluator does not
        uses a TMC and does not guarantee we will sample the subset necessary for LOO.
        It is more accurate and efficient to just compute LOO explicitly as below

        :param int batch_size: Baseline training batch size, defaults to 32
        :param int epochs: Number of epochs for baseline training, defaults to 1
        """

        self.data_values = np.zeros((self.n_points,))
        indices = np.random.permutation(self.n_points)

        curr_model = copy.copy(self.pred_model)

        curr_model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        y_valid_hat = self.pred_model.predict(self.x_valid)
        baseline_score = self.evaluate(self.y_valid, y_valid_hat)

        for i in tqdm.tqdm(range(self.n_points)):
            loo_coalition = np.delete(indices, i)  # Deletes random point

            curr_model = copy.copy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=loo_coalition),
                Subset(self.y_train, indices=loo_coalition),
                batch_size=batch_size,
                epochs=epochs,
            )

            y_hat = curr_model.predict(self.x_valid)
            loo_score = self.evaluate(self.y_valid, y_hat)
            self.data_values[indices[i]] = baseline_score - loo_score

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values using the LOO data valuator.

        :return np.ndarray: predicted data values/selection for every input data point
        """
        return self.data_values



