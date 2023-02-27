import copy

import torch

from dataoob.dataval import DataEvaluator, Model


class KNNShapley(DataEvaluator):
    """Data valuation for nearest neighbor algorithms.
    Ref. https://arxiv.org/abs/1908.08619

    :param Model pred_model: Prediction model
    :param int k_neighbors: Number of neighbors to classify, defaults to 10
    """

    def __init__(
        self,
        pred_model: Model,
        use_prediction: bool = False,
        k_neighbors: int = 10,
    ):
        self.pred_model = copy.deepcopy(pred_model) if use_prediction else None
        self.use_prediction = use_prediction

        self.k_neighbors = k_neighbors

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """_Stores and transforms input data for KNNShapley

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Trains the KNN shapley model to assign datavalues to each datapoint.
        Can train either on the model predictions or raw datapoints

        :param int batch_size: pred_model training batch size, defaults to 32
        :param int epochs: Number of epochs for the pred_model,  defaults to 1
        """
        if self.use_prediction:
            self.pred_model.fit(
                self.x_train, self.y_train, batch_size=batch_size, epochs=epochs
            )

            y_hat_train = self.pred_model.predict(self.x_train)
            y_hat_valid = self.pred_model.predict(self.x_valid)
        else:
            y_hat_train, y_hat_valid = self.x_train, self.x_valid

        self.data_values = self._knn_shap(
            y_hat_train, self.y_train, y_hat_valid, self.y_valid
        )

        # coef = torch.tensor(self.pred_model.coefs_[0])  TODO discuss intuition is here
        # inter = torch.tensor(self.pred_model.intercepts_[0])

        # y_hat_train = F.relu(torch.matmul(self.x_train, coef) + inter)
        # y_hat_train = F.relu(torch.matmul(self.x_valid, coef) + inter)

    @staticmethod
    def match(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns 1. for all matching rows and 0. otherwise"""
        return (x == y).all(dim=1).float()

    def _knn_shap(
        self,
        y_hat_train: torch.Tensor,
        y_train: torch.Tensor,
        y_hat_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Computes KNN shapley, as implemented by the following
        Ref. https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py

        :param torch.Tensor y_hat_train: Data covariates or predicted values
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor y_hat_valid: Data covariates or predicted values
        :param torch.Tensor y_valid: Test+Held-out labels
        :return torch.Tensor: KNN shapley values for each datapoint
        """
        N = y_hat_train.size(dim=0)
        M = y_hat_valid.size(dim=0)

        # Computes Euclidean distance
        dist = torch.cdist(y_hat_train.view(N, -1), y_hat_valid.view(M, -1))

        # Arranges by distances
        sort_indices = torch.argsort(dist, dim=0, stable=True)
        y_train_sorted = y_train[sort_indices]

        score = torch.zeros_like(dist)
        score[sort_indices[N - 1], range(M)] = (
            self.match(y_train_sorted[N - 1], y_valid) / N
        )

        for i in range(N - 2, -1, -1):
            score[sort_indices[i], range(M)] = (
                score[sort_indices[i + 1], range(M)] +
                min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1)) *
                (self.match(i, y_valid) - self.match(y_train_sorted[i + 1], y_valid))
            )

        return score.mean(axis=1)

    def evaluate_data_values(self) -> torch.Tensor:
        """Returns data values using the data valuator model.

        :return torch.Tensor: Predicted data values/selection for every input data point
        """
        return self.data_values
