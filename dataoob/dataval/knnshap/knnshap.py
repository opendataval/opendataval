import copy

import numpy as np
import torch
import torch.nn.functional as F

from dataoob.dataval import DataEvaluator, Model


class KNNShapley(DataEvaluator):
    """_summary_

    :param Model pred_model: _description_
    :param callable metric: _description_
    :param int k_neighbors: _description_, defaults to 10
    """
    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        k_neighbors: int=10,
    ):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        self.k_neighbors = k_neighbors

    def input_data(self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """_summary_

        :param torch.Tensor x_train: _description_
        :param torch.Tensor y_train: _description_
        :param torch.Tensor x_valid: _description_
        :param torch.Tensor y_valid: _description_
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def train_data_values(self, batch_size: int=32, epochs: int=20):
        """_summary_

        :param int batch_size: _description_, defaults to 32
        :param int epochs: _description_, defaults to 20
        """
        self.pred_model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)

        y_hat_train = F.relu(self.pred_model.predict(self.x_train))
        y_hat_valid = F.relu(self.pred_model.predict(self.x_valid))

        self.data_values = self._knn_shap(y_hat_train, self.y_train, y_hat_valid, self.y_valid)

        # coef = torch.tensor(self.pred_model.coefs_[0])  TODO discuss what the intuition is here
        # inter = torch.tensor(self.pred_model.intercepts_[0])

        # y_hat_train = F.relu(torch.matmul(self.x_train, coef) + inter)
        # y_hat_train = F.relu(torch.matmul(self.x_valid, coef) + inter)

    def _knn_shap(self, y_hat_train: torch.Tensor, y_train: torch.Tensor, y_hat_valid: torch.Tensor, y_valid: torch.Tensor):
        """_summary_

        :param torch.Tensor y_hat_train: _description_
        :param torch.Tensor y_train: _description_
        :param torch.Tensor y_hat_valid: _description_
        :param torch.Tensor y_valid: _description_
        :return _type_: _description_
        """
        N = y_hat_train.size(dim=0)
        M = y_hat_valid.size(dim=0)

        # Computes Euclidean distance
        dist = torch.cdist(y_hat_train.view(N, -1), y_hat_valid.view(M, -1))

        # Arranges by distances
        sort_indices = torch.argsort(dist, dim=0, stable=True)
        y_train_sorted = y_train[sort_indices]
        print((y_train_sorted[N-1] == y_valid).float().size())

        score = torch.zeros_like(dist)
        score[sort_indices[N-1], range(M)] = (y_train_sorted[N-1] == y_valid).all(dim=1).float() / N

        for i in range(N-2, -1, -1):
            score[sort_indices[i], range(M)] = (
                score[sort_indices[i+1], range(M)] +
                ( (y_train_sorted[i] == y_valid).all(dim=1).float() - (y_train_sorted[i+1] == y_valid).all(dim=1).float() ) *
                min(self.k_neighbors, i+1) / (self.k_neighbors*(i+1))
            )

        return score.mean(axis=1)

    def evaluate_data_values(self):
        """_summary_

        :return _type_: _description_
        """
        return self.data_values



