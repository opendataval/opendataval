import numpy as np
import torch
from dataoob.dataval import DataEvaluator
from torch.utils.data import DataLoader


class KNNShapley(DataEvaluator):
    """Data valuation for nearest neighbor algorithms.
    Ref. https://arxiv.org/abs/1908.08619

    :param int k_neighbors: Number of neighbors to classify, defaults to 10
    """

    def __init__(
        self,
        k_neighbors: int = 10,
    ):
        self.k_neighbors = k_neighbors

    @property
    def pred_model(self):
        raise NotImplementedError("KNNShapley does not support a model, cn change")

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for KNNShapley

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """Returns 1. for all matching rows and 0. otherwise"""
        return (y == self.y_valid).all(dim=1).float()

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes KNN shapley, as implemented by the following
        Ref. https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py

        :param int batch_size: pred_model training batch size, defaults to 32
        :param int epochs: Number of epochs for the pred_model,  defaults to 1
        """
        N = len(self.x_train)
        M = len(self.x_valid)  # Throw something in to extract the values

        # Computes Euclidean distance by computing crosswise per batch, batch_size//2
        # Doesn't shuffle to maintain relative order, are views necessary
        x_train_view, x_valid_view = self.x_train.view(N, -1), self.x_valid.view(M, -1)

        dist_list = []  # Uses batching to only loand at most `batch_size` tensors
        for x_train_batch in DataLoader(x_train_view, batch_size, shuffle=False):
            dist_row = []
            for x_valid_batch in DataLoader(x_valid_view, batch_size, shuffle=False):
                dist_row.append(torch.cdist(x_train_batch, x_valid_batch))
            dist_list.append(torch.cat(dist_row, dim=1))
        dist = torch.cat(dist_list, dim=0)

        # Arranges by distances
        sort_indices = torch.argsort(dist, dim=0, stable=True)
        y_train_sort = self.y_train[sort_indices]

        score = torch.zeros_like(dist)
        score[sort_indices[N - 1], range(M)] = self.match(y_train_sort[N - 1]) / N

        for i in range(N - 2, -1, -1):
            score[sort_indices[i], range(M)] = (
                score[sort_indices[i + 1], range(M)] +
                min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1)) *
                (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
            )

        self.data_values = score.mean(axis=1)

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values using the KNN Shapley data valuator model.

        :return np.ndarray: Predicted data values/selection for every input data point
        """
        return self.data_values
