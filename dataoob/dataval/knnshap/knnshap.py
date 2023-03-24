import numpy as np
import torch
import tqdm
from dataoob.dataval import DataEvaluator
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader


class KNNShapley(DataEvaluator):
    """Data valuation using KNNShapley implementation

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. [Online]. Available: https://arxiv.org/abs/1908.08619.

    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors to group the data points, by default 10
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, k_neighbors: int = 10, random_state: RandomState = None):
        self.k_neighbors = k_neighbors
        self.random_state = check_random_state(random_state)

    @property
    def pred_model(self):
        raise NotImplementedError("KNNShapley does not support a model")

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """Returns :math:`1.` for all matching rows and :math:`0.` otherwise"""
        return (y == self.y_valid).all(dim=1).float()

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes KNN shapley data values, as implemented

        References
        ----------
        .. [1] PyTorch implementation
            <https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py>

        Parameters
        ----------
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        """
        N = len(self.x_train)
        M = len(self.x_valid)

        # Computes Euclidean distance by computing crosswise per batch, batch_size//2
        # Doesn't shuffle to maintain relative order
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

        # fmt: off
        for i in tqdm.tqdm(range(N - 2, -1, -1)):
            score[sort_indices[i], range(M)] = (
                score[sort_indices[i + 1], range(M)]
                + min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1))
                * (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
            )

        self.data_values = score.mean(axis=1)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values computed from KNN Shapley

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values
