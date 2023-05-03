import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model.api import Model


class KNNShapley(DataEvaluator, ModelLessMixin):
    """Data valuation using KNNShapley implementation.

    KNN Shapley is a model-less mixin. This means we cannot specify an underlying
    prediction model for the DataEvaluator. However, we can specify a pretrained
    embedding model.

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors to group the data points, by default 10
    batch_size : int, optional
        Batch size of tensors to load at a time during training, by default 32
    embedding_model : Model, optional
        Pre-trained embedding model used by DataEvaluator, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Model = None,
        random_state: RandomState = None,
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """:math:`1.` for all matching rows and :math:`0.` otherwise."""
        return (y == self.y_valid).all(dim=1).float()

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Computes KNN shapley data values, as implemented by the following. Ignores all
        positional and key word arguments.

        References
        ----------
        .. [1] PyTorch implementation
            <https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py>
        """
        n = len(self.x_train)
        m = len(self.x_valid)
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        # Computes Euclidean distance by computing crosswise per batch
        # Doesn't shuffle to maintain relative order
        x_train_view, x_valid_view = x_train.view(n, -1), x_valid.view(m, -1)

        dist_list = []  # Uses batching to only load at most `batch_size` tensors
        for x_train_batch in DataLoader(x_train_view, self.batch_size):  # No shuffle
            dist_row = []
            for x_val_batch in DataLoader(x_valid_view, self.batch_size):
                dist_row.append(torch.cdist(x_train_batch, x_val_batch))
            dist_list.append(torch.cat(dist_row, dim=1))
        dist = torch.cat(dist_list, dim=0)

        # Arranges by distances
        sort_indices = torch.argsort(dist, dim=0, stable=True)
        y_train_sort = self.y_train[sort_indices]

        score = torch.zeros_like(dist)
        score[sort_indices[m - 1], range(m)] = self.match(y_train_sort[n - 1]) / n

        # fmt: off
        for i in tqdm.tqdm(range(n - 2, -1, -1)):
            score[sort_indices[i], range(m)] = (
                score[sort_indices[i + 1], range(m)]
                + min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1))
                * (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
            )

        self.data_values = score.mean(axis=1)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using KNN Shapley data valuation

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values
