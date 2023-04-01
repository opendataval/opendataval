import numpy as np
import tqdm

import torch

from torch.utils.data import Subset
import copy

from dataoob.dataval import DataEvaluator
from numpy.random import RandomState
from sklearn.utils import check_random_state

from collections import defaultdict


class DataOob(DataEvaluator):
    """Data Out-of-Bag data valuation implementation.
    Input evaluation metrics are valid if we compare one data point across several
    predictions. Examples include: `accuracy` and `L2 distance`

    References
    ----------
    .. [1] Y. Kwon,
        #TODO,
        arXiv.org, 2023. [Online]. Available: #TODO.

    Parameters
    ----------
    num_models : int, optional
        Number of models to bag/aggregate, by default 1000
    proportion : float, optional
        Proportion of data points in the in-bag sample.
        sample_size = len(dataset) * proportion, by default 1.0
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: int = 1.0,
        random_state: RandomState = None,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for Data Out-Of-Bag Evaluator

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
        self.label_dim = 1 if self.y_train.dim() == 1 else self.y_train.size(dim=1)
        self.max_samples = round(self.proportion * self.num_points)
        return self

    def train_data_values(self, *args, **kwargs):
        """This implementation is closer to the original implementation, although
        it doesn't generalize to regressions

        """
        self.oob_pred = torch.zeros((0, self.label_dim), requires_grad=False)
        self.oob_indices = GroupingIndex()

        for i in tqdm.tqdm(range(self.num_models)):
            in_bag = np.random.randint(0, self.num_points, self.max_samples)
            # out_bag is the indices where the bincount is zero.
            out_bag = (np.bincount(in_bag, minlength=self.num_points) == 0).nonzero()[0]

            curr_model = copy.deepcopy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=in_bag),
                Subset(self.y_train, indices=in_bag),
                *args,
                **kwargs
            )

            y_hat = curr_model.predict(Subset(self.x_train, indices=out_bag))

            self.oob_pred = torch.cat((self.oob_pred, y_hat), dim=0)
            self.oob_indices.add_indices(out_bag)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        self.data_values = np.zeros(self.num_points)

        for i, indices in self.oob_indices.items():
            # Expands the label to the desired size, squeezes for regression
            oob_labels = self.y_train[i].expand((len(indices), -1)).squeeze(dim=1)
            self.data_values[i] = self.evaluate(oob_labels, self.oob_pred[indices])
        return self.data_values


class GroupingIndex(defaultdict[int, list[int]]):
    """Modified defaultdict to facilitate a groupings values and the corresponding
    position of insertion

    Parameters
    ----------
    start : int, optional
        Starting insertion position, increments after each insertion, by default 0
    """

    def __init__(self, start: int = 0):
        super(GroupingIndex, self).__init__(list)
        self.position = start

    def add_indices(self, indices: list[int]):
        for i in indices:
            self.__getitem__(i).append(self.position)
            self.position += 1
