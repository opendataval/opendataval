import numpy as np
from dataoob.dataval import DataEvaluator
from numpy.random import RandomState
import copy
import torch
from torch.utils.data import Subset
from sklearn.utils import check_random_state
import tqdm
class DataBanzhaf(DataEvaluator):
    """Data Banzhaf implementation
    Ref. https://arxiv.org/abs/2205.15466

    :param int samples: Number of samples from training set to take, defaults to 1000
    :param RandomState random_state: Random initial state, defaults to None
    """
    def __init__(
        self,
        samples: int = 1000,
        random_state: RandomState = None
    ):
        self.samples = samples
        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for Banzhaf

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        self.sample_utility = np.zeros(shape=(self.num_points, 2))  # number data points, if it's included, utility added at index 1, if not included utility at index 0
        self.sample_counts = np.zeros(shape=(self.num_points, 2))

        return self

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes the data values using Data Banzhaf

        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs for training, defaults to 1
        :return _type_: _description_
        """
        num_subsets = self.random_state.binomial(1, .5, size=(self.samples, self.num_points))

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
        """Returns data values using the Data Banzhaf data valuator.

        :return np.ndarray: predicted data values/selection for every input data point
        """
        msr = np.divide(self.sample_utility, self.sample_counts, where=self.sample_counts != 0.)
        return msr[:, 1] - msr[:, 0]




