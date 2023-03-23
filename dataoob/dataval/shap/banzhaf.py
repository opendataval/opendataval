import numpy as np
from dataoob.dataval import DataEvaluator
from dataoob.model import Model
import copy
import torch
from torch.utils.data import Subset
from sklearn.utils import check_random_state

class DataBanzhaf(DataEvaluator):
    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        samples: int = 1000,
        random_state: np.random.RandomState=None
    ):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        self.samples: int = samples
        self.random_state = check_random_state(random_state)


    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for DVRL

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
        self.sample_utility = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        # might be better to chunk these
        samples = self.random_state.binomial(1, .5, size=(self.samples, self.num_points))

        for i in range(self.samples):
            subset = samples[i].nonzero()[0]

            curr_model = copy.deepcopy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                batch_size=batch_size,
                epochs=epochs,
            )
            y_valid_hat = curr_model.predict(self.x_valid)

            curr_perf = self.evaluate(self.y_valid, y_valid_hat)
            self.sample_utility[range(self.num_points), samples[i]] += curr_perf
            self.sample_counts[range(self.num_points), samples[i]] += 1

    def evaluate_data_values(self) -> np.ndarray:
        msr = np.divide(self.sample_utility, self.sample_counts, where=self.sample_counts!=0.)
        return msr[:, 1] - msr[:, 0]




