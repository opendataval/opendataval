import numpy as np
from sklearn.utils import check_random_state

from sklearn.linear_model import LassoCV, LinearRegression
import torch
import copy
from numpy.random import RandomState
from dataoob.dataval import DataEvaluator
from torch.utils.data import Subset
from dataoob.dataval import DataEvaluator
import tqdm

from scipy.stats import zscore


class AME(DataEvaluator):
    def __init__(self, num_models: int = 10, random_state: RandomState = None):
        self.num_models = num_models
        self.random_state = check_random_state(random_state)

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        subsets, performance = [], []
        for proportion in [0.2, 0.4, 0.6, 0.8]:
            sub, perf = (
                BaggingEvaluator(self.num_models, proportion, self.random_state)
                .input_model_metric(self.pred_model, self.metric)
                .input_data(self.x_train, self.y_train, self.x_valid, self.y_valid)
                .train_data_values(batch_size, epochs)
                .get_subset_perf()
            )

            subsets.append(sub)
            performance.append(perf)

        self.subsets = np.vstack(subsets)
        self.performance = np.vstack(performance).reshape(-1)

        return self

    def evaluate_data_values(self):
        standard_subsets = zscore(self.subsets, axis=1)
        centered_perf = self.performance - np.mean(self.performance)

        dv_ame = LinearRegression(fit_intercept=False)
        dv_ame.fit(X=standard_subsets, y=centered_perf)
        return dv_ame.coef_


class BaggingEvaluator(DataEvaluator):
    """_summary_

    :param int num_models: _description_, defaults to 10
    :param float proportion: _description_, defaults to 1.0
    :param RandomState random_state: _description_, defaults to None
    """
    def __init__(
        self,
        num_models: int = 10,
        proportion: float = 1.0,
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
        """Stores and transforms input data for Bagging Evaluator

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
        return self

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        self.num_subsets = self.random_state.binomial(1, self.proportion, size=(self.num_models, self.num_points))
        self.performance = np.zeros((self.num_models,))

        for i in tqdm.tqdm(range(self.num_models)):
            subset = self.num_subsets[i].nonzero()[0]

            curr_model = copy.deepcopy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                batch_size=batch_size,
                epochs=epochs,
            )
            y_valid_hat = curr_model.predict(self.x_valid)

            curr_perf = self.evaluate(self.y_valid, y_valid_hat)
            self.performance[i] = curr_perf

        return self

    def evaluate_data_values(self):
        """
        With the held-out data (X_val, y_val), the performance of a model trained on a bootstrapped dataset is evaluated
        """
        standard_subsets = zscore(self.subsets, axis=1)
        standard_perf = self.performance - np.mean(self.performance)

        dv_ame = LassoCV()
        dv_ame.fit(X=standard_subsets, y=standard_perf)
        return dv_ame.coef_

    def get_subset_perf(self):
        return self.num_subsets, self.performance
