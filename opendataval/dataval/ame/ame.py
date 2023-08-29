from typing import Optional

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from scipy.stats import zscore
from sklearn.linear_model import LassoCV
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin


class AME(DataEvaluator, ModelMixin):
    """Implementation of Average Marginal Effect Data Valuation.

    References
    ----------
    .. [1] J. Lin, A. Zhang, M. Lecuyer, J. Li, A. Panda, and S. Sen,
        Measuring the Effect of Training Data on Deep Learning Predictions via
        Randomized Experiments,
        arXiv.org, 2022. Available: https://arxiv.org/abs/2206.10013.

    Parameters
    ----------
    num_models : int, optional
        Number of models to bag/aggregate, by default 1000
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self, num_models: int = 1000, random_state: Optional[RandomState] = None
    ):
        self.num_models = num_models
        self.random_state = check_random_state(random_state)

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the AME model by fitting bagging models on different proportions
        and aggregating the subsets and the performance metrics

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        subsets, performance = [], []
        for proportion in [0.2, 0.4, 0.6, 0.8]:
            sub, perf = (
                BaggingEvaluator(self.num_models, proportion, self.random_state)
                .input_model_metric(self.pred_model, self.metric)
                .input_data(self.x_train, self.y_train, self.x_valid, self.y_valid)
                .train_data_values(*args, **kwargs)
                .get_subset_perf()
            )

            subsets.append(sub)
            performance.append(perf)

        self.subsets = np.vstack(subsets)
        self.performance = np.vstack(performance).reshape(-1)

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the coefficients of the Lasso regression
        according to Lin et al.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        norm_subsets = zscore(self.subsets, axis=1)
        norm_subsets[np.isnan(norm_subsets)] = 0  # For when all elements are the same
        centered_perf = self.performance - np.mean(self.performance)

        dv_ame = LassoCV(random_state=self.random_state)
        dv_ame.fit(X=norm_subsets, y=centered_perf)
        return dv_ame.coef_


class BaggingEvaluator(DataEvaluator, ModelMixin):
    """Bagging Data Evaluator, samples data points from :math:`Bernouli(proportion)`.

    References
    ----------
    .. [1] J. Lin, A. Zhang, M. Lecuyer, J. Li, A. Panda, and S. Sen,
        Measuring the Effect of Training Data on Deep Learning Predictions via
        Randomized Experiments,
        arXiv.org, 2022. Available: https://arxiv.org/abs/2206.10013.

    Parameters
    ----------
    num_models : int, optional
        Number of models to bag/aggregate, by default 1000
    proportion : float, optional
        Proportion for bernoulli which data points are sampled, by default 1.0
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 1.0,
        random_state: Optional[RandomState] = None,
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
        """Store and transform input data for Bagging Evaluator.

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
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the Bagging model to get subsets and corresponding evaluations of
        the performance of those subsets to compute the data values

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        sample_dim = (self.num_models, self.num_points)
        self.subsets = self.random_state.binomial(1, self.proportion, size=sample_dim)
        self.performance = np.zeros((self.num_models,))

        for i in tqdm.tqdm(range(self.num_models)):
            subset = self.subsets[i].nonzero()[0]
            if not subset.any():
                continue

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                *args,
                **kwargs,
            )
            y_valid_hat = curr_model.predict(self.x_valid)

            curr_perf = self.evaluate(self.y_valid, y_valid_hat)
            self.performance[i] = curr_perf

        return self

    def evaluate_data_values(self):
        """Return data values for each training data point.

        Compute data values using the coefficients of the Lasso regression,
        as used by Lin et al. for the AME evaluator.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        norm_subsets = zscore(self.subsets, axis=1)
        norm_subsets[np.isnan(norm_subsets)] = 0
        centered_perf = self.performance - np.mean(self.performance)

        dv_ame = LassoCV(random_state=self.random_state)
        dv_ame.fit(X=norm_subsets, y=centered_perf)
        return dv_ame.coef_

    def get_subset_perf(self):
        """Return the subsets and performance, used by AME DataEvaluator."""
        return self.subsets, self.performance
