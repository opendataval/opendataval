from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.dataval.margcontrib.sampler import GrTMCSampler, Sampler


class ShapEvaluator(DataEvaluator, ModelMixin, ABC):
    """Abstract class for all semivalue-based methods of computing data values.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    .. [2]  Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Attributes
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contribution, by default uses
        TMC-Shapley with a Gelman-Rubin statistic terminator. Samplers are found in
        :py:mod:`~opendataval.margcontrib.sampler`

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        opendataval/margcontrib/sampler.py, by default GrTMCSampler and uses additonal
        arguments as constructor for sampler.
    gr_threshold : float, optional
        Convergence threshold for the Gelman-Rubin statistic.
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.05
    max_mc_epochs : int, optional
        Max number of outer epochs of MCMC sampling, by default 100
    models_per_iteration : int, optional
        Number of model fittings to take per iteration prior to checking GR convergence,
        by default 100
    mc_epochs : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, sampler: Sampler = None, *args, **kwargs):
        self.sampler = sampler

        if self.sampler is None:
            self.sampler = GrTMCSampler(*args, **kwargs)

    @abstractmethod
    def compute_weight(self) -> np.ndarray:
        """Compute the weights for each cardinality of training set."""

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Multiplies the marginal contribution with their respective weights to get
        data values for semivalue-based estimators

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every input data point
        """
        return np.sum(self.marg_contrib * self.compute_weight(), axis=1)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for semi-value samplers.

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

        # Sampler specific setup
        self.num_points = len(x_train)
        self.sampler.set_coalition(x_train)
        self.sampler.set_evaluator(self._evaluate_model)

        return self

    def train_data_values(self, *args, **kwargs):
        """Uses sampler to trains model to find marginal contribs and data values."""
        self.marg_contrib = self.sampler.compute_marginal_contribution(*args, **kwargs)
        return self

    def _evaluate_model(self, subset: list[int], *args, **kwargs):
        """Evaluate performance of the model on a subset of the training data set.

        Parameters
        ----------
        subset : list[int]
            indices of covariates/label to be used in training
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        float
            Performance of subset of training data set
        """
        curr_model = self.pred_model.clone()
        curr_model.fit(
            Subset(self.x_train, indices=subset),
            Subset(self.y_train, indices=subset),
            *args,
            **kwargs,
        )
        y_valid_hat = curr_model.predict(self.x_valid)

        curr_perf = self.evaluate(self.y_valid, y_valid_hat)
        return curr_perf
