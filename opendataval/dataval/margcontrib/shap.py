from abc import ABC, abstractmethod

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator


class ShapEvaluator(DataEvaluator, ABC):
    """Abstract class for all Shapley-based methods of computing data values.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    .. [2]  Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
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
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE = {}
    """Cached marginal contributions."""
    GR_MAX = 100
    """Default maximum Gelman-Rubin statistic. Used for burn-in."""

    def __init__(
        self,
        gr_threshold: float = 1.05,
        max_mc_epochs: int = 100,
        models_per_iteration: int = 100,
        mc_epochs: int = 1000,
        cache_name: str = None,
        random_state: RandomState = None,
    ):
        self.max_mc_epochs = max_mc_epochs
        self.gr_threshold = gr_threshold
        self.models_per_iteration = models_per_iteration
        self.mc_epochs = mc_epochs

        self.cache_name = cache_name

        self.random_state = check_random_state(random_state)

    @abstractmethod
    def compute_weight(self):
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
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Shapley-based predictors.

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

        # Additional parameters
        self.num_points = len(x_train)
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8
        self.marginal_increment_array_stack = np.zeros((0, self.num_points))

        return self

    def train_data_values(self, *args, **kwargs):
        """Compute the marginal contributions for semivalue based data evaluators.

        Computes the marginal contribution by sampling.
        Checks MCMC convergence every 100 iterations using Gelman-Rubin Statistic.
        NOTE if the marginal contribution has not been calculated, will look it up in
        a cache of already trained ShapEvaluators, otherwise will train from scratch.

        Parameters
        ----------
        args : tuple[Any], optional
             Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Notes
        -----
        marginal_increment_array_stack : np.ndarray
            Marginal increments when one data point is added.
        """
        # Checks cache if model name has been computed prior
        if (marg_contrib := ShapEvaluator.CACHE.get(self.cache_name)) is not None:
            self.marginal_contribution = marg_contrib
            return self

        print("Start: marginal contribution computation", flush=True)

        gr_stat = ShapEvaluator.GR_MAX  # Converges when < gr_threshold
        iteration = 0  # Iteration wise terminator, in case MCMC goes on for too long

        while iteration < self.max_mc_epochs and gr_stat > self.gr_threshold:
            # we check the convergence every 100 random samples.
            # we terminate iteration if Shapley value is converged.
            samples_array = [
                self._calculate_marginal_contributions(*args, **kwargs)
                for _ in tqdm.tqdm(range(self.models_per_iteration))
            ]
            self.marginal_increment_array_stack = np.vstack(
                [self.marginal_increment_array_stack, *samples_array],
            )

            gr_stat = self._compute_gr_statistic(self.marginal_increment_array_stack)
            iteration += 1  # Update terminating conditions
            print(f"{gr_stat=}")

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count
        ShapEvaluator.CACHE[self.cache_name] = self.marginal_contribution
        print("Done: marginal contribution computation", flush=True)

        return self

    def _calculate_marginal_contributions(
        self, *args, min_cardinality: int = 5, **kwargs
    ) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        min_cardinality : int, optional
            Minimum cardinality of a training set, must be passed as kwarg, by default 5
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-12  # Prevents overflow
        coalition = list(subset[:min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self._evaluate_model(coalition, *args, **kwargs)

        for cutoff, idx in enumerate(subset[min_cardinality:], start=min_cardinality):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self._evaluate_model(coalition, *args, **kwargs)
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1

            # if a new increment is not large enough, we terminate the valuation.
            distance = abs(curr_perf - prev_perf) / np.sum(marginal_increment)

            # update prev_perf
            prev_perf = curr_perf

            # If updates are too small then we assume it contributes 0.
            if distance < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                break

        return marginal_increment.reshape(1, -1)

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

    def _compute_gr_statistic(self, samples: np.ndarray, num_chains: int = 10) -> float:
        """Compute Gelman-Rubin statistic of the marginal contributions.

        References
        ----------
        .. [1] Y. Kwon and J. Zou,
            Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
            Machine Learning,
            arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

        .. [2] D. Vats and C. Knudson,
            Revisiting the Gelman-Rubin Diagnostic,
            arXiv.org, 2018. Available: https://arxiv.org/abs/1812.09384.

        Parameters
        ----------
        samples : np.ndarray
            Marginal incremental stack, used to find values for the num_chains variances
        num_chains : int, optional
            Number of chains to be made from the incremental stack, by default 10

        Returns
        -------
        float
            Gelman-Rubin statistic
        """
        if len(samples) < self.mc_epochs:
            return ShapEvaluator.GR_MAX  # If not burn-in, returns a high GR value

        # Set up
        num_samples, num_datapoints = samples.shape
        num_samples_per_chain, offset = divmod(num_samples, num_chains)
        samples = samples[offset:]  # Remove remainders from initial

        # Divides total sample into num_chains parallel chains
        mcmc_chains = samples.reshape(num_chains, num_samples_per_chain, num_datapoints)

        # Computes the average of the intra-chain sample variances
        s_term = np.mean(np.var(mcmc_chains, axis=1, ddof=1), axis=0)

        # Computes the variance of the sample_means of the chain
        sampling_mean = np.mean(mcmc_chains, axis=1, keepdims=False)
        b_term = num_samples_per_chain * np.var(sampling_mean, axis=0, ddof=1)

        gr_stats = np.sqrt(
            (num_samples_per_chain - 1) / num_samples_per_chain
            + (b_term / (s_term * num_samples_per_chain))
        )  # Ref. https://arxiv.org/pdf/1812.09384 (p.7, Eq.4)
        return np.max(gr_stats)
