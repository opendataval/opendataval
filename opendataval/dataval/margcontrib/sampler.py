from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional, TypeVar

import numpy as np
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state

Self = TypeVar("Self")


class Sampler(ABC):
    def set_evaluator(self, util_func: Callable[[list[int], ...], float]):
        """Given a list of indices, finds the  contribution of the coalition.

        This function sets the utility function for computing the utility of each
        coalition, different samplers can use this utility function differently.
        The following is an example of how the api would work:
        ::
            self.sampler.set_evaluator(self._evaluate_model)
        """
        self.compute_marg_contrib = util_func

    @abstractmethod
    def train_data_values(self) -> Self:
        """Computes the marginal contribution of each data point."""

    @abstractmethod
    def set_num_points(self, num_points: int) -> Self:
        """Initializes storage to find marginal contribution of each data point."""

    @abstractmethod
    def marg_contrib(self) -> np.ndarray:
        """Gets marginal contribution of each data point."""


class MonteCarloSampler(Sampler):
    """Monte Carlo sampler for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler. We take
    mc_epochs permutations and compute the marginal contributions. Simplest
    implementation but the least practical.

    Parameters
    ----------
    mc_epochs : int, optional
        Number of outer epochs of MCMC sampling, by default 1000
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 1000,
        cache_name: Optional[str] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.cache_name = cache_name or id(self)  # Unique identifier
        self.random_state = check_random_state(random_state)

    def set_num_points(self, num_points: int):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = num_points
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8

        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Uses TMC-Shapley sampling to find the marginal contribution of each data point,
        takes self.mc_epochs number of samples.
        """
        if (marg_contrib := TMCSampler.CACHE.get(self.cache_name)) is not None:
            self.data_values = marg_contrib
            return self

        for _ in tqdm.trange(self.mc_epochs):
            self._calculate_marginal_contributions(*args, **kwargs)

        self.data_values = self.marginal_contrib_sum / self.marginal_count
        MonteCarloSampler.CACHE[self.cache_name] = self.data_values
        return self

    def marg_contrib(self):
        if not hasattr(self, "data_values"):
            raise Exception("Sampler must be trained to find marginal contribution.")
        return self.data_values

    def _calculate_marginal_contributions(self, *args, **kwargs):
        """Compute marginal contribution through MC sampling.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        coalition = []
        curr_perf = prev_perf = 0

        for cutoff, idx in enumerate(subset):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_marg_contrib(coalition, *args, **kwargs)

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1
        return


class TMCSampler(Sampler):
    """TMCShapley sampler for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler.

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    mc_epochs : int, optional
        Number of outer epochs of MCMC sampling, by default 1000
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 1000,
        min_cardinality: int = 5,
        cache_name: Optional[str] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.cache_name = cache_name or id(self)  # Unique identifier
        self.random_state = check_random_state(random_state)

    def set_num_points(self, num_points: int):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = num_points
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8

        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Uses TMC-Shapley sampling to find the marginal contribution of each data point,
        takes self.mc_epochs number of samples.
        """
        if (marg_contrib := TMCSampler.CACHE.get(self.cache_name)) is not None:
            self.data_values = marg_contrib
            return self

        for _ in tqdm.trange(self.mc_epochs):
            self._calculate_marginal_contributions(*args, **kwargs)

        self.data_values = self.marginal_contrib_sum / self.marginal_count
        TMCSampler.CACHE[self.cache_name] = self.data_values
        return self

    def marg_contrib(self):
        if not hasattr(self, "data_values"):
            raise Exception("Sampler must be trained to find marginal contribution.")
        return self.data_values

    def _calculate_marginal_contributions(self, *args, **kwargs):
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        coalition = list(subset[: self.min_cardinality])
        marginal_increment = 1e-8
        truncation_counter = 0

        # Baseline at minimal cardinality
        curr_perf = self.compute_marg_contrib(coalition, *args, **kwargs)
        prev_perf = curr_perf

        for cutoff, idx in enumerate(
            subset[self.min_cardinality :], start=self.min_cardinality
        ):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_marg_contrib(coalition, *args, **kwargs)

            # When the cardinality of random set is 'n',
            marginal_increment += curr_perf - prev_perf
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1

            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_perf - prev_perf) / marginal_increment < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                break
        return


class GrTMCSampler(Sampler):
    """TMC Sampler with terminator for semivalue-based methods of computing data values.

    Evaluators that share marginal contributions should share a sampler.

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
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    GR_MAX = 100
    """Default maximum Gelman-Rubin statistic. Used for burn-in."""

    def __init__(
        self,
        gr_threshold: float = 1.05,
        max_mc_epochs: int = 100,
        models_per_iteration: int = 100,
        mc_epochs: int = 1000,
        min_cardinality: int = 5,
        cache_name: Optional[str] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.max_mc_epochs = max_mc_epochs
        self.gr_threshold = gr_threshold
        self.models_per_iteration = models_per_iteration
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality

        self.cache_name = cache_name or id(self)  # Unique identifier
        self.random_state = check_random_state(random_state)

    def set_num_points(self, num_points: int):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = num_points
        self.marginal_contrib_sum = np.zeros((self.num_points, self.num_points))
        self.marginal_count = np.zeros((self.num_points, self.num_points)) + 1e-8
        # Used for computing the GR-statistic
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
        if (marg_contrib := GrTMCSampler.CACHE.get(self.cache_name)) is not None:
            self.marginal_contribution = marg_contrib
            return self

        print("Start: marginal contribution computation", flush=True)

        gr_stat = GrTMCSampler.GR_MAX  # Converges when < gr_threshold
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
        GrTMCSampler.CACHE[self.cache_name] = self.marginal_contribution
        print("Done: marginal contribution computation", flush=True)

        return self

    def marg_contrib(self):  # TODO consider caching this outright
        if not hasattr(self, "marginal_contribution"):
            raise Exception("Sampler must be trained to find marginal contribution.")
        return self.marginal_contribution

    def _calculate_marginal_contributions(self, *args, **kwargs) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        marginal_increment = np.zeros(self.num_points) + 1e-8  # Prevents overflow
        coalition = list(subset[: self.min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self.compute_marg_contrib(coalition, *args, **kwargs)

        for cutoff, idx in enumerate(
            subset[self.min_cardinality :], start=self.min_cardinality
        ):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self.compute_marg_contrib(coalition, *args, **kwargs)
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[idx, cutoff] += curr_perf - prev_perf
            self.marginal_count[idx, cutoff] += 1

            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_perf - prev_perf) / np.sum(marginal_increment) < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                break

            # update performance
            prev_perf = curr_perf

        return marginal_increment.reshape(1, -1)

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
            return GrTMCSampler.GR_MAX  # If not burn-in, returns a high GR value

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
