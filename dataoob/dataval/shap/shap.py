import copy
from abc import abstractmethod, ABC
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, Subset

from dataoob.dataval import DataEvaluator
from dataoob.model import Model


class ShapEvaluator(DataEvaluator, ABC):
    """ShapEvaluator is an abstract class for all shapley-based methods of
    computing data values. Implements core computations of marginal contribution.
    Ref. https://arxiv.org/abs/1904.02868
    Ref. https://arxiv.org/abs/2110.14049 for TMC

    :param Model pred_model: Prediction model
    :param callable (torch.Tensor, torch.Tensor -> float) metric: Evaluation function
    to determine model performance
    :param float GR_threshold: Convergence threshold for the Gelman-Rubin statistic.
    Shapley values are NP-hard this is the approximation criteria
    :param int max_iterations: Max number of outer iterations of MCMC sampling,
    guarantees the training won't deadloop, defaults to 100
    :param int min_samples: Minimum samples before checking MCMC convergence
    """

    marg_contrib_dict = {}
    GR_MAX = 100

    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
    ):
        self.pred_model = copy.copy(pred_model)
        self.metric = metric

        self.max_iterations = max_iterations
        self.gr_threshold = gr_threshold
        self.min_samples = min_samples

        self.model_name = model_name

    @abstractmethod
    def compute_weight(self):
        """Computes the weights applied to the marginal contributions"""
        pass

    def evaluate_data_values(self) -> torch.Tensor:
        """Multiplies the marginal contribution with their respective weights to get
        data values for semivalue-based estimators

        :return np.ndarray: Predicted data values/selection for every input data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)

    @staticmethod
    def marginal_cache(model_name: str, marginal_contrib: np.ndarray = None):
        if model_name and marginal_contrib is not None:
            ShapEvaluator.marg_contrib_dict[model_name] = marginal_contrib
        elif model_name:
            return ShapEvaluator.marg_contrib_dict.get(model_name)
        return None


    def input_data(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        x_valid: torch.Tensor | Dataset,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for Shapley-based predictors

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        # Additional parameters
        self.n_points = len(x_train)


    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes the marginal contributions for Shapley values. Additionally checks
        termination conditions.

        TODO consider updating with Prof's Beta shapley algorithm, efficient computation,
        pros: more efficient, cons: may affect further sampling, redundant, consider cache

        marginal_increment_array_stack np.ndarray: Marginal increments when one data
        point is added. Average is Shapley as we consider a random permutation.

        :param int batch_size: Baseline training batch size, defaults to 32
        :param int epochs: Number of epochs for baseline training, defaults to 1
        """
        # Checks cache if model name has been computed prior
        if self.marginal_cache(self.model_name) is not None:
            self.marginal_contribution = self.marginal_cache(self.model_name)
            return

        print(f"Start: marginal contribution computation", flush=True)
        self.marginal_contrib_sum = np.zeros((self.n_points, self.n_points))
        self.marginal_count = np.zeros((self.n_points, self.n_points)) + 1e-8  # Overflow
        self.marginal_increment_array_stack = np.zeros((0, self.n_points))

        gr_stat = ShapEvaluator.GR_MAX  # Converges when < gr_threshold
        iteration = 0  # Iteration wise terminator, in case MCMC goes on for too long

        while iteration < self.max_iterations and gr_stat > self.gr_threshold:
            # we check the convergence every 100 random sets.
            # we terminate iteration if Shapley value is converged.

            for _ in tqdm.tqdm(range(100)):
                marginal_increment_array = self._calculate_marginal_contributions(
                    batch_size=batch_size, epochs=epochs
                )
                self.marginal_increment_array_stack = np.concatenate(
                    [self.marginal_increment_array_stack, marginal_increment_array],
                    axis=0,
                )

            gr_stat = self._compute_gr_statistics(self.marginal_increment_array_stack)
            iteration += 1  # Update terminating conditions

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_count
        self.marginal_cache(self.model_name, self.marginal_contribution)
        print(f"Done: marginal contribution computation", flush=True)


    def _calculate_marginal_contributions(
        self, batch_size=32, epochs: int = 1, min_cardinality: int = 5
    ):
        """Computes marginal contribution through TMC-Shapley algorithm

        :param int batch_size: Baseline training batch size, defaults to 32
        :param int epochs: Number of epochs for baseline training, defaults to 1
        :param int min_cardinality: Minimum cardinality of a training set, defaults to 5
        :return np.ndarray: An array of marginal increments when one data point is added.
        Average of this value is Shapley as we consider a random permutation.
        """
        # for each iteration, we use random permutation for our MCMC
        indices = np.random.permutation(self.n_points)
        marginal_increment = np.zeros(self.n_points) + 1e-12  # Prevents overflow
        coalition = list(indices[:min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = curr_perf = self._evaluate_model(coalition, batch_size, epochs)

        for cutoff, idx in enumerate(indices[min_cardinality:], start=min_cardinality):
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self._evaluate_model(coalition, batch_size, epochs)
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[cutoff, idx] += (curr_perf - prev_perf)
            self.marginal_count[cutoff, idx] += 1

            # if a new increment is not large enough, we terminate the valuation.
            distance = np.abs(curr_perf - prev_perf) / np.sum(marginal_increment)

            # Update terminating conditions
            prev_perf = curr_perf
            # If updates are too small then we assume it contributes 0.
            if distance < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                # print(f'Among {self.n_points}, {n} samples are observed!', flush=True)
                break

        return marginal_increment.reshape(1, -1)

    def _evaluate_model(self, indices: list[int], batch_size: int = 32, epochs: int = 1):
        """Trains and evaluates the performance of the model

        :param list[int] x_batch: Data covariates+labels indices
        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        :return float: returns current performance of model given the batch
        """

        # Trains the model
        curr_model = copy.copy(self.pred_model)
        curr_model.fit(
            Subset(self.x_train, indices=indices),
            Subset(self.y_train, indices=indices),
            batch_size=batch_size,
            epochs=epochs,
        )

        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.evaluate(self.y_valid, y_valid_hat)

        return curr_perf

    def _compute_gr_statistics(self, samples: np.ndarray, num_chains: int=10):
        """Computes Gelman-Rubin statistic of the marginal contributions
        Ref. https://arxiv.org/pdf/1812.09384

        :param np.ndarray mem: Marginal incremental stack, used to calculate values for
        the num_chains variances
        :param int num_chains: Number of chains to be made from the incremental stack,
        defaults to 10
        :return float: Gelman-Rubin statistic
        """

        if len(samples) < self.min_samples:
            return ShapEvaluator.GR_MAX  # If not enough samples, return a high GR value

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
            (num_samples_per_chain - 1) / num_samples_per_chain +
            (b_term / (s_term * num_samples_per_chain))
        )  # Ref. https://arxiv.org/pdf/1812.09384 (p.7, Eq.4)
        return np.max(gr_stats)
