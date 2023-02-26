import copy

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Subset

from dataoob.dataval import DataEvaluator, Model


class ShapEvaluator(DataEvaluator):
    """Shap Evaluator is an abstract class for all shapley-based methods of
    computing data values. While this method is abstract, it implements much
    of the core computations for specific implementations to access. It also
    caches the marginal contributions per model.
    Ref. https://arxiv.org/abs/1904.02868
    Ref. https://arxiv.org/abs/2110.14049

    :param Model pred_model: Prediction model
    :param callable (torch.Tensor, torch.Tensor -> float) metric: Evaluation function
    to determine model performance
    :param float GR_threshold: Convergence threshold for the Gelman-Rubin statistic.
    Shapley values are NP-hard this is the approximation criteria
    :param int max_iterations: Max number of outer iterations of MCMC sampling,
    guarantees the training won't deadloop, defaults to 100
    """

    marg_contrib_dict = {}

    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        gr_threshold: float = 1.01,
        max_iterations=100,
        model_name: str = None,
        *args,
        **kwargs,
    ):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        self.max_iterations = max_iterations
        self.gr_threshold = gr_threshold

        self.model_name = model_name

    def compute_weight(self, *args, **kwargs):
        return 1.0 / self.n_points

    @staticmethod
    def marginal_cache(model_name: str, marignal_contrib: np.ndarray = None):
        if model_name and marignal_contrib is not None:
            ShapEvaluator.marg_contrib_dict[model_name] = marignal_contrib
        elif model_name:
            return ShapEvaluator.marg_contrib_dict.get(model_name)
        return None

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Computes the marginal contributions for Shapley values. Additionally checks
        termination conditions.

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
        self.marginal_count = np.zeros((self.n_points, self.n_points)) + 1e-8  #Overflow
        self.marginal_increment_array_stack = np.zeros((0, self.n_points))

        gr_stat = 100  # MCMC terminator initial value, converges when < gr_threshold
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

        self.marginal_contribution = self.marginal_contrib_sum / self.marginal_counts
        self.marginal_cache(self.model_name, self.marginal_contribution)
        print(f"Done: marginal contribution computation", flush=True)

    def evaluate_data_values(self, *args, **kwargs) -> torch.Tensor:
        """Multiplies the marginal contribution with their respective weights to get
        NOTE torch has GPU support so if we get into a situation where computing the
        gr_threshold is a bottleneck I can swap the underlying array -> Tensor

        :return torch.Tensor: Predicted data values/selection for every input data point
        """
        return torch.tensor(np.sum(
            self.marginal_contribution * self.compute_weight(*args, **kwargs), axis=1
        ))

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
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

        # Additional paramters
        self.n_points = x_train.size(dim=0)

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
            self.marginal_contrib_sum[cutoff, idx] += curr_perf - prev_perf
            self.marginal_counts[cutoff, idx] += 1

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

    def _evaluate_model(self, indices: list, batch_size=32, epochs: int = 1):
        """Trains and evaluates the performance of the model

        :param list[int] x_batch: Data covariates+labels indices
        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        :return float: returns current performance of model given the batch
        """

        # Trains the model
        curr_model = copy.deepcopy(self.pred_model)
        if isinstance(curr_model, nn.Module):
            curr_model.fit(
                Subset(self.x_train, indices=indices),
                Subset(self.y_train, indices=indices),
                batch_size=batch_size,
                epochs=epochs,
            )
        else:
            curr_model.fit(self.x_train[indices], self.y_train[indices])

        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.evaluate(self.y_valid, y_valid_hat)

        return curr_perf

    def _compute_gr_statistics(self, mem: np.array, n_chains: int=10):
        """Comoputes Gelman-Rubin statistic of the marginal contributions
        Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)

        :param np.ndarray mem: Marginal incremental stack, used to calculate values for
        the n_chains variances
        :param int n_chains: Number of chains to be made from the incremental stack,
        defaults to 10
        :return float: Gelman-Rubin statistic
        """

        # Set up
        (N, n_to_be_valued) = mem.shape
        n_MC_sample, offset = N // n_chains, N % n_chains

        mem = mem[offset:]  # Remove remainders from initial, (think burnout)

        # Vector optimized
        mem_tmp = mem.reshape(n_chains, n_MC_sample, n_to_be_valued)

        mem_mean = np.mean(mem_tmp, axis=1, keepdims=True)
        s_term = np.sum((mem_tmp - mem_mean) ** 2, axis=(0, 1)) / (
            n_chains * (n_MC_sample - 1)
        )

        mu_hat = np.mean(mem_tmp, axis=(0, 1))
        B_term = (
            n_MC_sample * np.sum((mem_mean - mu_hat) ** 2, axis=(0, 1)) / (n_chains - 1)
        )

        GR_stats = np.sqrt(
            ((n_MC_sample - 1) / n_MC_sample) + (B_term / (s_term * n_MC_sample))
        )
        return np.max(GR_stats)
