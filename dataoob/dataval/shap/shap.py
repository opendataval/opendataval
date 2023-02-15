import copy
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import tqdm

from dataoob.dataval import Classifier, Evaluator, Model


class ShapEvaluator(Evaluator):
    """Shap Evaluator is an abstract class for all shapley-based methods of
    computing data values. While this method is abstract, it implements much
    of the core computations for specific implementations to access. It also
    caches the marginal contributions per model.
    Ref. https://arxiv.org/abs/1904.02868
    Ref. https://arxiv.org/abs/2110.14049

    :param Model pred_model: Prediction model
    :param callable (torch.tensor, torch.tensor -> float) metric: Evaluation function
    to determine model performance
    :param float GR_threshold: Convergence threshold for the Gelman-Rubin statistic.
    Shapley values are NP-hard this is the approximation criteria
    :param int max_iterations: Max number of outer iterations of MCMC sampling, guarantees
    the training won't deadloop, defaults to 100
    """
    # marg_contrib_dict = LRUCache(size=3) TODO

    def __init__(self, pred_model: Model, metric: callable, GR_threshold: float, max_iterations=100, *args, **kwargs):
        self.pred_model = pred_model
        self.metric = metric

        self.max_iterations = max_iterations
        self.GR_threshold = GR_threshold

    @classmethod  # TODO
    def model_to_marg_contrib(cls, model: Model):
        if model not in cls.Model_To_Marg_Contribs:
            cls.Model_To_Marg_Contribs[model] = {}
        return cls.Model_To_Marg_Contribs[model]

    def train_data_values(
        self,
        batch_size: int=32,
        epochs: int=1
    ):
        """Computes the marginal contributions for Shapley values. Additionally checks
        termination conditions.

        marginal_increment_array_stack : an array of marginal increments when one data
        point (idx) is added. Average  is Shapley as we consider a random permutation.

        :param int batch_size: Baseline training batch size, defaults to 32
        :param int epochs: Number of epochs for baseline training, defaults to 1
        """
        print(f"Start: marginal contribution computation", flush=True)
        self.marginal_contrib_sum = np.zeros((self.n_points, self.n_points))
        self.marginal_contrib_count = np.zeros((self.n_points, self.n_points)) + 1e-8  # Prevents overflow
        self.marginal_increment_array_stack = np.zeros((0, self.n_points))

        GR_stat = 100  # Initial value, for MCMC terminator, converges when < GR_threshold
        iteration = 0  # Iteration wise terminator, incase MCMC goes on for too long


        while iteration < self.max_iterations and GR_stat > self.GR_threshold:
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

            GR_stat = self._compute_GR_statistics(self.marginal_increment_array_stack)
            iteration += 1  # Update terminating conditions

        self.marginal_contribution = self.marginal_contrib_sum / (self.marginal_contrib_count)
        print(f"Done: marginal contribution computation", flush=True)

    def evaluate_data_values(self, x: torch.tensor, y: torch.tensor, *args, **kwargs):
        """Multiplies the marginal contribution with their respective weights to get

        :param torch.tensor x: _description_
        :param torch.tensor y: _description_
        :return _type_: _description_
        """
        return (
            np.sum(self.marginal_contribution * self.compute_weight(*args, **kwargs), axis=1)
        )

    @abstractmethod
    def compute_weight(self, *args, **kwargs):
        return 1

    def input_data(
        self,
        x_train: torch.tensor,
        y_train: torch.tensor,
        x_valid: torch.tensor,
        y_valid: torch.tensor,
    ):
        """Stores and transforms input data for Shapley-based predictors

        :param torch.tensor x_train: Data covariates
        :param torch.tensor y_train: Data labels
        :param torch.tensor x_valid: Test+Held-out covariates
        :param torch.tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        # Additional paramters
        self.n_points = x_train.size(dim=0)



    def _calculate_marginal_contributions(self, batch_size=32, epochs: int=1, min_cardinality: int=5):
        """Computes marginal contribution through uniform MCMC sampling

        :param int batch_size: Baseline training batch size, defaults to 32
        :param int epochs: Number of epochs for baseline training, defaults to 1
        :param int min_cardinality: Minimum cardinality of a training set, defaults to 5
        :return np.array: An array of marginal increments when one data point (idx) is added.
        Average of this value is Shapley as we consider a random permutation.
        """
        # for each iteration, we use random permutation of indices sampling from uniform for our MCMC
        indices = np.random.permutation(self.n_points)
        marginal_increment = np.zeros(self.n_points) + 1e-12  # Prevents overflow
        coalition = list(indices[:min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        prev_perf = self._evaluate_model(
            self.x_train[coalition],
            self.y_train_onehot[coalition],
            batch_size=batch_size,
            epochs=epochs
        )

        for cutoff, idx in enumerate(indices[min_cardinality:], start=min_cardinality):  # TODO consider using dataloader
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_perf = self._evaluate_model(
                self.x_train[coalition],
                self.y_train_onehot[coalition],
                batch_size=batch_size,
                epochs=epochs
            )
            marginal_increment[idx] = curr_perf - prev_perf

            # When the cardinality of random set is 'n',
            self.marginal_contrib_sum[cutoff, idx] += (curr_perf - prev_perf)
            self.marginal_contrib_count[cutoff, idx] += 1

            # if a new increment is not large enough, we terminate the valuation.
            distance_to_full_score = np.abs(
                marginal_increment[idx] / (np.sum(marginal_increment))
            )

            # Update terminating conditions
            prev_perf = curr_perf
            # If updates are too small then we assume it contributes 0.
            if distance_to_full_score < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                # print(f'Among {self.n_points}, {n} samples are observed!', flush=True)
                break

        return  marginal_increment.reshape(1, -1)

    def _evaluate_model(self, x_batch: torch.tensor, y_batch: torch.tensor, batch_size=32, epochs: int=1):
        """Trains and evaluates the performance of the model

        :param torch.tensor x_batch: Data covariates
        :param torch.tensor y_batch: Data labels
        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        :return float: returns current performance of model given the batch
        """

        # Trains the model
        curr_model = copy.deepcopy(self.pred_model)
        if isinstance(curr_model, nn.Module):
            curr_model.fit(
                x_batch,
                y_batch,
                batch_size=batch_size,
                epochs=epochs,
            )
        else:
            curr_model.fit(x_batch, y_batch)


        y_valid_hat = curr_model.predict(self.x_valid)
        curr_perf = self.evaluate(self.y_valid, y_valid_hat)

        return curr_perf

    def _compute_GR_statistics(self, mem: np.array, n_chains=10):
        """Comoputes Gelman-Rubin statistic of the marginal contributions
        Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)  TODO

        :param np.array mem: _description_
        :param int n_chains: _description_, defaults to 10
        :return float: Gelman-Rubin statistic
        """
        # if len(mem) < 1000:  # Magic number
        #     return 100

        # Set up
        (N, n_to_be_valued) = mem.shape
        n_MC_sample, offset = N // n_chains, N % n_chains

        mem = mem[offset:]  # Remove burnout

        # Vector optimized
        mem_tmp = mem.reshape(n_chains, n_MC_sample, n_to_be_valued)
        mem_mean = np.mean(mem_tmp, axis=1, keepdims=True)
        s_term = np.sum( (mem_tmp - mem_mean)**2, axis=(0, 1)) / (n_chains * (n_MC_sample - 1) )
        mu_hat = np.mean(mem_tmp, axis=(0,1))

        B_term = ( n_MC_sample * np.sum((mem_mean - mu_hat) ** 2, axis=(0, 1)) / (n_chains-1) )

        GR_stats = np.sqrt(((n_MC_sample - 1) / n_MC_sample) + (B_term / (s_term * n_MC_sample)))
        return np.max(GR_stats)
