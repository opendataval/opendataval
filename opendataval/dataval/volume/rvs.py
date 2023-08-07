import math
from collections import Counter, defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator, ModelLessMixin


class RobustVolumeShapley(DataEvaluator, ModelLessMixin):
    """Robust Volume Shapley and Volume Shapley data valuation implementation.

    While the following DataEvaluator uses the same TMC-Shapley algorithm used by
    semivalue evaluators, the following implementation does not utilize the GR statistic
    to check for convergence. Instead a fixed number of samples is taken, which is
    closer to the original implementation here:
    https://github.com/ZhaoxuanWu/VolumeBased-DataValuation/tree/main

    References
    ----------
    .. [1] X. Xu, Z. Wu, C. S. Foo, and B. Kian,
        Validation Free and Replication Robust Volume-based Data Valuation,
        Advances in Neural Information Processing Systems,
        vol. 34, pp. 10837-10848, Dec. 2021.

    Parameters
    ----------
    mc_epochs : int, optional
        Number of samples from TMC-Shapley, the total number of iterations will equal
        len(x_train) * mc_epochs, by default 1000.
    robust : bool, optional
        If the robust volume measure will be used which trades off a "more refined
        representation of diversity for greater robustness to replication",
        by default True
    omega : Optional[float], optional
        Width/discretization coefficient for x_train to be split into a set of d-cubes,
        required if `robust` is True, by default 0.05
    random_state : Optional[RandomState], optional
        Random initial state, by default None

    Mixins
    ------
    ModelLessMixin
        Mixin for a data evaluator that doesn't require a model or evaluation metric.
    """

    def __init__(
        self,
        mc_epochs: int = 1000,
        robust: bool = True,
        omega: Optional[float] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.robust = robust
        self.omega = omega if robust and omega is not None else 0.05

        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for volume-based evaluators.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates, unused parameter
        y_valid : torch.Tensor
            Test+Held-out labels, unused parameter
        """
        self.x_train, _ = self.embeddings(x_train, x_valid)
        self.y_train, _ = y_train, y_valid

        # Additional parameters
        self.num_points = len(x_train)
        self.marginal_contrib = np.zeros((self.num_points,))
        self.marginal_contrib_sum = 0.0
        self.marginal_count = np.zeros((self.num_points,)) + 1e-8

        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Uses TMC-Shapley sampling to find the marginal contribution to volume of each
        data point, takes self.mc_epochs number of samples.
        """
        for _ in tqdm.trange(self.mc_epochs):
            self._calculate_marginal_volume()

        self.data_values = self.marginal_contrib / self.marginal_count
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/marginal contribution for every training data point
        """
        return self.data_values.flatten()

    def _calculate_marginal_volume(self, min_cardinality: int = 5):
        """Compute marginal volume through TMC-Shapley algorithm.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        min_cardinality : int, optional
            Minimum cardinality of a training set, must be passed as kwarg, by default 5
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # for each iteration, we use random permutation for our MCMC
        subset = self.random_state.permutation(self.num_points)
        coalition = list(subset[:min_cardinality])
        truncation_counter = 0

        # Baseline at minimal cardinality
        curr_vol = self._evaluate_volume(coalition)
        prev_vol = curr_vol

        for idx in subset[min_cardinality:]:
            # Increment the batch_size and evaluate the change compared to prev model
            coalition.append(idx)
            curr_vol = self._evaluate_volume(coalition)
            marginal = curr_vol - prev_vol
            prev_vol = curr_vol

            self.marginal_contrib[idx] += marginal
            self.marginal_contrib_sum += marginal
            self.marginal_count[idx] += 1

            # If a new increment is not large enough, we terminate the valuation.
            # If updates are too small then we assume it contributes 0.
            if abs(curr_vol - prev_vol) / self.marginal_contrib_sum < 1e-8:
                truncation_counter += 1
            else:
                truncation_counter = 0

            if truncation_counter == 10:  # If enter space without changes to model
                break
        return

    def _evaluate_volume(self, subset: Sequence[int]):
        x_train = self.x_train[subset]  # potential BUG with PyTorch Subsets
        if self.robust:
            x_tilde, cubes = compute_x_tilde_and_counts(x_train, self.omega)
            return compute_robust_volumes(x_tilde, cubes)
        else:
            return torch.sqrt(torch.linalg.det(x_train.T @ x_train).abs() + 1e-8)


def compute_x_tilde_and_counts(x: torch.Tensor, omega: float):
    """Compresses the original feature matrix x to x_tilde with the specified omega.

    Returns
    -------
    np.ndarray
        Compressed form of x as a d-cube
    dict[tuple, int]
        A dictionary of cubes with the respective counts in each dcube
    """
    assert 0 <= omega <= 1.0, "`omega` must be in range [0, 1]"
    cubes = Counter()  # a dictionary to store the freqs
    omega_dict = defaultdict(list)
    min_ds = torch.min(x, axis=0).values

    # a dictionary to store cubes of not full size
    for entry in x:
        cube_key = tuple(math.floor(ent.item() / omega) for ent in entry - min_ds)
        cubes[cube_key] += 1
        omega_dict[cube_key].append(entry)

    x_tilde = torch.stack([torch.stack(value).mean(0) for value in omega_dict.values()])
    return x_tilde, cubes


def compute_robust_volumes(x_tilde: torch.Tensor, hypercubes: dict[tuple, int]):
    alpha = 1.0 / (10 * len(x_tilde))  # it means we set beta = 10

    flat_data = x_tilde.reshape(-1, x_tilde.shape[1])
    volume = torch.sqrt(torch.linalg.det(flat_data.T @ flat_data).abs() + 1e-8)
    rho_omega_prod = 1.0

    for freq_count in hypercubes.values():
        rho_omega_prod *= (1 - alpha ** (freq_count + 1)) / (1 - alpha)

    return volume * rho_omega_prod
