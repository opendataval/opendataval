import math
from collections import Counter, defaultdict
from typing import Optional, Sequence

import numpy as np
import torch

from opendataval.dataval.api import DataEvaluator, EmbeddingMixin
from opendataval.dataval.margcontrib import Sampler, TMCSampler


class RobustVolumeShapley(DataEvaluator, EmbeddingMixin):
    """Robust Volume Shapley and Volume Shapley data valuation implementation.

    While the following DataEvaluator uses the same TMC-Shapley algorithm used by
    semivalue evaluators, the following implementation defaults to the non-GR statistic
    implementation. Instead a fixed number of samples is taken, which is
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
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.GrTMCSampler`.
    robust : bool, optional
        If the robust volume measure will be used which trades off a "more refined
        representation of diversity for greater robustness to replication",
        by default True
    omega : Optional[float], optional
        Width/discretization coefficient for x_train to be split into a set of d-cubes,
        required if `robust` is True, by default 0.05

    Mixins
    ------
    EmbeddingMixin
        Mixin for a data evaluator to use an embedding model.
    """

    def __init__(
        self,
        sampler: Sampler = None,
        robust: bool = True,
        omega: Optional[float] = None,
        *args,
        **kwargs
    ):
        self.sampler = sampler
        self.robust = robust
        self.omega = omega if robust and omega is not None else 0.05

        if sampler is None:
            self.sampler = TMCSampler(*args, **kwargs)

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

        # Sampler parameters
        self.num_points = len(self.x_train)
        self.sampler.set_coalition(x_train)
        self.sampler.set_evaluator(self._evaluate_volume)
        return self

    def train_data_values(self, *args, **kwargs):
        self.marg_contrib = self.sampler.compute_marginal_contribution(*args, **kwargs)
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return np.sum(self.marg_contrib / self.num_points, axis=1)

    def _evaluate_volume(self, subset: Sequence[int]):
        x_train = self.x_train[subset]  # TODO PyTorch Subsets
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
