import numpy as np
from numpy.random import RandomState

from dataoob.dataval.margcontrib.shap import ShapEvaluator


class DataShapley(ShapEvaluator):
    """Data Shapley implementation.

    References
    ----------
    .. [1] A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    gr_threshold : float, optional
        Convergence threshold for the Gelman-Rubin statistic.
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.05
    max_iterations : int, optional
        Max number of outer iterations of MCMC sampling, by default 100
    samples_per_iteration : int, optional
        Number of samples to take per iteration prior to checking GR convergence,
        by default 100
    min_samples : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        gr_threshold: float = 1.05,
        max_iterations: int = 100,
        samples_per_iteration: int = 100,
        min_samples: int = 1000,
        cache_name: str = None,
        random_state: RandomState = None,
    ):
        super().__init__(
            gr_threshold=gr_threshold,
            max_iterations=max_iterations,
            samples_per_iteration=samples_per_iteration,
            min_samples=min_samples,
            cache_name=cache_name,
            random_state=random_state,
        )

    def compute_weight(self) -> float:
        """Compute weights (uniform) for each cardinality of training set.

        Shapley values take a simple average of the marginal contributions across
        all different cardinalities.

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """
        return 1 / self.num_points

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Multiplies the marginal contribution with their respective weights to get
        Data Shapley data values

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)