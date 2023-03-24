import numpy as np
from numpy.random import RandomState

from dataoob.dataval.shap.shap import ShapEvaluator


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
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.01
    max_iterations : int, optional
        Max number of outer iterations of MCMC sampling, by default 100
    min_samples : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    model_name : str, optional
        Unique name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
        random_state: RandomState = None,
    ):
        super(DataShapley, self).__init__(
            gr_threshold, max_iterations, min_samples, model_name, random_state
        )

    def compute_weight(self) -> float:
        """Computes weights (uniform) for each cardinality of training set.
        Shapley values take a simple average of the marginal contributions across
        all different cardinalities.

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """
        return 1 / self.num_points

    def evaluate_data_values(self) -> np.ndarray:
        """Multiplies the marginal contribution with their respective weights to get
        Data Shapley data values

        :return np.ndarray: Predicted data values/selection for every input data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)
