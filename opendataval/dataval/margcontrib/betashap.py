import numpy as np
from numpy.random import RandomState
from scipy.special import beta

from opendataval.dataval.margcontrib.shap import ShapEvaluator


class BetaShapley(ShapEvaluator):
    """Beta Shapley implementation. Must specify alpha/beta values for beta function.

    References
    ----------
    .. [1] Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
    gr_threshold : float, optional
        Convergence threshold for the Gelman-Rubin statistic.
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.05
    max_mc_epochs : int, optional
        Max number of outer iterations of MCMC sampling, by default 100
    models_per_iteration : int, optional
        Number of model fittings to take per iteration prior to checking GR convergence,
        by default 100
    mc_epochs : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    cache_name : str, optional
        Unique cache_name of the model, caches marginal contributions, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    alpha : int, optional
        Alpha parameter for beta distribution used in the weight function, by default 4
    beta : int, optional
        Beta parameter for beta distribution used in the weight function, by default 1
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        gr_threshold: float = 1.05,
        max_mc_epochs: int = 100,
        models_per_iteration: int = 100,
        mc_epochs: int = 1000,
        cache_name: str = None,
        alpha: int = 4,
        beta: int = 1,
        random_state: RandomState = None,
    ):
        super().__init__(
            gr_threshold=gr_threshold,
            max_mc_epochs=max_mc_epochs,
            models_per_iteration=models_per_iteration,
            mc_epochs=mc_epochs,
            cache_name=cache_name,
            random_state=random_state,
        )
        self.alpha, self.beta = alpha, beta  # Beta distribution parameters

    def compute_weight(self) -> np.ndarray:
        r"""Compute weights for each cardinality of training set.

        Uses :math:`\alpha`, :math:`beta` are parameters to the beta distribution.
        [1] BetaShap weight computation, :math:`j` is cardinality, Equation (3) and (5).

        .. math::
            w(j) := \frac{1}{n} w^{(n)}(j) \tbinom{n-1}{j-1}
            \propto \frac{Beta(j + \beta - 1, n - j + \alpha)}{Beta(\alpha, \beta)}
            \tbinom{n-1}{j-1}

        References
        ----------
        .. [1] Y. Kwon and J. Zou,
            Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
            Machine Learning,
            arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """
        weight_list = [
            beta(j + self.beta, self.num_points - (j + 1) + self.alpha)
            / beta(j + 1, self.num_points - j)
            for j in range(self.num_points)
        ]

        return np.array(weight_list) / np.sum(weight_list)

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Multiplies the marginal contribution with their respective weights to get
        Beta Shapley data values.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)
