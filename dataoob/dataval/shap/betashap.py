import numpy as np
from numpy.random import RandomState
from scipy.special import beta

from dataoob.dataval.shap.shap import ShapEvaluator


class BetaShapley(ShapEvaluator):
    """Beta Shapley implementation. Must specify alpha/beta values for beta function

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
        Shapley values are NP-hard so we resort to MCMC sampling, by default 1.01
    max_iterations : int, optional
        Max number of outer iterations of MCMC sampling, by default 100
    min_samples : int, optional
        Minimum samples before checking MCMC convergence, by default 1000
    model_name : str, optional
        Unique name of the model, caches marginal contributions, by default None
    alpha : int, optional
        Alpha parameter for beta distribution used in the weight function, by default 16
    beta : int, optional
        Beta parameter for beta distribution used in the weight function, by default 1
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
        alpha: int = 16,
        beta: int = 1,
        random_state: RandomState = None,
    ):
        super(BetaShapley, self).__init__(
            gr_threshold, max_iterations, min_samples, model_name, random_state
        )
        self.alpha, self.beta = alpha, beta  # Beta distribution parameters

    def compute_weight(self) -> np.ndarray:
        r"""Computes weights for each cardinality of training set. Uses :math:`\alpha`,
        :math:`beta` are parameters to the beta distribution

        [1] Beta Shap weight computation, :math:`j` is cardinality, Equation (3) and (5)

        .. math::
            w(j) := \frac{1}{n} * w^{(n)}(j) * \tbinom{n-1}{j-1}
            \propto \frac{Beta(j + \beta - 1, n - j + \alpha)}{Beta(\alpha, \beta)} *
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
        """Multiplies the marginal contribution with their respective weights to get
        Beta Shapley data values

        :return np.ndarray: Predicted data values/selection for every input data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)
