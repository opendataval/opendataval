import numpy as np
from scipy.special import beta

from opendataval.dataval.margcontrib.shap import Sampler, ShapEvaluator


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
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.GrTMCSampler`.
    alpha : int, optional
        Alpha parameter for beta distribution used in the weight function, by default 4
    beta : int, optional
        Beta parameter for beta distribution used in the weight function, by default 1
    """

    def __init__(
        self, sampler: Sampler = None, alpha: int = 4, beta: int = 1, *args, **kwargs
    ):
        super().__init__(sampler=sampler, *args, **kwargs)
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
