import torch
import numpy as np
from scipy.special import beta

from dataoob.dataval.shap.shap import ShapEvaluator
from dataoob.model import Model

class BetaShapley(ShapEvaluator):
    """Beta Shapley implementation. Must specify alpha/beta values for beta function
    Ref. https://arxiv.org/pdf/2110.14049

    :param Model pred_model: Prediction model
    :param callable (torch.Tensor, torch.Tensor -> float) metric: Evaluation function
    to determine model performance
    :param float GR_threshold: Convergence threshold for the Gelman-Rubin statistic.
    Shapley values are NP-hard this is the approximation criteria
    :param int max_iterations: Max number of outer iterations of MCMC sampling,
    guarantees the training won't deadloop, defaults to 100
    :param int min_samples: Minimum samples before checking MCMC convergence
    :param float alpha: _description_, defaults to 16
    :param float beta: _description_, defaults to 1
    """

    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
        alpha: int = 16,
        beta: int = 1
    ):
        super(BetaShapley, self).__init__(
            pred_model,
            metric,
            gr_threshold,
            max_iterations,
            min_samples,
            model_name
        )
        self.alpha, self.beta = alpha, beta  # Beta distribution parameters

    def compute_weight(self) -> np.ndarray:
        """_summary_
        Ref. https://arxiv.org/pdf/2110.14049Equation (3) and (5)

        Since
        w^{n_points}(j)*binom{n_points-1}{j-1}/n_points
        = Beta(j+beta_param-1, n_points-j+alpha_param)*binom{n_points-1}{j-1}/
        Beta(alpha_param, beta_param)
        = Constant*Beta(j+beta_param-1, n_points-j+alpha_param)/Beta(j, n_points-j+1)
        where $Constant = 1/(n_points*Beta(alpha_param, beta_param))$.

        :return np.ndarray: _description_
        """
        weight_list = np.array([
            beta(j + self.beta, self.n_points - (j + 1) + self.alpha) /
            beta(j + 1, self.n_points - j)
            for j in range(self.n_points)
        ])
        return weight_list / np.sum(weight_list)


    def evaluate_data_values(self) -> np.ndarray:
        """Multiplies the marginal contribution with their respective weights to get
        Beta Shapley data values

        :return np.ndarray: Predicted data values/selection for every input data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)