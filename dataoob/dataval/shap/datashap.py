from dataoob.dataval.shap.shap import ShapEvaluator
from dataoob.model import Model

class DataShapley(ShapEvaluator):
    """Data Shapley implementation.
    Ref. https://arxiv.org/abs/1904.02868

    :param Model pred_model: Prediction model
    :param callable (torch.Tensor, torch.Tensor -> float) metric: Evaluation function
    to determine model performance
    :param float GR_threshold: Convergence threshold for the Gelman-Rubin statistic.
    Shapley values are NP-hard this is the approximation criteria
    :param int max_iterations: Max number of outer iterations of MCMC sampling,
    guarantees the training won't deadloop, defaults to 100
    :param int min_samples: Minimum samples before checking MCMC convergence
    """

    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
    ):
        super(DataShapley, self).__init__(
            pred_model,
            metric,
            gr_threshold,
            max_iterations,
            min_samples,
            model_name
        )

    def compute_weight(self) -> float:
        """Shapley values evaluate the average marginal contribution for each
        cardinality. It can be seen as sampling uniformly from the set of all
        combinations of datapoints

        :return float: _description_
        """
        return 1 / len(self.n_points)