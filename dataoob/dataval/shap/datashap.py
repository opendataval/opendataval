import numpy as np
from dataoob.dataval.shap.shap import ShapEvaluator
from numpy.random import RandomState



class DataShapley(ShapEvaluator):
    """Data Shapley implementation.
    Ref. https://arxiv.org/abs/1904.02868

    :param float gr_threshold: Convergence threshold for the Gelman-Rubin statistic.
    Shapley values are NP-hard this is the approximation criteria
    :param int max_iterations: Max number of outer iterations of MCMC sampling,
    guarantees the training won't deadloop, defaults to 100
    :param int min_samples: Minimum samples before checking MCMC convergence
    :param str model_name: Unique name of the model, used to cache computed marginal contributions, defaults to None
    :param RandomState random_state: Random initial state, defaults to None
    """

    def __init__(
        self,
        gr_threshold: float = 1.01,
        max_iterations: int = 100,
        min_samples: int = 1000,
        model_name: str = None,
        random_state: RandomState = None
    ):
        super(DataShapley, self).__init__(
            gr_threshold, max_iterations, min_samples, model_name, random_state
        )

    def compute_weight(self) -> float:
        """Computes weight function for each cardinality.
        It can be seen as sampling uniformly from the set of all combinations of
        data points because Shapley values take a mean.

        :return float: Marginal contribution weight
        """
        return 1 / self.num_points

    def evaluate_data_values(self) -> np.ndarray:
        """Multiplies the marginal contribution with their respective weights to get
        Data Shapley data values

        :return np.ndarray: Predicted data values/selection for every input data point
        """
        return np.sum(self.marginal_contribution * self.compute_weight(), axis=1)
