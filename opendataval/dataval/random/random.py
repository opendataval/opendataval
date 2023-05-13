import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator


class RandomEvaluator(DataEvaluator):
    """Completely Random DataEvaluator for baseline comparison purposes.

    Generates Random data values from Uniform[0.0, 1.0].

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, random_state: RandomState = None):
        self.random_state = check_random_state(random_state)

    def train_data_values(self, *args, **kwargs):
        """RandomEval does not train to find the training values."""
        pass

    def evaluate_data_values(self) -> np.ndarray:
        """Return random data values for each training data point."""
        return self.random_state.uniform(size=(len(self.x_train),))
