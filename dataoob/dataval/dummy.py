from dataoob.dataval import DataEvaluator
import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state


class DummyEvaluator(DataEvaluator):
    def __init__(self, random_state: RandomState = None):
        self.random_state = check_random_state(random_state)

    def train_data_values(self, *args, **kwargs):
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self.random_state.rand(len(self.x_train))
