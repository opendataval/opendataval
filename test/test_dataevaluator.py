import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from dataoob.dataval import DataEvaluator
from dataoob.model import Model


class DummyModel(Model):
    def __init__(self, random_state: RandomState = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def __eq__(self, rhs):
        """Since a unique identifier on creation is random state. We also will copy
        the random state, so to make sure they're the same we just check the next number
        """
        return self.random_state.tomaxint() == rhs.random_state.tomaxint()

    def fit(self, *args, **kwargs):
        return self

    def predict(self, x, *args, **kwargs):
        return torch.zeros_like(x)


class DummyDataEvaluator(DataEvaluator):
    def __init__(self, random_state: RandomState = None):
        self.random_state = check_random_state(random_state)
        self.trained = False

    def train_data_values(self, *args, **kwargs):
        self.trained = True
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return np.array([1.0] * self.x_train.shape[0])


class DummyDataLoader:
    def __init__(self, x_train, y_train, x_valid, y_valid):
        self.datapoints = x_train, y_train, x_valid, y_valid


class TestDataEvaluator(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=0)
        self.x_train = torch.randn(100, 10)
        self.y_train = torch.randn(100, 1)
        self.x_valid = torch.randn(20, 10)
        self.y_valid = torch.randn(20, 1)

        self.model = DummyModel(input_dim=10, output_dim=1)
        self.metric = MagicMock(return_value=1.0)
        self.loader = DummyDataLoader(
            self.x_train, self.y_train, self.x_valid, self.y_valid
        )

    def test_init_(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        # Same underlying object, so we do an equality check on the objects
        self.assertEqual(evaluator.random_state, self.random_state)

    def test_evaluate(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        y = torch.randn(100, 1)
        y_hat = torch.randn(100, 1)

        with self.assertRaises(ValueError):
            evaluator.evaluate(y, y_hat)

        evaluator.input_model_metric(self.model, self.metric)

        self.assertEqual(evaluator.evaluate(y, y_hat), 1.0)
        self.metric.assert_called_once_with(y, y_hat)

    def test_input_model_metric(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_model_metric(self.model, self.metric)

        self.assertEqual(evaluator.pred_model, self.model)
        self.assertEqual(evaluator.metric, self.metric)

    def test_input_data(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_data(
            self.x_train, self.y_train, self.x_valid, self.y_valid
        )

        self.assertTrue((evaluator.x_train == self.x_train).all().item())
        self.assertTrue((evaluator.y_train == self.y_train).all().item())
        self.assertTrue((evaluator.x_valid == self.x_valid).all().item())
        self.assertTrue((evaluator.y_valid == self.y_valid).all().item())
        self.assertFalse(evaluator.trained)

    def test_train(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.train(
            self.x_train, self.y_train, self.x_valid, self.y_valid
        )

        self.assertTrue((evaluator.x_train == self.x_train).all().item())
        self.assertTrue((evaluator.y_train == self.y_train).all().item())
        self.assertTrue((evaluator.x_valid == self.x_valid).all().item())
        self.assertTrue((evaluator.y_valid == self.y_valid).all().item())
        self.assertTrue(evaluator.trained)

    def test_input_dataloader(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_dataloader(self.loader)

        self.assertTrue((evaluator.x_train == self.x_train).all().item())
        self.assertTrue((evaluator.y_train == self.y_train).all().item())
        self.assertTrue((evaluator.x_valid == self.x_valid).all().item())
        self.assertTrue((evaluator.y_valid == self.y_valid).all().item())
        self.assertFalse(evaluator.trained)

    def test_train_evaluate_data_values(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_model_metric(self.model, self.metric)
        evaluator = evaluator.input_data(
            self.x_train, self.y_train, self.x_valid, self.y_valid
        )
        self.assertTrue(evaluator.train_data_values().evaluate_data_values().all())
        self.assertTrue(evaluator.trained)


if __name__ == "__main__":
    unittest.main()
