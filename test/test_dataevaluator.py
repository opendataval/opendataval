import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher
from opendataval.dataval import DataEvaluator, ModelMixin
from opendataval.model import Model


class DummyModel(Model):
    def __init__(self, random_state: RandomState = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def __eq__(self, rhs):
        """Check equality with unique identifier random state."""
        return self.random_state.tomaxint() == rhs.random_state.tomaxint()

    def fit(self, *args, **kwargs):
        return self

    def predict(self, x, *args, **kwargs):
        return torch.zeros_like(x)


class DummyDataEvaluator(DataEvaluator, ModelMixin):
    def __init__(self, random_state: RandomState = None):
        self.random_state = check_random_state(random_state)
        self.trained = False

    def train_data_values(self, *args, **kwargs):
        self.trained = True
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return np.array([1.0] * self.x_train.shape[0])


class TestDataEvaluator(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=0)
        train = self.random_state.rand(100, 10), self.random_state.rand(100, 1)
        valid = self.random_state.rand(20, 10), self.random_state.rand(20, 1)
        test = self.random_state.rand(20, 10), self.random_state.rand(20, 1)

        self.model = DummyModel(input_dim=10, output_dim=1)
        self.metric = MagicMock(return_value=1.0)
        self.fetcher = DataFetcher.from_data_splits(
            *train,
            *valid,
            *test,
            one_hot=False,
        )

        self.x_train, self.y_train = (torch.tensor(t).float() for t in train)
        self.x_valid, self.y_valid = (torch.tensor(t).float() for t in valid)
        self.x_test, self.y_test = (torch.tensor(t).float() for t in test)

    def test_init_(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        # Same underlying object, so we do an equality check on the objects
        self.assertEqual(evaluator.random_state, self.random_state)

    def test_evaluate(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        y = torch.randn(100, 1)
        y_hat = torch.randn(100, 1)
        evaluator.input_model(self.model).input_metric(self.metric)

        self.assertEqual(evaluator.evaluate(y, y_hat), 1.0)
        self.metric.assert_called_once_with(y, y_hat)

    def test_input_model_metric(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_model(self.model).input_metric(self.metric)

        self.assertEqual(evaluator.pred_model, self.model)
        self.assertEqual(evaluator.metric, self.metric)

    def test_input_data(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_data(
            self.x_train, self.y_train, self.x_valid, self.y_valid
        )

        self.assertTrue(torch.equal(evaluator.x_train, self.x_train))
        self.assertTrue(torch.equal(evaluator.y_train, self.y_train))
        self.assertTrue(torch.equal(evaluator.x_valid, self.x_valid))
        self.assertTrue(torch.equal(evaluator.y_valid, self.y_valid))
        self.assertFalse(evaluator.trained)

    def test_train(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.train(self.fetcher, self.model)

        print(evaluator.x_train)
        print(self.x_train)
        self.assertTrue(torch.equal(evaluator.x_train, self.x_train))
        self.assertTrue(torch.equal(evaluator.y_train, self.y_train))
        self.assertTrue(torch.equal(evaluator.x_valid, self.x_valid))
        self.assertTrue(torch.equal(evaluator.y_valid, self.y_valid))
        self.assertEqual(evaluator.evaluate(evaluator.y_train, self.y_train), 0.0)
        self.assertTrue(evaluator.trained)

        # Tests categorical default
        self.fetcher.one_hot = True
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.train(self.fetcher, self.model)

        self.assertTrue(torch.equal(evaluator.x_train, self.x_train))
        self.assertTrue(torch.equal(evaluator.y_train, self.y_train))
        self.assertTrue(torch.equal(evaluator.x_valid, self.x_valid))
        self.assertTrue(torch.equal(evaluator.y_valid, self.y_valid))
        self.assertEqual(evaluator.evaluate(evaluator.y_train, self.y_train), 1.0)
        self.assertTrue(evaluator.trained)

    def test_input_fetcher(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_fetcher(self.fetcher)

        self.assertTrue(torch.equal(evaluator.x_train, self.x_train))
        self.assertTrue(torch.equal(evaluator.y_train, self.y_train))
        self.assertTrue(torch.equal(evaluator.x_valid, self.x_valid))
        self.assertTrue(torch.equal(evaluator.y_valid, self.y_valid))
        self.assertFalse(evaluator.trained)

    def test_train_evaluate_data_values(self):
        evaluator = DummyDataEvaluator(random_state=self.random_state)
        evaluator = evaluator.input_model(self.model).input_metric(self.metric)
        evaluator = evaluator.input_data(
            self.x_train, self.y_train, self.x_valid, self.y_valid
        )
        self.assertTrue(evaluator.train_data_values().evaluate_data_values().all())
        self.assertTrue(evaluator.trained)


if __name__ == "__main__":
    unittest.main()
