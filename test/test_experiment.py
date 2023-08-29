import unittest
from unittest.mock import Mock, call

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher, Register, mix_labels
from opendataval.dataval import DataEvaluator, ModelMixin
from opendataval.experiment import ExperimentMediator
from opendataval.model import Model
from opendataval.model.mlp import ClassifierMLP


class DummyModel(Model):
    def fit(self, *args, **kwargs):
        return self

    def predict(self, x_train):
        return torch.ones((len(x_train), 1))


class BrokenDummyModel(Model):
    def fit(self, *args, **kwargs):
        raise ValueError("Broken exception raised")

    def predict(self, x_train):
        return torch.ones((len(x_train), 1))


class DummyEvaluator(DataEvaluator, ModelMixin):
    """Random data evaluator. Mainly used for testing purposes."""

    def __init__(self, random_state: RandomState = None):
        self.trained = False
        self.random_state = check_random_state(random_state)

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values."""
        self.trained = True
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values (random) for each training data point."""
        return self.random_state.rand(len(self.x_train))


class BrokenEvaluator(DataEvaluator):
    """Random data evaluator. Mainly used for testing purposes."""

    def __init__(self, random_state: RandomState = None):
        self.random_state = check_random_state(random_state)

    def train_data_values(self, *args, **kwargs):
        raise Exception("Evaluator broken")

    def evaluate_data_values(self) -> np.ndarray:
        raise Exception("Evaluator broken")


Register("test_dataset").from_numpy(np.array([[1, 2], [3, 4], [5, 6]]), 1)


class TestExperimentMediator(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher.setup(
            "test_dataset", "dummy_dir", False, 10, 0.7, 0.2, 0.1
        )
        self.dataevaluator = DummyEvaluator()

    def test_experiment_mediator(self):
        experimentmediator = ExperimentMediator(
            self.fetcher,
            DummyModel(),
            train_kwargs={"epochs": 10},
            metric_name="accuracy",
        ).compute_data_values(data_evaluators=[self.dataevaluator])
        self.assertIsInstance(experimentmediator.fetcher, DataFetcher)
        self.assertIsInstance(experimentmediator.data_evaluators[0], DataEvaluator)
        self.assertEqual(experimentmediator.train_kwargs, {"epochs": 10})
        self.assertEqual(experimentmediator.metric, "accuracy")
        self.assertTrue(self.dataevaluator.trained)

    def test_experiment_mediator_pass_args(self):
        experimentmediator = ExperimentMediator(
            self.fetcher,
            DummyModel(),
            metric_name="accuracy",
        ).compute_data_values(data_evaluators=[self.dataevaluator], epochs=10)
        self.assertIsInstance(experimentmediator.fetcher, DataFetcher)
        self.assertIsInstance(experimentmediator.data_evaluators[0], DataEvaluator)
        self.assertEqual(experimentmediator.metric, "accuracy")
        self.assertTrue(self.dataevaluator.trained)

    def test_experiment_mediator_default(self):
        experimentmediator = ExperimentMediator(
            self.fetcher, DummyModel()
        ).compute_data_values([self.dataevaluator])
        self.assertEqual(experimentmediator.metric, "neg_mse")
        self.assertTrue(self.dataevaluator.trained)

    def test_experiment_mediator_create_fetcher(self):
        experimentmediator = ExperimentMediator.setup(
            dataset_name="test_dataset",
            force_download=False,
            train_count=0.7,
            valid_count=0.2,
            test_count=0.1,
            add_noise=mix_labels,
            noise_kwargs={"noise_rate": 0.2},
            pred_model=DummyModel(),
            train_kwargs={"epochs": 5},
            metric_name="accuracy",
        ).compute_data_values(data_evaluators=[self.dataevaluator])
        self.assertEqual(experimentmediator.metric, "accuracy")
        self.assertTrue(self.dataevaluator.trained)

    def test_experiment_mediator_exceptions(self):
        with self.assertRaises(ValueError):
            ExperimentMediator.setup(
                dataset_name="test_dataset",
                force_download=False,
                train_count=0.8,
                valid_count=3.1,
                test_count=0.0,
                add_noise=mix_labels,
                noise_kwargs={"noise_rate": 0.2},
                pred_model=DummyModel(),
                train_kwargs={"epochs": 5},
                metric_name="accuracy",
            ).compute_data_values(data_evaluators=[DummyEvaluator()])

        with self.assertRaises(ValueError):
            ExperimentMediator.setup(
                dataset_name="test_dataset",
                force_download=False,
                train_count=0.8,
                valid_count=0.1,
                test_count=0,
                add_noise=mix_labels,
                noise_kwargs={"noise_rate": 0.2},
                pred_model=DummyModel(),
                train_kwargs={"epochs": 5},
                metric_name="accuracy",
            ).compute_data_values(data_evaluators=[BrokenDummyModel()])
        self.assertFalse(self.dataevaluator.trained)

    def test_evaluate_mediator(self):
        mock_func = Mock(
            side_effect=[{"a": [1, 2], "b": [3, 4]}, {"a": [5, 6], "b": [7, 8]}]
        )
        kwargs = {"c": 1, "d": "2"}  # Makes sure the undesired kwargs are filtered out
        dummies = [DummyEvaluator(1), DummyEvaluator(2)]
        experimentmediator = ExperimentMediator(
            self.fetcher, DummyModel()
        ).compute_data_values(dummies)
        res = experimentmediator.evaluate(exper_func=mock_func, **kwargs)
        mock_func.assert_has_calls(
            [
                call(dummies[0], self.fetcher),
                call(dummies[1], self.fetcher),
            ]
        )
        print(res)
        self.assertTrue(res.loc[str(dummies[0])]["a"].eq([1, 2]).all())
        self.assertTrue(res.loc[str(dummies[0])]["b"].eq([3, 4]).all())
        self.assertTrue(res.loc[str(dummies[1])]["a"].eq([5, 6]).all())
        self.assertTrue(res.loc[str(dummies[1])]["b"].eq([7, 8]).all())

    def test_experiment_mediator_model_factory_setup(self):
        exper_med = ExperimentMediator.model_factory_setup(
            dataset_name="test_dataset",
            force_download=False,
            train_count=0.7,
            valid_count=0.2,
            test_count=0.1,
            add_noise=mix_labels,
            noise_kwargs={"noise_rate": 0.2},
            model_name="ClassifierMLP",
            train_kwargs={"epochs": 5},
            metric_name="accuracy",
        )
        exper_med = exper_med.compute_data_values(data_evaluators=[self.dataevaluator])
        self.assertEqual(exper_med.metric, "accuracy")
        self.assertTrue(self.dataevaluator.trained)
        self.assertIsInstance(exper_med.data_evaluators[0].pred_model, ClassifierMLP)


if __name__ == "__main__":
    unittest.main()
