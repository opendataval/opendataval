import unittest
from unittest.mock import Mock, call

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from dataoob.dataloader import DataLoader, Register, mix_labels
from dataoob.dataval import DataEvaluator
from dataoob.evaluator import (
    DataEvaluatorArgs,
    DataEvaluatorFactoryArgs,
    DataLoaderArgs,
    ExperimentMediator,
)
from dataoob.model import Model


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


class DummyEvaluator(DataEvaluator):
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


class TestDataLoaderArgs(unittest.TestCase):
    def test_data_loader_args(self):
        args = DataLoaderArgs(
            dataset="test_dataset", force_download=False, device=torch.device("cpu")
        )
        self.assertEqual(args.dataset, "test_dataset")
        self.assertEqual(args.force_download, False)
        self.assertEqual(args.device.type, "cpu")

    def test_data_loader_args_default(self):
        args = DataLoaderArgs(dataset="test_dataset")
        self.assertEqual(args.train_count, 0.8)
        self.assertEqual(args.valid_count, 0.2)


class TestDataEvaluatorArgs(unittest.TestCase):
    def test_data_evaluator_args(self):
        args = DataEvaluatorArgs(pred_model=DummyModel(), train_kwargs={"epochs": 5})
        self.assertIsInstance(args.pred_model, Model)
        self.assertEqual(args.train_kwargs["epochs"], 5)

    def test_data_evaluator_args_default(self):
        args = DataEvaluatorArgs(pred_model=DummyModel())
        self.assertEqual(args.metric_name, "accuracy")


class TestDataEvaluatorFactoryArgs(unittest.TestCase):
    def test_data_evaluator_factory_args(self):
        args = DataEvaluatorFactoryArgs(
            pred_model_factory=lambda a, b, c: DummyModel(),
            train_kwargs={"epochs": 5},
            metric_name="f1_score",
            device=torch.device("cpu"),
        )
        self.assertEqual(args.metric_name, "f1_score")
        self.assertEqual(args.device.type, "cpu")

    def test_data_evaluator_factory_args_default(self):
        args = DataEvaluatorFactoryArgs(pred_model_factory=lambda a, b, c: DummyModel())
        self.assertEqual(args.metric_name, "accuracy")
        self.assertEqual(args.device.type, "cpu")


class TestExperimentMediator(unittest.TestCase):
    def setUp(self):
        self.dataloader = DataLoader(
            "test_dataset", False, torch.device("cpu")
        ).split_dataset(0.8, 0.2)
        self.dataevaluator = DummyEvaluator()

    def test_experiment_mediator(self):
        experimentmediator = ExperimentMediator(
            self.dataloader,
            [self.dataevaluator],
            DummyModel(),
            train_kwargs={"epochs": 10},
            metric_name="accuracy",
        )
        self.assertIsInstance(experimentmediator.loader, DataLoader)
        self.assertIsInstance(experimentmediator.data_evaluators[0], DataEvaluator)
        self.assertIsInstance(experimentmediator.train_kwargs, dict)
        self.assertEqual(experimentmediator.metric_name, "accuracy")
        self.assertTrue(self.dataevaluator.trained)

    def test_experiment_mediator_default(self):
        experimentmediator = ExperimentMediator(
            self.dataloader, [self.dataevaluator], DummyModel()
        )
        self.assertEqual(experimentmediator.metric_name, "accuracy")
        self.assertTrue(self.dataevaluator.trained)

    def test_experminet_mediator_create_dataloader(self):
        experimentmediator = ExperimentMediator.create_dataloader(
            dataset="test_dataset",
            force_download=False,
            train_count=0.8,
            valid_count=0.2,
            add_noise_func=mix_labels,
            noise_kwargs={"noise_rate": 0.2},
            device=torch.device("cpu"),
            pred_model=DummyModel(),
            train_kwargs={"epochs": 5},
            metric_name="accuracy",
            data_evaluators=[self.dataevaluator],
        )
        self.assertEqual(experimentmediator.metric_name, "accuracy")
        self.assertTrue(self.dataevaluator.trained)

    def test_experiment_mediator_exceptions(self):
        with self.assertRaises(ValueError):
            ExperimentMediator.create_dataloader(
                dataset="test_dataset",
                force_download=False,
                train_count=0.8,
                valid_count=1.1,
                add_noise_func=mix_labels,
                noise_kwargs={"noise_rate": 0.2},
                device=torch.device("cpu"),
                pred_model=DummyModel(),
                train_kwargs={"epochs": 5},
                metric_name="accuracy",
                data_evaluators=[DummyEvaluator()],
            )
        self.assertFalse(self.dataevaluator.trained)

        self.assertWarns(
            Warning,
            ExperimentMediator.preset_setup,
            DataLoaderArgs(
                dataset="test_dataset",
                force_download=False,
                device=torch.device("cpu"),
                noise_kwargs={"noise_rate": 0.2},
            ),
            DataEvaluatorFactoryArgs(
                pred_model_factory=lambda a, b, c: BrokenDummyModel(),
                device=torch.device("cpu"),
            ),
            data_evaluators=[BrokenEvaluator()],
        )

    def test_evaluate_mediator(self):
        mock_func = Mock(side_effect=[{"a": [1, 2], "b": [3]}, {"a": [4, 5], "b": [6]}])
        kwargs = {"c": 1, "d": "2"}
        dummies = [DummyEvaluator(1), DummyEvaluator(2)]
        experimentmediator = ExperimentMediator(self.dataloader, dummies, DummyModel())
        res = experimentmediator.evaluate(exper_func=mock_func, **kwargs)
        mock_func.assert_has_calls(
            [
                call(dummies[0], self.dataloader, **kwargs),
                call(dummies[1], self.dataloader, **kwargs),
            ]
        )

        self.assertTrue(res.loc["a", dummies[0].plot_title].eq([1, 2]).all())
        self.assertTrue(res.loc["b", dummies[0].plot_title][0] == 3)
        self.assertTrue(res.loc["a", dummies[1].plot_title].eq([4, 5]).all())
        self.assertTrue(res.loc["b", dummies[1].plot_title][0] == 6)


if __name__ == "__main__":
    unittest.main()
