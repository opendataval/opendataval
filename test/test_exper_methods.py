import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import RandomState
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher
from opendataval.dataloader.noisify import mix_labels
from opendataval.dataval import DataEvaluator, ModelMixin
from opendataval.experiment.exper_methods import (
    discover_corrupted_sample,
    increasing_bin_removal,
    noisy_detection,
    remove_high_low,
    save_dataval,
)
from opendataval.metrics import Metrics
from opendataval.model import Model
from opendataval.util import get_name, set_random_state


class DummyModel(Model):
    def fit(self, *args, **kwargs):
        return self

    def predict(self, x_train):
        return torch.ones((len(x_train), 1))


class DummyEvaluator(DataEvaluator, ModelMixin):
    def __init__(self, random_state: RandomState = None):
        self.pred_model = DummyModel()
        self.random_state = check_random_state(random_state)

    def evaluate(self, y_pred, y_true):
        return (y_true == y_pred).float().mean()

    def train_data_values(self, *args, **kwargs):
        self.data_values = self.random_state.random(len(self.x_train))
        return self

    def evaluate_data_values(self) -> np.ndarray:
        return self.data_values


class TestExperiment(unittest.TestCase):
    def setUp(self):
        random_state = set_random_state(10)
        covar, labels = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=random_state,
        )

        self.fetcher = (
            DataFetcher.from_data(covar, labels, False, random_state=random_state)
            .split_dataset_by_indices(range(20), range(20, 40), range(40, 60))
            .noisify(mix_labels, noise_rate=0.25)
        )
        self.data_evaluator = (
            DummyEvaluator(random_state).input_fetcher(self.fetcher).train_data_values()
        )
        self.num_points = 20
        self.train_kwargs = {}
        self.plot = plt.subplot(1, 1, 1)

    def test_noisy_detection(self):
        result = noisy_detection(self.data_evaluator, self.fetcher)
        self.assertIn("kmeans_f1", result)
        self.assertIsInstance(result["kmeans_f1"], float)
        self.assertGreaterEqual(result["kmeans_f1"], 0.0)
        self.assertLessEqual(result["kmeans_f1"], 1.0)

    def test_increasing_bin_removal(self):
        metric_name = Metrics.ACCURACY
        result = increasing_bin_removal(
            self.data_evaluator,
            self.fetcher,
            bin_size=1,
            metric=metric_name,
            plot=self.plot,
            train_kwargs=self.train_kwargs,
        )
        self.assertIn("axis", result)
        self.assertIn("frac_datapoints_explored", result)
        self.assertIn(f"{get_name(metric_name)}_at_datavalues", result)

    def test_save_dataval(self):
        data_values = self.data_evaluator.evaluate_data_values()
        result = save_dataval(self.data_evaluator, self.fetcher)
        self.assertListEqual(list(range(20)), result["indices"].tolist())
        self.assertListEqual(data_values.tolist(), result["data_values"].tolist())

    def test_discover_corrupted_sample(self):
        result = discover_corrupted_sample(
            self.data_evaluator,
            self.fetcher,
            percentile=0.5,
            plot=self.plot,
        )
        keys = ["optimal", "random", "corrupt_found"]
        self.assertIn("axis", result)
        axis_len = len(result["axis"])
        for key in keys:
            self.assertIn(key, result)
            self.assertEqual(axis_len, len(result[key]), msg=f"len(axis) != len({key})")

    def test_remove_high_low(self):
        metric = Metrics.ACCURACY
        result = remove_high_low(
            self.data_evaluator,
            self.fetcher,
            metric=metric,
            percentile=0.05,
            plot=self.plot,
            train_kwargs=self.train_kwargs,
        )
        keys = [
            f"remove_least_influential_first_{get_name(metric)}",
            f"remove_most_influential_first_{get_name(metric)}",
        ]
        self.assertIn("axis", result)
        axis_len = len(result["axis"])

        for key in keys:
            self.assertIn(key, result)
            self.assertEqual(axis_len, len(result[key]), f"len(axis)!=len({key})")


if __name__ == "__main__":
    unittest.main()
