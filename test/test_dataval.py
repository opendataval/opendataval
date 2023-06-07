import unittest
import warnings

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher, mix_labels
from opendataval.dataval.random import RandomEvaluator
from opendataval.experiment import ExperimentMediator, discover_corrupted_sample
from opendataval.model import Model
from opendataval.presets import dummy_evaluators
from opendataval.util import set_random_state


class DummyModel(Model):
    def __init__(self, num_classes: int, random_state: RandomState = None):
        self.num_classes = num_classes
        torch.manual_seed(check_random_state(random_state).tomaxint())

    def fit(self, *args, **kwargs):
        return self

    def predict(self, x_train):
        return torch.rand(len(x_train), self.num_classes)


class TestDataEvaluatorDryRun(unittest.TestCase):
    """Quick dry run to ensure all data evaluators are working as intended."""

    def test_dry_run(self):
        random_state = set_random_state(10)
        fetcher = (
            DataFetcher("iris", random_state=random_state)
            .split_dataset_by_count(8, 2, 2)
            .noisify(mix_labels, noise_rate=0.2)
        )

        # Checks that all evaluators in `dummy_evaluators` can have at least
        # a dry run with low data. Basically a sanity check.

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exper_med = ExperimentMediator(
                fetcher=fetcher,
                pred_model=DummyModel(3, 10),
                metric_name="accuracy",
            ).compute_data_values(data_evaluators=dummy_evaluators)

            exper_med.evaluate(discover_corrupted_sample)

    def test_random(self):
        fetcher = (
            DataFetcher("iris", random_state=25)
            .split_dataset_by_count(3, 2, 2)
            .noisify(mix_labels, noise_rate=0.2)
        )

        data_val = (
            RandomEvaluator(10)
            .input_model(DummyModel(3, 10))
            .input_metric(lambda *_: 1.0)  # Dummy metric
            .input_fetcher(fetcher)
        )

        self.assertTrue(
            np.array_equal(
                set_random_state(10).uniform(size=(len(fetcher.x_train),)),
                data_val.evaluate_data_values(),
            )
        )
