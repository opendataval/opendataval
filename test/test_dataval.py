import unittest
import warnings

import numpy as np
import torch

from opendataval.dataloader import DataFetcher, mix_labels
from opendataval.dataval.random import RandomEvaluator
from opendataval.experiment import ExperimentMediator, discover_corrupted_sample
from opendataval.model import Model
from opendataval.presets import dummy_evaluators
from opendataval.util import set_random_state


class DummyModel(Model):
    def fit(self, *args, **kwargs):
        return self

    def predict(self, x_train):
        return torch.rand((len(x_train), 1))


class TestDataEvaluatorDryRun(unittest.TestCase):
    """Quick dry run to ensure all data evaluators are working as intended."""

    def test_dry_run(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # temporarily ignores warnings
            random_state = set_random_state(10)
            fetcher = (
                DataFetcher("iris", random_state=random_state)
                .split_dataset_by_count(3, 2, 2)
                .noisify(mix_labels, noise_rate=0.2)
            )

            # Checks that all evaluators in `dummy_evaluators` can have at least
            # a dry run with low data. Basically a sanity check.

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                exper_med = ExperimentMediator(
                    fetcher=fetcher,
                    pred_model=DummyModel(),
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
            .input_model_metric(DummyModel(), lambda *_: 1.0)  # Dummy metric as well
            .input_fetcher(fetcher)
        )

        self.assertTrue(
            np.array_equal(
                set_random_state(10).uniform(size=(len(fetcher.x_train),)),
                data_val.evaluate_data_values(),
            )
        )
