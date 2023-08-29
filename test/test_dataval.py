import unittest
import warnings

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher, mix_labels

# Data Evaluators
from opendataval.dataval import (
    AME,
    DVRL,
    BetaShapley,
    ClassWiseShapley,
    DataBanzhaf,
    DataBanzhafMargContrib,
    DataOob,
    DataShapley,
    InfluenceFunction,
    InfluenceSubsample,
    KNNShapley,
    LavaEvaluator,
    LeaveOneOut,
    RandomEvaluator,
    RobustVolumeShapley,
    TMCSampler,
)
from opendataval.experiment import ExperimentMediator, discover_corrupted_sample
from opendataval.model import GradientModel
from opendataval.util import set_random_state


class DummyModel(GradientModel):
    def __init__(self, num_classes: int, random_state: RandomState = None):
        self.num_classes = num_classes
        torch.manual_seed(check_random_state(random_state).tomaxint())

    def fit(self, *args, **kwargs):
        return self

    def predict(self, x_train):
        return torch.rand(len(x_train), self.num_classes)

    def grad(self, x_data, y_data):
        """Dummy gradient method, returns random values for a 3 layer model."""
        for _ in range(len(x_data)):
            yield tuple(torch.rand((2, 3)) for _ in range(3))


class TestDataEvaluatorDryRun(unittest.TestCase):
    """Quick dry run to ensure all data evaluators are working as intended."""

    def test_dry_run(self):
        random_state = set_random_state(10)
        fetcher = (
            DataFetcher("iris", random_state=random_state)
            .split_dataset_by_count(16, 16, 2)
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

        data_val = RandomEvaluator(10).input_fetcher(fetcher)

        self.assertTrue(
            np.array_equal(
                set_random_state(10).uniform(size=(len(fetcher.x_train),)),
                data_val.evaluate_data_values(),
            )
        )


# fmt: off
# ruff: noqa: E501 D103
# Dummy evaluators used for low iteration training, for testing
RANDOM_STATE = set_random_state(10)  # Constant random state for testing

first_sampler = TMCSampler(2, cache_name="cache_preset", random_state=RANDOM_STATE)
second_sampler = TMCSampler(2, random_state=RANDOM_STATE)

dummy_evaluators = [
    AME(2, random_state=RANDOM_STATE),  # For lasso, minimum needs 5 for split
    DVRL(1, rl_epochs=1, random_state=RANDOM_STATE),
    DataOob(1, random_state=RANDOM_STATE),
    InfluenceSubsample(1, random_state=RANDOM_STATE),
    InfluenceFunction(),
    KNNShapley(5, random_state=RANDOM_STATE),
    LeaveOneOut(random_state=RANDOM_STATE),
    LavaEvaluator(random_state=RANDOM_STATE),
    DataBanzhaf(num_models=1, random_state=RANDOM_STATE),
    DataBanzhafMargContrib(max_mc_epochs=2, models_per_iteration=1, cache_name="cache_preset", random_state=RANDOM_STATE),
    BetaShapley(max_mc_epochs=2, models_per_iteration=1, cache_name="cache_preset", random_state=RANDOM_STATE),
    DataShapley(first_sampler),
    DataShapley(max_mc_epochs=2, models_per_iteration=1, random_state=RANDOM_STATE),
    ClassWiseShapley(mc_epochs=2, random_state=RANDOM_STATE),
    RandomEvaluator(random_state=RANDOM_STATE),
    RobustVolumeShapley(second_sampler, robust=False)
]
