"""Presets for data valuation.

Configurations for experiment presets.
"""

# Data Evaluators
from opendataval.dataval.ame.ame import AME
from opendataval.dataval.dvrl.dvrl import DVRL
from opendataval.dataval.influence.influence import InfluenceFunctionEval
from opendataval.dataval.knnshap import KNNShapley
from opendataval.dataval.margcontrib.banzhaf import DataBanzhaf, DataBanzhafMargContrib
from opendataval.dataval.margcontrib.betashap import BetaShapley
from opendataval.dataval.margcontrib.datashap import DataShapley
from opendataval.dataval.margcontrib.loo import LeaveOneOut
from opendataval.dataval.oob.oob import DataOob
from opendataval.dataval.random.random import RandomEvaluator
from opendataval.util import set_random_state

RANDOM_STATE = set_random_state(10)  # Constant random state for testing

# fmt: off
# ruff: noqa: E501 D103
# Dummy evaluators used for low iteration training, for testing
dummy_evaluators = [
    AME(2, random_state=RANDOM_STATE),  # For lasso, minimum needs 5 for split
    DVRL(1, rl_epochs=1, random_state=RANDOM_STATE),
    DataOob(1, random_state=RANDOM_STATE),
    InfluenceFunctionEval(1, random_state=RANDOM_STATE),
    KNNShapley(5, random_state=RANDOM_STATE),
    LeaveOneOut(random_state=RANDOM_STATE),
    DataBanzhaf(num_models=1, random_state=RANDOM_STATE),
    DataBanzhafMargContrib(99, max_mc_epochs=2, models_per_iteration=1, cache_name="cache_preset", random_state=RANDOM_STATE),
    BetaShapley(99, max_mc_epochs=2, models_per_iteration=1, cache_name="cache_preset", random_state=RANDOM_STATE),
    DataShapley(cache_name="cache_preset", random_state=RANDOM_STATE),
    DataShapley(99, max_mc_epochs=2, models_per_iteration=1, cache_name="cache_preset_other", random_state=RANDOM_STATE),
    RandomEvaluator(random_state=RANDOM_STATE)
]
