"""Presets for data valuation.

Configurations for experiment presets.
"""

# Data Evaluators
from dataoob.dataval.ame.ame import AME
from dataoob.dataval.dvrl.dvrl import DVRL
from dataoob.dataval.influence.influence import InfluenceFunctionEval
from dataoob.dataval.margcontrib.banzhaf import DataBanzhaf, DataBanzhafMargContrib
from dataoob.dataval.margcontrib.betashap import BetaShapley
from dataoob.dataval.margcontrib.datashap import DataShapley
from dataoob.dataval.margcontrib.loo import LeaveOneOut
from dataoob.dataval.oob.oob import DataOob
from dataoob.util import set_random_state

RANDOM_STATE = set_random_state(10)  # Constant random state for testing

# fmt: off
# ruff: noqa: E501 D103
# Dummy evaluators used for low iteration training, for testing
dummy_evaluators = [
    AME(2, random_state=RANDOM_STATE),  # For lasso, minimum needs 5 for split
    DVRL(1, rl_epochs=1, random_state=RANDOM_STATE),
    DataOob(1, random_state=RANDOM_STATE),
    InfluenceFunctionEval(1, random_state=RANDOM_STATE),
    LeaveOneOut(random_state=RANDOM_STATE),
    DataBanzhaf(samples=1, random_state=RANDOM_STATE),
    DataBanzhafMargContrib(99, max_iterations=2, samples_per_iteration=1, cache_name="t", random_state=RANDOM_STATE),
    BetaShapley(99, max_iterations=2, samples_per_iteration=1, cache_name="t", random_state=RANDOM_STATE),
    DataShapley(cache_name="t", random_state=RANDOM_STATE),
    DataShapley(99, max_iterations=2, samples_per_iteration=1, cache_name="r", random_state=RANDOM_STATE),
]
