"""Presets for data valuation.

Configurations for experiment presets, linting and black formatting
is turned off in this file. No good code found below
# TODO look into how openxai does these presets
"""
import torch

# Data Evaluators
from dataoob.dataval import DataEvaluator
from dataoob.dataval.ame.ame import AME
from dataoob.dataval.dvrl.dvrl import DVRL
from dataoob.dataval.influence.influence import InfluenceFunctionEval
from dataoob.dataval.knnshap.knnshap import KNNShapley
from dataoob.dataval.oob.oob import DataOob
from dataoob.dataval.shap.banzhaf import DataBanzhaf, DataBanzhafMargContrib
from dataoob.dataval.shap.betashap import BetaShapley
from dataoob.dataval.shap.datashap import DataShapley
from dataoob.dataval.shap.loo import LeaveOneOut

# Data classes for wrapping EvaluatorMediator Args
from dataoob.evaluator import DataEvaluatorFactoryArgs as DEFA
from dataoob.evaluator import DataFetcherArgs as DLA
from dataoob.evaluator import ExperimentMediator
from dataoob.model.ann import BinaryANN, ClassifierANN

# Models
from dataoob.model.api import Model
from dataoob.model.logistic_regression import BinaryLogisticRegression as BLR
from dataoob.model.logistic_regression import LogisticRegression as LR

RANDOM_STATE = 10  # NOTE this random state should be changed, for now it's a constant

# API functions to be interacted with
def new_evaluator(preset_name: str, new_evaluator: DataEvaluator) -> ExperimentMediator:
    """Load preset model and data set to test an evaluator.

    For a new DataEvaluator, trains the DataEvaluator on the preset model and data set
    and returns an ExperimentMediator to run experiments on

    Parameters
    ----------
    preset_name : str
        Name of preset for the data set and the model
    new_evaluator : DataEvaluator
        A new DataEvaluator to be trained using the pred_model and data set form above.

    Returns
    -------
    ExperimentMediator
        New ExperimentMediator with a single DataValuator, the specified input
        trained on the preset data set and model.
    """
    data_evaluators = [new_evaluator]
    loader_args, evaluator_args = experiment_presets[preset_name]

    return ExperimentMediator.preset_setup(loader_args, evaluator_args, data_evaluators)


def from_presets(preset_name: str, evaluators_name: str) -> ExperimentMediator:
    """Load preset model, data sets, and data evaluator.

    Loads existing presets for model, data set and data evaluator. These default
    arguments represent values we believe would be relevant in determining evaluator's
    performance. NOTE make sure to test with evaluators_name=``'dummy'`` to make sure
    the data set is valid for your evaluator

    Parameters
    ----------
    preset_name : str
        Name of preset for the data set and the model
    evaluators_name : str
        Name of preset for data valuators

    Returns
    -------
    ExperimentMediator
        New ExperimentMediator with DataValuators specified by ``evaluator_name``
        trained on the preset data set and model.
    """
    data_evaluators = data_evaluator_presets[evaluators_name]
    loader_args, evaluator_args = experiment_presets[preset_name]

    return ExperimentMediator.preset_setup(loader_args, evaluator_args, data_evaluators)


# fmt: off
# ruff: noqa: E501 D103
def ann_class_fac(covar_dim: int, label_dim: int, device: torch.Tensor) -> Model:
    if label_dim == 2:
        return BinaryANN(covar_dim).to(device)
    else:
        return ClassifierANN(covar_dim, label_dim).to(device)

def lr_class_fac(covar_dim: int, label_dim: int, device: torch.Tensor) -> LR:
    if label_dim == 2:
        return BLR(covar_dim).to(device)
    else:
        return LR(covar_dim, label_dim).to(device)

experiment_presets = {
    'iris_low_noise_ann': (DLA("iris", noise_kwargs={'noise_rate': 0.05} ), DEFA(ann_class_fac, train_kwargs={'batch_size': 25, 'epochs': 25})),
    'iris_mid_noise_ann': (DLA("iris", noise_kwargs={'noise_rate': 0.20}), DEFA(ann_class_fac, train_kwargs={ 'batch_size': 25, 'epochs': 25})),
    'iris_high_noise_ann': (DLA("iris", noise_kwargs={'noise_rate': 0.30}), DEFA(ann_class_fac, train_kwargs={ 'batch_size': 25, 'epochs': 25})),
}

dummy_evaluators = [  # Used for quick testing and run throughs
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

data_evaluators = [
    AME(num_models=1500, random_state=RANDOM_STATE),
    DataOob(random_state=RANDOM_STATE),  # 1000 samples
    DVRL(rl_epochs=3000, random_state=RANDOM_STATE),  # RL requires torch device
    InfluenceFunctionEval(5000, random_state=RANDOM_STATE),
    DataBanzhaf(5000, random_state=RANDOM_STATE),
    BetaShapley(gr_threshold=1.05, min_samples=500, cache_name="cached", random_state=RANDOM_STATE),
    DataShapley(gr_threshold=1.05, min_samples=500, cache_name="cached", random_state=RANDOM_STATE),
]

data_evaluator_presets = {
    'experiment': data_evaluators,
    'model_less': [*data_evaluators, KNNShapley],
    'dummy': dummy_evaluators
}
