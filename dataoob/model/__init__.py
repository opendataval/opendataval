"""Prediction models to be trained, predict, and evaluated.

Models
======

.. currentmodule:: dataoob.model

:py:class:`Model` is an ABC used to take an existing model and make it compatible with
the :py:class:`~dataoob.dataval.DataEvaluator` and other related objects.

API
---
.. autosummary::
    :toctree: generated/

    Model
    ModelFactory

Torch Mixins
------------
.. autosummary::
    :toctree: generated/

    TorchClassMixin
    TorchRegressMixin
    TorchPredictMixin

Sci-kit learn wrappers
----------------------
.. autosummary::
    :toctree: generated/

    ClassifierSkLearnWrapper
    ClassifierUnweightedSkLearnWrapper
    RegressionSkLearnWrapper
"""
# Model Factory imports
import torch

from dataoob.dataloader import DataFetcher
from dataoob.model.api import (
    ClassifierSkLearnWrapper,
    ClassifierUnweightedSkLearnWrapper,
    Model,
    TorchClassMixin,
    TorchPredictMixin,
)
from dataoob.model.logistic_regression import LogisticRegression
from dataoob.model.mlp import ClassifierMLP, RegressionMLP


def ModelFactory(
    model_name: str,
    fetcher: DataFetcher = None,
    device: torch.device = torch.device("cpu"),
) -> Model:
    """Factory to create prediction models from specified presets

    Model Factory that creates a specified mode, based on the input parameters, it is
    recommended to import the specific model and specify additional arguments instead of
    relying on the factory.

    Parameters
    ----------
    model_name : str
        Name of prediction model
    covar_dim : tuple[int, ...]
        Dimensions of the covariates, typically the shape besides first dimension
    label_dim : tuple[int, ...]
        Dimensions of the labels, typically the shape besides first dimension
    device : torch.device, optional
        Tensor device for acceleration, some models do not use this argument,
        by default torch.device("cpu")

    Returns
    -------
    Model
        Preset model with the specified dimensions on the specified tensor device

    Raises
    ------
    ValueError
        Raises exception when
    """
    model_name = model_name.lower()
    covar_dim, label_dim = fetcher.covar_dim, fetcher.label_dim

    if model_name == "logreg":
        return LogisticRegression(*covar_dim, *label_dim).to(device=device)
    elif model_name == "mlpclass":
        return ClassifierMLP(*covar_dim, *label_dim).to(device=device)
    elif model_name == "mlpregress":
        return RegressionMLP(*covar_dim, *label_dim).to(device=device)
    elif model_name == "bert":
        # Temporary fix while I figure out a better way for model factory
        from dataoob.model.bert import BertClassifier

        return BertClassifier(num_classes=label_dim[0]).to(device=device)
    else:
        raise ValueError(f"{model_name} is not a valid predefined model")
