r"""Prediction models to be trained, predict, and evaluated.

Models
======

.. currentmodule:: opendataval.model

:py:class:`Model` is an ABC used to take an existing model and make it compatible with
the :py:class:`~opendataval.dataval.DataEvaluator` and other related objects.

API
---
.. autosummary::
    :toctree: generated/

    Model
    GradientModel
    ModelFactory

Torch Mixins
------------
.. autosummary::
    :toctree: generated/

    TorchClassMixin
    TorchRegressMixin
    TorchPredictMixin
    TorchGradMixin

Sci-kit learn wrappers
----------------------
.. autosummary::
    :toctree: generated/

    ClassifierSkLearnWrapper
    ClassifierUnweightedSkLearnWrapper
    RegressionSkLearnWrapper


Default Hyperparameters
-----------------------

.. math::

    \newcommand\T{\Rule{0pt}{1em}{.3em}}

    \begin{array}{llll}
    \hline
    \textbf{Algorithm} & \textbf{Hyperparameter} & \textbf{Default Value} & \textbf{Key word argument} \\
    \hline
    \mbox{Logistic Regression}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\ & \mbox{loss function}
        & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{MLP Classification}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{BERT Classification}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.001 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAMW} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{LeNet-5 Classification}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Cross Entropy} & \mbox{no} \\
    \hline
    \mbox{MLP Regression}
        & \mbox{epochs} & 1 & \mbox{yes} \\
        & \mbox{batch size} & 32 & \mbox{yes} \\
        & \mbox{learning rate} & 0.01 & \mbox{yes} \\
        & \mbox{optimizer} & \mbox{ADAM} & \mbox{no} \\
        & \mbox{loss function} & \mbox{Mean Square Error} & \mbox{no} \\
    \hline
    \end{array}
"""
# Model Factory imports
from typing import Optional

import torch
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from opendataval.dataloader import DataFetcher
from opendataval.model.api import (
    ClassifierSkLearnWrapper,
    ClassifierUnweightedSkLearnWrapper,
    Model,
    RegressionSkLearnWrapper,
    TorchClassMixin,
    TorchPredictMixin,
    TorchRegressMixin,
)
from opendataval.model.bert import BertClassifier
from opendataval.model.grad import GradientModel, TorchGradMixin
from opendataval.model.lenet import LeNet
from opendataval.model.logistic_regression import LogisticRegression
from opendataval.model.mlp import ClassifierMLP, RegressionMLP


def ModelFactory(  # noqa: C901 model factory tries to match name with long if-else
    model_name: str,
    fetcher: Optional[DataFetcher] = None,
    device: torch.device = torch.device("cpu"),
    *args,
    **kwargs,
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
    args : tuple[Any]
        Additional positional arguments passed to the Model constructor
    kwargs : tuple[Any]
        Additional key word arguments passed to the Model constructor

    Returns
    -------
    Model
        Preset model with the specified dimensions on the specified tensor device

    Raises
    ------
    ValueError
        Raises exception when model name is not matched
    """
    covar_dim, label_dim = fetcher.covar_dim, fetcher.label_dim
    if model_name is None:
        return None
    model_name = model_name.lower()

    if model_name == "logisticregression":
        return LogisticRegression(*covar_dim, *label_dim, *args, **kwargs).to(device)
    elif model_name == "classifiermlp":
        return ClassifierMLP(*covar_dim, *label_dim, *args, **kwargs).to(device)
    elif model_name == "regressionmlp":
        return RegressionMLP(*covar_dim, *label_dim, *args, **kwargs).to(device)
    elif model_name == "bertclassifier":
        return BertClassifier(num_classes=label_dim[0], *args, **kwargs).to(device)
    elif model_name == "lenet":
        return LeNet(
            num_classes=label_dim[0],
            gray_scale=covar_dim[0] == 1,  # 1 means grey
            *args,
            **kwargs,
        ).to(device)
    elif model_name == "sklogreg":
        return ClassifierSkLearnWrapper(SkLogReg, label_dim[0], *args, **kwargs)
    elif model_name == "skmlp":
        return ClassifierUnweightedSkLearnWrapper(
            MLPClassifier, label_dim[0], *args, **kwargs
        )
    elif model_name == "skknn":
        return ClassifierUnweightedSkLearnWrapper(
            KNeighborsClassifier, label_dim[0], label_dim[0], *args, **kwargs
        )
    elif model_name == "sklinreg":
        return RegressionSkLearnWrapper(LinearRegression, *args, **kwargs)
    else:
        raise ValueError(f"{model_name} is not a valid predefined model")
