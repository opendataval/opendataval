"""Prediction models to be trained, predict, and evaluated.

Models
======

.. currentmodule:: dataoob.model

:py:class:`Model` is an ABC used to take an existing model and make it compatible with
the :py:class:`~dataoob.dataval.DataEvaluator` and other related objects.

ABC
---
.. autosummary::
    :toctree: generated/

    Model

Torch Mixins
------------
.. autosummary::
    :toctree: generated/

    TorchBinClassMixin
    TorchClassMixin
    TorchRegressMixin
    TorchPredictMixin

Sci-kit learn wrappers
----------------------
.. autosummary::
    :toctree: generated/

    ClassifierSkLearnWrapper
    ClassifierUnweightedSkLearnWrapper
"""
from dataoob.model.api import *
