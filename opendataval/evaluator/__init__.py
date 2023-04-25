"""Run experiments on :py:class:`~opendataval.dataval.DataEvaluator`.

Evaluator
=========

.. currentmodule:: opendataval.evaluator

:py:class:`ExperimentMediator` provides an API to set up an experiment.
In :py:mod:`~opendataval.evaluator.exper_methods`, there are several functions that can
be used with :py:class:`ExperimentMediator` to test a
:py:class:`~opendataval.dataval.DataEvaluator`.

Experiment Setup
----------------
.. autosummary::
    :toctree: generated/

    ExperimentMediator

Experiments
-----------
.. autosummary::
    :toctree: generated/

    exper_methods
"""

from opendataval.evaluator.api import ExperimentMediator
from opendataval.evaluator.exper_methods import *
