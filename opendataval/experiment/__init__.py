"""Run experiments on :py:class:`~opendataval.dataval.DataEvaluator`.

Experiment
==========

.. currentmodule:: opendataval.experiment

:py:class:`ExperimentMediator` provides an API to set up an experiment.
In :py:mod:`~opendataval.experiment.exper_methods`, there are several functions that can
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

from opendataval.experiment.api import ExperimentMediator
from opendataval.experiment.exper_methods import *
