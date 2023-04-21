"""Run predefined and ad-hoc experiments on :py:class:`~dataoob.dataval.DataEvaluator`.

Evaluator
=========

.. currentmodule:: dataoob.evaluator

:py:class:`ExperimentMediator` provides an API to set up an experiment.
In :py:mod:`~dataoob.evaluator.exper_methods`, there are several functions that can be
used with :py:class:`ExperimentMediator` to test a
:py:class:`~dataoob.dataval.DataEvaluator`.

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

from dataoob.evaluator.api import ExperimentMediator
from dataoob.evaluator.exper_methods import *
