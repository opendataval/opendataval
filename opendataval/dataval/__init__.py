"""Create :py:class:`~opendataval.dataval.DataEvaluator` to quantify the value of data.

Data Evaluator
==============

.. currentmodule:: opendataval.dataval

Provides an ABC for DataEvaluator to inherit from. The work flow is as follows:
:py:class:`~opendataval.dataloader.Register`,
:py:class:`~opendataval.dataloader.DataFetcher`
-> :py:class:`~opendataval.dataval.DataEvaluator`
-> :py:mod:`~opendataval.experiment.exper_methods`



Catalog
-------
.. autosummary::
    :toctree: generated/

    DataEvaluator
    AME
    DVRL
    InfluenceFunctionEval
    KNNShapley
    DataOob
    DataBanzhaf
    BetaShapley
    DataShapley
    LeaveOneOut
    ShapEvaluator
    RandomEvaluator
"""
from opendataval.dataval.ame import AME
from opendataval.dataval.api import DataEvaluator
from opendataval.dataval.dvrl import DVRL
from opendataval.dataval.influence import InfluenceFunctionEval
from opendataval.dataval.knnshap import KNNShapley
from opendataval.dataval.margcontrib import (
    BetaShapley,
    DataBanzhaf,
    DataShapley,
    LeaveOneOut,
    ShapEvaluator,
)
from opendataval.dataval.oob import DataOob
from opendataval.dataval.random import RandomEvaluator
