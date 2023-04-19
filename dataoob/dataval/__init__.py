"""Create :py:class:`~dataoob.dataval.DataEvaluator` to quantify the value of data.

Data Evaluator
==============

.. currentmodule:: dataoob.dataval

Provides an ABC for DataEvaluator to inherit from. The work flow is as follows:
:py:class:`~dataoob.dataloader.Register`, :py:class:`~dataoob.dataloader.DataFetcher`
-> :py:class:`~dataoob.dataval.DataEvaluator`
-> :py:mod:`~dataoob.evaluator.exper_methods`



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
"""
from dataoob.dataval.ame import AME
from dataoob.dataval.api import DataEvaluator
from dataoob.dataval.dvrl import DVRL
from dataoob.dataval.influence import InfluenceFunctionEval
from dataoob.dataval.knnshap import KNNShapley
from dataoob.dataval.margcontrib import (
    BetaShapley,
    DataBanzhaf,
    DataShapley,
    LeaveOneOut,
    ShapEvaluator,
)
from dataoob.dataval.oob import DataOob
