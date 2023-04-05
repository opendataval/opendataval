"""Create :py:class:`~dataoob.dataval.DataEvaluator` to quantify the value of data.

Data Evaluator
==============

.. currentmodule:: dataoob.dataval

Provides an ABC for DataEvaluator to inherit from. The work flow is as follows:
(:py:class:`~dataoob.dataloader.DataLoader`, :py:class:`~dataoob.dataloader.DataLoader`)
-> :py:class:`~dataoob.dataval.DataEvaluator`
-> :py:mod:`~dataoob.evaluator.exper_methods`



Catalog
-------
.. autosummary::
    :toctree: generated/

    DataEvaluator
    AME
    DVRL
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
from dataoob.dataval.knnshap import KNNShapley
from dataoob.dataval.oob import DataOob
from dataoob.dataval.shap import (
    BetaShapley,
    DataBanzhaf,
    DataShapley,
    LeaveOneOut,
    ShapEvaluator,
)
