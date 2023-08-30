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
    ModelMixin
    EmbeddingMixin
    AME
    DVRL
    InfluenceFunction
    InfluenceSubsample
    KNNShapley
    DataOob
    DataBanzhaf
    BetaShapley
    DataShapley
    LavaEvaluator
    LeaveOneOut
    ShapEvaluator
    RandomEvaluator
    RobustVolumeShapley
    Sampler
    TMCSampler
    GrTMCSampler
"""
from opendataval.dataval.ame import AME
from opendataval.dataval.api import DataEvaluator, EmbeddingMixin, ModelMixin
from opendataval.dataval.csshap import ClassWiseShapley
from opendataval.dataval.dvrl import DVRL
from opendataval.dataval.influence import InfluenceFunction, InfluenceSubsample
from opendataval.dataval.knnshap import KNNShapley
from opendataval.dataval.lava import LavaEvaluator
from opendataval.dataval.margcontrib import (
    BetaShapley,
    DataBanzhaf,
    DataBanzhafMargContrib,
    DataShapley,
    GrTMCSampler,
    LeaveOneOut,
    Sampler,
    ShapEvaluator,
    TMCSampler,
)
from opendataval.dataval.oob import DataOob
from opendataval.dataval.random import RandomEvaluator
from opendataval.dataval.volume import RobustVolumeShapley
