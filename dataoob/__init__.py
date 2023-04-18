"""Framework with data sets, experiments, and evaluators to quantify the worth of data.

dataoob
=======

.. currentmodule:: dataoob

:py:mod:`dataoob` provides a framework to evaluate the worth of data. The framework is
easily extendable via adding/registering new datasets via
:py:class:`~dataoob.dataloader.DataFetcher` + :py:class:`~dataoob.dataloader.Register`,
creating your own :py:class:`~dataoob.dataval.DataEvaluator` via inheritance, or
creating new experiments to be run by :py:class:`~dataoob.evaluator.ExperimentMediator`.
The framework provides a robust and replicable way of loading data, selecting a model,
training (several) data evaluators, and running an experiment to determine performance
on all of them.

Modules
-------
.. autosummary::
    :toctree: generated/

    dataloader
    dataval
    model
    evaluator
"""
__version__ = "0.0.1"
