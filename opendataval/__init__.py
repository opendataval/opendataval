"""Framework with data sets, experiments, and evaluators to quantify the worth of data.

opendataval
=======

.. currentmodule:: opendataval

:py:mod:`opendataval` provides a framework to evaluate the worth of data. The framework
is easily extendable via adding/registering new datasets via
:py:class:`~opendataval.dataloader.DataFetcher` +
:py:class:`~opendataval.dataloader.Register`, creating your own
:py:class:`~opendataval.dataval.DataEvaluator` via inheritance, or creating new
experiments to be run by :py:class:`~opendataval.experiment.ExperimentMediator`.
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
    experiment

Utils
-----
.. autosummary::
    :toctree: generated/

    ~opendataval.util.set_random_state
    ~opendataval.util.load_mediator_output
    __version__
"""
__version__ = "1.0.0"
"""Version release number."""
