"""Data sets registered with :py:class:`~opendataval.dataloader.register.Register`.

Data sets
=========
.. autosummary::
    :toctree: generated/

    datasets
    imagesets
    nlpsets

Catalog of registered data sets that can be used with
:py:class:`~opendataval.dataloader.fetcher.DataFetcher`. Pass in the ``str`` name
registering the data set to load the data set as needed.
.
"""
from opendataval.dataloader.datasets import (
    challenge,
    cleanlab,
    datasets,
    imagesets,
    nlpsets,
)
