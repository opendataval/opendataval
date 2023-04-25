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

NOTE :py:mod:`~opendataval.dataloader.datasets.imagesets` and
:py:class:`~opendataval.dataloader.datasets.nlpsets` have external dependencies,
run `make install-extra`.
"""
from opendataval.dataloader.datasets import datasets

try:
    from opendataval.dataloader.datasets import nlpsets
except ImportError as e:
    print(
        f"Failed to import nlpsets or imagesets, likely optional dependency not found."
        f"Error message is as follows: {e}"
    )
