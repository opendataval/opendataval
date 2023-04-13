"""Data sets registered with :py:class:`~dataoob.dataloader.register.Register`.

Data sets
=========
.. autosummary::
    :toctree: generated/

    datasets
    ~dataoob.dataloader.datasets.imagesets
    ~dataoob.dataloader.datasets.nlpsets

Catalog of registered data sets that can be used with
:py:class:`~dataoob.dataloader.loader.DataLoader`. Pass in the ``str`` name registering
the data set to load the data set as needed.

NOTE :py:mod:`~dataoob.dataloader.datasets.imagesets` and
:py:class:`~dataoob.dataloader.datasets.nlpsets` are not explicitly imported, you must
import manually as they have external dependencies, run `make install-extra`.
"""
from dataoob.dataloader.datasets import datasets
