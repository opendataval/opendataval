"""Create data sets and loads with :py:class:`~opendataval.dataloader.DataFetcher`.

Data Loader
===========

.. currentmodule:: opendataval.dataloader

Provides an API to add new data sets and load them with the data loader.
To create a new data set, create a :py:class:`Register` object to register the data set
with a name. Then load the data set with :py:class:`DataFetcher`. This allows us the
flexibility to call the dataset later and to define separate functions/classes
for the covariates and labels of a data set

Creating/Loading data sets
--------------------------
.. autosummary::
    :toctree: generated/

    Register
    DataFetcher
    datasets

Utils
-----
.. autosummary::
   :toctree: generated/

    cache
    mix_labels
    one_hot_encode
    CatDataset
"""
from opendataval.dataloader import datasets
from opendataval.dataloader.fetcher import DataFetcher
from opendataval.dataloader.noisify import NoiseFunc, add_gauss_noise, mix_labels
from opendataval.dataloader.register import Register, cache, one_hot_encode
from opendataval.dataloader.util import CatDataset
