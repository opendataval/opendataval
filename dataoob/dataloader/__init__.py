"""Create data sets and loads with :py:class:`~dataoob.dataloader.DataLoader`.

Data Loader
===========

.. currentmodule:: dataoob.dataloader

Provides an API to add new data sets and load them with the data loader.
To create a new data set, create a :py:class:`Register` object to register the data set
with a name. Then load the data set with :py:class:`DataLoader`. This allows us the
flexibility to call the dataset later and to define separate functions/classes
for the covariates and labels of a data set

Creating/Loading data sets
--------------------------
.. autosummary::
    :toctree: generated/

    Register
    DataLoader
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
from dataoob.dataloader import datasets
from dataoob.dataloader.loader import DataLoader
from dataoob.dataloader.noisify import add_gauss_noise, mix_labels
from dataoob.dataloader.register import Register, cache, one_hot_encode
from dataoob.dataloader.util import CatDataset
