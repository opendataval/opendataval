import os
import warnings
from functools import partial
from typing import Callable, TypeVar, Union

import numpy as np
import pandas as pd
import requests
import tqdm
from torch.utils.data import Dataset

DatasetFunc = Callable[..., Union[Dataset, np.ndarray, tuple[np.ndarray, np.ndarray]]]
Self = TypeVar("Self")


def cache(url: str, cache_dir: str, file_name: str, force_download: bool) -> str:
    """Download a file if it it is not present and returns the file_path.

    Parameters
    ----------
    url : str
        URL of the file to be downloaded
    cache_dir : str
        Directory to cache downloaded files
    file_name : str, optional
        File name within the cache directory of the downloaded file, by default ""
    force_download : bool, optional
        Forces a download regardless if file is present, by default False

    Returns
    -------
    str
        File path to the downloaded file
    """
    if file_name is None:
        file_name = os.path.basename(url)

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    filepath = os.path.join(cache_dir, file_name)

    if not os.path.isfile(filepath) or force_download:
        with requests.get(url, stream=True, timeout=60) as r, open(filepath, "wb") as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192), "Downloading:"):
                f.write(chunk)

    return filepath


def one_hot_encode(data: np.ndarray) -> np.ndarray:
    """One hot encodes a 1D numpy array."""
    num_values = np.max(data) + 1
    return np.eye(num_values)[data]


def _read_csv(file_path: str, label_columns: Union[str, list]) -> DatasetFunc:
    """Create data set from csv file path, nested functions for api consistency."""
    return lambda: _from_pandas(pd.read_csv(file_path), label_columns)()


def _from_pandas(df: pd.DataFrame, label_columns: Union[str, list]) -> DatasetFunc:
    """Create data set from pandas dataframe, nested functions for api consistency."""
    if all(isinstance(col, int) for col in label_columns):
        label_columns = df.columns[label_columns]
    return lambda: (df.drop(label_columns, axis=1).values, df[label_columns].values)


def _from_numpy(array, label_columns: Union[str, list[int]]) -> DatasetFunc:
    """Create data set from numpy array, nested functions for api consistency."""
    if isinstance(label_columns, int):
        label_columns = [label_columns]
    return lambda: (np.delete(array, label_columns, axis=1), array[:, label_columns])


class Register:
    """Register a data set by defining its name and adding functions to retrieve data.

    Registers data sets to be fetched by the DataFetcher. Also allows specific
    transformations to be applied on a data set. This gives the benefit of creating
    :py:class:`Register` objects to distinguish separate data sets

    Parameters
    ----------
    dataset_name : str
        Data set name
    categorical : bool, optional
        Whether the data set is categorically labeled, by default False
    cacheable : bool, optional
        Whether data set can be downloaded and cached, by default False

    Warns
    ------
    Warning
        :py:class:`Register` keeps track of all data set names registered and all must
        be unique. If there are any duplicates, warns user.
    """

    CACHE_DIR = "data_files"

    Datasets: dict[str, Self] = {}
    """Creates a directory for all registered/downloadable data set functions."""

    def __init__(
        self, dataset_name: str, categorical: bool = False, cacheable: bool = False
    ):
        if dataset_name in Register.Datasets:
            warnings.warn(f"{dataset_name} has been registered, names must be unique")

        self.dataset_name = dataset_name
        self.categorical = categorical

        self.covar_transform = None
        self.label_transform = None
        if categorical:
            self.label_transform = one_hot_encode

        if cacheable:
            if not os.path.isdir(Register.CACHE_DIR):
                os.mkdir(Register.CACHE_DIR)

            self.download_dir = os.path.join(
                os.getcwd(), f"{Register.CACHE_DIR}/{dataset_name}/"
            )

        Register.Datasets[dataset_name] = self

    def from_csv(self, file_path: str, label_columns: Union[str, list]):
        """Register data set from csv file."""
        self.covar_label_func = _read_csv(file_path, label_columns)
        return self

    def from_pandas(self, df: pd.DataFrame, label_columns: Union[str, list]):
        """Register data set from pandas data frame."""
        self.covar_label_func = _from_pandas(df, label_columns)
        return self

    def from_numpy(self, df: pd.DataFrame, label_columns: Union[str, list[int]]):
        """Register data set from numpy array."""
        self.covar_label_func = _from_numpy(df, label_columns)
        return self

    def __call__(self, func: DatasetFunc, *args, **kwargs) -> DatasetFunc:
        """Majority of provided datasets are in `from_covar_label_func` format."""
        return self.from_covar_label_func(func, *args, **kwargs)

    def from_covar_label_func(self, func: DatasetFunc, *args, **kwargs) -> DatasetFunc:
        """Register data set from Callable -> (covariates, labels)."""
        self.covar_label_func = partial(func, *args, **kwargs)
        return func

    def from_covar_func(self, func: DatasetFunc, *args, **kwargs) -> DatasetFunc:
        """Register data set from 2 Callables, registers covariates Callable."""
        self.cov_func = partial(func, *args, **kwargs)
        return func

    def from_label_func(self, func: DatasetFunc, *args, **kwargs) -> DatasetFunc:
        """Register data set from 2 Callables, registers labels Callable."""
        self.label_func = partial(func, *args, **kwargs)
        return func

    def add_covar_transform(self, transform: Callable[[np.ndarray], np.ndarray]):
        """Add covariate transform after data is fetched."""
        self.covar_transform = transform
        return self

    def add_label_transform(self, transform: Callable[[np.ndarray], np.ndarray]):
        """Add label transform after data is fetched."""
        self.label_transform = transform
        return self

    def load_data(self, force_download: bool = False) -> tuple[Dataset, np.ndarray]:
        """Retrieve data from specified data input functions.

        Loads the covariates and labels from the registered callables, applies
        transformations, and returns the covariates and labels.

        Parameters
        ----------
        force_download : bool, optional
            Forces download from source URL, by default False

        Returns
        -------
        (np.ndarray | Dataset, np.ndarray)
            Transformed covariates and labels of the data set
        """
        dataset_kwargs = {}
        if hasattr(self, "download_dir"):
            dataset_kwargs["cache_dir"] = self.download_dir
            dataset_kwargs["force_download"] = force_download

        if hasattr(self, "covar_label_func"):
            covar, labels = self.covar_label_func(**dataset_kwargs)
        else:
            covar = self.cov_func(**dataset_kwargs)
            labels = self.label_func(**dataset_kwargs)

        if self.covar_transform:
            if isinstance(covar, Dataset):
                covar.transform = self.covar_transform
            else:
                covar = self.covar_transform(covar)

        if self.label_transform:
            labels = self.label_transform(labels)

        return covar, labels
