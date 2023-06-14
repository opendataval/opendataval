import os
import warnings
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import requests
import tqdm
from torch.utils.data import Dataset

DatasetFunc = Callable[..., Union[Dataset, np.ndarray, tuple[np.ndarray, np.ndarray]]]
Self = TypeVar("Self")


@lru_cache()
def _request_session():
    return requests.Session()


def cache(
    url: str, cache_dir: Path, file_name: str = None, force_download: bool = False
) -> Path:
    """Download a file if it it is not present and returns the filepath.

    Parameters
    ----------
    url : str
        URL of the file to be downloaded
    cache_dir : str
        Directory to cache downloaded files
    file_name : str, optional
        File name within the cache directory of the downloaded file, by default None
    force_download : bool, optional
        Forces a download regardless if file is present, by default False

    Returns
    -------
    str
        File path to the downloaded file

    Raises
    ------
    HTTPError
        HTTP error occurred during downloading the dataset.
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if file_name is None:
        file_name = os.path.basename(url)

    filepath = cache_dir / file_name
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists() or force_download:
        filepath.touch(exist_ok=True)
        session = _request_session()

        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192), "Downloading:"):
                    f.write(chunk)

    return filepath


def one_hot_encode(data: np.ndarray) -> np.ndarray:
    """One hot encodes a numpy array.

    Raises
    ------
    ValueError
        When the input array is not of shape (N,), (N,1), (N,1,1)...
    """
    data = data.reshape(len(data))  # Reduces shape to (N,) array
    num_values = np.max(data) + 1
    return np.eye(num_values)[data]


def _read_csv(filepath: str, label_columns: Union[str, list]):
    """Create data set from csv file path, nested functions for api consistency."""
    return _from_pandas(pd.read_csv(filepath), label_columns)


def _from_pandas(df: pd.DataFrame, labels: Union[str, list]):
    """Create data set from pandas dataframe, nested functions for api consistency."""
    if all(isinstance(col, int) for col in labels):
        labels = df.columns[labels]
    return df.drop(labels, axis=1).values, df[labels].values


def _from_numpy(array: np.ndarray, label_columns: Union[int, list[int]]):
    """Create data set from numpy array, nested functions for api consistency."""
    if isinstance(label_columns, int):
        label_columns = [label_columns]
    return np.delete(array, label_columns, axis=1), array[:, label_columns]


class Register:
    """Register a data set by defining its name and adding functions to retrieve data.

    Registers data sets to be fetched by the DataFetcher. Also allows specific
    transformations to be applied on a data set. This gives the benefit of creating
    :py:class:`Register` objects to distinguish separate data sets

    Parameters
    ----------
    dataset_name : str
        Data set name
    one_hot : bool, optional
        Whether the data set should be one hot encoded labeled, by default False
    cacheable : bool, optional
        Whether data set can be downloaded and cached, by default False
    presplit : bool, optional
        Whether the data set was presplit, by default False

    Warns
    ------
    Warning
        :py:class:`Register` keeps track of all data set names registered and all must
        be unique. If there are any duplicates, warns user.
    """

    CACHE_DIR = "data_files"
    """Default directory to cache downloads to."""

    Datasets: dict[str, Self] = {}
    """Creates a directory for all registered/downloadable data set functions."""

    def __init__(
        self,
        dataset_name: str,
        one_hot: bool = False,
        cacheable: bool = False,
        presplit: bool = False,
    ):
        if dataset_name in Register.Datasets:
            warnings.warn(f"{dataset_name} has been registered, names must be unique")

        self.dataset_name = dataset_name
        self.one_hot = one_hot
        self.presplit = presplit

        self.covar_transform = None
        self.label_transform = None

        if self.one_hot:
            self.label_transform = one_hot_encode

        self.cacheable = cacheable

        Register.Datasets[dataset_name] = self

    def from_csv(self, filepath: str, label_columns: Union[str, list]):
        """Register data set from csv file."""
        self.covar_label_func = lambda: _read_csv(filepath, label_columns)
        return self

    def from_pandas(self, df: pd.DataFrame, label_columns: Union[str, list]):
        """Register data set from pandas data frame."""
        self.covar_label_func = lambda: _from_pandas(df, label_columns)
        return self

    def from_numpy(self, array: np.ndarray, label_columns: Union[int, Sequence[int]]):
        """Register data set from covariate and label numpy array."""
        self.covar_label_func = lambda: _from_numpy(array, label_columns)
        return self

    def from_data(self, covar: np.ndarray, label: np.ndarray, one_hot: bool = None):
        """Register data set from covariate and label numpy array."""
        self.covar_label_func = lambda: (covar, label)
        # Overrides default one_hot if specified
        if one_hot is not None:
            self.one_hot = one_hot
            self.cacheable = False
            self.label_transform = one_hot_encode if one_hot else self.label_transform

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

    def load_data(
        self, cache_dir: str = None, force_download: bool = False
    ) -> tuple[Dataset, np.ndarray]:
        """Retrieve data from specified data input functions.

        Loads the covariates and labels from the registered callables, applies
        transformations, and returns the covariates and labels.

        Parameters
        ----------
        cache_dir : str, optional
            Directory of where to cache the loaded data, by default None which uses
            :py:attr:`Register.CACHE_DIR`
        force_download : bool, optional
            Forces download from source URL, by default False

        Returns
        -------
        (np.ndarray | Dataset, np.ndarray)
            Transformed covariates and labels of the data set
        """
        dataset_kwargs = {}

        if self.cacheable:
            cache_dir = Path(cache_dir if cache_dir is not None else Register.CACHE_DIR)
            cache_dir.mkdir(parents=True, exist_ok=True)

            full_path = cache_dir / self.dataset_name
            dataset_kwargs["cache_dir"] = str(full_path)
            dataset_kwargs["force_download"] = force_download

        if hasattr(self, "covar_label_func"):
            covar, label = self.covar_label_func(**dataset_kwargs)
        else:
            covar = self.cov_func(**dataset_kwargs)
            label = self.label_func(**dataset_kwargs)

        # Wraps response in tuple in case data is not presplit
        covar_tup = covar if self.presplit else (covar,)
        label_tup = label if self.presplit else (label,)

        if self.covar_transform:
            if isinstance(covar, Dataset):
                for cov in covar_tup:
                    cov.transform = self.covar_transform
            else:
                covar_tup = tuple(self.covar_transform(cov) for cov in covar_tup)

        if self.label_transform:
            label_tup = tuple(self.label_transform(lab) for lab in label_tup)

        return *covar_tup, *label_tup
