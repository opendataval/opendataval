import os
from typing import Any, Callable, Self

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

DatasetCallable = Callable[..., Dataset | np.ndarray | tuple[np.ndarray, np.ndarray]]


def one_hot_encode(data: np.ndarray) -> np.ndarray:
    """One hot encodes a 1D numpy array."""
    num_values = np.max(data) + 1
    return np.eye(num_values)[data]


def _read_csv(file_path: str, label_columns: str | list) -> DatasetCallable:
    """Create data set from csv file path, nested functions for api consistency."""
    return lambda: _from_pandas(pd.read_csv(file_path), label_columns)()


def _from_pandas(df: pd.DataFrame, label_columns: str | list) -> DatasetCallable:
    """Create data set from pandas dataframe, nested functions for api consistency."""
    if all(isinstance(col, int) for col in label_columns):
        label_columns = df.columns[label_columns]
    return lambda: (df.drop(label_columns, axis=1).values, df[label_columns].values)


def _from_numpy(array, label_columns: int | list[int]) -> DatasetCallable:
    """Create data set from numpy array, nested functions for api consistency."""
    if isinstance(label_columns, int):
        label_columns = [label_columns]
    return lambda: (np.delete(array, label_columns, axis=1), array[:, label_columns])


class Register:
    """Register a data set by defining its name and adding functions to retrieve data.

    Registers data sets to be loaded by the DataLoader. Also allows specific
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
    dataset_kwargs : dict[str, Any], optional
        Keyword arguments to pass to the data set functions, by default None

    Raises
    ------
    KeyError
        :py:class:`Register` keeps track of all data set names registered and all must
        be unique. If there are any duplicates, raises KeyError.
    """

    CACHE_DIR = "data_files"

    Datasets: dict[str, Self] = {}
    """Creates a directory for all registered/downloadable data set functions."""

    def __init__(
        self,
        dataset_name: str,
        categorical: bool = False,
        cacheable: bool = False,
        dataset_kwargs: dict[str, Any] = None,
    ):
        if dataset_name in Register.Datasets:
            raise KeyError(f"{dataset_name} has been registered, names must be unique")

        self.dataset_name = dataset_name
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.categorical = categorical

        self.covar_transform = None
        self.label_transform = None
        if categorical:
            self.label_transform = one_hot_encode

        if cacheable:
            self.download_dir = os.path.join(
                os.getcwd(), f"{Register.CACHE_DIR}/{dataset_name}"
            )

        Register.Datasets[dataset_name] = self

    def from_csv(self, file_path: str, label_columns: str | list):
        """Register data set from csv file."""
        self.covar_label_func = _read_csv(file_path, label_columns)
        return self

    def from_pandas(self, df: pd.DataFrame, label_columns: str | list):
        """Register data set from pandas data frame."""
        self.covar_label_func = _from_pandas(df, label_columns)
        return self

    def from_numpy(self, df: pd.DataFrame, label_columns: int | list[int]):
        """Register data set from numpy array."""
        self.covar_label_func = _from_numpy(df, label_columns)
        return self

    def from_covar_label_func(self, func: DatasetCallable) -> DatasetCallable:
        """Register data set from Callable -> (covariates, labels)."""
        self.covar_label_func = func
        return func

    def from_covar_func(self, func: DatasetCallable) -> DatasetCallable:
        """Register data set from 2 Callables, registers covariates Callable."""
        self.cov_func = func
        return func

    def from_label_func(self, func: DatasetCallable) -> DatasetCallable:
        """Register data set from 2 Callables, registers labels Callable."""
        self.label_func = func
        return func

    def add_covar_transform(self, transform: Callable[[np.ndarray], np.ndarray]):
        """Add covariate transform after data is loaded."""
        self.covar_transform = transform
        return self

    def add_label_transform(self, transform: Callable[[np.ndarray], np.ndarray]):
        """Add label transform after data is loaded."""
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
        if hasattr(self, "download_dir"):
            self.dataset_kwargs["cache_dir"] = self.download_dir
            self.dataset_kwargs["force_download"] = force_download

        if hasattr(self, "covar_label_func"):
            covar, labels = self.covar_label_func(**self.dataset_kwargs)
        else:
            covar = self.cov_func(**self.dataset_kwargs)
            labels = self.label_func(**self.dataset_kwargs)

        if self.covar_transform:
            if isinstance(covar, Dataset):
                covar.transform = self.covar_transform
            else:
                covar = self.covar_transform(covar)

        if self.label_transform:
            labels = self.label_transform(labels)

        return covar, labels
