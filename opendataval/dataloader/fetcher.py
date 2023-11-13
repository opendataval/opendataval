import json
import warnings
from itertools import accumulate, chain
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, Dataset, Subset

from opendataval.dataloader.register import Register
from opendataval.dataloader.util import CatDataset

Self = TypeVar("Self", bound="DataFetcher")


class DataFetcher:
    """Load data for an experiment from an input data set name.

    Facade for :py:class:`Register` object, prepares the data and provides an API for
    subsequent splitting, adding noise, and transforming into a tensor.

    Parameters
    ----------
    dataset_name : str
        Name of the data set, must be registered with :py:class:`Register`
    cache_dir : str, optional
        Directory of where to cache the loaded data, by default None which uses
        :py:attr:`Register.CACHE_DIR`
    force_download : bool, optional
        Forces download from source URL, by default False
    random_state : RandomState, optional
        Random initial state, by default None

    Attributes
    ----------
    datapoints : tuple[torch.Tensor, ...]
        Train+Valid+Test covariates and labels
    covar_dim : tuple[int, ...]
        Covariates dimension of the loaded data set.
    label_dim : tuple[int, ...]
        Label dimension of the loaded data set.
    num_points : int
        Number of data points in the total data set
    one_hot : bool
        If True, the data set has categorical labels as one hot encodings
    [train/valid/test]_indices : np.ndarray[int]
        The indices of the original data set used to make the training data set.
    noisy_train_indices : np.ndarray[int]
        The indices of training data points with noise added to them.
    covar : Dataset | np.ndarray
        Covariate dataset a dataset function
    lables : np.ndarray
        Corresponding labels for covariates from a dataset function
    [x/y]_[train/valid/test] : np.ndarray
        Access to the raw split of the [covariate/label] [train/valid/test] data set
        prior being transformed into a tensor. Useful for adding noise to functions.

    Raises
    ------
    KeyError
        In order to use a data set, you must register it by creating a
        :py:class:`Register`
    ValueError
        Loaded Data set covariates and labels must be of same length.
    ValueError
        All covariates must be of same dimension. All labels must be of same dimension.
    ValueError
        Splits must not exceed the length of the data set. In other words, if
        the splits are ints, the values must be less than the length. If they are
        floats they must be less than 1.0. If they are anything else, raises error
    ValueError
        Specified indices must not repeat and must not be outside range of the data set
    """

    def __init__(
        self,
        dataset_name: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        random_state: Optional[RandomState] = None,
    ):
        if dataset_name not in Register.Datasets:
            raise KeyError(
                "Must register data set in register_dataset."
                "Ensure the data set is imported and optional dependencies installed."
            )

        self.dataset = Register.Datasets[dataset_name]
        self.one_hot = self.dataset.one_hot

        if self.dataset.presplit:
            self._presplit_data(*self.dataset.load_data(cache_dir, force_download))
        else:
            self._add_data(*self.dataset.load_data(cache_dir, force_download))

        self.random_state = check_random_state(random_state)
        self.noisy_train_indices = np.array([])

    def _presplit_data(self, x_train, x_valid, x_test, y_train, y_valid, y_test):
        if not len(x_train) == len(y_train):
            raise ValueError("Training Covariates and Labels must be of same length.")
        if not len(x_valid) == len(y_valid):
            raise ValueError("Validation Covariates and Labels must be of same length.")
        if not len(x_test) == len(y_test):
            raise ValueError("Testing Covariates and Labels must be of same length.")

        if not (x_train[0].shape == x_valid[0].shape == x_test[0].shape):
            raise ValueError("Covariates must be of same shape.")
        if not (y_train[0].shape == y_valid[0].shape == y_test[0].shape):
            raise ValueError("Labels must be of same shape.")

        self.x_train, self.x_valid, self.x_test = x_train, x_valid, x_test
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test

        tr, val, test = len(self.x_train), len(self.x_valid), len(self.x_test)
        self.train_indices = np.fromiter(range(tr), dtype=int)
        self.valid_indices = np.fromiter(range(tr, tr + val), dtype=int)
        self.test_indices = np.fromiter(range(tr + val, tr + val + test), dtype=int)

    def _add_data(self, covar, labels):
        if not len(covar) == len(labels):
            raise ValueError("Covariates and Labels must be of same length.")
        self.covar, self.labels = covar, labels

    @staticmethod
    def datasets_available() -> set[str]:
        """Get set of available data set names."""
        return set(Register.Datasets.keys())

    @classmethod
    def setup(
        cls,
        dataset_name: str,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        random_state: Optional[RandomState] = None,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
        add_noise: Optional[Callable[[Self, Any, ...], dict[str, Any]]] = None,
        noise_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Create, split, and add noise to DataFetcher from input arguments."""
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        split_types = (type(train_count), type(valid_count), type(test_count))
        if split_types == (int, int, int):
            return (
                cls(dataset_name, cache_dir, force_download, random_state)
                .split_dataset_by_count(train_count, valid_count, test_count)
                .noisify(add_noise, **noise_kwargs)
            )
        elif split_types == (float, float, float):
            return (
                cls(dataset_name, cache_dir, force_download, random_state)
                .split_dataset_by_prop(train_count, valid_count, test_count)
                .noisify(add_noise, **noise_kwargs)
            )
        else:
            raise ValueError(
                f"Expected split types to all of int or float but got "
                f"{type(train_count)=}|{type(valid_count)}|{type(test_count)=}"
            )

    @classmethod
    def from_data(
        cls,
        covar: Union[Dataset, np.ndarray],
        labels: np.ndarray,
        one_hot: bool,
        random_state: Optional[RandomState] = None,
    ):
        """Return DataFetcher from input Covariates and Labels.

        Parameters
        ----------
        covar : Union[Dataset, np.ndarray]
            Input covariates
        labels : np.ndarray
            Input labels, no transformation is applied, therefore if the input data
            should be one hot encoded, the transform is not applied
        one_hot : bool
            Whether the input data has already been one hot encoded. This is just a flag
            and not transform will be applied
        random_state : RandomState, optional
            Initial random state, by default None

        Raises
        ------
        ValueError
            Input covariates and labels are of different length, no 1-to-1 mapping.
        """
        fetcher = cls.__new__(cls)
        fetcher._add_data(covar, labels)

        fetcher.one_hot = one_hot
        fetcher.random_state = check_random_state(random_state)

        return fetcher

    @classmethod
    def from_data_splits(
        cls,
        x_train: Union[Dataset, np.ndarray],
        y_train: np.ndarray,
        x_valid: Union[Dataset, np.ndarray],
        y_valid: np.ndarray,
        x_test: Union[Dataset, np.ndarray],
        y_test: np.ndarray,
        one_hot: bool,
        random_state: Optional[RandomState] = None,
    ):
        """Return DataFetcher from already split data.

        Parameters
        ----------
        x_train : Union[Dataset, np.ndarray]
            Input training covariates
        y_train : np.ndarray
            Input training labels
        x_valid : Union[Dataset, np.ndarray]
            Input validation covariates
        y_valid : np.ndarray
            Input validation labels
        x_test : Union[Dataset, np.ndarray]
            Input testing covariates
        y_test : np.ndarray
            Input testing labels
        one_hot : bool
            Whether the label data has already been one hot encoded. This is just a flag
            and not transform will be applied
        random_state : RandomState, optional
            Initial random state, by default None

        Raises
        ------
        ValueError
            Loaded Data set covariates and labels must be of same length.
        ValueError
            All covariates must be of same dimension.
            All labels must be of same dimension.
        """
        fetcher = cls.__new__(cls)
        fetcher._presplit_data(x_train, x_valid, x_test, y_train, y_valid, y_test)

        fetcher.one_hot = one_hot
        fetcher.random_state = check_random_state(random_state)

        return fetcher

    @property
    def datapoints(self):
        """Return split data points to be input into a DataEvaluator as tensors.

        Returns
        -------
        (torch.Tensor | Dataset, torch.Tensor)
            Training Covariates, Training Labels
        (torch.Tensor | Dataset, torch.Tensor)
            Validation Covariates, Valid Labels
        (torch.Tensor | Dataset, torch.Tensor)
            Test Covariates, Test Labels
        """
        x_trn, x_val, x_test = self.x_train, self.x_valid, self.x_test

        if not isinstance(self.x_train, Dataset) and not isinstance(
            self.x_train, torch.Tensor
        ):  # Turns arrays -> cpu tensors
            x_trn = torch.tensor(x_trn, dtype=torch.float).view(-1, *self.covar_dim)
            x_val = torch.tensor(x_val, dtype=torch.float).view(-1, *self.covar_dim)
            x_test = torch.tensor(x_test, dtype=torch.float).view(-1, *self.covar_dim)

        y_trn = torch.tensor(self.y_train, dtype=torch.float).view(-1, *self.label_dim)
        y_val = torch.tensor(self.y_valid, dtype=torch.float).view(-1, *self.label_dim)
        y_test = torch.tensor(self.y_test, dtype=torch.float).view(-1, *self.label_dim)

        return x_trn, y_trn, x_val, y_val, x_test, y_test

    @property
    def covar_dim(self) -> tuple[int, ...]:
        """Get covar dimensions."""
        data = self.covar if hasattr(self, "covar") else self.x_train
        if isinstance(data, Dataset):
            return (1,) if isinstance(data[0], (str, float, int)) else data[0].shape
        else:
            return (1,) if data.ndim == 1 else data.shape[1:]

    @property
    def label_dim(self) -> tuple[int, ...]:
        """Get label dimensions."""
        data = self.labels if hasattr(self, "labels") else self.y_train
        if isinstance(data, Dataset):
            return (1,) if isinstance(data[0], (str, float, int)) else data[0].shape
        else:
            return (1,) if data.ndim == 1 else data.shape[1:]

    @property
    def num_points(self) -> int:
        """Get total number of data points."""
        if hasattr(self, "covar"):
            return len(self.covar)
        else:
            return len(self.x_train) + len(self.x_valid) + len(self.x_test)

    def split_dataset_by_prop(
        self,
        train_prop: float = 0.0,
        valid_prop: float = 0.0,
        test_prop: float = 0.0,
    ):
        """Split the covariates and labels to the specified proportions."""
        train_count, valid_count, test_count = (
            round(self.num_points * p) for p in (train_prop, valid_prop, test_prop)
        )
        return self.split_dataset_by_count(train_count, valid_count, test_count)

    def split_dataset_by_count(
        self,
        train_count: int = 0,
        valid_count: int = 0,
        test_count: int = 0,
    ):
        """Split the covariates and labels to the specified counts.

        Parameters
        ----------
        train_count : int
            Number/proportion training points
        valid_count : int
            Number/proportion validation points
        test_count : int
            Number/proportion test points

        Returns
        -------
        self : object
            Returns a DataFetcher with covariates, labels split into train/valid/test.

        Raises
        ------
        ValueError
            Invalid input for splitting the data set, either the proportion is more
            than 1 or the total splits are greater than the len(dataset)
        """
        if hasattr(self, "dataset") and self.dataset.presplit:
            warnings.warn("Dataset is already presplit, no need to split data.")
            return self

        if sum((train_count, valid_count, test_count)) > self.num_points:
            raise ValueError(
                f"Split totals must be < {self.num_points=} and of the same type: "
            )
        sp = list(accumulate((train_count, valid_count, test_count)))

        # Extra underscore to unpack any remainders
        idx = self.random_state.permutation(self.num_points)
        self.train_indices, self.valid_indices, self.test_indices, _ = np.split(idx, sp)

        if isinstance(self.covar, Dataset):
            self.x_train = Subset(self.covar, self.train_indices)
            self.x_valid = Subset(self.covar, self.valid_indices)
            self.x_test = Subset(self.covar, self.test_indices)
        else:
            self.x_train = self.covar[self.train_indices]
            self.x_valid = self.covar[self.valid_indices]
            self.x_test = self.covar[self.test_indices]

        self.y_train = self.labels[self.train_indices]
        self.y_valid = self.labels[self.valid_indices]
        self.y_test = self.labels[self.test_indices]

        return self

    def split_dataset_by_indices(
        self,
        train_indices: Optional[Sequence[int]] = None,
        valid_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
    ):
        """Split the covariates and labels to the specified indices.

        Parameters
        ----------
        train_indices : Sequence[int]
            Indices of training data set
        valid_indices : Sequence[int]
            Indices of valid data set
        test_indices : Sequence[int]
            Indices of test data set

        Returns
        -------
        self : object
            Returns a DataFetcher with covariates, labels split into train/valid/test.

        Raises
        ------
        ValueError
            Invalid input for indices of the train, valid, or split data set, leak
            of at least 1 data point in the indices.
        """
        train_indices = [] if train_indices is None else train_indices
        valid_indices = [] if valid_indices is None else valid_indices
        test_indices = [] if test_indices is None else test_indices

        idx = chain(train_indices, valid_indices, test_indices)
        seen = set()
        for index in idx:
            if not (0 <= index < len(self.covar)) or index in seen:
                raise ValueError(
                    f"{index=} is repeated or is out of range for {self.num_points=}"
                )
            seen.add(index)

        if isinstance(self.covar, Dataset):
            self.x_train = Subset(self.covar, train_indices)
            self.x_valid = Subset(self.covar, valid_indices)
            self.x_test = Subset(self.covar, test_indices)
        else:
            self.x_train = self.covar[train_indices]
            self.x_valid = self.covar[valid_indices]
            self.x_test = self.covar[test_indices]

        self.y_train = self.labels[train_indices]
        self.y_valid = self.labels[valid_indices]
        self.y_test = self.labels[test_indices]

        self.train_indices = np.array(train_indices, dtype=int)
        self.valid_indices = np.array(valid_indices, dtype=int)
        self.test_indices = np.array(test_indices, dtype=int)

        return self

    def noisify(
        self,
        add_noise: Union[Callable[[Self, Any, ...], Self], str, None] = None,
        *noise_args,
        **noise_kwargs,
    ):
        """Add noise to the data points.

        Adds noise to the data set and saves the indices of the noisy data.
        Return object of `add_noise` is a dict with keys to signify how the
        data are updated:
        {'x_train','y_train','x_valid','y_valid','x_test','y_test','noisy_train_indices'}

        Parameters
        ----------
        add_noise : Callable
            If None, no changes are made. Takes as argument required arguments
            DataFetcher modifies the DataFetcher in place with updated parameters.
        args : tuple[Any]
            Additional positional arguments passed to ``add_noise``
        kwargs: dict[str, Any]
            Additional key word arguments passed to ``add_noise``

        Returns
        -------
        self : object
            Returns a DataFetcher with noise added to the data set.
        """
        if add_noise is None:
            return self
        if isinstance(add_noise, str):
            from opendataval.dataloader.noisify import NoiseFunc

            add_noise = NoiseFunc(add_noise)

        # Passes the DataFetcher to the noise_func, updates DataFetcher in-place
        return add_noise(fetcher=self, *noise_args, **noise_kwargs)

    def export_dataset(
        self,
        covariates_names: list[str],
        labels_names: list[str],
        output_directory: Path = Path.cwd(),
    ):
        if isinstance(covariates_names, str):
            covariates_names = [covariates_names]
        if isinstance(labels_names, str):
            labels_names = [labels_names]
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            output_directory.mkdir(parents=True)

        columns = covariates_names + labels_names
        x_train, x_valid, x_test = self.x_train, self.x_valid, self.x_test
        y_train, y_valid, y_test = self.y_train, self.y_valid, self.y_test

        if self.one_hot:
            y_train = np.argmax(y_train, axis=1, keepdims=True) if y_train.size else []
            y_valid = np.argmax(y_valid, axis=1, keepdims=True) if y_valid.size else []
            y_test = np.argmax(y_test, axis=1, keepdims=True) if y_test.size else []

        def generate_data(covariates, labels):
            data = CatDataset(covariates, labels)
            for cov, lab in DataLoader(data, batch_size=1, shuffle=False):
                yield from np.hstack((cov, lab))

        def save_to_csv(data, file_name):
            file_path = output_directory / file_name
            pd.DataFrame(data, columns=columns).to_csv(file_path, index=False)

        save_to_csv(generate_data(x_train, y_train), "train.csv")
        save_to_csv(generate_data(x_valid, y_valid), "valid.csv")
        save_to_csv(generate_data(x_test, y_test), "test.csv")

        noisy_indices = (
            self.noisy_train_indices.tolist()
            if hasattr(self, "noisy_train_indices")
            else []
        )
        out_path = output_directory / f"noisy-indices-{self.dataset.dataset_name}.json"
        with open(out_path, "w+") as f:
            json.dump(noisy_indices, f)

        return
