from itertools import accumulate, chain
from typing import Any, Callable, Sequence, TypeVar, Union

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Dataset, Subset

from opendataval.dataloader.register import Register

Self = TypeVar("Self")


class DataFetcher:
    """Load data for an experiment from an input data set name.

    Facade for :py:class:`Register` object, prepares the data and provides an API for
    subsequent splitting, adding noise, and transforming into a tensor.

    Parameters
    ----------
    dataset_name : str
        Name of the data set, must be registered with :py:class:`Register`
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
    train_indices : np.ndarray[int]
        The indices of the original data set used to make the training data set.
    valid_indices : np.ndarray[[int]
        The indices of the original data set used to make the validation data set.
    test_indices : np.ndarray[[int]
        The indices of the original data set used to make the test data set.
    noisy_train_indices : np.ndarray[[int]
        The indices of training data points with noise added to them.
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
        Splits must not exceed the length of the data set. In other words, if
        the splits are ints, the values must be less than the length. If they are
        floats they must be less than 1.0. If they are anything else, raises error
    ValueError
        Specified indices must not repeat and must not be outside range of the data set
    """

    def __init__(
        self,
        dataset_name: str,
        force_download: bool = False,
        random_state: RandomState = None,
    ):
        if dataset_name not in Register.Datasets:
            raise KeyError(
                "Must register data set in register_dataset."
                "Ensure the data set is imported and installed optional dependencies."
            )

        dataset = Register.Datasets[dataset_name]
        self.covar, self.labels = dataset.load_data(force_download)
        if not len(self.covar) == len(self.labels):
            raise ValueError("Covariates and Labels must be of same length.")

        self.random_state = check_random_state(random_state)

    @staticmethod
    def datasets_available() -> list[str]:
        """Get list of available data set names."""
        return list(Register.Datasets.keys())

    @classmethod
    def setup(
        cls,
        dataset_name: str,
        force_download: bool = False,
        random_state: RandomState = None,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
        add_noise_func: Callable[[Self, Any, ...], dict[str, Any]] = None,
        noise_kwargs: dict[str, Any] = None,
    ):
        """Create, split, and add noise to DataFetcher from input arguments."""
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        return (
            cls(dataset_name, force_download, random_state)
            .split_dataset(train_count, valid_count, test_count)
            .noisify(add_noise_func, **noise_kwargs)
        )

    @classmethod
    def from_data(
        cls,
        covar: Union[Dataset, np.ndarray],
        labels: np.ndarray,
        random_state: RandomState = None,
    ):
        """Return DataFetcher from input Covariates and Labels."""
        fetcher = cls.__new__(cls)
        fetcher.covar, fetcher.labels = covar, labels
        if not len(fetcher.covar) == len(fetcher.labels):
            raise ValueError("Covariates and Labels must be of same length.")

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
        random_state: RandomState = None,
    ):
        """Return DataFetcher from already split data."""
        if not (
            len(x_train) == len(y_train)
            and len(x_valid) == len(y_valid)
            and len(x_test) == len(y_test)
        ):
            raise ValueError("Covariates and Labels must be of same length.")

        if not (
            x_train[0].shape == x_valid[0].shape == x_test[0].shape
            and y_train[0].shape == y_valid[0].shape == y_test[0].shape
        ):
            raise ValueError("Covariates and Labels inputs must be of same shape.")

        fetcher = cls.__new__(cls)
        fetcher.x_train, fetcher.y_train = x_train, y_train
        fetcher.x_valid, fetcher.y_valid = x_valid, y_valid
        fetcher.x_test, fetcher.y_test = x_test, y_test

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

        if not isinstance(self.x_train, Dataset):  # Turns arrays -> cpu tensors
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
        return (1,) if isinstance(data[0], str) or data.ndim == 1 else data.shape[1:]

    @property
    def label_dim(self) -> tuple[int, ...]:
        """Get label dimensions."""
        data = self.labels if hasattr(self, "labels") else self.y_train
        return (1,) if isinstance(data[0], str) or data.ndim == 1 else data.shape[1:]

    @property
    def num_points(self) -> int:
        """Get total number of data points."""
        if hasattr(self, "covar"):
            return len(self.covar)
        else:
            return len(self.x_train) + len(self.x_valid) + len(self.x_test)

    def split_dataset(
        self,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
    ):
        """Split the covariates and labels to the specified counts/proportions.

        Parameters
        ----------
        train_count : Union[int, float]
            Number/proportion training points
        valid_count : Union[int, float]
            Number/proportion validation points
        test_count : Union[int, float]
            Number/proportion test points

        Returns
        -------
        self : object
            Returns a DataFetcher with covariates, labels split into train/valid/test.

        Raises
        ------
        AttributeError
            No specified Covariates or labels. Ensure that the Register object
            has been fetched and your data set correctly
        ValueError
            Invalid input for splitting the data set, either the proportion is more
            than 1 or the total splits are greater than the len(dataset)
        """
        tr, val, tes = train_count, valid_count, test_count
        type_tuple = (type(tr), type(val), type(tes))  # Fix without structral match
        if sum((tr, val, tes)) <= self.num_points and type_tuple == (int, int, int):
            sp = list(accumulate((tr, val, tes)))
        elif sum((tr, val, tes)) <= 1.0 and type_tuple == (float, float, float):
            splits = (round(self.num_points * prob) for prob in (tr, val, tes))
            sp = list(accumulate(splits))
        else:
            raise ValueError(
                f"Splits must be < {self.num_points=} and of the same type: "
                f"{type(train_count)=}|{type(valid_count)=}|{type(test_count)=}."
            )

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
        train_indices: Sequence[int] = None,
        valid_indices: Sequence[int] = None,
        test_indices: Sequence[int] = None,
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
        add_noise_func: Callable[[Self, Any, ...], dict[str, Any]] = None,
        *noise_args,
        **noise_kwargs,
    ):
        """Add noise to the data points.

        Adds noise to the data set and saves the indices of the noisy data.
        Return object of `add_noise_func` is a dict with keys to signify how the
        data are updated:
        {'x_train','y_train','x_valid','y_valid','x_test','y_test','noisy_train_indices'}

        Parameters
        ----------
        add_noise_func : Callable
            If None, no changes are made. Takes as argument required arguments
            DataFetcher and adds noise to those the data points of DataFetcher as
            needed. Returns dict[str, np.ndarray] that has the updated np.ndarray in a
            dict to update the data loader with the following keys:

            - **"x_train"** -- Updated training covariates with noise, optional
            - **"y_train"** -- Updated training labels with noise, optional
            - **"x_valid"** -- Updated validation covariates with noise, optional
            - **"y_valid"** -- Updated validation labels with noise, optional
            - **"x_test"** -- Updated testing covariates with noise, optional
            - **"y_test"** -- Updated testing labels with noise, optional
            - **"noisy_train_indices"** -- Indices of training data set with noise.
        args : tuple[Any]
            Additional positional arguments passed to ``add_noise_func``
        kwargs: dict[str, Any]
            Additional key word arguments passed to ``add_noise_func``

        Returns
        -------
        self : object
            Returns a DataFetcher with noise added to the data set.
        """
        if add_noise_func is None:
            return self

        # Passes the DataFetcher to the noise_func, has access to all instance variables
        noisy_data = add_noise_func(fetcher=self, *noise_args, **noise_kwargs)

        self.x_train = noisy_data.get("x_train", self.x_train)
        self.y_train = noisy_data.get("y_train", self.y_train)
        self.x_valid = noisy_data.get("x_valid", self.x_valid)
        self.y_valid = noisy_data.get("y_valid", self.y_valid)
        self.x_test = noisy_data.get("x_test", self.x_test)
        self.y_test = noisy_data.get("y_test", self.y_test)
        self.noisy_train_indices = noisy_data.get("noisy_train_indices", np.array([]))

        return self
