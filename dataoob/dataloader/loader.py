from itertools import accumulate
from typing import Any, Callable, Self

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Dataset, Subset

from dataoob.dataloader.register import Register


class DataLoader:
    """Load data for an experiment from an input data set name.

    Facade for Register object, prepares the data and provides an API for subsequent
    splitting, adding noise, and transforming into a tensor.

    Parameters
    ----------
    dataset_name : str
        Name of the data set, can be registered in `datasets.py`
    force_download : bool, optional
        Forces download from source URL, by default False
    noise_rate : float, optional
        Ratio of data to add noise to, by default 0.0
    device : int, optional
        Tensor device for acceleration, by default torch.device("cpu")

    Raises
    ------
    KeyError
        In order to use a data set, you must register it by creating a
        :py:class:`Register`
    AttributeError
        No specified Covariates or labels. Ensure that the Register object
        has loaded your data set correctly
    ValueError
        Splits must not exceed the length of the data set. In other words, if
        the splits are ints, the values must be less than the length. If they are
        floats they must be less than 1.0. If they are anything else, raises error.
    ValueError

    """

    def __init__(
        self,
        dataset_name: str,
        force_download: bool = False,
        device: torch.device = torch.device("cpu"),
        random_state: RandomState = None,
    ):
        if dataset_name not in Register.Datasets:
            raise KeyError("Must register data set in register_dataset")

        dataset = Register.Datasets[dataset_name]
        self.covar, self.labels = dataset.load_data(force_download)

        self.device = device
        self.random_state = check_random_state(random_state)

    @staticmethod
    def datasets_available() -> list[str]:
        """Get list of available data set names."""
        return list(Register.Datasets.keys())

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
        if isinstance(self.covar, Dataset):
            x_train, x_valid, x_test = self.x_train, self.x_valid, self.x_test
        else:
            x_train = torch.tensor(self.x_train, device=self.device, dtype=torch.float)
            x_valid = torch.tensor(self.x_valid, device=self.device, dtype=torch.float)
            x_test = torch.tensor(self.x_test, device=self.device, dtype=torch.float)

        y_train = torch.tensor(self.y_train, device=self.device, dtype=torch.float)
        y_valid = torch.tensor(self.y_valid, device=self.device, dtype=torch.float)
        y_test = torch.tensor(self.y_test, device=self.device, dtype=torch.float)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def split_dataset(
        self,
        train_count: int | float = 0,
        valid_count: int | float = 0,
        test_count: int | float = 0,
    ):
        """Split the covariates and labels to the specified counts/proportions.

        Parameters
        ----------
        train_count : int | float
            Number/proportion training points
        valid_count : int | float
            Number/proportion validation points
        test_count : int | float
            Number/proportion test points

        Returns
        -------
        self : object
            Returns a DataLoader with covariates, labels split into train/valid.

        Raises
        ------
        AttributeError
            No specified Covariates or labels. Ensure that the Register object
            has loaded your data set correctly
        ValueError
            Invalid input for splitting the data set, either the proportion is more
            than 1 or the total splits are greater than the len(dataset)
        """
        if not (hasattr(self, "covar") and hasattr(self, "labels")):
            raise AttributeError(
                "No attribute covar, labels found make sure Register object is valid."
            )

        if not len(self.covar) == len(self.labels):
            raise ValueError("covariates and labels must be of same length.")

        num_points = len(self.covar)

        match (train_count, valid_count, test_count):
            case int(tr), int(val), int(tes) if sum((tr, val, tes)) <= num_points:
                splits = accumulate((tr, val, tes))
            case float(tr), float(val), float(tes) if sum((tr, val, tes)) <= 1.0:
                splits = (round(num_points * prob) for prob in (tr, val, tes))
                splits = accumulate(splits)
            case _:
                raise ValueError(
                    "Split can't exceed length and must be same type, def type is int."
                )

        # Extra underscore to unpack any remainders
        indices = self.random_state.permutation(num_points)
        train_indices, valid_indices, test_indices, _ = np.split(indices, list(splits))

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

        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices
        return self

    def noisify(
        self,
        add_noise_func: Callable[[Self, Any, ...], dict[str, np.ndarray | Dataset]],
        *noise_args,
        **noise_kwargs
    ):
        """Add noise to the data points.

        Adds noise to the data set and saves the indices of the noisy data.
        Return object of `add_noise_func` is a dict with keys to signify how the
        data are updated:
        {'x_train','y_train','x_valid','y_valid','x_test','y_test','noisy_indices'}

        Parameters
        ----------
        add_noise_func : Callable
            Takes as argument required arguments x_train, y_train, x_valid, y_valid
            and adds noise to those data points as needed. Returns dict[str, np.ndarray]
            that has the updated np.ndarray in a dict to update the data loader with the
            following keys:
            {'x_train','y_train','x_valid','y_valid','x_test','y_test','noisy_indices'}

        Returns
        -------
        self : object
            Returns a DataLoader with noise added to the data set.
        """
        # Passes the DataLoader to the noise_func, has access to all instance variables
        noisy_datapoints = add_noise_func(*noise_args, loader=self, **noise_kwargs)

        self.x_train = noisy_datapoints.get("x_train", self.x_train)
        self.y_train = noisy_datapoints.get("y_train", self.y_train)
        self.x_valid = noisy_datapoints.get("x_valid", self.x_valid)
        self.y_valid = noisy_datapoints.get("y_valid", self.y_valid)
        self.x_test = noisy_datapoints.get("x_test", self.x_test)
        self.y_test = noisy_datapoints.get("y_test", self.y_test)
        self.noisy_indices = noisy_datapoints.get("noisy_indices", np.array([]))

        return self


def mix_labels(loader: DataLoader, noise_rate: float) -> dict[str, np.ndarray]:
    """Mixes y_train labels of a DataLoader, adding noise to data.

    Parameters
    ----------
    loader : DataLoader
        DataLoader object housing the data to have noise added to
    noise_rate : float
        Proportion of labels to add noise to

    Returns
    -------
    dict[str, np.ndarray]
        dictionary of updated data points
    """
    y_train = loader.y_train
    rs = check_random_state(loader.random_state)
    num_points = len(y_train)
    replace = rs.choice(num_points, round(num_points * noise_rate), replace=False)
    target = rs.choice(num_points, round(num_points * noise_rate), replace=False)
    y_train[replace] = y_train[target]

    return {"y_train": y_train, "noisy_indices": replace}
