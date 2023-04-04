from itertools import accumulate

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state
import torch
from torch.utils.data import Dataset, Subset

from dataoob.dataloader.datasets import Register
from typing import Any, Callable, Self


class DataLoader:
    """DataLoader for dataoob, given input dataset name, prepares the data and provides
    an API for subsequent splitting and adding noise

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

    @property
    def datapoints(self):
        """Returns split data points to be input into a DataEvaluator as tensors

        Returns
        -------
        (torch.Tensor | Dataset, torch.Tensor)
            Training Covariates, Training Labels
        (torch.Tensor | Dataset, torch.Tensor)
            Validation+Test Covariates, Valid+test Labels
        """
        if isinstance(self.covar, Dataset):
            x_train, x_valid = self.x_train, self.x_valid
        else:
            x_train = torch.tensor(self.x_train, device=self.device, dtype=torch.float)
            x_valid = torch.tensor(self.x_valid, device=self.device, dtype=torch.float)

        y_train = torch.tensor(self.y_train, device=self.device, dtype=torch.float)
        y_valid = torch.tensor(self.y_valid, device=self.device, dtype=torch.float)

        return x_train, y_train, x_valid, y_valid

    def split_dataset(self, train_count: int | float = 0, valid_count: int | float = 0):
        """Splits the covariates and labels to the specified counts/proportions

        Parameters
        ----------
        train_count : int | float
            Number/proportion training points
        valid_count : int | float
            Number/proportion validation points

        Returns
        -------
        self : object
            Returns a DataLoader with covariates, labels split into train/valid.

        Raises
        ------
        ValueError
            Invalid input for splitting the data set, either the proportion is more
            than 1 or the total splits are greater than the len(dataset)
        """
        assert hasattr(self, "covar") and hasattr(self, "labels")
        assert len(self.covar) == len(self.labels)
        num_points = len(self.covar)

        match (train_count, valid_count):
            case int(train), int(valid) if sum((train, valid)) <= num_points:
                splits = accumulate((train, valid))
            case float(train), float(valid) if sum((train, valid)) <= 1.0:
                splits = (round(num_points * prob) for prob in (train, valid))
                splits = accumulate(splits)
            case _:
                raise ValueError("Invalid split")

        # Extra underscore to unpack any remainders
        indices = self.random_state.permutation(num_points)
        train_idx, valid_idx, _ = np.split(indices, list(splits))

        if isinstance(self.covar, Dataset):
            self.x_train = Subset(self.covar, train_idx)
            self.x_valid = Subset(self.covar, valid_idx)
        else:
            self.x_train, self.x_valid = self.covar[train_idx], self.covar[valid_idx]
        self.y_train, self.y_valid = self.labels[train_idx], self.labels[valid_idx]

        return self

    def noisify(
        self,
        add_noise_func: Callable[[Self, Any, ...], dict[str, np.ndarray | Dataset]],
        *noise_args,
        **noise_kwargs
    ):
        """Adds noise to the data set and saves the indices of the noisy data

        Parameters
        ----------
        add_noise_func : Callable
            Takes as argument required arguments x_train, y_train, x_valid, y_valid
            and adds noise to those data points as needed. Returns dict[str, np.ndarray]
            that has the updated np.ndarray in a dict to update the data loader

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
        self.noisy_indices = noisy_datapoints.get("noisy_indices", np.array([]))

        return self


def mix_labels(loader: DataLoader, noise_rate: float) -> dict[str, np.ndarray]:
    """Mixes y_train labels of a DataLoader, adding noise to data

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
