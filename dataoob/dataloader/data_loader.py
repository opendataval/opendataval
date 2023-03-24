from itertools import accumulate

import numpy as np
import torch
from dataoob.dataloader.datasets import Register
from torch.utils.data import Dataset, Subset


def DataLoader(
    dataset_name: str,
    force_redownload: bool = False,
    train_count: int | float = 0,
    valid_count: int | float = 0,
    test_count: int | float = 0,
    noise_rate: float = 0.0,
    device: int = torch.device("cpu"),
) -> tuple[torch.Tensor]:
    """Dataloader for dataoob, input the data set name to receive covariates and labels
    split into train, valid, and test sets. Also returns noisy indices if any

    Parameters
    ----------
    dataset_name : str
        Name of the data set, can be registered in `datasets.py`
    force_redownload : bool, optional
        Forces redownload from source URL, by default False
    train_count : int | float, optional
        Number/proportion training points, by default 0
    valid_count : int | float, optional
        Number/proportion validation points, by default 0
    test_count : int | float, optional
        Number/proportion testing points, by default 0
    noise_rate : float, optional
        Ratio of data to add noise to, by default 0.0
    device : int, optional
        Tensor device for acceleration, by default torch.device("cpu")

    Returns
    -------
    (torch.Tensor | Dataset, torch.Tensor)
        Training Covariates, Training Labels
    (torch.Tensor | Dataset, torch.Tensor)
        Validation Covariates, Valid Labels
    (torch.Tensor | Dataset, torch.Tensor)
        Test Covariates, Test Labels
    (np.ndarray)
        Indices of noisified Training labels
    """
    x, y = load_dataset(dataset_name, device, force_redownload)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = split_dataset(
        x, y, train_count, valid_count, test_count
    )

    # Noisify the data
    y_train, noisy_indices = noisify(y_train, noise_rate)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), noisy_indices


def load_dataset(
    dataset_name: str, device: int = torch.device("cpu"), force_redownload: bool = False
) -> tuple[torch.Tensor | Dataset, torch.Tensor]:
    """Loads the data set from the dataset registry and loads as tensor on specified device

    Parameters
    ----------
    dataset_name : str
        Name of a registered data set
    device : int, optional
        Tensor device for acceleration, by default torch.device("cpu")
    force_redownload : bool, optional
        Forces redownload from source URL, by default False, by default False

    Returns
    -------
    (torch.Tensor | Dataset, torch.Tensor)
        Covariates and Labels of the data set

    Raises
    ------
    KeyError
        In order to use a data set, you must register it by creating a
        :py:class:`Register`
    """
    if dataset_name not in Register.Datasets:
        raise KeyError("Must register data set in register_dataset")

    covariates, labels = Register.Datasets[dataset_name].load_data(force_redownload)

    if not isinstance(covariates, Dataset):
        covariates = torch.tensor(covariates).to(dtype=torch.float32, device=device)
    labels = torch.tensor(labels).to(dtype=torch.float32, device=device)

    return covariates, labels


def noisify(
    labels: torch.Tensor, noise_rate: float = 0.0
) -> tuple[torch.Tensor, np.ndarray]:  # TODO leave for now change later
    if noise_rate == 0.0:
        return labels, np.array([])
    elif 0 <= noise_rate <= 1.0:
        num_points = labels.size(dim=0)

        noise_count = round(num_points * noise_rate)
        replace = np.random.choice(num_points, noise_count, replace=False)
        target = np.random.choice(num_points, noise_count, replace=False)
        labels[replace] = labels[target]

        return labels, replace
    else:
        raise Exception()


def split_dataset(
    x: torch.Tensor | Dataset,
    y: torch.Tensor,
    train_count: int | float,
    valid_count: int | float,
    test_count: int | float,
):
    """Splits the covariates and labels according to the specified counts/proportions

    Parameters
    ----------
    x : torch.Tensor | Dataset
        Data+Test+Held-out covariates
    y : torch.Tensor
        Data+Test+Held-out labels
    train_count : int | float
        Number/proportion training points
    valid_count : int | float
        Number/proportion validation points
    test_count : int | float
        Number/proportion testing points

    Returns
    -------
    (torch.Tensor | Dataset, torch.Tensor)
        Training Covariates, Training Labels
    (torch.Tensor | Dataset, torch.Tensor)
        Validation Covariates, Valid Labels
    (torch.Tensor | Dataset, torch.Tensor)
        Test Covariates, Test Labels

    Raises
    ------
    ValueError
        Invalid input for splitting the data set, either the proporition is more than 1.
        or the total splits are greater than the len(dataset)
    """
    assert len(x) == len(y)
    num_points = len(x)

    match (train_count, valid_count, test_count):
        case int(tr), int(val), int(tst) if sum((tr, val, tst)) <= num_points:
            splits = accumulate((tr, val, tst))
        case float(tr), float(val), float(tst) if sum((tr, val, tst)) <= 1.0:
            splits = (round(num_points * p) for p in (tr, val, tst))
            splits = accumulate(splits)
        case _:
            raise ValueError("Invalid split")

    # Extra underscore to unpack any remainders
    indices = np.random.permutation(num_points)
    train_idx, valid_idx, test_idx, _ = np.split(indices, list(splits))

    if isinstance(x, Dataset):
        x_train, y_train = Subset(x, train_idx), y[train_idx]
        x_valid, y_valid = Subset(x, valid_idx), y[valid_idx]
        x_test, y_test = Subset(x, test_idx), y[test_idx]
    else:
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]
        x_test, y_test = x[test_idx], y[test_idx]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
