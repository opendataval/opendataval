from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F
from dataoob.dataloader.datasets import DatasetDirectory
from torch.utils.data import Dataset, Subset


def DataLoader(
    dataset_name: str,
    train_count: int | float = 0,
    valid_count: int | float = 0,
    test_count: int | float = 0,
    categorical: bool = False,
    scaler: str = None,
    noise_rate: float = 0.0,
    device: int = torch.device("cpu"),
):
    """Dataloader for dataoob, input the dataset name and some additional parameters
    receive a dataset compatible for Data-oob.  TODO i don't love the api i built
    revisit when it comes time to rebuild. Ways to improve would be adding a numpy like
    modifications to it

    :param str dataset_name: Name of the dataset, can be registered in `datasets.py`
    :param bool force_redownload: Forces redownload from source URL, defaults to False
    :param int | float train_count: Number/proportion training points, defaults to 0
    :param int | float valid_count: Number/proportion validation points, defaults to 0
    :param int | float test_count: Number/proportion testing points, defaults to 0
    :param bool categorical: Whether the data is categorical, defaults to False
    :param callable (np.ndarray(m x n) -> np.ndarray(m x n)) scaler: Scaler that
    normalizes the data. NOTE This likely will change as the underlying datasets change,
    defaults to None
    :param float noise_rate: Ratio of noise to add to the data TODO think
    of other ways to add noise to the data, defaults to 0.
    :param torch.device device: Tensor device for acceleration, defaults to
    torch.device("cpu")

    :return torch.Tensor | Dataset, torch.Tensor: Training Covariates, Training Labels
    :return torch.Tensor | Dataset, torch.Tensor: Validation Covariates, Valid Labels
    :return torch.Tensor | Dataset, torch.Tensor: Test Covariates, Test Labels
    :return np.ndarray: Indices of noisified Training labels
    """
    x, y = load_dataset(dataset_name=dataset_name, device=device)
    # TODO pass in device, download and load functions are necessary

    # Scale the data
    if scaler:  # TODO API unification, maybe wrap in a class idk
        if isinstance(x, Dataset):
            x.transform = scaler
        else:
            x = scaler(x)
    y = one_hot_encode(y, device) if categorical else scaler(y)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = split_dataset(
        x, y, train_count=train_count, valid_count=valid_count, test_count=test_count
    )

    # Noisify the data
    y_train, noisy_indices = noisify(y_train, noise_rate)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), noisy_indices


def load_dataset(
    dataset_name: str, device: int = torch.device("cpu")
) -> tuple[torch.Tensor | Dataset, torch.Tensor]:
    if dataset_name not in DatasetDirectory:
        raise Exception("Must register Dataset in register_dataset")

    covariates, labels = DatasetDirectory[dataset_name](
        False
    )  # Pass in force download and device

    if not isinstance(covariates, Dataset):
        covariates = torch.tensor(covariates).to(dtype=torch.float32, device=device)
    labels = torch.tensor(labels).to(dtype=torch.float32, device=device)

    return covariates, labels


def one_hot_encode(
    data: torch.Tensor, device: int = torch.device("cpu")
) -> torch.Tensor:
    num_classes = int(torch.max(data).item()) + 1
    return F.one_hot(data.long(), num_classes).to(dtype=torch.float32, device=device)


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
) -> torch.Tensor:
    """Splits the covariates and labels according to the specified counts/proportions

    :param torch.Tensor | Dataset x: Data+Test+Held-out covariates
    :param torch.Tensor y: Data+Test+Held-out labels
    :param int | float train_count: Number/proportion training points, defaults to 0
    :param int | float valid_count: Number/proportion validation points, defaults to 0
    :param int | float test_count: Number/proportion testing points, defaults to 0
    :raises Exception: Raises exception when there's an invalid splitting of the dataset,
    ie, more datapoints than the total dataset requested.
    :return torch.Tensor, torch.Tensor: Training Covariates, Training Labels
    :return torch.Tensor, torch.Tensor: Validation Covariates, Validation Labels
    :return torch.Tensor, torch.Tensor: Test Covariates, Test Labels
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
            raise Exception()  # TODO

    # Extra underscore to unpack any remainders
    indices = np.random.permutation(num_points)
    train_idx, valid_idx, test_idx, _ = np.split(indices, list(splits))

    # TODO consider using torch subsets to make the split a little easier/generalizable
    if isinstance(x, Dataset):
        x_train, y_train = Subset(x, train_idx), y[train_idx]
        x_valid, y_valid = Subset(x, valid_idx), y[valid_idx]
        x_test, y_test = Subset(x, test_idx), y[test_idx]
    else:
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]
        x_test, y_test = x[test_idx], y[test_idx]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
