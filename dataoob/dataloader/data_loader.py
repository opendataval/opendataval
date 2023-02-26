from itertools import accumulate

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from dataoob.dataloader import datasets


def DataLoader(
    dataset_name: str,
    force_redownload: bool = False,
    train_count: int | float = 0,
    valid_count: int | float = 0,
    test_count: int | float = 0,
    categorical=False,
    scaler: str = StandardScaler().fit_transform,
    noise_rate: float = 0.0,
    device: int = torch.device("cpu"),
):
    """Dataloader for dataoob, input the dataset name and some additional parameters
    receive a dataset compatible for Data-oob.

    :param str dataset_name: Name of the dataset, can be registered in `datasets.py`
    :param bool force_redownload: Forces redownload from source URL, defaults to False
    :param int | float train_count: Number/proportion training points, defaults to 0
    :param int | float valid_count: Number/proportion validation points, defaults to 0
    :param int | float test_count: Number/proportion testing points, defaults to 0
    :param bool categorical: Whether the data is categorical, defaults to False
    :param callable (np.ndarray(m x n) -> np.ndarray(m x n)) scaler: Scaler that
    normalizes the data. NOTE This likely will change as the underlying datasets change,
    defaults to StandardScaler().fit_transform
    :param float noise_rate: Ratio of noise to add to the data TODO think
    of other ways to add noise to the data, defaults to 0.
    :param torch.device device: Tensor device for accelearation, defaults to
    torch.device("cpu")
    :return torch.Tensor, torch.Tensor: Training Covariates, Training Labels
    :return torch.Tensor, torch.Tensor: Validation Covariates, Validation Labels
    :return torch.Tensor, torch.Tensor: Test Covariates, Test Labels
    :return np.ndarray: Indices of noisified Training labels
    """
    x, y = datasets.download_dataset(
        dataset_name=dataset_name, force_redownload=force_redownload
    )

    # Scale the data
    x, y = scaler(x), one_hot_encode(y) if categorical else scaler(y)
    x = torch.tensor(x).to(dtype=torch.float32, device=device)
    y = torch.tensor(y).to(dtype=torch.float32, device=device)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = split_dataset(
        x, y, train_count=train_count, valid_count=valid_count, test_count=test_count
    )  # TODO consider loading images, will have to change the apis

    # Noisify the data
    y_train, noisy_indices = noisify(y_train, noise_rate)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), noisy_indices


def one_hot_encode(data) -> torch.Tensor:
    data = data.to_numpy()
    label_dim = int(np.max(data) + 1)
    return np.eye(label_dim)[np.squeeze(data)]


def noisify(
    labels: torch.Tensor, noise_rate: float = 0.0
) -> tuple[torch.Tensor, np.ndarray]:
    if noise_rate == 0.0:
        return labels, np.array([])
    if 0 <= noise_rate <= 1.0:
        n_points = labels.size(dim=0)

        noise_count = round(n_points * noise_rate)
        replace = np.random.choice(n_points, noise_count, replace=False)
        target = np.random.choice(n_points, noise_count, replace=False)
        labels[replace] = labels[target]

        return labels, replace
    else:
        raise Exception()


def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    train_count: int | float,
    valid_count: int | float,
    test_count: int | float,
) -> torch.Tensor:
    """Splits the Covariates and labels according to the specified counts/proportions

    :param torch.Tensor x: Data+Test+Held-out covariates
    :param torch.Tensor y: Data+Test+Held-out labels
    :param int | float train_count: Number/proportion training points, defaults to 0
    :param int | float valid_count: Number/proportion validation points, defaults to 0
    :param int | float test_count: Number/proportion testing points, defaults to 0
    :raises Exception: Raises exception when there's an invalid splitting of the datset,
    ie, more datapoints than the total dataset requested.
    :return torch.Tensor, torch.Tensor: Training Covariates, Training Labels
    :return torch.Tensor, torch.Tensor: Validation Covariates, Validation Labels
    :return torch.Tensor, torch.Tensor: Test Covariates, Test Labels
    """
    assert len(x) == len(y)
    n_points = len(x)

    match (train_count, valid_count, test_count):
        case int(tr), int(val), int(tst) if sum((tr, val, tst)) <= n_points:
            splits = accumulate((tr, val, tst))
        case float(tr), float(val), float(tst) if sum((tr, val, tst)) <= 1.0:
            splits = (round(n_points * p) for p in (tr, val, tst))
            splits = accumulate(splits)
        case _:
            raise Exception()  # TODO

    # Extra underscore to unpack any remainders
    indices = np.random.permutation(n_points)
    train_idx, valid_idx, test_idx, _ = np.split(indices, list(splits))

    # TODO consider using torch subsets to make the split a little easier/generalizable
    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
