from itertools import accumulate

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from dataoob.dataloader import datasets


def DataLoader(
        dataset_name: str,
        force_redownload: bool=False,
        train_count: int | float=0,
        valid_count: int | float=0,
        test_count: int | float=0,
        categorical=False,
        scaler: str = StandardScaler().fit_transform,
        noise_rate: float = 0.,
        device=torch.device("cpu")
    ):
    """_summary_

    :param str dataset_name: _description_
    :param bool force_redownload: _description_, defaults to False
    :param int | float train_count: _description_, defaults to 0
    :param int | float valid_count: _description_, defaults to 0
    :param int | float test_count: _description_, defaults to 0
    :param bool categorical: _description_, defaults to False
    :param callable (np.array(m x n) -> np.array(m x n)) scaler: Normalizes the data
    with an np.array -> np.array transform, defaults to StandardScaler().fit_transform
    :param float noise_rate: _description_, defaults to 0.
    :param _type_ device: _description_, defaults to torch.device("cpu")
    :return _type_: _description_
    """
    x, y = datasets.download_dataset(dataset_name=dataset_name, force_redownload=force_redownload)

    # Scale the data
    x, y = scaler(x), one_hot_encode(y) if categorical else scaler(y)
    x = torch.tensor(x).to(dtype=torch.float32, device=device)
    y = torch.tensor(y).to(dtype=torch.float32, device=device)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = split_dataset(
        x=x,
        y=y,
        train_count=train_count,
        valid_count=valid_count,
        test_count=test_count,
    )  # TODO consider loading images, will have to change the apis of the models to accept indices as arguments

    # Noisify the data
    y_train, noisy_indices = noisify(y_train, noise_rate)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), noisy_indices


def one_hot_encode(data) -> torch.tensor:
    data = data.to_numpy()
    label_dim = int(np.max(data) + 1)
    return np.eye(label_dim)[np.squeeze(data)]

def noisify(labels: torch.tensor, noise_rate: float=0.) -> tuple[torch.tensor, np.array]:
    if noise_rate == 0.:
        return labels, np.array([])
    if  0 <= noise_rate <= 1.:
        n_points = labels.size(dim=0)

        noise_count = round(n_points * noise_rate)
        replace = np.random.choice(n_points, noise_count, replace=False)
        target = np.random.choice(n_points, noise_count, replace=False)
        labels[replace] = labels[target]

        return labels, replace
    else:
        raise Exception()


def split_dataset(
    x: torch.tensor,
    y: torch.tensor,
    train_count: int | float,
    valid_count: int | float,
    test_count: int | float,
) -> torch.tensor:
    """_summary_

    :param torch.tensor x: Data+Test+Held-out covariates
    :param torch.tensor y: Data+Test+Held-out labels
    :param int | float train_count: Number or porportion of train data points
    :param int | float valid_count: Number or porportion of train valid points
    :param int | float test_count: Number or porportion of train test points
    :param bool categorical: If the data is categorical, defaults to False
    :param int device: Tensor device, defaults to torch.device("cpu")
    :raises Exception: _description_
    :return torch.tensor: _description_
    """
    assert len(x) == len(y)
    n_points  = len(x)

    match (train_count, valid_count, test_count):
        case int(tr), int(val), int(tst) if sum((tr, val, tst)) <= n_points:
            splits = accumulate((tr, val, tst))
        case float(tr), float(val), float(tst) if sum((tr, val, tst)) <= 1.:
            splits = (round(n_points * p) for p in (tr, val, tst))
            splits = accumulate(splits)
        case _:
            raise Exception()

    # Extra underscore to unpack any remainders
    indices = np.random.permutation(n_points)
    train_idx, valid_idx, test_idx, _ = np.split(indices, list(splits))

    x_train, y_train = x[train_idx], y[train_idx]
    x_valid, y_valid = x[valid_idx], y[valid_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)




