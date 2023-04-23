from typing import Union

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from dataoob.dataloader.fetcher import DataFetcher
from dataoob.dataloader.util import IndexTransformDataset


def mix_labels(fetcher: DataFetcher, noise_rate: float = 0.2) -> dict[str, np.ndarray]:
    """Mixes y_train labels of a DataFetcher, adding noise to data.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher object housing the data to have noise added to
    noise_rate : float
        Proportion of labels to add noise to

    Returns
    -------
    dict[str, np.ndarray]
        dictionary of updated data points

        - **"y_train"** -- Updated training labels mixed
        - **"y_valid"** -- Updated validation labels mixed
        - **"noisy_train_indices"** -- Indices of training data set with mixed labels
    """
    rs = check_random_state(fetcher.random_state)

    y_train, y_valid = fetcher.y_train, fetcher.y_valid
    num_train, num_valid = len(y_train), len(y_valid)

    replace_train = rs.choice(num_train, round(num_train * noise_rate), replace=False)
    target_train = rs.choice(num_train, round(num_train * noise_rate), replace=False)
    replace_valid = rs.choice(num_valid, round(num_valid * noise_rate), replace=False)
    target_valid = rs.choice(num_valid, round(num_valid * noise_rate), replace=False)

    y_train[replace_train] = y_train[target_train]
    y_valid[replace_valid] = y_valid[target_valid]

    return {
        "y_train": y_train,
        "y_valid": y_valid,
        "noisy_train_indices": replace_train,
    }


def add_gauss_noise(
    fetcher: DataFetcher, noise_rate: float = 0.2, mu: float = 0.0, sigma: float = 1.0
) -> dict[str, Union[Dataset, np.ndarray]]:
    """Add gaussian noise to covariates.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher object housing the data to have noise added to
    noise_rate : float
        Proportion of labels to add noise to
    mu : float, optional
        Center of gaussian distribution which noise is generated from, by default 0
    sigma : float, optional
        Standard deviation of gaussian distribution, by default 1

    Returns
    -------
    dict[str, np.ndarray]
        dictionary of updated data points
        - **"x_train"** -- Updated training covariates with added gaussian noise
        - **"noisy_train_indices"** -- Indices of training data set with mixed labels
    """
    rs = check_random_state(fetcher.random_state)

    x_train, x_valid = fetcher.x_train, fetcher.x_valid
    num_train, num_valid = len(x_train), len(x_valid)
    feature_dim = fetcher.covar_dim

    noisy_train_idx = rs.choice(num_train, round(num_train * noise_rate), replace=False)
    noisy_valid_idx = rs.choice(num_valid, round(num_valid * noise_rate), replace=False)
    noise_train = rs.normal(mu, sigma, size=(len(noisy_train_idx), *feature_dim))
    noise_valid = rs.normal(mu, sigma, size=(len(noisy_valid_idx), *feature_dim))

    if isinstance(x_train, Dataset):
        # We add a zero tensor at the top because noise only some indices have noise
        # added. For those that do not, they have the zero tensor added -> no change
        padded_noise_train = np.vstack([np.zeros(shape=(1, *feature_dim)), noise_train])
        padded_noise_valid = np.vstack([np.zeros(shape=(1, *feature_dim)), noise_valid])
        noise_add_train = torch.tensor(padded_noise_train, dtype=torch.float)
        noise_add_valid = torch.tensor(padded_noise_valid, dtype=torch.float)

        # A remapping to noisy index, in noise array, offset by 1 for non-noisy data
        # as the 0th index is the zero tensor from above
        remap_train = np.zeros((num_train,), dtype=int)
        remap_valid = np.zeros((num_valid,), dtype=int)
        remap_train[noisy_train_idx] = range(1, len(noisy_train_idx) + 1)
        remap_valid[noisy_valid_idx] = range(1, len(noisy_valid_idx) + 1)

        x_train = IndexTransformDataset(
            x_train, lambda data, ind: (data + noise_add_train[remap_train[ind]])
        )
        x_valid = IndexTransformDataset(
            x_valid, lambda data, ind: (data + noise_add_valid[remap_valid[ind]])
        )
    else:
        x_train[noisy_train_idx] = x_train[noisy_train_idx] + noise_train
        x_valid[noisy_valid_idx] = x_valid[noisy_valid_idx] + noise_valid

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "noisy_train_indices": noisy_train_idx,
    }
