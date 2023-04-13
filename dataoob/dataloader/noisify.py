import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from dataoob.dataloader.loader import DataLoader
from dataoob.dataloader.util import IndexTransformDataset


def mix_labels(loader: DataLoader, noise_rate: float = 0.2) -> dict[str, np.ndarray]:
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
    rs = check_random_state(loader.random_state)

    y_train = loader.y_train
    num_points = len(y_train)

    replace = rs.choice(num_points, round(num_points * noise_rate), replace=False)
    target = rs.choice(num_points, round(num_points * noise_rate), replace=False)
    y_train[replace] = y_train[target]

    return {"y_train": y_train, "noisy_indices": replace}


def add_gauss_noise(
    loader: DataLoader, noise_rate: float = 0.2, mu: float = 0.0, sigma: float = 1.0
) -> dict[str, np.ndarray]:
    """Adds gaussian noise to covariates.

    Parameters
    ----------
    loader : DataLoader
        DataLoader object housing the data to have noise added to
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
    """
    rs = check_random_state(loader.random_state)

    x_train = loader.x_train
    num_points = len(x_train)
    [*feature_dim] = x_train[0].shape  # Unpacks dims of tensors and numpy array

    noisy_indices = rs.choice(num_points, round(num_points * noise_rate), replace=False)
    noise = rs.normal(mu, sigma, size=(len(noisy_indices), *feature_dim))

    if isinstance(x_train, Dataset):
        # We add a zero tensor at the top because noise only some indices have noise
        # added. For those that do not, they have the zero tensor added -> no change
        padded_noise = np.vstack([np.zeros(shape=(1, *feature_dim)), noise])
        noise_add = torch.tensor(padded_noise, dtype=torch.float)

        # A remapping to noisy index, in noise array, offset by 1 for non-noisy data
        # as the 0th index is the zero tensor from above
        remap = np.zeros((num_points,), dtype=int)
        remap[noisy_indices] = range(1, len(noisy_indices) + 1)

        x_train = IndexTransformDataset(
            x_train, lambda data, ind: (data + noise_add[remap[ind]]).to(loader.device)
        )
    else:
        x_train[noisy_indices] = x_train[noisy_indices] + noise

    return {"x_train": x_train, "noisy_indices": noisy_indices}
