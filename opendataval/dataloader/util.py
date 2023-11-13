import pickle
from bisect import bisect_right
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import torch
from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)


class CatDataset(Dataset[tuple[Dataset, ...]]):
    """Data set wrapping indexable Datasets.

    Parameters
    ----------
    datasets : tuple[Dataset]
        Tuple of data sets we would like to concat together, must be same length

    Raises
    ------
    ValueError
        If all input data sets are not the same length
    """

    def __init__(self, *datasets: list[Dataset[Any]]):
        self.datasets = [ds for ds in datasets if ds is not None]

        if not all(len(datasets[0]) == len(ds) for ds in self.datasets):
            raise ValueError("Size mismatch between data sets")

    def __getitem__(self, index) -> tuple[Any, ...]:
        """Return tuple of indexed element or tensor value on first axis."""
        return tuple(ds[index] for ds in self.datasets)

    def __len__(self) -> int:
        return len(self.datasets[0])


class IndexTransformDataset(Dataset[T_co]):
    """Data set wrapper that allows a per-index transform to be applied.

    Primarily useful when adding noise to specific subset of indices. If a transform
    is defined, it will apply the transformation but also pass in the indices
    (what is passed into __getitem__) as well.

    Parameters
    ----------
    dataset : Dataset[T_co]
        Data set with transform to be applied
    index_transformation : Callable[[T_co, Sequence[int]], T_co], optional
        Function that takes input sequence of ints and data and applies
        the specific transform per index, by default None which is no transform.

    """

    def __init__(
        self,
        dataset: Dataset[T_co],
        index_transformation: Optional[Callable[[T_co, int], T_co]] = None,
    ):
        self.dataset = dataset
        self._transform = index_transformation

    @property
    def transform(self) -> Callable[[T_co, int], T_co]:
        """Gets the transform function, if None, no transformation applied."""
        if self._transform is None:
            return lambda data, _: data
        return self._transform

    @transform.setter
    def transform(self, index_transformation: Callable[[T_co, int], T_co]):
        """Assign new transform to the dataset."""
        self._transform = index_transformation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        """Apply specified transform at indices onto data and returns it."""
        data = self.dataset.__getitem__(index)
        return self.transform(data, index)


class ListDataset(Dataset[T_co]):
    """Data set wrapping a list.

    ListDataset is primarily useful to when you want to pass back a list but also
    want to get around the type checks of Datasets. This is intended to be used
    with NLP data sets as the the axis 1 dimension is variable and BERT tokenizers take
    inputs as only lists.

    Parameters
    ----------
    input_list : Sequence[T_co]
        Input sequence to be used as data set.
    """

    def __init__(self, input_list: Sequence[T_co]):
        self.data = input_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> list[T_co]:
        return self.data[index]


def load_tensor(tensor_path: Path) -> torch.Tensor:
    with torch.no_grad():
        return torch.load(tensor_path, map_location="cpu").detach()


class FolderDataset(Dataset):
    """Dataset for tensors within a folder."""

    BATCH_CACHE = 5

    def __init__(self, folder_path: Path, sizes: Optional[list[int]] = None):
        self.folder_path = Path(folder_path)
        self.folder_path.mkdir(exist_ok=True, parents=True)
        self.sizes = sizes or [0]

    @cached_property
    def shape(self) -> tuple[int, ...]:
        batch_size, *shape = self.get_batch(0).shape

        assert (
            batch_size == self.sizes[1] - self.sizes[0]
        ), f"Unexpected batch size, {batch_size=} != {self.sizes[1] - self.sizes[0]}"

        return (len(self), *shape)

    def __len__(self) -> int:
        return self.sizes[-1]

    def format_batch_path(self, batch_index: int) -> str:
        return self.folder_path / f"{batch_index:03}.pt"

    @lru_cache(BATCH_CACHE)
    def get_batch(self, batch_index: int) -> torch.Tensor:
        if batch_index < 0:
            raise ValueError(f"Batch {batch_index} must be greater than 0")
        elif not batch_index < len(self.sizes):
            raise KeyError(f"Batch {batch_index} is not in range {len(self.sizes)}")
        return load_tensor(self.format_batch_path(batch_index))

    def __getitem__(self, i: int) -> torch.Tensor:
        batch_index = bisect_right(self.sizes, i) - 1
        if not (0 <= batch_index < len(self.sizes) - 1):
            raise KeyError(f"Index {i} is not within range [0, {self.sizes[-1]})")
        displace = i - self.sizes[batch_index]
        return self.get_batch(batch_index)[displace]

    def write(self, batch_number: int, data: torch.Tensor):
        self.sizes.append(self.sizes[-1] + len(data))
        torch.save(data.detach(), self.format_batch_path(batch_number))

    @property
    def metadata(self) -> dict[str, Any]:
        """Important metadata defining a GradientDataset, used for loading."""
        return {"folder_path": self.folder_path, "sizes": self.sizes}

    def save(self):
        """Saves metadata to disk, allows us to load GradientDataset as needed."""
        with open(self.folder_path / ".metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: Path):
        """Loads existing gradient dataset metadata from path/.metadata.pkl"""
        with open(Path(path) / ".metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        # If errors are raised, error with pickling
        return cls(**metadata)

    @staticmethod
    def exists(path: Path):
        return (Path(path) / ".metadata.pkl").exists()
