from typing import Any, Callable, Sequence, TypeVar

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
        index_transformation: Callable[[T_co, int], T_co] = None,
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
