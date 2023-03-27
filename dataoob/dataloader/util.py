from typing import Any

from torch.utils.data import Dataset


class CatDataset(Dataset[tuple[Dataset, ...]]):
    """Data set wrapping data sets. Wrapper data set of other data sets to be able to
    align data points. This helps keep APIs more consistent.

    Parameters
    ----------
    datasets : tuple[Dataset]
        Tuple of data sets we would like to concat together, must be same length

    Raises
    ------
    ValueError
        If all input data sets are not the same length
    """

    def __init__(self, *datasets: list[Dataset]):
        self.datasets = [ds for ds in datasets if ds is not None]
        if not all(len(datasets[0]) == len(ds) for ds in self.datasets):
            raise ValueError("Size mismatch between data sets")

    def __getitem__(self, index) -> tuple[Any, ...]:
        """Returns tuple of the element at the index for each data set"""
        return tuple(ds[index] for ds in self.datasets)

    def __len__(self) -> int:
        return len(self.datasets[0])
