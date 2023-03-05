from typing import Tuple

from torch.utils.data import Dataset


class CatDataset(Dataset[Tuple[Dataset, ...]]):
    """Dataset wrapping datasets

    Wrapper dataset of other datasets to be able to align data points
    This helps keep APIs more consistent. NOTE this is a copy of TensorDataset except
    it uses `len` instead of `.size(0)`, allowing us to have a CatDataset of a dataset,
    which is explicitly useful when we take a subset of a larger dataset such as images.
    I'm not entirely sure why this wasn't in torch
    """
    def __init__(self, *datasets: Dataset):
        self.datasets = [ds for ds in datasets if ds is not None]
        if not all(len(datasets[0]) == len(ds) for ds in self.datasets):
            raise Exception("Size mismatch between datasets")

    def __getitem__(self, index):
        return tuple(ds[index] for ds in self.datasets)

    def __len__(self):
        return len(self.datasets[0])