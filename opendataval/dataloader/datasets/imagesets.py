from abc import ABC, abstractmethod
from typing import Sequence, Type, TypeVar

import matplotlib as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, VisionDataset

Self = TypeVar("Self")

MAX_DATASET_SIZE = 10
"""Data Valuation algorithms can take a long time for large data sets, thus cap size."""


def resnet_embeddings(
    image_set: Type[VisionDataset], size: tuple[int, int] = (224, 224)
):
    """Convert PIL Images into embeddings with ResNet18 model.

    Given a PIL Images, passes through ResNet18 (as done by prior Data Valuation papers)
    and saves the vector embeddings. The embeddings are extracted from the ``avgpool``
    layer of ResNet18. The extraction is through the PyTorch forward hook feature.

    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun,
        Deep Residual Learning for Image Recognition,
        2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
        Jun. 2016, doi: https://doi.org/10.1109/cvpr.2016.90.
    .. [2] A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.
    """

    def wrapper(
        cache_dir: str, force_download: bool, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Methods: `@christiansafka <https://github.com/christiansafka/img2vec>`_."""
        from torchvision.models.resnet import ResNet18_Weights, resnet18

        img2vec_transforms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                # Means and std as specified by @christiansafka
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Gets the avgpool layer, the outputs of this layer are our embeddings
        embedder = resnet18(weights=ResNet18_Weights.DEFAULT)
        embedding_layer = embedder._modules.get("avgpool")

        # We will register a hook to extract the ouput of avgpool layers.
        image_embeddings = torch.zeros(0, 512, 1, 1)

        def extract(_model, _inputs, output: torch.Tensor):
            nonlocal image_embeddings  # Allows us to reassign to image_embeddings
            image_embeddings = torch.cat((image_embeddings, output.detach()))

        hook = embedding_layer.register_forward_hook(extract)
        labels_list = []

        # Resnet inputs expect `BASE_TRANSFORM`ed images as input
        dataset = image_set(root=cache_dir, download=force_download, **kwargs)
        dataset.transform = img2vec_transforms

        with torch.no_grad():  # Passes through model, and our hook extracts outputs
            for img, labels in DataLoader(dataset, 64):
                embedder(img)
                labels_list.extend(labels)
                if len(image_embeddings) > MAX_DATASET_SIZE:  # Caps data set size
                    break

        hook.remove()  # Cleans up the hook

        return image_embeddings.numpy(force=True), np.array(labels)

    return wrapper


def show_image(imgs: list[Image.Image] | Image.Image) -> None:
    """Displays an image or a list of images."""
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return


class DatasetAdapter(Dataset, ABC):
    """Abstract class to adapt a PyTorch data set to separate covariates from labels."""

    @classmethod
    def __new__(cls, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        return dataset, dataset.labels()

    def labels(self):
        return np.array(self.targets, dtype=int)

    def __len__(self) -> int:
        """Gets length of data set."""
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, index: int | Sequence[int]) -> torch.Tensor:
        """Extract covariates only of a data set."""
        raise NotImplementedError


class MnistAdapter(DatasetAdapter, MNIST):
    """Adapter for PyTorch MNIST data sets. Valid input for Register.__call__

    Parameters
    ----------
    torch_dataset : VisionDatasetClass
        MNIST data set class provided by torchvision.
    """

    def __init__(self, cache_dir, force_download, *args, **kwargs):
        super().__init__(root=cache_dir, download=force_download, *args, **kwargs)

    def __getitem__(self, index: int | Sequence[int]) -> torch.Tensor:
        """Getitem from MNIST except we do not return the label."""
        img = self.ds.data[index]

        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == "__main__":
    MnistAdapter("data_files/mnist", force_download=True)

# numbers = Register(
#     "mnist", categorical=True, cacheable=True
# ).add_covar_transform(BASE_TRANSFORM)(MnistAdapter(MNIST))
# """Template for registering any MNIST data set."""

# fashion = Register(
#     "fashionmnist", categorical=True, cacheable=True
# ).add_covar_transform(BASE_TRANSFORM)(MnistAdapter(FashionMNIST))

# cifar100 = Register(
#     "cifar100", categorical=True, cacheable=True
# ).add_covar_transform(BASE_TRANSFORM)(MnistAdapter(CIFAR100))

# cifar10 = Register(
#     "cifar10", categorical=True, cacheable=True
# ).add_covar_transform(BASE_TRANSFORM)(MnistAdapter(CIFAR10))
