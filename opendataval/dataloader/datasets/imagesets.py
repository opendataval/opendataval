"""TorchVision data sets.

Uses `torchvision <https://github.com/pytorch/vision>`_. as a dependency.
"""

import os
from pathlib import Path
from typing import TypeVar, Union

import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
    STL10,
    SVHN,
    FashionMNIST,
    VisionDataset,
)

from opendataval.dataloader.register import Register

Self = TypeVar("Self")

MAX_DATASET_SIZE = 50000


def ResnetEmbeding(
    dataset_class: type[VisionDataset], size: tuple[int, int] = (224, 224)
):
    """Convert PIL color Images into embeddings with ResNet50 model.

    Given a PIL Images, passes through ResNet50 (as done by prior Data Valuation papers)
    and saves the vector embeddings. The embeddings are extracted from the ``avgpool``
    layer of ResNet50. The extraction is through the PyTorch forward hook feature.

    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun,
        Deep Residual Learning for Image Recognition,
        2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
        Jun. 2016, doi: https://doi.org/10.1109/cvpr.2016.90.
    .. [2] A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    image_set : type[VisionDataset]
        Class of Dataset to compute the embeddings of.
    size : tuple[int, int], optional
        Size to resize images to, by default (224, 224)

    Returns
    -------
    Callable
        Wrapped function when called returns a covariate embedding array and label array
    """

    def wrapper(
        cache_dir: str, force_download: bool, *args, **kwargs
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Methods: `@christiansafka <https://github.com/christiansafka/img2vec>`_."""
        from torchvision.models.resnet import ResNet50_Weights, resnet50

        img2vec_transforms = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                # Means and std as specified by @christiansafka
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        cache_dir = Path(cache_dir)
        embed_file_name = f"{dataset_class.__name__}_{MAX_DATASET_SIZE}_embed.pt"
        embed_path = cache_dir / embed_file_name

        # Resnet inputs expect `img2vec_transforms`ed images as input
        dataset = dataset_class(
            root=cache_dir,
            download=force_download or not cache_dir.exists(),
            transform=img2vec_transforms,
            *args,
            **kwargs,
        )
        subset = np.random.RandomState(10).permutation(len(dataset))

        if embed_path.exists():
            image_embeddings = torch.load(embed_path)
            return (
                image_embeddings,
                np.fromiter(
                    (dataset.targets[i] for i in subset),
                    count=len(image_embeddings),
                    dtype=int,
                ),
            )

        dataset = Subset(dataset, subset[:MAX_DATASET_SIZE])

        # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Gets the avgpool layer, the outputs of this layer are our embeddings
        embedder = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        embedder.fc = nn.Identity()

        # We will register a hook to extract the ouput of avgpool layers.
        image_embeddings = torch.zeros(0, 2048)
        labels_list = []

        with torch.no_grad():  # Passes through model, and our hook extracts outputs
            for img, labels in tqdm.tqdm(
                DataLoader(dataset, 256, pin_memory=True, num_workers=4)
            ):
                img = img.to(device)
                embedding = embedder(img).detach().cpu()
                image_embeddings = torch.cat((image_embeddings, embedding), dim=0)
                labels_list.extend(labels)

        image_embeddings = image_embeddings.detach()
        embed_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(image_embeddings, embed_path)
        return image_embeddings, np.array(labels_list)

    return wrapper


def show_image(imgs: Union[list[Image.Image], Image.Image]) -> None:
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


class VisionAdapter(Dataset):
    """Adapter for PyTorch vision data sets. __call__ is called by :py:class:`Register`.

    Adapter for MNIST data sets. __init__ inputs the class and __call__ initializes the
    Dataset and extracts labels. __call__ returns tuple[Self, np.array] where Self is
    a Dataset of covariates and np.array is an array of labels.

    Parameters
    ----------
    dataset_class : type[VisionDataset]
        Torchvision data set class provided.
    """

    def __init__(self, dataset_class: type[VisionDataset]):
        self.dataset_class = dataset_class
        self.transform = None  # Additional transforms applied to the wrapper Dataset.

    def __call__(
        self, cache_dir: str, force_download: bool, *args, **kwargs
    ) -> tuple[Self, np.ndarray]:
        """Return covariates as PyTorch Dataset and labels as np.array.

        Parameters
        ----------
        cache_dir : str
            Directory to download cached files to.
        force_download : bool
            Whether to force a download of the data files.

        Returns
        -------
        tuple[Self, np.ndarray]
            Returns covariates as PyTorch Dataset and labels as np.array. This approach
            was chosen because we need to perform vectorized operations on the labels
            in some data valuators but not necessarily on the covariates, thus, to save
            memory, we leave the Covariates as a PyTorch Dataset.
        """
        # force_download is set to true if  directory doesn't exist, initial download
        force_download = force_download or not os.path.exists(cache_dir)
        self.dataset = self.dataset_class(
            root=cache_dir, download=force_download, *args, **kwargs
        )
        labels = np.array(self.dataset.targets, dtype=int)

        # Incase we forget to apply transform, ensures output is tensor
        if self.dataset.transform is None:
            self.transform = transforms.ToTensor()

        return self, labels

    def __getitem__(self, index: int) -> torch.Tensor:
        """Getitem extracts only the covariates.

        Parameters
        ----------
        index : int
            Index to get covariate from the dataset

        Returns
        -------
        torch.Tensor
            Tensor representing the image with transforms added
        """
        img, _ = self.dataset.__getitem__(index)  # Ignores label
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.dataset)


numbers = Register("mnist", True, True)(VisionAdapter(MNIST))
"""Vision Classification data set registered as ``"mnist"``, from TorchVision."""

fashion = Register("fashion", True, True)(VisionAdapter(FashionMNIST))
"""Vision Classification data set registered as ``"fashion"``, from TorchVision."""

cifar100 = Register("cifar100", True, True)(VisionAdapter(CIFAR100))
"""Vision Classification data set registered as ``"cifar100"``, from TorchVision."""

cifar10 = Register("cifar10", True, True)(VisionAdapter(CIFAR10))
"""Vision Classification registered as ``"cifar10"``, from TorchVision."""

cifar10_embed = Register("cifar10-embeddings", True, True)(ResnetEmbeding(CIFAR10))
"""Vision Classification registered as ``"cifar10-embeddings"`` ResNet50 embeddings"""

stl10_embed = Register("stl10-embeddings", True, True)(ResnetEmbeding(STL10))
"""Vision Classification registered as ``"stl10-embeddings"`` ResNet50 embeddings"""

svhn_embed = Register("svhn-embeddings", True, True)(ResnetEmbeding(SVHN))
"""Vision Classification registered as ``"svhn-embeddings"`` ResNet50 embeddings"""
