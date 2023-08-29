from abc import abstractmethod
from typing import Iterator, Union

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset

from opendataval.dataloader import CatDataset
from opendataval.model.api import Model, TorchModel


class GradientModel(Model):
    """Provides access to gradients of a :py:class:`~opendataval.model.api.Model`

    TODO Some data evaluators may benefit from higher-order gradients or hessians.
    """

    @abstractmethod
    def grad(
        self,
        x_data: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        *args,
        **kwargs
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Given input data, iterates through the computed gradients of the model.

        Will yield a tuple with gradients for each layer of the model for each input
        data. The data the underlying model is trained on does not have to be the data
        the gradients of the model are computed for. An iterator is used because
        storing the computed gradient for each data point use up lots of memory.

        Parameters
        ----------
        x_data : Union[torch.Tensor, Dataset]
            Data covariates
        y_data : Union[torch.Tensor, Dataset]
            Data labels

        Yields
        ------
        Iterator[tuple[torch.Tensor, ...]]
            Computed gradients (for each layer as tuple) yielded by data point in order
        """


class TorchGradMixin(GradientModel, TorchModel):
    """Gradient Mixin for Torch Neural Networks."""

    def grad(
        self,
        x_data: Union[torch.Tensor, Dataset],
        y_data: Union[torch.Tensor, Dataset],
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Given input data, yields the computed gradients for a torch model

        Parameters
        ----------
        x_data : Union[torch.Tensor, Dataset]
            Data covariates
        y_data : Union[torch.Tensor, Dataset]
            Data labels

        Yields
        ------
        Iterator[tuple[torch.Tensor, ...]]
            Computed gradients (for each layer as tuple) yielded by data point in order
        """
        criterion = F.binary_cross_entropy if self.num_classes == 2 else F.cross_entropy
        dataset = CatDataset(x_data, y_data)

        # Explicitly setting batch_size to 1
        for x_batch, y_batch in DataLoader(dataset, 1, shuffle=False, pin_memory=True):
            # Moves data to correct device
            x_batch = x_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)

            outputs = self.__call__(x_batch)
            batch_loss = criterion(outputs, y_batch, reduction="mean")
            batch_grad = grad(batch_loss, self.parameters())
            yield batch_grad
        return
