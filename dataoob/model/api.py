from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataoob.util import CatDataset


class Model(ABC):
    @abstractmethod
    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, *args, **kwargs):
        pass


class ClassifierNN(Model, nn.Module):
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor=None,
        batch_size=32,
        epochs=1,
        verbose=False,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = F.binary_cross_entropy  # TODO update when we move from binary classification
        dataset = CatDataset(x_train, y_train, sample_weight)

        for epoch in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(dataset, batch_size, shuffle=True):
                optimizer.zero_grad()  # Setting our stored gradients equal to zero
                outputs = self.__call__(x_batch)

                if sample_weight is not None:
                   loss = criterion(outputs, y_batch, weight=weights[0])
                else:
                    loss = criterion(outputs, y_batch)

                loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
                optimizer.step()  # Updates weights and biases with the optimizer (SGD)

def to_cpu(tensor: torch.Tensor):
    """Mini functioin to move tensor to CPU for SKlearn"""
    assert isinstance(tensor, torch.Tensor), "Not a valid input for Wrapper"
    return tensor.detach().cpu()

class ClassifierSkLearnWrapper(Model):
    def __init__(self, base_model, device: torch.device=torch.device('cpu')):
        self.model = base_model
        self.device = device

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor,  sample_weight: torch.Tensor=None, *args, **kwargs):
        x_train, y_train = to_cpu(x_train), to_cpu(y_train)
        self.model.fit(
            x_train, torch.argmax(y_train, dim=1),
            None if sample_weight is None else torch.squeeze(to_cpu(sample_weight)),
            *args,
            **kwargs
        )

    def predict(self, x: torch.Tensor):
        x = to_cpu(x)
        output = self.model.predict_proba(x)
        return torch.from_numpy(output).to(dtype=torch.float32, device=self.device)

class ClassifierUnweightedSkLearnWrapper(ClassifierSkLearnWrapper):
    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor,  sample_weight: torch.Tensor=None, *args, **kwargs):
        x_train, y_train  = to_cpu(x_train), to_cpu(y_train)

        if sample_weight is not None:
            n_samples = x_train.size(dim=0)
            sample_weight = torch.squeeze(sample_weight)
            indices = np.random.choice(n_samples, size=(n_samples), replace=True, p=sample_weight/sample_weight.sum())

            self.model.fit(x_train[indices], torch.argmax(y_train, dim=1), *args, **kwargs)
        else:
            self.model.fit(x_train, torch.argmax(y_train, dim=1), *args, **kwargs)
