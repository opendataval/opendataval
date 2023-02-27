from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataoob.dataloader.util import CatDataset


class Model(ABC):
    """Abstract class of Models. Provides a template of how build models should be
    designed and methods to be impolemented
    """
    @abstractmethod
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weights: torch.Tensor=None,
        batch_size=32,
        epochs=1,
        *args,
        **kwargs
    ):
        """Fits the model on the input data

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor sample_weights: Weights associated with each datapoint,
        defaults to None
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Predicts the label from the input covariates data

        :param torch.Tensor x_train: Data covariates
        """
        pass


class ClassifierNN(Model, nn.Module):
    """Classifier for Torch Neural Networks"""
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor=None,
        batch_size=32,
        epochs=1,
        verbose=False,
    ):
        """Fits the torch classifier"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # TODO update when we move from binary classification
        criterion = F.binary_cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)

        for epoch in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(dataset, batch_size, shuffle=True):
                optimizer.zero_grad()
                outputs = self.__call__(x_batch)

                if sample_weight is not None:
                   loss = criterion(outputs, y_batch, weight=weights[0])
                else:
                    loss = criterion(outputs, y_batch)

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Mini functioin to move tensor to CPU for SKlearn"""
    assert isinstance(tensor, torch.Tensor), "Not a valid input for Wrapper"
    return tensor.detach().cpu()

class ClassifierSkLearnWrapper(Model):
    """Wrapper for SciKit-learn classifiers that can have weighted fit methods"""
    def __init__(self, base_model, device: torch.device=torch.device('cpu')):
        self.model = base_model
        self.device = device

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor=None,
        batch_size=32,
        epochs=1,
        *args,
        **kwargs
    ):
        """Fits the sk-learn model with sample weights"""
        x_train, y_train = to_cpu(x_train), to_cpu(y_train)
        self.model.fit(
            x_train, torch.argmax(y_train, dim=1),
            None if sample_weight is None else torch.squeeze(to_cpu(sample_weight)),
            *args,
            **kwargs
        )

    def predict(self, x: torch.Tensor):
        """Predicts labels from sk-learn model"""
        x = to_cpu(x)
        output = self.model.predict_proba(x)
        return torch.from_numpy(output).to(dtype=torch.float32, device=self.device)

class ClassifierUnweightedSkLearnWrapper(ClassifierSkLearnWrapper):
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor=None,
        batch_size=32,
        epochs=1,
        *args,
        **kwargs
    ):
        """Fits the sk-learn model without the sample_weights parameter, instead
        fits by sampling from the model with those weights"""
        x_train, y_train  = to_cpu(x_train), to_cpu(y_train)

        if sample_weight is not None:
            n_samples = x_train.size(dim=0)
            weights = torch.squeeze(sample_weight)
            weights = weights/weights.sum()
            idx = np.random.choice(n_samples, size=(n_samples), replace=True, p=weights)

            self.model.fit(
                x_train[idx], torch.argmax(y_train[idx], dim=1), *args, **kwargs
            )
        else:
            self.model.fit(x_train, torch.argmax(y_train, dim=1), *args, **kwargs)
