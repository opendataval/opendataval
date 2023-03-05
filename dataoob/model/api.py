from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, dataloader
from dataoob.dataloader.util import CatDataset


class Model(ABC):
    """Abstract class of Models. Provides a template of how build models should be
    designed and methods to be implemented
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
        x_train: torch.Tensor | Dataset,
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

def to_cpu(tensors: torch.Tensor) -> torch.Tensor:
    """Mini function to move tensor to CPU for sk-learn"""
    return tuple(t.detach().cpu() for t in dataloader.default_collate(tensors))

class ClassifierSkLearnWrapper(Model):
    """Wrapper for sk-learn classifiers that can have weighted fit methods"""
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
        # Using a dataset and dataloader (despite loading all the data) for better
        # API consistency, such as passing datasets to a sk-learn  model
        dataset = CatDataset(x_train, y_train, sample_weight)
        num_samples = len(dataset)
        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_cpu)
        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))

        self.model.fit(
            x_train, torch.argmax(y_train, dim=1),
            None if sample_weight is None else torch.squeeze(weights[0]),
            *args,
            **kwargs
        )

    def predict(self, x: torch.Tensor):
        """Predicts labels from sk-learn model"""
        x = x.detach().cpu()
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

        # Using a dataset and dataloader (despite loading all the data) for better
        # API consistency, such as passing datasets to a sk-learn  model
        dataset = CatDataset(x_train, y_train, sample_weight)
        num_samples = len(dataset)
        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_cpu)
        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))

        if sample_weight is not None:
            weights = torch.squeeze(sample_weight)/sample_weight.sum()
            idx = np.random.choice(num_samples, size=(num_samples), replace=True, p=weights)

            self.model.fit(
                x_train[idx], torch.argmax(y_train[idx], dim=1), *args, **kwargs
            )
        else:
            self.model.fit(x_train, torch.argmax(y_train, dim=1), *args, **kwargs)
