from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataoob.dataloader.util import CatDataset
from sklearn.dummy import DummyClassifier
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              dataloader)


class Model(ABC):
    """Abstract class of Models. Provides a template of how build models should be
    designed and methods to be implemented # TODO consider building mixins to make api better
    """

    @abstractmethod
    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weights: torch.Tensor = None,
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


class BinaryClassifierNNMixin(Model, nn.Module):
    """Binary Classifier Mixin for Torch Neural Networks"""

    def fit(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor = None,
        batch_size=32,
        epochs=1,
    ):
        """Fits the torch classifier"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # TODO update when we move from binary classification
        criterion = F.binary_cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)

        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(dataset, batch_size):
                optimizer.zero_grad()
                outputs = self.__call__(x_batch)

                if sample_weight is not None:
                    loss = criterion(outputs, y_batch, weight=weights[0])
                else:
                    loss = criterion(outputs, y_batch)

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights


class ClassifierNNMixin(Model, nn.Module):
    """Classifier Mixin for Torch Neural Networks"""

    def fit(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor = None,
        batch_size=32,
        epochs=1,
    ):
        """Fits the torch classifier"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # TODO update when we move from binary classification
        criterion = F.cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)

        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(dataset, batch_size):
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

    def __init__(
        self, base_model, num_classes: int, device: torch.device = torch.device("cpu")
    ):
        self.model = base_model
        self.num_classes = num_classes
        self.device = device

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor = None,
        batch_size=32,
        epochs=1,
        *args,
        **kwargs
    ):
        # Using a dataset and dataloader (despite loading all the data) for better
        # API consistency, such as passing datasets to a sk-learn  model
        dataset = CatDataset(x_train, y_train, sample_weight)
        num_samples = len(dataset)
        if num_samples == 0:
            self.model = DummyClassifier(strategy="constant", constant=0).fit([0], [0])
            return
        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_cpu)

        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))
        y_train = torch.argmax(y_train, dim=1)
        y_train_unique = torch.unique(y_train, sorted=True)

        if len(y_train_unique) != self.num_classes:  # All labels must be in sample
            self.model = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
        elif sample_weight is not None:
            self.model.fit(x_train, y_train, torch.squeeze(weights[0]), *args, **kwargs)
        else:
            self.model.fit(x_train, y_train, *args, **kwargs)

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
        sample_weight: torch.Tensor = None,
        batch_size=32,
        epochs=1,
        *args,
        **kwargs
    ):
        """Fits the sk-learn model without the sample_weights parameter, instead
        fits by sampling from the model with those weights"""

        # Using a dataset and dataloader (despite loading all the data) for better
        # API consistency, such as passing datasets to a sk-learn  model
        dataset = CatDataset(x_train, y_train)
        num_samples = len(dataset)
        if num_samples == 0:
            self.model = DummyClassifier(strategy="constant", constant=0).fit([0], [0])
            return
        ws = None  # Weighted sampler, if it's None, uses all samples

        if sample_weight is not None:
            ws = WeightedRandomSampler(sample_weight, num_samples, replacement=True)

        dataloader = DataLoader(dataset, num_samples, sampler=ws, collate_fn=to_cpu)

        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train = next(iter(dataloader))
        y_train = torch.argmax(y_train, dim=1)
        y_train_unique = torch.unique(y_train_unique, sorted=True)

        if len(y_train_unique) != self.num_classes:  # All labels must be in sample
            print("Insufficient classes")
            self.model = DummyClassifier(strategy="most_frequent")
        else:
            self.model.fit(x_train, y_train, *args, **kwargs)
