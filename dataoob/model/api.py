from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, dataloader

from dataoob.dataloader.util import CatDataset


class Model(ABC):
    """Abstract class of Models. Provides a template for models"""

    @abstractmethod
    def fit(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        *args,
        sample_weights: torch.Tensor = None,
        **kwargs
    ):
        """Fits the model on the training data

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        args : tuple[Any]
            Additional positional args
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        kwargs : dict[str, Any]
            Addition key word args
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor | Dataset, *args, **kwargs) -> torch.Tensor:
        """Predicts the label from the input covariates data

        Parameters
        ----------
        x : torch.Tensor | Dataset
            Input data covariates


        Returns
        -------
        torch.Tensor
            Output predictions based on the input
        """
        pass


class BinaryClassifierNNMixin(Model, nn.Module):
    """Binary Classifier Mixin for Torch Neural Networks"""

    def fit(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor = None,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        """Fits a torch binary classifier Model object using SGD

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, by default None
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

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

    def predict(self, x: torch.Tensor | Dataset) -> torch.Tensor:
        """Predicts output from input tensor/data set

        Parameters
        ----------
        x : torch.Tensor
            Input covariates

        Returns
        -------
        torch.Tensor
            Predicted tensor output
        """
        if isinstance(x, Dataset):
            x = next(iter(DataLoader(x, batch_size=len(x))))
        y_hat = self.forward(x)
        return y_hat


class ClassifierNNMixin(Model, nn.Module):
    """Classifier Mixin for Torch Neural Networks"""

    def fit(
        self,
        x_train: torch.Tensor | Dataset,
        y_train: torch.Tensor,
        sample_weight: torch.Tensor = None,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        """Fits a torch classifier Model object using SGD

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, by default None
        """
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
                    # F.cross_entropy doesn't support sample_weights
                    loss = criterion(outputs, y_batch, reduction="none")
                    loss = (loss * weights[0]).mean()
                else:
                    loss = criterion(outputs, y_batch, reduction="mean")

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

    def predict(self, x: torch.Tensor | Dataset) -> torch.Tensor:
        """Predicts output from input tensor/data set

        Parameters
        ----------
        x : torch.Tensor
            Input covariates

        Returns
        -------
        torch.Tensor
            Predicted tensor output
        """
        if isinstance(x, Dataset):
            x = next(iter(DataLoader(x, batch_size=len(x))))
        y_hat = self.forward(x)
        return y_hat


def to_cpu(tensors: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    """Mini function to move tensor to CPU for sk-learn"""
    return tuple(t.numpy(force=True) for t in dataloader.default_collate(tensors))


class ClassifierSkLearnWrapper(Model):
    """Wrapper for sk-learn classifiers that can have weighted fit methods

    Parameters
    ----------
    base_model : BaseModel
        Any sk-learn model that supports ``sample_weights``
    num_classes : int
        Label dimensionality
    device : torch.device, optional
        Device output tensor is moved to, by default torch.device("cpu")

    .. highlight:: python
    ::
        wrapped = ClassifierSkLearnWrapper(LinearRegression(), 2, torch.device('cuda'))

    """

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
        *args,
        sample_weight: torch.Tensor = None,
        **kwargs
    ):
        """Fits a sk-learn wrapped classifier Model. If there are less classes in the
        sample than num_classes, uses dummy model.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        args : tuple[Any]
            Additional positional args
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        kwargs : dict[str, Any]
            Addition key word args

        .. highlight:: python
        ::
            wrapped = ClassifierSkLearnWrapper(
                LogisticRegression(), 2, torch.device('cuda')
            )

        """
        # Using a data set and dataloader (despite loading all the data) consistency
        dataset = CatDataset(x_train, y_train, sample_weight)
        num_samples = len(dataset)
        if num_samples == 0:
            self.model = DummyClassifier(strategy="constant", constant=0).fit([0], [0])
            return
        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_cpu)

        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))
        y_train = np.argmax(y_train, axis=1)
        y_train_unique = np.unique(y_train)

        if len(y_train_unique) != self.num_classes:  # All labels must be in sample
            self.model = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
        elif sample_weight is not None:
            self.model.fit(x_train, y_train, np.squeeze(weights[0]), *args, **kwargs)
        else:
            self.model.fit(x_train, y_train, sample_weight=None, *args, **kwargs)

    def predict(self, x: torch.Tensor | Dataset) -> torch.Tensor:
        """Predicts labels from sk-learn model"""
        # Extracts the input into a cpu tensor
        if isinstance(x, Dataset):
            x = next(iter(DataLoader(x, len(x), collate_fn=to_cpu)))[0]
        else:
            x = x.numpy(force=True)
        output = self.model.predict_proba(x)

        return torch.from_numpy(output).to(device=self.device, dtype=torch.float)


class ClassifierUnweightedSkLearnWrapper(ClassifierSkLearnWrapper):
    """Wrapper for sk-learn classifiers that can don't have weighted fit methods

    Parameters
    ----------
    base_model : BaseModel
        Any sk-learn model that supports ``sample_weights``
    num_classes : int
        Label dimensionality
    device : torch.device, optional
        Device output tensor is moved to, by default torch.device("cpu")

    .. highlight:: python
    ::
        wrapped = ClassifierSkLearnWrapper(
            RandomForestClassifier(), 2, torch.device('cuda')
        )

    """

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        *args,
        sample_weight: torch.Tensor = None,
        **kwargs
    ):
        """Fits a sk-learn wrapped classifier Model without sample weight. It uses
        weighted random sampling to bypass this. If there are less classes in the
        sample than num_classes, uses dummy model.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        args : tuple[Any]
            Additional positional args
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        kwargs : dict[str, Any]
            Addition key word args
        """
        # Using a data set and dataloader (despite loading all the data) for better
        # API consistency, such as passing data sets to a sk-learn  model
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
        y_train = np.argmax(y_train, axis=1)
        y_train_unique = np.unique(y_train)

        if len(y_train_unique) != self.num_classes:  # All labels must be in sample
            print("Insufficient classes")
            self.model = DummyClassifier(strategy="most_frequent")
        else:
            self.model.fit(x_train, y_train, *args, **kwargs)
