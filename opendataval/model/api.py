import copy
import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier, DummyRegressor
from torch.utils.data import DataLoader, Dataset, default_collate

from opendataval.dataloader.util import CatDataset

Self = TypeVar("Self")


class Model(ABC):
    """Abstract class of Models. Provides a template for models."""

    Models: ClassVar[dict[str, Self]] = {}

    def __init_subclass__(cls, *args, **kwargs):
        """Registers Model types, used as part of the CLI."""
        super().__init_subclass__(*args, **kwargs)
        cls.Models[cls.__name__.lower()] = cls

    @abstractmethod
    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        *args,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Self:
        """Fits the model on the training data.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        args : tuple[Any]
            Additional positional args
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        kwargs : dict[str, Any]
            Addition key word args

        Returns
        -------
        self : object
            Returns self for api consistency with sklearn.
        """
        return self

    @abstractmethod
    def predict(self, x: Union[torch.Tensor, Dataset], *args, **kwargs) -> torch.Tensor:
        """Predict the label from the input covariates data.

        Parameters
        ----------
        x : torch.Tensor | Dataset
            Input data covariates


        Returns
        -------
        torch.Tensor
            Output predictions based on the input
        """

    def clone(self) -> Self:
        """Clone Model object.

        Copy and returns object representing current state. We often take a base
        model and train it several times, so we need to have the same initial conditions
        Default clone implementation.

        Returns
        -------
        self : object
            Returns deep copy of model.
        """
        return copy.deepcopy(self)


class TorchModel(Model, nn.Module):
    """Torch Models have a device they belong to and shared behavior"""

    @property
    def device(self):
        return next(self.parameters()).device


class TorchClassMixin(TorchModel):
    """Classifier Mixin for Torch Neural Networks."""

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
    ):
        """Fits the model on the training data.

        Fits a torch classifier Model object using ADAM optimizer and cross
        categorical entropy loss.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        criterion = F.binary_cross_entropy if self.num_classes == 2 else F.cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)

        self.train()
        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=True
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                optimizer.zero_grad()
                outputs = self.__call__(x_batch)

                if sample_weight is not None:
                    # F.cross_entropy doesn't support sample_weights
                    loss = criterion(outputs, y_batch, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(outputs, y_batch, reduction="mean")

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

        return self


class TorchRegressMixin(TorchModel):
    """Regressor Mixin for Torch Neural Networks."""

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
    ):
        """Fits the regression model on the training data.

        Fits a torch regression Model object using ADAM optimizer and MSE loss.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        criterion = F.mse_loss
        dataset = CatDataset(x_train, y_train, sample_weight)

        self.train()
        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                pin_memory=True,
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                optimizer.zero_grad()
                y_hat = self.__call__(x_batch)

                if sample_weight is not None:
                    # F.cross_entropy doesn't support sample_weight
                    loss = criterion(y_hat, y_batch, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(y_hat, y_batch, reduction="mean")

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

        return self


class TorchPredictMixin(TorchModel):
    """Torch ``.predict()`` method mixin for Torch Neural Networks."""

    def predict(self, x: Union[torch.Tensor, Dataset]) -> torch.Tensor:
        """Predict output from input tensor/data set.

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
            x = next(iter(DataLoader(x, batch_size=len(x), pin_memory=True)))
        x = x.to(device=self.device)

        self.eval()
        with torch.no_grad():
            y_hat = self.__call__(x)

        return y_hat


def to_numpy(tensors: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    """Mini function to move tensor to CPU for sk-learn."""
    return tuple(t.numpy(force=True) for t in default_collate(tensors))


class ClassifierSkLearnWrapper(Model):
    """Wrapper for sk-learn classifiers that can have weighted fit methods.

    Example:
    ::
        wrapped = ClassifierSkLearnWrapper(LinearRegression(), 2)

    Parameters
    ----------
    base_model : BaseModel
        Any sk-learn model that supports ``sample_weights``
    num_classes : int
        Label dimensionality
    """

    def __init__(self, base_model, num_classes: int, *args, **kwargs):
        self.model = base_model(*args, **kwargs)
        self.num_classes = num_classes

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        *args,
        sample_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Fits the model on the training data.

        Fits a sk-learn wrapped classifier Model. If there are less classes in the
        sample than num_classes, uses dummy model.
        ::
            wrapped = ClassifierSkLearnWrapper(MLPClassifier, 2)

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        args : tuple[Any]
            Additional positional args
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        kwargs : dict[str, Any]
            Addition key word args
        """
        # Using a data set and dataloader (despite loading all the data) consistency
        dataset = CatDataset(x_train, y_train, sample_weight)
        num_samples = len(dataset)

        if num_samples == 0:
            self.model = DummyClassifier(strategy="constant", constant=0).fit([0], [0])
            self.model.n_classes_ = self.num_classes
            return self

        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_numpy)
        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))
        y_train = np.argmax(y_train, axis=1)
        y_train_unique = np.unique(y_train)

        with warnings.catch_warnings():  # Ignores warnings in the following block
            warnings.simplefilter("ignore")

            if len(y_train_unique) != self.num_classes:  # All labels must be in sample
                dummy_strat = "most_frequent"
                self.model = DummyClassifier(strategy=dummy_strat).fit(x_train, y_train)
                self.model.n_classes_ = self.num_classes
            elif sample_weight is not None:
                weights = np.squeeze(weights[0])
                self.model.fit(x_train, y_train, *args, sample_weight=weights, **kwargs)
            else:
                self.model.fit(x_train, y_train, *args, sample_weight=None, **kwargs)

        return self

    def predict(self, x: Union[torch.Tensor, Dataset]) -> torch.Tensor:
        """Predict labels from sk-learn model.

        Makes a prediction based on the input tensor. Uses the `.predict_proba(x)`
        method on sk-learn classifiers. Output dim will match the input to
        the `.train(x, y)` method

        Parameters
        ----------
        x : torch.Tensor | Dataset
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Extracts the input into a cpu tensor
        if isinstance(x, Dataset):
            x = next(iter(DataLoader(x, len(x)))).numpy(force=True)
        else:
            x = x.numpy(force=True)
        output = self.model.predict_proba(x)

        return torch.from_numpy(output).to(dtype=torch.float)


class ClassifierUnweightedSkLearnWrapper(ClassifierSkLearnWrapper):
    """Wrapper for sk-learn classifiers that can don't have weighted fit methods.

    Example:
    ::
        wrapped = ClassifierSkLearnWrapper(KNeighborsClassifier, 2)

    Parameters
    ----------
    base_model : BaseModel
        Any sk-learn model that supports ``sample_weights``
    num_classes : int
        Label dimensionality
    """

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        *args,
        sample_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Fits the model on the training data.

        Fits a sk-learn wrapped classifier Model without sample weight.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
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
        dataset = CatDataset(x_train, y_train, sample_weight)

        if num_samples == 0:
            self.model = DummyClassifier(strategy="constant", constant=0).fit([0], [0])
            self.model.n_classes_ = self.num_classes
            return self

        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_numpy)
        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))
        y_train = np.argmax(y_train, axis=1)
        y_train_unique = np.unique(y_train)

        with warnings.catch_warnings():  # Ignores warnings in the following block
            warnings.simplefilter("ignore")

            if len(y_train_unique) != self.num_classes:  # All labels must be in sample
                dummy_strat = "most_frequent"
                self.model = DummyClassifier(strategy=dummy_strat).fit(x_train, y_train)
                self.model.n_classes_ = self.num_classes
            elif sample_weight is not None:
                indices = np.random.choice(  # Random sample of the train data set
                    num_samples,
                    size=(num_samples),
                    replace=True,
                    p=weights[0].squeeze() / weights[0].sum(),
                )
                self.model.fit(x_train[indices], y_train[indices], *args, **kwargs)
            else:
                self.model.fit(x_train, y_train, *args, **kwargs)

        return self


class RegressionSkLearnWrapper(Model):
    """Wrapper for sk-learn regression models.

    Example:
    ::
        wrapped = RegressionSkLearnWrapper(LinearRegression)

    Parameters
    ----------
    base_model : BaseModel
        Any sk-learn model that supports ``sample_weights``
    """

    def __init__(self, base_model, *args, **kwargs):
        self.model = base_model(*args, **kwargs)
        self.num_classes = 1

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        *args,
        sample_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Fits the model on the training data.

        Fits a sk-learn wrapped regression Model. If there is insufficient data to fit
        a regression (such as len(x_train)==0), will use DummyRegressor that predicts
        np.zeros((num_samples, self.num_classes))

        Parameters
        ----------
        x_train : torch.Tensor |  Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
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
        dataset = CatDataset(x_train, y_train, sample_weight)
        num_samples = len(dataset)

        if num_samples == 0:
            constant_return = np.zeros(shape=(1, self.num_classes))
            self.model = DummyRegressor(strategy="mean").fit([[0]], constant_return)
            return self

        dataloader = DataLoader(dataset, batch_size=num_samples, collate_fn=to_numpy)
        # *weights helps check if we passed weights into the Dataloader
        x_train, y_train, *weights = next(iter(dataloader))

        with warnings.catch_warnings():  # Ignores warnings in the following block
            warnings.simplefilter("ignore")

            if sample_weight is not None:
                weights = np.squeeze(weights[0])
                self.model.fit(x_train, y_train, *args, sample_weight=weights, **kwargs)
            else:
                self.model.fit(x_train, y_train, *args, sample_weight=None, **kwargs)

        return self

    def predict(self, x: Union[torch.Tensor, Dataset]) -> torch.Tensor:
        """Predict values from sk-learn regression model.

        Makes a prediction based on the input tensor. Uses the `.predict(x)`
        method on sk-learn regression models. Output dim will match ``self.num_classes``

        Parameters
        ----------
        x : torch.Tensor | Dataset
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Extracts the input into a cpu tensor
        if isinstance(x, Dataset):
            x = next(iter(DataLoader(x, len(x)))).numpy(force=True)
        else:
            x = x.numpy(force=True)
        output = self.model.predict(x).reshape(-1, self.num_classes)

        return torch.from_numpy(output).to(dtype=torch.float)
