import copy
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from dataoob.model import Model
from dataoob.dataloader import DataLoader


class DataEvaluator(ABC):
    """Abstract class of Data Evaluators. Facilitates Data Evaluation computation.

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    args : tuple[Any]
        DavaEvaluator positional arguments
    kwargs : Dict[str, Any]
        DavaEvaluator key word arguments

    The following is an example of how the api would work
    .. highlight:: python
    ::
        dataval = (
            DataEvaluator(*args, **kwargs)
            .input_model_metric(model, metric)
            .input_data(x_train, y_train, x_valid, y_valid)
            .train_data_values(batch_size, epochs)
            .evaluate_data_values()
        )
    """

    def __init__(self, random_state: RandomState = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor, metric: Callable = None):
        """Evaluates performance

        Parameters
        ----------
        y : torch.Tensor
            Labels to be evaluate performance of predictions
        y_hat : torch.Tensor
            Predictions of labels
        metric : Callable, optional
            Callable evaluating performance of labels and prediction, by default None

        Returns
        -------
        Any | float
            Performance metric

        Raises
        ------
        ValueError
            If metric is not specified either by the ``self.input_model_metric()`` or
            as an argument
        """
        if metric is None:
            return self.metric(y, y_hat)
        elif callable(metric):
            return metric(y, y_hat)
        raise ValueError("Metric not specified")

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Trains the Data Evaluator and the underlying prediction model. Wrapper for
        ``self.input_data`` and ``self.train_data_values`` under one method

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.input_data(x_train, y_train, x_valid, y_valid)
        self.train_data_values(*args, **kwargs)

        return self

    def input_model_metric(
        self, pred_model: Model, metric: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        """Inputs the prediction model and the evaluation metric

        Parameters
        ----------
        pred_model : Model
            Prediction model
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        return self

    def input_dataloader(self, loader: DataLoader):
        """Inputs data from a DataLoader object"""
        x_train, y_train, x_valid, y_valid = loader.datapoints
        return self.input_data(x_train, y_train, x_valid, y_valid)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and processes the data for DataEvaluator

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        return self

    @abstractmethod
    def train_data_values(self, *args, **kwargs):
        """Trains the DataEvaluator to compute data values

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a trained Data Evaluator.
        """
        return self

    @abstractmethod
    def evaluate_data_values(self) -> np.ndarray:
        """Computes the data values of the training data set.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        pass

    def __new__(cls, *args, **kwargs):
        """Saves the input arguments for better plot ttiles"""
        obj = object.__new__(cls)
        obj.__inputs = [str(arg) for arg in args]
        obj.__inputs.extend(f"{arg_name}={value}" for arg_name, value in kwargs.items())

        return obj

    @property
    def plot_title(self) -> str:  # For publication keep it simple
        return f"{self.__class__.__name__}({', '.join(self.__inputs)})"


def DE(method: DataEvaluator, model: Model, *args, **kwargs):
    # TODO Write If Else statements once it's populated with Data Evaluators
    return method(model, *args, **kwargs)
