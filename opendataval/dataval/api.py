from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from opendataval.dataloader import DataFetcher
from opendataval.model import Model


class DataEvaluator(ABC):
    """Abstract class of Data Evaluators. Facilitates Data Evaluation computation.

    The following is an example of how the api would work:
    ::
        dataval = (
            DataEvaluator(*args, **kwargs)
            .input_model_metric(model, metric)
            .input_data(x_train, y_train, x_valid, y_valid)
            .train_data_values(batch_size, epochs)
            .evaluate_data_values()
        )

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    args : tuple[Any]
        DavaEvaluator positional arguments
    kwargs : Dict[str, Any]
        DavaEvaluator key word arguments

    Attributes
    ----------
    pred_model : Model
        Prediction model to find how much each training datum contributes towards it.

    Raises
    ------
    ValueError
        If metric is not specified either by the ``self.input_model_metric()`` or
        as an argument
    """

    def __init__(self, random_state: RandomState = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor, metric: Callable = None):
        """Evaluate performance of the specified metric between label and predictions.

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
        if y.device != y_hat.device:
            y = y.to(device=y_hat.device)

        if metric is None and hasattr(self, "metric"):
            return self.metric(y, y_hat)
        elif callable(metric):
            return metric(y, y_hat)
        raise ValueError("Metric not specified.")

    def input_model_metric(
        self, pred_model: Model, metric: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        """Input the prediction model and the evaluation metric.

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
        self.pred_model = pred_model.clone()
        self.metric = metric

        return self

    def input_data(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: torch.Tensor,
        x_valid: Union[torch.Tensor, Dataset],
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for DataEvaluator.

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
        """Trains model to predict data values.

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
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """

    def input_fetcher(self, fetcher: DataFetcher):
        """Input data from a DataFetcher object. Alternative way of adding data."""
        x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints
        return self.input_data(x_train, y_train, x_valid, y_valid)

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Store and transform data, then train model to predict data values.

        Trains the Data Evaluator and the underlying prediction model. Wrapper for
        ``self.input_data`` and ``self.train_data_values`` under one method.

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

    def __new__(cls, *args, **kwargs):
        """Record the input arguments for unique identifier of DataEvaluator."""
        obj = object.__new__(cls)
        obj.__inputs = [str(arg) for arg in args]
        obj.__inputs.extend(f"{arg_name}={value}" for arg_name, value in kwargs.items())

        return obj

    def __str__(self) -> str:  # For publication keep it simple
        """Get unique string representation for a DataEvaluator."""
        return f"{self.__class__.__name__}({', '.join(self.__inputs)})"
