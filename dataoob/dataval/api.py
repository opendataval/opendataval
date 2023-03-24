import copy
from abc import ABC, abstractmethod

import torch
import numpy as np
from numpy.random import RandomState

from dataoob.model import Model
from sklearn.utils import check_random_state

from typing import Callable


class DataEvaluator(ABC):
    """Abstract class of Data Evaluators."""

    def __init__(self, random_state: RandomState = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor, metric: Callable = None):
        if metric is None:
            return self.metric(y, y_hat)
        elif callable(metric):
            return metric(y, y_hat)
        raise Exception("Metric not specified")

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        batch_size: int = 32,
        epochs: int = 1
    ):
        """Trains the Data Evaluator and the underlying prediction model. Wrapper for
       `self.input_data` and `self.train_data_values` under one method

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        :param int batch_size: Training batch size, defaults to 32
        """
        self.input_data(x_train, y_train, x_valid, y_valid)
        self.train_data_values(batch_size=batch_size, epochs=epochs)

        return self

    def input_model_metric(self, pred_model: Model, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        """Inputs the prediction model and the evaluation metric


        :param Model pred_model: _description_
        :param Callable[[torch.Tensor, torch.Tensor], float] metric: Evaluation function to determine model performance
        :return _type_: _description_
        """
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        return self


    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and processes the data for the given evaluator, helps
        separate the structure from the data

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        return self

    def get_data_points(self):
        return (self.x_train, self.y_train), (self.x_valid, self.y_valid)

    @abstractmethod
    def train_data_values(self, batch_size: int=32, epochs: int=1):
        """Trains the evaluator to compute data values of the model

        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        """
        return self

    @abstractmethod
    def evaluate_data_values(self) -> np.ndarray:
        """Evaluates the data values of the input training dataset. Outputs the
        data values as an array

        :return np.ndarray: Data values computed by the data valuator. Outputs a
        np.ndarray because many metrics expect an np.ndarray.
        """
        pass


def DE(method: DataEvaluator, model: Model, *args, **kwargs):
    # TODO Write If Else statements once it's populated with Data Evaluators
    return method(model, *args, **kwargs)
