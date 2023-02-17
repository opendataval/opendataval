import copy
from abc import ABC, abstractmethod

import torch

from dataoob.model import Model


class DataEvaluator(ABC):
    """Abstract class of Evaluators. Provides a template of how evaluators interact with the pred_model
    """
    def __init__(self, pred_model: Model, metric: callable):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

    def evaluate(self, y: torch.tensor, yhat: torch.tensor, metric: callable=None):
        if metric is None:
            return self.metric(y, yhat)
        elif isinstance(metric, callable):
            return metric(y, yhat)
        raise Exception("Metric not specified")

    def train(
        self,
        x_train: torch.tensor,
        y_train: torch.tensor,
        x_valid: torch.tensor,
        y_valid: torch.tensor,
        epochs: int = 1,
        batch_size: int = 32,
        *args,
        **kwargs
    ):
        """_summary_

        :param torch.tensor x_train: Data covariates
        :param torch.tensor y_train: Data labels
        :param torch.tensor x_valid: Test+Held-out covariates
        :param torch.tensor y_valid: Test+Held-out labels
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        :param int batch_size: Training batch size, defaults to 32
        """
        self.input_data(x_train, y_train, x_valid, y_valid)
        self.train_data_values(
            batch_size=batch_size,
            epochs=epochs,
            *args,
            **kwargs
        )

    @abstractmethod
    def input_data(
        self,
        x_train: torch.tensor,
        y_train: torch.tensor,
        x_valid: torch.tensor,
        y_valid: torch.tensor,
    ):
        """Stores and processes the data for the given evaluator, helps
        seperate the structure from the data

        :param torch.tensor x_train: Data covariates
        :param torch.tensor y_train: Data labels
        :param torch.tensor x_valid: Test+Held-out covariates
        :param torch.tensor y_valid: Test+Held-out labels
        """
        pass

    @abstractmethod
    def train_data_values(
        self,
        batch_size: int=32,
        epochs: int=1,
        *args,
        **kwargs
    ):
        """Trains the evaluator to compute data values of the model

        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        """
        pass

    @abstractmethod
    def evaluate_data_values(self) -> torch.tensor:
        """Evaluates the datavalues of the following tensors. NOTE this method may change
        due to the fact that inputs that differ from input tensors might not be allowed
        """
        pass


def DE(method: DataEvaluator, model: Model, *args, **kwargs):
    # TODO Write If Else statements once it's populated with Data Evaluators
    return method(model, *args, **kwargs)
