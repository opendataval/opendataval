from abc import ABC, abstractmethod

import torch

from dataoob.model.api import Model


class Evaluator(ABC):
    def __init__(self, pred_model: Model):
        self.pred_model = pred_model

    def train(
        self, x_train: torch.tensor, y_train: torch.tensor, x_valid: torch.tensor, y_valid: torch.tensor,
        pre_train_pred: bool=False, batch_size: int=32, epochs: int=1, pred_epochs: int=1, *args, **kwargs
    ):
        self.input_data(x_train, y_train, x_valid, y_valid)
        self.train_data_value(
            pre_train_pred=pre_train_pred,
            batch_size=batch_size,
            epochs=epochs,
            pred_epochs=pred_epochs,
            *args,
            **kwargs
        )

    @abstractmethod
    def input_data(self, x_train: torch.tensor, y_train: torch.tensor, x_valid: torch.tensor, y_valid: torch.tensor):
        pass

    @abstractmethod
    def train_data_value(self, pre_train_pred: bool=False, batch_size: int=32, epochs: int=1, pred_epochs: int=1, *args, **kwargs):
        pass

    @abstractmethod
    def predict_data_value(self, x: torch.tensor, y: torch.tensor, *args, **kwargs):
        pass


def DataEvaluator(method: Evaluator, model: Model, *args, **kwargs):
    # TODO Write If Else statements once it's populated with Data Evaluators
    return method(model, *args, **kwargs)
