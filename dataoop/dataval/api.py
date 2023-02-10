from abc import ABC, abstractmethod

import torch

from dataoop.model.api import Model


class Evaluator(ABC):
    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def data_value_evaluator(self, inputs: torch.tensor, label: torch.Tensor):
        pass


def DataEvaluator(method: Evaluator, model: Model, *args, **kwargs):
    # TODO Write If Else statements once it's populated with Data Evaluators
    return method(model, *args, **kwargs)
