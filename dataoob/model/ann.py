import torch
import torch.nn as nn

from dataoob.model import BinaryClassifierNNMixin, ClassifierNNMixin
from collections import OrderedDict
from typing import Callable


class ANN(ClassifierNNMixin):
    """Initializes the Artifical Neural Network."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Callable = nn.ReLU(),
    ):
        super(ANN, self).__init__()

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layers - 2)):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = act_fn

        mlp_layers[f"{i+1}_out_lin"] = nn.Linear(hidden_dim, num_classes)
        mlp_layers[f"{i+1}_out_acti"] = act_fn

        mlp_layers["output"] = nn.Softmax(-1)

        self.mlp = nn.Sequential(mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ANN

        :param torch.Tensor x: Input tensor
        :return torch.Tensor: Output Tensor of logistic regression
        """
        x = self.mlp(x)
        return x


class BinaryANN(BinaryClassifierNNMixin, ANN):
    """Initializes the Binary Artificial Neural Network. BinaryClassifier is first
    to override ANN's fit method"""

    def __init__(
        self,
        input_dim: int,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Callable = nn.ReLU(),
    ):
        super(BinaryANN, self).__init__(input_dim, 2, layers, hidden_dim, act_fn)
