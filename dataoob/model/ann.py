from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn

from dataoob.model.api import (
    TorchBinClassMixin,
    TorchClassMixin,
    TorchPredictMixin,
    TorchRegressMixin,
)


class ClassifierANN(TorchClassMixin, TorchPredictMixin):
    """Initializes the Artificial Neural Network."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Callable = None,
    ):
        super().__init__()

        act_fn = nn.ReLU() if act_fn is None else act_fn

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
        """Forward pass of Artificial Neural Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of ANN
        """
        x = self.mlp(x)
        return x


class BinaryANN(TorchBinClassMixin, ClassifierANN):
    """Initializes the BinaryANN. TorchBinClassMixin defines `.fit()`."""

    def __init__(
        self,
        input_dim: int,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Callable = None,
    ):
        super().__init__(input_dim, 2, layers, hidden_dim, act_fn)


class RegressionANN(TorchRegressMixin, TorchPredictMixin):
    """Initializes RegressionANN."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Callable = None,
    ):
        super().__init__()

        act_fn = nn.ReLU() if act_fn is None else act_fn

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layers - 2)):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = act_fn

        mlp_layers["output"] = nn.Linear(hidden_dim, num_classes)

        self.mlp = nn.Sequential(mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Artificial Neural Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of ANN
        """
        x = self.mlp(x)
        return x
