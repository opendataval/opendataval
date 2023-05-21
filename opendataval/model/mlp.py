from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn

from opendataval.model.api import (
    TorchClassMixin,
    TorchPredictMixin,
    TorchRegressMixin,
)


class ClassifierMLP(TorchClassMixin, TorchPredictMixin):
    """Initializes the Multilayer Perceptron  Classifier.

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the MLP
    num_classes : int
        Size of the output dimension of the MLP, outputs selection probabilities
    layers : int, optional
        Number of layers for the MLP, by default 5
    hidden_dim : int, optional
        Hidden dimension for the MLP, by default 25
    act_fn : Callable, optional
        Activation function for MLP, if none, set to nn.ReLU, by default None
    """

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
        self.num_classes = num_classes

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layers - 2)):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = act_fn

        mlp_layers[f"{i+1}_out_lin"] = nn.Linear(hidden_dim, num_classes)
        mlp_layers["output"] = nn.Softmax(-1)

        self.mlp = nn.Sequential(mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP Neural Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of MLP
        """
        x = self.mlp(x)
        return x


class RegressionMLP(TorchRegressMixin, TorchPredictMixin):
    """Initializes the Multilayer Perceptron Regression.

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the MLP
    num_classes : int
        Size of the output dimension of the MLP, >1 means multi dimension output
    layers : int, optional
        Number of layers for the MLP, by default 5
    hidden_dim : int, optional
        Hidden dimension for the MLP, by default 25
    act_fn : Callable, optional
        Activation function for MLP, if none, set to nn.ReLU, by default None
    """

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
        """Forward pass of Multilayer Perceptron.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of MLP
        """
        x = self.mlp(x)
        return x
