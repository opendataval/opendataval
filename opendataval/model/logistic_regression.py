import torch
import torch.nn as nn
import torch.nn.functional as F

from opendataval.model.api import TorchClassMixin, TorchPredictMixin


class LogisticRegression(TorchClassMixin, TorchPredictMixin):
    """Initialize LogisticRegression

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the LogisticRegression
    num_classes : int
        Size of the output dimension of the LR, outputs selection probabilities
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Logistic Regression.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of logistic regression
        """
        x = self.linear(x)
        x = F.sigmoid(x)
        return x
