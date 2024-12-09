import torch
import torch.nn as nn
import torch.nn.functional as F

from opendataval.model.api import TorchClassMixin, TorchPredictMixin
from opendataval.model.grad import TorchGradMixin


class LogisticRegression(TorchClassMixin, TorchPredictMixin, TorchGradMixin):
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
        if self.num_classes <= 2:
            x = F.sigmoid(x)
        else:
            # Equivalent to sigmoid for classes of size 2.
            x = F.softmax(x, dim=1)
        return x
