import torch
import torch.nn as nn
import torch.nn.functional as F

from dataoob.model import BinaryClassifierNNMixin, ClassifierNNMixin


class LogisticRegression(ClassifierNNMixin):
    """Initializes the LogisticRegression."""

    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = num_classes

        self.linear = nn.Linear(self.input_dim, self.num_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Logistic Regression

        :param torch.Tensor x: Input tensor
        :return torch.Tensor: Output Tensor of logistic regression
        """
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x


class BinaryLogisticRegression(BinaryClassifierNNMixin, LogisticRegression):
    """Initializes the Binary Logistic Regression. BinaryClassifier is first to
    overide LogisticRegression's fit method"""

    def __init__(self, input_dim: int):
        super(BinaryLogisticRegression, self).__init__(input_dim, 2)
