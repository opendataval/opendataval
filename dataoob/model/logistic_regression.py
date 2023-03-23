import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataoob.model import BinaryClassifierNNMixin


class LogisticRegression(BinaryClassifierNNMixin):
    def __init__(self, input_dim: int):
        """Initializes the LogisticRegression."""

        super(LogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = 2

        self.linear = nn.Linear(self.input_dim, self.num_of_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Logistic Regression

        :param torch.Tensor x: Input tensor
        :return torch.Tensor: Output Tensor of logistic regression
        """
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts output from input tensor

        :param torch.Tensor x: Input tensor
        :return torch.Tensor: Predicted tensor output
        """
        y = self.forward(x)
        return y
