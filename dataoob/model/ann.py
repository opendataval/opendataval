import torch
import torch.nn as nn
import torch.nn.functional as F

from dataoob.model import ClassifierNNMixin


class ANN(ClassifierNNMixin):
    def __init__(self, input_dim, num_of_classes=2):
        """Initializes the Artifical Neural Network."""

        super(ANN, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = num_of_classes
        self.linear1 = nn.Linear(self.input_dim, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linearout = nn.Linear(100, num_of_classes)
        self.output = nn.Softmax(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ANN

        :param torch.Tensor x: Input tensor
        :return torch.Tensor: Output Tensor of logistic regression
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linearout(x)
        x = self.output(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts output from input tensor

        :param torch.Tensor x: Input tensor
        :return torch.Tensor: Predicted tensor output
        """
        y = self.forward(x)
        return y
