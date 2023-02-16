import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataoob.model import ClassifierNN


class ANN(ClassifierNN):
    def __init__(self, input_dim, num_of_classes=2):
        """Initializes the Artifical Neural Network."""

        super(ANN, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = num_of_classes
        self.linear1 = nn.Linear(self.input_dim, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linearout = nn.Linear(100, num_of_classes)
        self.output = nn.Softmax(-1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linearout(x)
        x = self.output(x)
        return x

    def predict(self, x):
        """
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.array(x)).float()
        else:
            x = torch.squeeze(x)

        y = self.forward(x)
        return y
