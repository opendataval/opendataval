import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataoob.model import ClassifierNN


class LogisticRegression(ClassifierNN):
    def __init__(self, input_dim: int, num_of_classes: int = 2):
        """Initializes the LogisticRegression."""

        super(LogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = num_of_classes

        self.linear = nn.Linear(self.input_dim, self.num_of_classes)

    def forward(self, x: torch.Tensor):
        """_summary_

        :param torch.Tensor x: _description_
        :return _type_: _description_
        """
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x

    def predict(self, x: torch.Tensor):
        """_summary_

        :param torch.Tensor x: _description_
        :return _type_: _description_
        """ """
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
