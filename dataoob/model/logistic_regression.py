import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataoob.model import Classifier


class LogisticRegression(nn.Module, Classifier):
    def __init__(self, input_dim: int, num_of_classes: int = 2):
        ''' Initializes the LogisticRegression.
        '''

        super(LogisticRegression, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = num_of_classes

        self.linear = nn.Linear(self.input_dim, self.num_of_classes)

    def return_ground_truth_importance(self, x):
        """ Returns a vector containing the ground truth feature attributions for input x.
        """
        # the true feature attribution is the same for all points x
        return self.linear.weight[1, :] - self.linear.weight[0, :]

    def forward(self, x: torch.tensor):
        """_summary_

        :param torch.tensor x: _description_
        :return _type_: _description_
        """
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x

    def predict_proba(self, x: torch.tensor):
        """_summary_

        :param torch.tensor x: _description_
        :return _type_: _description_
        """        """
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

