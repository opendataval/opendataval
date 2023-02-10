import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataoob.model import Classifier


class ANN(nn.Module, Classifier):
    def __init__(self, input_dim, num_of_classes = 2):
        ''' Initializes the LogisticRegression.
        '''

        super(ANN, self).__init__()

        self.input_dim = input_dim
        self.num_of_classes = num_of_classes
        self.linear1 = nn.Linear(self.input_dim, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 100)
        self.linearout  = nn.Linear(100, num_of_classes)
        self.output = nn.Softmax(1)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x =F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)

        x = self.linearout(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    def predict_proba(self, x):
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