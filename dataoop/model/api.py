from abc import ABC, abstractmethod

import torch
from tqdm import tqdm


class Model(ABC):
    @abstractmethod
    def fit(self, x_train, y_train, *args, **kwargs):
        pass
    @abstractmethod
    def predict(self, x):
        pass

class Classifier(Model):
    @abstractmethod
    def predict_proba(self, x):
        pass
    def predict(self, x):
        """  TODO Modify to return torch tensor
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        return torch.argmax(self.predict_proba(x), dim=1)
    def fit(self, x_train, y_train, sample_weight=None, batch_size=32, epochs=1,  verbose=False):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()



        for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
            permutation = torch.randperm(x_train.size(axis=0))

            for i in range(0, x_train.size(axis=0), batch_size):
                indices = permutation[i: i+batch_size]
                x_batch, y_batch = x_train[indices], y_train[indices]
                optimizer.zero_grad() # Setting our stored gradients equal to zero
                outputs = self.__call__(x_batch)

                loss = criterion(torch.squeeze(outputs), y_batch)
                if sample_weight is not None:
                    loss = torch.mul(loss, sample_weight[indices])
                loss.sum().backward(retain_graph=True) # Computes the gradient of the given tensor w.r.t. the weights/bias
                optimizer.step() # Updates weights and biases with the optimizer (SGD)


