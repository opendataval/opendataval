from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Model(ABC):
    @abstractmethod
    def fit(self, x_train, y_train, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class Classifier(Model):
    @abstractmethod
    def predict(self, x):
        """TODO Modify to return torch tensor
        predict method for CFE-Models which need this method.
        :param data: torch or list
        :return: np.array with prediction
        """
        pass

    def fit(
        self,
        x_train,
        y_train,
        sample_weight=None,
        batch_size=32,
        epochs=1,
        verbose=False,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = F.binary_cross_entropy

        for epoch in range(int(epochs)):
            permutation = torch.randperm(x_train.size(axis=0))

            for i in range(0, x_train.size(axis=0), batch_size):
                # Consider dataloader, more idiomatic
                indices = permutation[i : i + batch_size]
                x_batch, y_batch = x_train[indices], y_train[indices]

                optimizer.zero_grad()  # Setting our stored gradients equal to zero
                outputs = self.__call__(x_batch)
                if sample_weight is None:
                   loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(outputs, y_batch, weight=sample_weight[indices])

                loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
                optimizer.step()  # Updates weights and biases with the optimizer (SGD)
