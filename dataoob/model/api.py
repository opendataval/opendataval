from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_cpu(x):
    return x.detach().cpu()

class Model(ABC):
    @abstractmethod
    def fit(self, x_train: torch.tensor, y_train: torch.tensor, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x: torch.tensor, *args, **kwargs):
        pass


class ClassifierNN(Model, nn.Module):
    def fit(
        self,
        x_train,
        y_train,
        sample_weight: torch.tensor=None,
        batch_size=32,
        epochs=1,
        verbose=False,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = F.binary_cross_entropy  # TODO update when we move from binary classification

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

class ClassifierSkLearnWrapper(Model):
    def __init__(self, base_model, device: torch.device=torch.device('cpu')):
        self.model = base_model
        self.device = device

    def fit(self, x_train: torch.tensor, y_train: torch.tensor,  sample_weight: torch.tensor=None, *args, **kwargs):
        x_train, y_train = to_cpu(x_train), to_cpu(y_train)

        self.model.fit(
            x_train, torch.argmax(y_train, dim=1),
            None if sample_weight is None else torch.squeeze(to_cpu(sample_weight)),
            *args,
            **kwargs
        )

    def predict(self, x: torch.tensor):
        x = to_cpu(x)
        output = self.model.predict_proba(x)
        return torch.from_numpy(output).to(dtype=torch.float32, device=self.device)

class ClassifierUnweightedSkLearnWrapper(ClassifierSkLearnWrapper):
    def fit(self, x_train: torch.tensor, y_train: torch.tensor,  sample_weight: torch.tensor=None, *args, **kwargs):
        x_train, y_train  = to_cpu(x_train), to_cpu(y_train)

        if sample_weight is not None:
            n_samples = x_train.size(dim=0)
            sample_weight = torch.squeeze(sample_weight)
            indices = np.random.choice(n_samples, size=(n_samples), replace=True, p=sample_weight/sample_weight.sum())

            self.model.fit(x_train[indices], torch.argmax(y_train, dim=1), *args, **kwargs)
        else:
            self.model.fit(x_train, torch.argmax(y_train, dim=1), *args, **kwargs)
