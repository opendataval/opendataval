import torch
import copy
import numpy as np
class DummyEvaluator:
    """Abstract class of Data Evaluators. Provides a template of how evaluators interact
    with the pred_model and specific methods each evaluator should implement
    """
    def __init__(self, pred_model, metric: callable, *args, **kwargs):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor, metric: callable=None):
        if metric is None:
            return self.metric(y, y_hat)
        elif isinstance(metric, callable):
            return metric(y, y_hat)
        raise Exception("Metric not specified")

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        batch_size: int = 32,
        epochs: int = 1
    ):
        """Trains the Data Evaluator and the underlying prediction model. Wrapper for
       `self.input_data` and `self.train_data_values` under one method

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        :param int batch_size: Training batch size, defaults to 32
        """
        self.input_data(x_train, y_train, x_valid, y_valid)
        self.train_data_values(batch_size=batch_size, epochs=epochs)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and processes the data for the given evaluator, helps
        separate the structure from the data

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        (self.x_train, self.y_train) = (x_train, y_train)
        (self.x_valid, self.y_valid) = (x_valid, y_valid)

    def get_data_points(self):
        return (self.x_train, self.y_train), (self.x_valid, self.y_valid)

    def train_data_values(self, batch_size: int=32, epochs: int=1):
        """Trains the evaluator to compute data values of the model

        :param int batch_size: Training batch size, defaults to 32
        :param int epochs: Number of epochs to train the pred_model, defaults to 1
        """
        pass

    def evaluate_data_values(self) -> np.ndarray:
        """Evaluates the data values of the input training dataset. Outputs the
        data values as an array

        :return np.ndarray: Data values computed by the data valuator. Outputs a
        np.ndarray because many metrics expect an np.ndarray.
        """
        return np.ones((len(self.x_train),))