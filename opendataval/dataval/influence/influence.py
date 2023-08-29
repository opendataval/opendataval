from functools import partial

import numpy as np
import torch

from opendataval.dataval.api import DataEvaluator, ModelMixin
from opendataval.model import GradientModel


class InfluenceFunction(DataEvaluator, ModelMixin):
    """Influence Function Data evaluation implementation.

    TODO it may be useful to compute gradients of the validation dataset in batches
    to save time/space.
    TODO H^{-1} implementation, Current implementation is for first-order gradients

    References
    ----------
    .. [1] P. W. Koh and P. Liang,
        Understanding Black-box Predictions via Influence Functions,
        arXiv.org, 2017. https://arxiv.org/abs/1703.04730.

    Parameters
    ----------
    grad_args : tuple, optional
        Positional arguments passed to the model.grad function
    grad_kwargs : dict[str, Any], optional
        Key word arguments passed to the model.grad function
    """

    def __init__(self, *grad_args, **grad_kwargs):
        self.args = grad_args
        self.kwargs = grad_kwargs

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Function Data Valuation.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.influence = np.zeros(len(x_train))
        return self

    def input_model(self, pred_model: GradientModel):
        """Input the prediction model with gradient.

        Parameters
        ----------
        pred_model : GradientModel
            Prediction model with a gradient
        """
        assert (  # In case model doesn't inherit but still wants the grad function
            isinstance(pred_model, GradientModel)
            or callable(getattr(pred_model, "grad"))
        ), ("Model with gradient required.")

        self.pred_model = pred_model.clone()
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to compute influence of each data point (data values).

        References
        ----------
        .. [1] Implementation inspired by `valda <https://github.com/uvanlp/valda>`_.
            <https://github.com/uvanlp/valda/blob/main/src/valda/inf_func.py>

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Trains model on training data so we can find gradients of trained model
        self.pred_model.fit(self.x_train, self.y_train, *args, **kwargs)
        iter_grad = partial(self.pred_model.grad, *self.args, **self.kwargs)

        # Outer loop remains an iterator
        # Inner loop pre-computes and stores gradients for speed up.
        valid_grad_list = list(iter_grad(self.x_valid, self.y_valid))

        for i, train_grads in enumerate(iter_grad(self.x_train, self.y_train)):
            for valid_grads in valid_grad_list:
                inf = sum(torch.sum(t * v) for t, v in zip(train_grads, valid_grads))
                self.influence[i] += inf

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return influence (data values) for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values for training input data point
        """
        return self.influence
