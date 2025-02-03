from collections import OrderedDict
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from opendataval.dataloader import CatDataset
from opendataval.dataval import DataEvaluator, ModelMixin
from opendataval.dataval.lossval import loss_functions
from opendataval.model import (
    ClassifierMLP,
    Model,
    RegressionMLP,
    TorchGradMixin,
    TorchPredictMixin,
)
from opendataval.model.api import TorchModel


class LossValMLP(TorchPredictMixin, TorchGradMixin):
    """Pytorch MLP for LossVal
    Can be used for both regression and classification. Accepts the same parameters as
    the RegressionMLP and ClassifierMLP.

    The size of the training data set must be specified!
    """

    def __init__(
        self,
        input_dim: int,
        training_set_size: int,
        is_classification: bool,
        num_classes: int = 1,
        layers: int = 5,
        hidden_dim: int = 25,
        act_fn: Optional[Callable] = None,
    ):
        """
        Initializes an MLP for LossVal that learns the importance scores too.
        The implementation supports both regression and classification.

        Parameters
        ----------
        input_dim : int
            Size of the input dimension of the MLP
        training_set_size : int
            Size of the training set
        is_classification : bool
            Whether the model is used for classification
        num_classes : int, optional
            Number of classes for classification, by default 1
        layers : int, optional
            Number of layers for the MLP, by default 5
        hidden_dim : int, optional
            Hidden dimension for the MLP, by default 25
        act_fn : Callable, optional
            Activation function for MLP, if none, set to nn.ReLU, by default None
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = num_classes
        self.training_set_size = training_set_size
        self.nr_layers = layers
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.is_classification = is_classification

        # Initialize the data weights with ones
        self.data_weights = nn.Parameter(
            torch.ones(training_set_size), requires_grad=True
        )

        self.layers = layers
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn

        act_fn = nn.ReLU() if act_fn is None else act_fn

        mlp_layers = OrderedDict()
        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layers - 2)):
            mlp_layers[f"{i + 1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i + 1}_acti"] = act_fn

        if is_classification:
            mlp_layers[f"{i + 1}_out_lin"] = nn.Linear(hidden_dim, num_classes)
            mlp_layers["output"] = nn.Softmax(-1)
        else:
            mlp_layers["output"] = nn.Linear(hidden_dim, num_classes)

        self.mlp = nn.Sequential(mlp_layers)

        self.is_classification = is_classification
        self.weight_history = None

    def forward(self, x):
        x = self.mlp(x)
        return x

    def get_data_weights(self) -> np.ndarray:
        return self.data_weights.detach().cpu().numpy().copy()

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
        val_X: torch.Tensor = None,
        val_y: torch.Tensor = None,
        loss_function: Optional[Callable] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Fits the model on the training data.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        if (
            val_X is None or val_y is None
        ):  # This is necessary to enable the data addition and removal experiments!
            print(
                "Warning: No validation data provided! Assuming Data Removal "
                "experiment and training a standard MLP without validation."
            )
            if self.is_classification:
                return ClassifierMLP(
                    input_dim=self.input_dim,
                    num_classes=self.output_dim,
                    layers=self.nr_layers,
                    hidden_dim=self.hidden_dim,
                    act_fn=self.act_fn,
                ).fit(x_train, y_train, sample_weight, batch_size, epochs, lr)
            else:
                return RegressionMLP(
                    input_dim=self.input_dim,
                    num_classes=self.output_dim,
                    layers=self.nr_layers,
                    hidden_dim=self.hidden_dim,
                    act_fn=self.act_fn,
                ).fit(x_train, y_train, sample_weight, batch_size, epochs, lr)

        assert sample_weight is None, "Sample weights are not supported for LossVal."

        def move_dataset_to_device(dataset_, device_):
            data_loader = DataLoader(
                dataset=dataset_, batch_size=len(dataset_), shuffle=False
            )

            # Extract the full batch from DataLoader and put data on the device
            sample_ids_, x_data, targets = next(iter(data_loader))
            sample_ids_ = sample_ids_.to(device_)
            x_data = x_data.to(device_)
            targets = targets.to(device_)

            return torch.utils.data.TensorDataset(sample_ids_, x_data, targets)

        if loss_function is None:
            if self.is_classification:
                loss_function = loss_functions.LossVal_cross_entropy
            else:
                loss_function = loss_functions.LossVal_mse

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        indices = torch.arange(len(x_train))
        dataset = CatDataset(indices, x_train, y_train)
        val_X, val_y = val_X.to(self.device), val_y.to(self.device)

        # Already load the data on the device; the datasets are small enough
        dataset = move_dataset_to_device(dataset, self.device)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        self.train()
        for _ in range(int(epochs)):
            for sample_ids, x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_hat = self.__call__(x_batch)

                # Here the modified loss is called.
                loss = loss_function(
                    train_y_pred=y_hat,
                    train_y_true=y_batch,
                    train_X=x_batch,
                    val_X=val_X,
                    val_y=val_y,
                    weights=self.data_weights,
                    sample_ids=sample_ids,
                    device=self.device,
                )

                loss.backward()
                # Important: This step also updates the sample weights
                # (the data valuation)
                optimizer.step()

        return self


class LossValEvaluator(DataEvaluator, ModelMixin):
    def __init__(
        self,
        is_classification,
        loss_function: Optional[Callable] = None,
        device=torch.device("cpu"),
        nr_epochs=1,
        lr=None,
        mlp_args=None,
        *args,
        **kwargs,
    ):
        super(LossValEvaluator, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.device = device
        self.mlp_args = (
            mlp_args if mlp_args is not None else {}
        )  # MLP arguments to override the default values
        self.is_classification = is_classification

        self.nr_epochs = nr_epochs
        self.lr = lr

        self.pred_model = None

        # If the loss is None, it will be set to the correct loss function by the MLP
        self.criterion = loss_function

    def input_model(self, pred_model: Model):
        """Input the prediction model with gradient.

        Parameters
        ----------
        pred_model : GradientModel
            Prediction model with a gradient
        """
        # Ignore the input model. This evaluator creates its own model.
        if isinstance(pred_model, TorchModel):
            self.device = pred_model.device
        return self

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data

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

        self.num_points = len(x_train)

        # We need the info about the data to instantiate the MLP for LossVal
        self.pred_model = LossValMLP(
            input_dim=x_train.shape[1],
            training_set_size=self.num_points,
            num_classes=y_train.shape[1],
            is_classification=self.is_classification,
            **self.mlp_args,
        )

        return self

    def train_data_values(self, *args, **kwargs):
        """Calculate the data values.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Check that the model is a torch model:
        assert isinstance(self.pred_model, TorchModel)
        assert isinstance(self.pred_model, nn.Module)
        assert isinstance(self.pred_model, LossValMLP)
        self.pred_model: LossValMLP

        # Move copy of all data to the device
        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.x_valid = self.x_valid.to(self.device)
        self.y_valid = self.y_valid.to(self.device)

        # Train the model
        self.pred_model = self.pred_model.to(self.device)

        if "epochs" in kwargs:
            # Just in case there are conflicting parameters (when one is passed by the
            # ExperimentMediator)
            kwargs.pop("epochs")

        if self.lr is None:
            self.pred_model.fit(
                self.x_train,
                self.y_train,
                epochs=self.nr_epochs,
                loss_function=self.criterion,
                val_X=self.x_valid,
                val_y=self.y_valid,
                *args,
                **kwargs,
            )
        else:
            self.pred_model.fit(
                self.x_train,
                self.y_train,
                epochs=self.nr_epochs,
                lr=self.lr,
                loss_function=self.criterion,
                val_X=self.x_valid,
                val_y=self.y_valid,
                *args,
                **kwargs,
            )

        return self

    def evaluate_data_values(self):
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        return self.pred_model.get_data_weights()

    def __repr__(self):
        lr_str = "" if self.lr is None else f", lr={self.lr}"
        return f"LossValEvaluator(nr_epochs={self.nr_epochs}{lr_str})"

    def __str__(self):
        return self.__repr__()
