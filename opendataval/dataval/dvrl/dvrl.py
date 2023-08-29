from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, RandomSampler

from opendataval.dataloader.util import CatDataset
from opendataval.dataval.api import DataEvaluator, ModelMixin


class DVRL(DataEvaluator, ModelMixin):
    """Data valuation using reinforcement learning class, implemented with PyTorch.

    References
    ----------
    .. [1] J. Yoon, S. Arik, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1909.11671.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimensions for the RL Multilayer Perceptron Value Estimator (VE)
        (details in :py:class:`DataValueEstimatorRL` class), by default 100
    layer_number : int, optional
        Number of hidden layers for the Value Estimator (VE), by default 5
    comb_dim : int, optional
        After concat inputs how many layers, much less than `hidden_dim`, by default 10
    rl_epochs : int, optional
        Number of training epochs for the VE, by default 1000
    rl_batch_size : int, optional
        Batch size for training the VE, by default 32
    lr : float, optional
        Learning rate for the VE, by default 0.01
    threshold : float, optional
        Search rate threshold, the VE may get stuck in certain bounds close to
        :math:`[0, 1]`, thus outside of :math:`[1-threshold, threshold]` we encourage
        searching, by default 0.9
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        hidden_dim: int = 100,
        layer_number: int = 5,
        comb_dim: int = 10,
        rl_epochs: int = 1000,
        rl_batch_size: int = 32,
        lr: float = 0.01,
        threshold: float = 0.9,
        device: torch.device = torch.device("cpu"),
        random_state: Optional[RandomState] = None,
    ):
        # Value estimator parameters
        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.comb_dim = comb_dim
        self.device = device

        # Training parameters
        self.rl_epochs = rl_epochs
        self.rl_batch_size = rl_batch_size
        self.lr = lr
        self.threshold = threshold

        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for DVRL.

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

        self.num_points, [*self.feature_dim] = len(x_train), x_train[0].shape
        [*self.label_dim] = (1,) if self.y_train.ndim == 1 else self.y_train[0].shape

        self.value_estimator = DataValueEstimatorRL(
            x_dim=np.prod(self.feature_dim),
            y_dim=np.prod(self.label_dim),
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            random_state=self.random_state,
        ).to(self.device)

        return self

    def _evaluate_baseline_models(self, *args, **kwargs):
        """Load and train baseline models.

        Baseline performance information is necessary to compute the reward.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Final model
        self.final_model = self.pred_model.clone()

        # Train baseline model with input data
        self.ori_model = self.pred_model.clone()
        self.ori_model.fit(self.x_train, self.y_train, *args, **kwargs)

        # Trains validation model
        self.val_model = self.ori_model.clone()
        self.val_model.fit(self.x_valid, self.y_valid, *args, **kwargs)

        # Eval performance
        # Baseline performance
        y_valid_hat = self.ori_model.predict(self.x_valid)
        self.valid_perf = self.evaluate(self.y_valid, y_valid_hat)

        # Compute diff
        y_pred = self.val_model.predict(self.x_train).cpu()

        self.y_pred_diff = torch.abs(self.y_train - y_pred)

    def train_data_values(self, *args, num_workers: int = 0, **kwargs):
        """Trains model to predict data values.

        Trains the VE to assign probabilities of each data point being selected
        using a signal from the evaluation performance.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        num_workers : int, optional
            Number of workers used to load data, by default 0, loaded in main process
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        batch_size = min(self.rl_batch_size, len(self.x_train))
        self._evaluate_baseline_models(*args, **kwargs)

        # Solver
        optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr)
        criterion = DveLoss(threshold=self.threshold)

        gen = torch.Generator(self.device).manual_seed(self.random_state.tomaxint())
        cpu_gen = torch.Generator("cpu").manual_seed(self.random_state.tomaxint())

        data = CatDataset(self.x_train, self.y_train, self.y_pred_diff)
        rs = RandomSampler(data, True, self.rl_epochs * batch_size, generator=cpu_gen)
        dataloader = DataLoader(
            data,
            batch_size,
            sampler=rs,
            generator=cpu_gen,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        for x_batch, y_batch, y_hat_batch in tqdm.tqdm(dataloader):
            # Moves tensors to actual device
            x_batch_ve = x_batch.to(device=self.device)
            y_batch_ve = y_batch.to(device=self.device)
            y_hat_batch_ve = y_hat_batch.to(device=self.device)

            optimizer.zero_grad()

            # Generates selection probability
            pred_dataval = self.value_estimator(x_batch_ve, y_batch_ve, y_hat_batch_ve)

            # Samples the selection probability
            select_prob = torch.bernoulli(pred_dataval, generator=gen)
            if select_prob.sum().item() == 0:  # Exception (select probability is 0)
                pred_dataval = 0.5 * torch.ones_like(pred_dataval, requires_grad=True)
                select_prob = torch.bernoulli(pred_dataval, generator=gen)

            # Prediction and training
            new_model = self.pred_model.clone()
            new_model.fit(
                x_batch,
                y_batch,
                *args,
                sample_weight=select_prob.detach().cpu(),  # Expects cpu tensors
                **kwargs,
            )

            # Reward computation
            y_valid_hat = new_model.predict(self.x_valid)
            dvrl_perf = self.evaluate(self.y_valid, y_valid_hat)
            # NOTE must want to maximize the metric (IE for MSE use -MSE)
            reward_curr = dvrl_perf - self.valid_perf

            # Trains the VE
            loss = criterion(pred_dataval, select_prob, reward_curr)
            loss.backward(retain_graph=True)
            optimizer.step()

        weights = torch.zeros(0, 1, device=self.device)
        for x_batch, y_batch, y_hat_batch in DataLoader(
            data, batch_size=self.rl_batch_size, shuffle=False
        ):
            # Moves tensors to actual device
            x_batch = x_batch.to(device=self.device)
            y_batch = y_batch.to(device=self.device)
            y_hat_batch = y_hat_batch.to(device=self.device)

            data_values = self.value_estimator(x_batch, y_batch, y_hat_batch)
            weights = torch.cat([weights, data_values])

        self.final_model = self.pred_model.clone()
        self.final_model.fit(
            self.x_train,
            self.y_train,
            *args,
            sample_weight=weights.detach().cpu(),  # Expects cpu tensors
            **kwargs,
        )

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values for DVRL using the Value Estimator MLP.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        y_valid_pred = self.final_model.predict(self.x_train).cpu()
        y_hat = torch.abs(self.y_train - y_valid_pred)
        response = torch.zeros(0, 1, device=self.device)

        # Estimates data value
        with torch.no_grad():  # No dropout layers so no need to set to eval
            data = CatDataset(self.x_train, self.y_train, y_hat)
            for x_batch, y_batch, y_hat_batch in DataLoader(
                data, batch_size=self.rl_batch_size, shuffle=False
            ):
                # Moves tensors to actual device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                y_hat_batch = y_hat_batch.to(device=self.device)

                data_values = self.value_estimator(x_batch, y_batch, y_hat_batch)
                response = torch.cat([response, data_values])

        return response.squeeze().numpy(force=True)


class DataValueEstimatorRL(nn.Module):
    """Value Estimator model.

    Here, we assume a simple multi-layer perceptron architecture for the data
    value evaluator model. For data types like tabular, multi-layer perceptron
    is already efficient at extracting the relevant information.
    For high-dimensional data types like images or text,
    it is important to introduce inductive biases to the architecture to
    extract information efficiently. In such cases, there are two options:
    (i) Input the encoded representations (e.g. the last layer activations of
    ResNet for images, or the last layer activations of BERT for  text) and use
    the multi-layer perceptron on top of it. The encoded representations can
    simply come from a pre-trained predictor model using the entire dataset.
    (ii) Modify the data value evaluator model definition below to have the
    appropriate inductive bias (e.g. using convolutions layers for images,
    or attention layers text).

    References
    ----------
    .. [1] J. Yoon, Sercan O, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1909.11671.

    Parameters
    ----------
    x_dim : int
        Data covariates dimension, can be flatten dimension size
    y_dim : int
        Data labels dimension, can be flatten dimension size
    hidden_dim : int
        Hidden dimensions for the Value Estimator
    layer_number : int
        Number of hidden layers for the Value Estimator
    comb_dim : int
        After concat inputs how many layers, much less than `hidden_dim`, by default 10
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        layer_number: int,
        comb_dim: int,
        random_state: Optional[RandomState] = None,
    ):
        super().__init__()

        if random_state is not None:  # Can't pass generators to nn.Module layers
            torch.manual_seed(check_random_state(random_state).tomaxint())

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(x_dim + y_dim, hidden_dim)
        mlp_layers["input_acti"] = nn.ReLU()

        for i in range(int(layer_number - 3)):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = nn.ReLU()

        mlp_layers[f"{i+1}_out_lin"] = nn.Linear(hidden_dim, comb_dim)
        mlp_layers[f"{i+1}_out_acti"] = nn.ReLU()

        self.mlp = nn.Sequential(mlp_layers)

        yhat_combine = OrderedDict()

        # Combines with y_hat
        yhat_combine["reduce_lin"] = nn.Linear(comb_dim + y_dim, comb_dim)
        yhat_combine["reduce_acti"] = nn.ReLU()

        yhat_combine["out_lin"] = nn.Linear(comb_dim, 1)
        yhat_combine["out_acti"] = nn.Sigmoid()  # Sigmoid for binary selection
        self.yhat_comb = nn.Sequential(yhat_combine)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of inputs through value estimator for data values of input.

        Forward pass through Value Estimator. Returns selection probabilities.
        Concats the difference between labels and predicted labels to compute
        selection probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Data covariates
        y : torch.Tensor
            Data labels
        y_hat : torch.Tensor
            Data label predictions (from prediction model)

        Returns
        -------
        torch.Tensor
            Selection probabilities per covariate data point
        """
        # Flattens input dimension in case it is more than 2D
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        y_hat = y_hat.flatten(start_dim=1)

        out = torch.concat((x, y), dim=1)
        out = self.mlp(out)
        out = torch.cat((out, y_hat), dim=1)
        out = self.yhat_comb(out)
        return out


class DveLoss(nn.Module):
    """Compute Loss for Value Estimator.

    Custom loss function for the value estimator RL Model. Uses BCE Loss and
    checks average is within threshold to encourage exploration

    Parameters
    ----------
    threshold : float, optional
        Search rate threshold, the VE may get stuck in certain bounds close to
        :math:`[0, 1]`, thus outside of :math:`[1-threshold, threshold]` we encourage
        searching, by default 0.9
    exploration_weight : float, optional
        Large constant to encourage exploration in the Value Estimator, by default 1e3
    """

    def __init__(self, threshold: float = 0.9, exploration_weight: float = 1e3):
        super().__init__()
        self.threshold = threshold
        self.exploration_weight = exploration_weight

    def forward(
        self,
        pred_dataval: torch.Tensor,
        selector_input: torch.Tensor,
        reward_input: float,
    ) -> torch.Tensor:
        """Compute the loss for the Value Estimator.

        Uses REINFORCE Algorithm to compute a loss for the Value Estimator.
        `pred_dataval` is the data values. `selector_input` is a bernoulli random
        variable with `p=pred_dataval`. Computes a BCE between `pred_dataval` and
        `selector_input` and multiplies by the reward signal. Adds an additional loss
        if the Value Estimator is getting stuck outside the threshold.

        References
        ----------
        .. [1] R. J. Williams,
            Simple statistical gradient-following algorithms for connectionist
            reinforcement learning,
            Machine Learning, vol. 8, no. 3-4, pp. 229-256, May 1992,
            doi: https://doi.org/10.1007/bf00992696.


        Parameters
        ----------
        pred_dataval : torch.Tensor
            Predicted values from value estimator
        selector_input : torch.Tensor
            `1` for selected `0` for not selected, bernoulli random variable
        reward_input : float
            Reward/performance signal of prediction model trained on `selector_input`.
            If positive, indicates better than naive model of full sample.

        Returns
        -------
        torch.Tensor
            Computed loss tensor for Value Estimator
        """
        loss = F.binary_cross_entropy(pred_dataval, selector_input, reduction="sum")

        reward_loss = reward_input * loss
        search_loss = (  # Additional loss when VE is stuck outside threshold range
            F.relu(torch.mean(pred_dataval) - self.threshold)
            + F.relu((1 - self.threshold) - torch.mean(pred_dataval))
        )

        return reward_loss + (self.exploration_weight * search_loss)
