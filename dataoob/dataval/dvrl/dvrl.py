import copy
from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, RandomSampler

from dataoob.dataloader.util import CatDataset
from dataoob.dataval import DataEvaluator


class DVRL(DataEvaluator):
    """Data valuation using reinforcement learning class, implemented with PyTorch

    References
    ----------
    .. [1] J. Yoon, Arik, Sercan O, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. [Online]. Available: https://arxiv.org/abs/1909.11671.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimensions for the RL Multilayer Perceptron Value Estimator (VE)
        (details in :py:class:`DataValueEstimatorRL` class), by default 100
    layer_number : int, optional
        Number of hidden layers for the Value Estimator (VE), by default 5
    comb_dim : int, optional
        After concat inputs how many layers, much less than `hidden_dim`, by default 10
    act_fn : Callable[[torch.Tensor], torch.Tensor], optional
        Activation function for VE, by default nn.ReLU()
    rl_epochs : int, optional
        Number of training epochs for the VE, by default 1000
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
        act_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
        rl_epochs: int = 1000,
        lr: float = 0.01,
        threshold: float = 0.9,
        device: torch.device = torch.device("cpu"),
        random_state: RandomState = None,
    ):
        # Value estimator parameters
        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.comb_dim = comb_dim
        self.act_fn = act_fn
        self.device = device

        # Training parameters
        self.rl_epochs = rl_epochs
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
        """Stores and transforms input data for DVRL

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

        self.value_estimator = DataValueEstimatorRL(
            x_dim=len(x_train[0]),  # In case x_train is a data set
            y_dim=y_train.size(dim=1),
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            act_fn=self.act_fn,
            random_state=self.random_state,
        ).to(self.device)

        return self

    def _evaluate_baseline_models(self, batch_size: int = 32, epochs: int = 1):
        """Loads and trains baseline models. Baseline performance information is
        necessary to compute the reward.

        Parameters
        ----------
        batch_size : int, optional
            Baseline training batch size, by default 32
        epochs : int, optional
            Number of epochs for baseline training, by default 1
        """
        # Final model
        self.final_model = copy.deepcopy(self.pred_model)

        # Train baseline model with input data
        self.ori_model = copy.deepcopy(self.pred_model)
        self.ori_model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
        )

        # Trains validation model
        self.val_model = copy.deepcopy(self.ori_model)
        self.val_model.fit(
            self.x_valid,
            self.y_valid,
            batch_size=batch_size,
            epochs=epochs,
        )

        # Eval performance
        # Baseline performance
        y_valid_hat = self.ori_model.predict(self.x_valid)
        self.valid_perf = self.evaluate(self.y_valid, y_valid_hat)

        # Compute diff
        y_train_valid_pred = self.val_model.predict(self.x_train)

        self.y_pred_diff = (  # Predicted differences as input to value estimator
            torch.abs(self.y_train - y_train_valid_pred)
            / torch.sum(self.y_train, axis=1, keepdim=True)
        )

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Trains the VE to assign probabilities of each data point being selected
        using a signal from the evaluation performance.

        Parameters
        ----------
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs (total = rl_epochs * epochs), by default 1
        """
        batch_size = min(batch_size, len(self.x_train))
        self._evaluate_baseline_models(batch_size=batch_size, epochs=epochs)

        # Solver
        optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr)
        criterion = DveLoss(threshold=self.threshold)

        dataset = CatDataset(self.x_train, self.y_train, self.y_pred_diff)
        gen = torch.Generator(self.device).manual_seed(self.random_state.tomaxint())
        rs = RandomSampler(
            dataset,
            replacement=True,
            num_samples=self.rl_epochs * batch_size,
            generator=gen,
        )

        for x_batch, y_batch, y_hat_batch in tqdm.tqdm(
            DataLoader(dataset, sampler=rs, batch_size=batch_size, generator=gen)
        ):
            optimizer.zero_grad()

            # Generates selection probability
            pred_dataval = self.value_estimator(x_batch, y_batch, y_hat_batch)

            # Samples the selection probability
            sel_prob_weight = torch.bernoulli(pred_dataval, generator=gen)
            # Exception (When selection probability is 0)
            if torch.sum(sel_prob_weight) == 0:
                pred_dataval = 0.5 * torch.ones(pred_dataval.size())
                sel_prob_weight = torch.bernoulli(pred_dataval, generator=gen)
            sel_prob_weight = sel_prob_weight.detach()

            # Prediction and training
            new_model = copy.deepcopy(self.pred_model)
            new_model.fit(
                x_batch,
                y_batch,
                sample_weight=sel_prob_weight,
                batch_size=batch_size,
                epochs=epochs,
            )

            # Reward computation
            y_valid_hat = new_model.predict(self.x_valid)
            dvrl_perf = self.evaluate(self.y_valid, y_valid_hat)
            # NOTE must want to maximize the metric (IE for MSE use -MSE)
            reward_curr = dvrl_perf - self.valid_perf

            # Trains the VE
            loss = criterion(
                pred_dataval.squeeze(), sel_prob_weight.squeeze(), reward_curr
            )
            loss.backward(retain_graph=True)
            optimizer.step()

        final_data_value_weights = self.value_estimator(
            self.x_train, self.y_train, self.y_pred_diff
        ).detach()
        # Trains final model
        self.final_model.fit(
            self.x_train,
            self.y_train,
            sample_weight=final_data_value_weights,
            batch_size=batch_size,
            epochs=epochs,
        )
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values using the Value Estimator

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        y_valid_pred = self.final_model.predict(self.x_train)
        y_hat = (  # Computes the diff from predicted as input to value estimator
            torch.abs(self.y_train - y_valid_pred)
            / torch.sum(self.y_train, axis=1, keepdim=True)
        )

        # Estimates data value
        final_data_value = self.value_estimator(self.x_train, self.y_train, y_hat)

        return np.array(torch.squeeze(final_data_value).detach().cpu())


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
    .. [1] J. Yoon, Arik, Sercan O, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. [Online]. Available: https://arxiv.org/abs/1909.11671.

    Parameters
    ----------
    x_dim : int
        Data covariates dimension
    y_dim : int
        Data labels dimension
    hidden_dim : int
        Hidden dimensions for the Value Estimator
    layer_number : int
        Number of hidden layers for the Value Estimator
    comb_dim : int
        After concat inputs how many layers, much less than `hidden_dim`, by default 10
    act_fn : Callable, optional
        Activation function for VE, by default nn.ReLU(), by default nn.ReLU()
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
        act_fn: Callable = nn.ReLU(),
        random_state: RandomState = None,
    ):
        super(DataValueEstimatorRL, self).__init__()

        if random_state is not None:
            torch.manual_seed(check_random_state(random_state).tomaxint())

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(x_dim + y_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layer_number - 3)):
            mlp_layers[f"{i+1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i+1}_acti"] = act_fn

        mlp_layers[f"{i+1}_out_lin"] = nn.Linear(hidden_dim, comb_dim)
        mlp_layers[f"{i+1}_out_acti"] = act_fn

        self.mlp = nn.Sequential(mlp_layers)

        yhat_combine = OrderedDict()

        # Combines with y_hat
        yhat_combine["reduce_lin"] = nn.Linear(comb_dim + y_dim, comb_dim)
        yhat_combine["reduce_acti"] = act_fn

        yhat_combine["out_lin"] = nn.Linear(comb_dim, 1)
        yhat_combine["out_acti"] = nn.Sigmoid()  # Sigmoid for binary selection
        self.yhat_comb = nn.Sequential(yhat_combine)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through Value Estimator. Returns selection probabilities.
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
        x = torch.concat((x, y), axis=1)
        x = self.mlp(x)
        x = torch.cat((x, y_hat), axis=1)
        x = self.yhat_comb(x)
        return x


class DveLoss(nn.Module):
    """Custom loss function for the value estimator RL Model. uses BCE Loss and
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
        super(DveLoss, self).__init__()
        self.threshold = threshold
        self.exploration_weight = exploration_weight

    def forward(
        self,
        pred_dataval: torch.Tensor,
        selector_input: torch.Tensor,
        reward_input: float,
    ):
        """Computes the loss for the Value Estimator, uses the reward signal from the
        prediction model, BCE loss, and whether Value Estimator is getting stuck
        outside of the threshold bounds

        Parameters
        ----------
        pred_dataval : torch.Tensor
            Predicted values from value estimator
        selector_input : torch.Tensor
            `1` for selected `0` for not selected
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
