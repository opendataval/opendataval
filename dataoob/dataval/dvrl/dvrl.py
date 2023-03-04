import copy
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, RandomSampler

from dataoob.dataloader.util import CatDataset
from dataoob.dataval import DataEvaluator, Model


class DVRL(DataEvaluator):
    """Data valuation using reinforcement learning class, implemented with PyTorch
    Ref. https://arxiv.org/abs/1909.11671

    :param Model pred_model: Prediction model
    :param callable (torch.Tensor, torch.Tensor -> float) metric: Evaluation function
    to determine model performance
    :param int hidden_dim: Hidden dimensions for the RL Multilayer Perceptron
    (details in `DataValueEstimatorRL` class)
    :param int layer_number: Number of hidden layers for the Value Estimator (VE)
    :param int comb_dim: After combining the input in the VE how many layers,
    much less than `hidden_dim`
    :param callable (torch.Tensor -> torch.Tensor) act_fn: Activation function for VE
    :param int rl_epochs: Number of epochs for the VE, defaults to 1
    :param float lr: Learning rate for the VE, defaults to 0.01
    :param float threshold: Search rate threshold, the VE may get stuck in certain
    bounds close to [0., 1.] because it samples from a binomial, thus outside of
    [1-threshold, threshold] we encourage the VE to search, defaults to 0.9
    """

    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        hidden_dim: int = 100,
        layer_number: int = 5,
        comb_dim: int = 10,
        act_fn: callable = nn.ReLU(),
        rl_epochs: int = 1000,
        lr: float = 0.01,
        threshold: float = 0.9,
        device: torch.device = torch.device("cpu"),
    ):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        # MLP parameters
        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.comb_dim = comb_dim
        self.act_fn = act_fn
        self.device = device

        # Training parameters
        self.rl_epochs = rl_epochs
        self.lr = lr
        self.threshold = threshold

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for DVRL

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.value_estimator = DataValueEstimatorRL(
            x_dim=len(x_train[0]),  # Incase x_train is a dataset
            y_dim=y_train.size(dim=1),
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            act_fn=self.act_fn,
        ).to(self.device)

    def evaluate_baseline_models(self, batch_size: int = 32, epochs: int = 1):
        """Loads and trains baseline models. Baseline performance information
        is necessary to compute the reward.

        :param int batch_size: Baseline training batch size, defaults to 32
        :param int epochs: Number of epochs for baseline training, defaults to 1
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

        self.y_pred_diff = (
            torch.abs(self.y_train - y_train_valid_pred) /
            torch.sum(self.y_train, axis=1, keepdim=True)
        )

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Trains the DVRL model to assign probabilities of each data point
        being selected.

        :param int batch_size: pred_model training batch size, defaults to 32
        :param int epochs: Number of epochs for the pred_model, per training
        (this will equal rl_epochs * epochs), defaults to 1
        """
        batch_size = min(batch_size, len(self.x_train))
        self.evaluate_baseline_models(batch_size=batch_size, epochs=epochs)

        # Solver
        optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr)
        criterion = DveLoss(threshold=self.threshold)

        dataset = CatDataset(self.x_train, self.y_train, self.y_pred_diff)
        rs = RandomSampler(
            dataset, replacement=True, num_samples=(self.rl_epochs * batch_size)
        )

        for x_batch, y_batch, y_hat_batch in tqdm.tqdm(
            DataLoader(dataset, sampler=rs, batch_size=batch_size)
        ):
            optimizer.zero_grad()

            # Generates selection probability
            est_dv_curr = self.value_estimator(x_batch, y_batch, y_hat_batch)

            # Samples the selection probability
            sel_prob_curr = torch.bernoulli(est_dv_curr)
            # Exception (When selection probability is 0)
            if torch.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * torch.ones(est_dv_curr.size())
                sel_prob_curr = torch.bernoulli(est_dv_curr)
            sel_prob_curr_weight = sel_prob_curr.detach()

            # Prediction and training
            new_model = copy.deepcopy(self.pred_model)
            new_model.fit(
                x_batch,
                y_batch,
                sample_weight=sel_prob_curr_weight,
                batch_size=batch_size,
                epochs=epochs,
            )

            # Reward computation
            y_valid_hat = new_model.predict(self.x_valid)
            dvrl_perf = self.evaluate(self.y_valid, y_valid_hat)

            # NOTE must want to maximize the metric (IE for MSE use -MSE)
            reward_curr = dvrl_perf - self.valid_perf

            # Trains the generator
            loss = criterion(
                torch.squeeze(est_dv_curr),
                torch.squeeze(sel_prob_curr_weight),
                reward_curr,
            )
            loss.backward(retain_graph=True)
            optimizer.step()


        # Trains DVRL predictor
        # Generate data values
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
            verbose=False,
        )


    def evaluate_data_values(self) -> np.ndarray:
        """Returns data values using the data valuator model.

        :return torch.Tensor: Predicted data values/selection for every input data point
        """
        # One-hot encoded labels
        # Generates y_train_hat
        y_valid_pred = self.final_model.predict(self.x_train)
        y_hat = (
            torch.abs(self.y_train - y_valid_pred) /
            torch.sum(self.y_train, axis=1, keepdim=True)
        )

        # Estimates data value
        final_data_value = torch.squeeze(self.value_estimator(self.x_train, self.y_train, y_hat))

        return np.array(final_data_value.detach().cpu())


class DataValueEstimatorRL(nn.Module):
    """Returns data value evaluator model.
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
    appropriate inductive bias (e.g. using convolutional layers for images,
    or attention layers text).

    :param int x_dim: Data covariates dimension
    :param int y_dim: Data labels dimension
    :param int hidden_dim: number of dims per hidden layer
    :param int layer_number: number of hidden layers
    :param int comb_dim: number of layers after combining with y_hat_pred
    :param callable (torch.Tensor -> torch.Tensor) act_fn: activation function
    between hidden layers
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        layer_number: int,
        comb_dim: int,
        act_fn: callable = F.relu,
    ):
        super(DataValueEstimatorRL, self).__init__()

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

    def forward(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        """Forward pass through Dvrl

        :param torch.Tensor x: Data covariates
        :param torch.Tensor y: Data labels
        :param torch.Tensor y_hat: Data label predictions (from another model)
        :return torch.Tensor: Predicted selection probabilities per datapoint
        """
        x = torch.concat((x, y), axis=1)
        x = self.mlp(x)
        x = torch.cat((x, y_hat), axis=1)
        x = self.yhat_comb(x)
        return x


class DveLoss(nn.Module):
    """Custom loss function for the Genearive RL Model. Computes a BCE Loss
    with the binomial values as the target and multiplies by the reward. Encourages
    searching with a exploration loss.

    :param float threshold: Threshold porportion which encourages exploration,
    gradient might get stuck above `threshold` or below `1-threshold`, defaults to .9
    :param float exploration_weight: Weight used in loss computation for exploration.
    """

    def __init__(self, threshold: float = 0.9, exploration_weight: float = 1e3):
        super(DveLoss, self).__init__()
        self.threshold = threshold
        self.exploration_weight = exploration_weight

    def forward(
        self,
        predicted_data_val: torch.Tensor,
        selector_input: torch.Tensor,
        reward_input: float,
    ):
        """Computes the loss for the Value Estimator and takes in account the reward

        :param torch.Tensor predicted_data_val: Predicted values from value estimator
        :param torch.Tensor selector_input: `1` for selected `0` for not selected
        :param float reward_input: Reward/performance of model trained on
        `selector_input` sample weights. If positive, indicates better than naive model.
        """
        likelihood = F.binary_cross_entropy(
            predicted_data_val, selector_input, reduction="sum"
        )

        reward_loss = reward_input * likelihood
        search_loss = F.relu(torch.mean(predicted_data_val) - self.threshold) + F.relu(
            (1 - self.threshold) - torch.mean(predicted_data_val)
        )
        return reward_loss + (self.exploration_weight * search_loss)
