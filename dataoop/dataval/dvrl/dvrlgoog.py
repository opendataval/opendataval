# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data Valuation using Reinforcement Learning (DVRL). Adapted for PyTorch"""

import copy
import os
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn import metrics


class DataValueEstimatorRL(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        layer_number: int,
        comb_dim: int,
        act_fn,
    ):
        super().__init__()

        mlp_layers = OrderedDict()

        mlp_layers["input"] = nn.Linear(
            x_dim + y_dim, hidden_dim
        )  # Option for code deduplication
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
        yhat_combine["out_acti"] = nn.Sigmoid()
        self.yhat_comb = nn.Sequential(yhat_combine)

    def forward(self, x, y_input, y_hat_input):
        x = torch.concat((x, y_input), axis=1)
        x = self.mlp(x)
        x = torch.cat((x, y_hat_input), axis=1)
        x = self.yhat_comb(x)
        return x


class Dvrl(object):
    """Data Valuation using Reinforcement Learning (DVRL) class.
    Attributes:
      x_train: training feature
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      problem: 'regression' or 'classification'
      pred_model: predictive model (object)
      parameters: network parameters such as hidden_dim, iterations,
                  activation function, layer_number, learning rate
      checkpoint_file_name: File name for saving and loading the trained model
      flags: flag for training with stochastic gradient descent (flag_sgd)
             and flag for using pre-trained model (flag_pretrain)
    """

    def __init__(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        problem,
        pred_model,
        parameters,
        checkpoint_file_name,
        flags,
    ):
        """Initializes DVRL."""

        # Inputs
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.problem = problem

        self.num_labels = int(max(torch.max(y_train), torch.max(y_valid)) + 1)

        if self.problem == "classification":
            self.y_train_onehot = F.one_hot(
                y_train.to(torch.int64), num_classes=self.num_labels
            ).to(torch.float32)
            self.y_valid_onehot = F.one_hot(
                y_valid.to(torch.int64), num_classes=self.num_labels
            ).to(torch.float32)
        elif self.problem == "regression":
            self.y_train_onehot = torch.reshape(
                y_train.to(torch.int64), [y_train.size(dim=0), 1]
            ).to(torch.float32)
            self.y_valid_onehot = torch.reshape(
                y_valid.to(torch.int64), [y_valid.size(dim=0), 1]
            ).to(torch.float32)

        # Network parameters
        self.hidden_dim = parameters["hidden_dim"]
        self.comb_dim = parameters["comb_dim"]
        self.outer_iterations = parameters["iterations"]
        self.act_fn = parameters["activation"]
        self.layer_number = parameters["layer_number"]
        self.batch_size = np.min([parameters["batch_size"], x_train.size(axis=0)])
        self.learning_rate = parameters["learning_rate"]

        # Basic parameters
        self.epsilon = 1e-8  # Adds to the log to avoid overflow
        self.threshold = 0.9  # Encourages exploration

        # Flags
        self.flag_sgd = flags["sgd"]
        self.flag_pretrain = flags["pretrain"]

        # If the pred_model uses stochastic gradient descent (SGD) for training
        if self.flag_sgd:
            self.inner_iterations = parameters["inner_iterations"]
            self.batch_size_predictor = np.min(
                [parameters["batch_size_predictor"], x_valid.size(dim=0)]
            )

        # Checkpoint file name
        self.checkpoint_file_name = checkpoint_file_name

        # Basic parameters
        self.data_dim = x_train.size(dim=1)
        self.label_dim = self.num_labels

        # Pred model (Note that any model architecture can be used as the predictor
        # model, either randomly initialized or pre-trained with the training data.
        # The condition for predictor model to have fit (e.g. using certain number
        # of back-propagation iterations) and predict functions as its subfunctions.
        self.pred_model = pred_model

        # Final model
        self.final_model = pred_model

        # With randomly initialized predictor
        if (not self.flag_pretrain) and self.flag_sgd:
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            pred_model.fit(
                self.x_train, self.y_train_onehot, batch_size=self.x_train.size(axis=0), epochs=0
            )
            # Saves initial randomization
            torch.save(pred_model.state_dict(), "tmp/pred_model.pt")
            # With pre-trained model, pre-trained model should be saved as
            # 'tmp/pred_model.h5'

        # Baseline model
        self.ori_model = copy.copy(self.pred_model)
        self.ori_model.load_state_dict(torch.load("tmp/pred_model.pt"))
        if self.flag_sgd:
            # Trains the model
            self.ori_model.fit(
                self.x_train,
                self.y_train_onehot,
                batch_size=self.batch_size_predictor,
                epochs=self.inner_iterations,
                verbose=False,
            )

        else:
            self.ori_model.fit(x_train, y_train)

        # Valid baseline model
        self.val_model = copy.copy(self.pred_model)
        self.val_model.load_state_dict(torch.load("tmp/pred_model.pt"))
        if isinstance(self.pred_model, nn.Module):
            self.val_model.fit(
                x_valid,
                self.y_valid_onehot,
                batch_size=self.batch_size_predictor,
                epochs=self.inner_iterations,
                verbose=False,
            )
        else:
            self.val_model.fit(x_valid, y_valid)

    @lru_cache(None)
    def data_value_evaluator(self) -> DataValueEstimatorRL:
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
        Returns:
          dve: data value estimations
        """
        return DataValueEstimatorRL(
            x_dim=self.data_dim,
            y_dim=self.label_dim,
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            act_fn=self.act_fn,
        )

    def train_dvrl(self, metric):
        """Trains DVRL based on the specified objective function.
        Args:
          metric: performance metric function to evaluate how well the model is doing
        """

        # Generates selected probability
        est_data_value = self.data_value_evaluator()

        # Solver
        dve_solver = torch.optim.Adam(
            est_data_value.parameters(), lr=self.learning_rate
        )

        # Baseline performance
        if self.flag_sgd:
            y_valid_hat = self.ori_model.predict(self.x_valid)
        else:
            if self.problem == "classification":
                y_valid_hat = self.ori_model.predict_proba(self.x_valid)
            elif self.problem == "regression":
                y_valid_hat = self.ori_model.predict(self.x_valid)

        valid_perf = metric(self.y_valid, y_valid_hat)

        # Prediction differences
        # if self.flag_sgd:
        #   y_train_valid_pred = self.val_model.predict(self.x_train)
        # else:
        if self.problem == "classification":
            y_train_valid_pred = self.val_model.predict_proba(self.x_train)
        elif self.problem == "regression":
            y_train_valid_pred = self.val_model.predict(self.x_train)
            y_train_valid_pred = torch.reshape(y_train_valid_pred, [-1, 1])

        if self.problem == "classification":

            y_pred_diff = torch.abs(self.y_train_onehot - y_train_valid_pred)
        elif self.problem == "regression":
            y_pred_diff = (
                torch.abs(self.y_train_onehot - y_train_valid_pred)
                / self.y_train_onehot
            )

        for epoch in tqdm.tqdm(range(self.outer_iterations)):
            batch_idx = np.random.permutation(self.x_train.size(axis=0))[
                : self.batch_size
            ]

            x_batch = self.x_train[batch_idx, :]
            y_batch_onehot = self.y_train_onehot[batch_idx]
            y_batch = self.y_train[batch_idx]
            y_hat_batch = y_pred_diff[batch_idx]

            dve_solver.zero_grad()
            # Generates selection probability
            est_dv_curr = est_data_value(x_batch, y_batch_onehot, y_hat_batch)

            # Samples the selection probability
            sel_prob_curr = torch.bernoulli(est_dv_curr)

            # Exception (When selection probability is 0)
            if torch.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * torch.ones(est_dv_curr.size())
                sel_prob_curr = torch.bernoulli(est_dv_curr)

            # Trains predictor
            # If the predictor is neural network
            new_model = copy.copy(self.pred_model)
            new_model.load_state_dict(torch.load("tmp/pred_model.pt"))
            if isinstance(self.pred_model, nn.Module):
                # Train the model

                new_model.fit(
                    x_batch,
                    y_batch_onehot,
                    sample_weight=sel_prob_curr[:, 0],
                    batch_size=self.batch_size_predictor,
                    epochs=self.inner_iterations,
                )

            else:
                new_model.fit(x_batch, y_batch, sel_prob_curr[:, 0])

            # Prediction
            if self.flag_sgd:
                y_valid_hat = new_model.predict(self.x_valid)
            else:
                if self.problem == "classification":
                    y_valid_hat = new_model.predict_proba(self.x_valid)
                elif self.problem == "regression":
                    y_valid_hat = new_model.predict(self.x_valid)

            # Reward computation
            dvrl_perf = metric(self.y_valid, y_valid_hat)

            if self.problem == "classification":
                reward_curr = dvrl_perf - valid_perf
            elif self.problem == "regression":
                reward_curr = valid_perf - dvrl_perf

            # Trains the generator
            loss = dve_loss(est_dv_curr, reward_curr, sel_prob_curr)
            loss.backward(retain_graph=True)
            dve_solver.step()

        # Saves trained model
        torch.save(
            {
                "epoch": epoch,
                "rl_model_state_dict": est_data_value.state_dict(),
                "optimizer_state_dict": dve_solver.state_dict(),
                "loss": loss,
            },
            self.checkpoint_file_name,
        )

        # Trains DVRL predictor
        # Generate data values
        final_data_value_weights = est_data_value(
            self.x_train, self.y_train_onehot, y_pred_diff
        ).detach()


        # Trains final model
        # If the final model is neural network
        if isinstance(self.final_model, nn.Module):
            self.final_model.load_state_dict(torch.load("tmp/pred_model.pt"))
            # Train the model
            self.final_model.fit(
                self.x_train,
                self.y_train_onehot,
                sample_weight=final_data_value_weights,
                batch_size=self.batch_size_predictor,
                epochs=self.inner_iterations,
                verbose=False,
            )
        else:
            self.final_model.fit(self.x_train, self.y_train, final_data_value_weights)

    def data_valuator(self, x_train, y_train):
        """Returns data values using the data valuator model.
        Args:
          x_train: training features
          y_train: training labels
        Returns:
          final_dat_value: final data values of the training samples
        """

        # One-hot encoded labels
        if self.problem == "classification":
            y_train_onehot = F.one_hot(
                y_train.to(torch.int64), num_classes=self.num_labels
            ).to(torch.float32)
            y_train_valid_pred = self.val_model.predict_proba(x_train)
        elif self.problem == "regression":
            y_train_onehot = np.reshape(y_train, [len(y_train), 1])
            y_train_valid_pred = np.reshape(self.val_model.predict(x_train), [-1, 1])

        # Generates y_train_hat
        if self.problem == "classification":
            y_train_hat = torch.abs(y_train_onehot - y_train_valid_pred)
        elif self.problem == "regression":
            y_train_hat = (
                torch.abs(y_train_onehot - y_train_valid_pred) / y_train_onehot
            )

        # Estimates data value
        est_data_value = self.data_value_evaluator()
        est_data_value.load_state_dict(
            torch.load(self.checkpoint_file_name)["rl_model_state_dict"]
        )

        final_data_value = est_data_value(x_train, y_train_onehot, y_train_hat)[:, 0]

        return final_data_value

    def dvrl_predictor(self, x_test):
        """Returns predictions using the predictor model.
        Args:
          x_test: testing features
        Returns:
          y_test_hat: predictions of the predictive model with DVRL
        """

        if self.problem == "classification":
            y_test_hat = self.final_model.predict_proba(x_test)
        elif self.problem == "regression":
            y_test_hat = self.final_model.predict(x_test)

        return y_test_hat


def dve_loss(est_data_value, reward_input, sel_prob_curr, epsilon=1e-8, threshhold=0.9):
    prob = torch.sum(
        sel_prob_curr * torch.log(est_data_value + epsilon)
        + (1 - sel_prob_curr) * torch.log(1 - est_data_value + epsilon)
    )
    return (
        (-reward_input * prob)
        + 1e3 * (max(torch.mean(est_data_value) - threshhold, 0))
        + max((1 - threshhold) - torch.mean(est_data_value), 0)
    )
