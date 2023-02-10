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



        self.problem = problem

        # Basic parameters
        self.data_dim = x_train.size(dim=1)
        self.label_dim = int(max(torch.max(y_train), torch.max(y_valid)) + 1)

        if self.problem == "classification":
            self.y_train_onehot = F.one_hot(
                y_train.long(), num_classes=self.label_dim
            ).to(torch.float32)
            self.y_valid_onehot = F.one_hot(
                y_valid.long(), num_classes=self.label_dim
            ).to(torch.float32)
        elif self.problem == "regression":
            self.y_train_onehot = torch.reshape(
                y_train.long(), [y_train.size(dim=0), 1]
            ).to(torch.float32)
            self.y_valid_onehot = torch.reshape(
                y_valid.long(), [y_valid.size(dim=0), 1]
            ).to(torch.float32)

        # Network parameters


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

        # Pred model (Note that any model architecture can be used as the predictor
        # model, either randomly initialized or pre-trained with the training data.
        # The condition for predictor model to have fit (e.g. using certain number
        # of back-propagation iterations) and predict functions as its subfunctions.
        self.pred_model = pred_model





    @lru_cache(None)
    def data_value_evaluator(self) -> DataValueEstimatorRL:

        return DataValueEstimatorRL(
            x_dim=self.data_dim,
            y_dim=self.label_dim,
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            act_fn=self.act_fn,
        ).to(torch.device("mps"))

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
        if self.problem == "classification":
            y_valid_hat = self.ori_model.predict_proba(self.x_valid)
        elif self.problem == "regression":
            y_valid_hat = self.ori_model.predict(self.x_valid)


        valid_perf = metric(self.y_valid, y_valid_hat)

        # Comptue diff
        if self.problem == "classification":
            y_train_valid_pred = self.val_model.predict_proba(self.x_train)
            y_pred_diff = torch.abs(self.y_train_onehot - y_train_valid_pred)

        elif self.problem == "regression":
            y_train_valid_pred = self.val_model.predict(self.x_train)
            y_train_valid_pred = torch.reshape(y_train_valid_pred, [-1, 1])

            y_pred_diff = (
                torch.abs(self.y_train_onehot - y_train_valid_pred)
                / self.y_train_onehot
            )

        for epoch in tqdm.tqdm(range(self.outer_iterations)):
            indices = np.random.permutation(self.x_train.size(axis=0))[
                : self.batch_size
            ]
            dve_solver.zero_grad()

            # Set up batch
            x_batch = self.x_train[indices]
            y_batch_onehot = self.y_train_onehot[indices]
            y_batch = self.y_train[indices]
            y_hat_batch = y_pred_diff[indices]


            # Generates selection probability
            est_dv_curr = est_data_value(x_batch, y_batch_onehot, y_hat_batch)

            # Samples the selection probability
            sel_prob_curr = torch.bernoulli(est_dv_curr)
            # Exception (When selection probability is 0)
            if torch.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * torch.ones(est_dv_curr.size())
                sel_prob_curr = torch.bernoulli(est_dv_curr)
            sel_prob_curr = sel_prob_curr.detach()


            # Prediction and training
            new_model = copy.deepcopy(self.pred_model)

            if self.problem == "classification":
                new_model.load_state_dict(torch.load("tmp/pred_model.pt"))
                new_model.fit(
                    x_batch,
                    y_batch_onehot,
                    sample_weight=sel_prob_curr,
                    batch_size=self.batch_size_predictor,
                    epochs=self.inner_iterations,
                )
                y_valid_hat = new_model.predict_proba(self.x_valid)
            elif self.problem == "regression":
                new_model.fit(x_batch, y_batch, sel_prob_curr)
                y_valid_hat = new_model.predict(self.x_valid)


            # Reward computation
            dvrl_perf = metric(self.y_valid, y_valid_hat)

            if self.problem == "classification":
                reward_curr =  dvrl_perf - valid_perf
            elif self.problem == "regression":
                reward_curr = valid_perf - dvrl_perf

            # Trains the generator
            loss = DveLoss(torch.squeeze(est_dv_curr), reward_curr, torch.squeeze(sel_prob_curr[:, 0]))
            if epoch % 10 == 0:
                print(f"{dvrl_perf=}")
                print(f"{valid_perf=}")
                print(f"{reward_curr=}")
                print(f"{torch.sum(sel_prob_curr)/len(sel_prob_curr)=}")
                print(f"{loss=}")
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
            f"tmp/{self.checkpoint_file_name}",
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
        return final_data_value_weights

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
                y_train.long(), num_classes=self.label_dim
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
            torch.load(f"tmp/{self.checkpoint_file_name}")["rl_model_state_dict"]
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


def DveLoss(est_data_value, reward_input, s_input, epsilon=1e-8, threshhold=0.90):
    """Generator Loss"""
    prob = torch.sum(
        s_input * torch.log(est_data_value + epsilon) +
        (1 - s_input) * torch.log(1 - est_data_value + epsilon)
    )

    return (
        (-reward_input * prob) +
        1e3 * (
            F.relu(torch.mean(est_data_value) - threshhold) +
            F.relu((1 - threshhold) - torch.mean(est_data_value))
        )
    )

