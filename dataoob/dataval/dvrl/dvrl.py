import copy
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn import metrics

from dataoob.dataval import Evaluator, Model


class DVRL(Evaluator):
    """_summary_

        :param Model pred_model: _description_
        :param int x_dim: _description_
        :param int y_dim: _description_
        :param int hidden_dim: _description_
        :param int layer_number: _description_
        :param int comb_dim: _description_
        :param callable (torch.tensor -> torch.tensor) act_fn: _description_
        :param str checkpoint_file_name: _description_, defaults to "checkpoint.pt"
        """
    def __init__(
        self, pred_model: Model,
        x_dim: int,
        y_dim: int,
        # metric: callable[[torch.tensor, torch.tensor], float],
        hidden_dim: int,
        layer_number: int,
        comb_dim: int,
        act_fn: callable,
        checkpoint_file_name: str="checkpoint.pt"
    ):
        self.problem = "classification"
        self.metric = lambda y, yh: metrics.roc_auc_score(y.detach().cpu(), yh.detach()[:, 1].cpu())  # Find away to infer these two from the parent evaluator,

        self.pred_model = pred_model
        self.selection_predictor = DataValueEstimatorRL(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dim=hidden_dim,
            layer_number=layer_number,
            comb_dim=comb_dim,
            act_fn=act_fn,
        ).to(torch.device('mps'))
        self.checkpoint_file_name = checkpoint_file_name

    def input_data(self, x_train: torch.tensor, y_train: torch.tensor, x_valid: torch.tensor, y_valid: torch.tensor):
        """Stores and transforms input data

        :param torch.tensor x_train: x training set
        :param torch.tensor y_train: y training set
        :param torch.tensor x_valid: x validation set
        :param torch.tensor y_valid: y validation sets
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

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


    def train_baseline_models_for_dv(self, pre_train: bool=False, batch_size: int=32, epochs: int=1):
        """_summary_

        :param _type_ pre_train: _description_
        :param int batch_size: _description_
        :param int epochs: _description_
        """
        # With randomly initialized predictor
        if (not pre_train) and isinstance(self.pred_model, nn.Module):
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            self.pred_model.fit(
                self.x_train, self.y_train_onehot, batch_size=self.x_train.size(axis=0), epochs=0
            )
            # Saves initial randomization
            torch.save(self.pred_model.state_dict(), "tmp/pred_model.pt")
            # With pre-trained model, pre-trained model should be saved as
            # 'tmp/pred_model.pt'

        # Final model
        self.final_model = copy.deepcopy(self.pred_model)

        # Train baseline model with input data
        self.ori_model = copy.deepcopy(self.pred_model)
        if isinstance(self.ori_model, nn.Module):
            # Trains the model
            self.ori_model.load_state_dict(torch.load("tmp/pred_model.pt"))
            self.ori_model.fit(
                self.x_train,
                self.y_train_onehot,
                batch_size=batch_size,
                epochs=epochs,
                verbose=False,
            )
        else:
            self.ori_model.fit(self.x_train, self.y_train)

        # Trains validation model
        self.val_model = copy.deepcopy(self.ori_model)
        if isinstance(self.val_model, nn.Module):
            self.val_model.load_state_dict(torch.load("tmp/pred_model.pt"))
            self.val_model.fit(
                self.x_valid,
                self.y_valid_onehot,
                batch_size=batch_size,
                epochs=epochs,
                verbose=False,
            )
        else:
            self.val_model.fit(self.x_valid, self.y_valid)

        # Eval performance
        # Baseline performance
        if self.problem == "classification":
            y_valid_hat = self.ori_model.predict_proba(self.x_valid)
        elif self.problem == "regression":
            y_valid_hat = self.ori_model.predict(self.x_valid)

        self.valid_perf = self.metric(self.y_valid, y_valid_hat)

        # Comptue diff
        if self.problem == "classification":
            y_train_valid_pred = self.val_model.predict_proba(self.x_train)
            self.y_pred_diff = torch.abs(self.y_train_onehot - y_train_valid_pred)

        elif self.problem == "regression":
            y_train_valid_pred = self.val_model.predict(self.x_train)
            y_train_valid_pred = torch.reshape(y_train_valid_pred, [-1, 1])

            self.y_pred_diff = (
                torch.abs(self.y_train_onehot - y_train_valid_pred)
                / self.y_train_onehot
            )

    def train_data_value(self, pre_train_pred: bool=False, batch_size: int=32, epochs: int=1, pred_epochs: int=1, lr: float=0.01, threshold: float=0.9):
        """_summary_

        :param bool pre_train_pred: _description_, defaults to False
        :param int batch_size: _description_, defaults to 32
        :param int epochs: _description_, defaults to 1
        :param int pred_epochs: _description_, defaults to 1
        :param float lr: _description_, defaults to 0.01
        :param float threshold: _description_, defaults to 0.9
        :return _type_: _description_
        """
        batch_size = min(batch_size, self.x_train.size(axis=0))
        self.train_baseline_models_for_dv(
            pre_train_pred, batch_size=batch_size, epochs=pred_epochs
        )

        # Solver
        optimizer = torch.optim.Adam(
            self.selection_predictor.parameters(), lr=lr
        )
        criterion = DveLoss(threshold=threshold)

        for epoch in tqdm.tqdm(range(epochs)):
            indices = np.random.permutation(self.x_train.size(axis=0))[: batch_size]
            optimizer.zero_grad()

            # Set up batch
            x_batch = self.x_train[indices]
            y_batch_onehot = self.y_train_onehot[indices]
            y_batch = self.y_train[indices]
            y_hat_batch = self.y_pred_diff[indices]

            # Generates selection probability
            est_dv_curr = self.selection_predictor(x_batch, y_batch_onehot, y_hat_batch)

            # Samples the selection probability
            sel_prob_curr = torch.bernoulli(est_dv_curr)
            # Exception (When selection probability is 0)
            if torch.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * torch.ones(est_dv_curr.size())
                sel_prob_curr = torch.bernoulli(est_dv_curr)
            sel_prob_curr_weight = sel_prob_curr.detach()


            # Prediction and training
            new_model = copy.deepcopy(self.pred_model)

            if self.problem == "classification":
                new_model.load_state_dict(torch.load("tmp/pred_model.pt"))
                new_model.fit(
                    x_batch,
                    y_batch_onehot,
                    sample_weight=sel_prob_curr_weight,
                    batch_size=batch_size,
                    epochs=pred_epochs,
                )
                y_valid_hat = new_model.predict_proba(self.x_valid)

            elif self.problem == "regression":
                new_model.fit(x_batch, y_batch, sel_prob_curr_weight)
                y_valid_hat = new_model.predict(self.x_valid)


            # Reward computation
            dvrl_perf = self.metric(self.y_valid, y_valid_hat)

            if self.problem == "classification":
                reward_curr =  dvrl_perf - self.valid_perf
            elif self.problem == "regression":
                reward_curr = self.valid_perf - dvrl_perf

            # Trains the generator
            loss = criterion(torch.squeeze(est_dv_curr), torch.squeeze(sel_prob_curr), reward_curr)
            if epoch % 10 == 0:
                print(f"{dvrl_perf=}")
                print(f"{reward_curr=}")
                print(f"{torch.sum(sel_prob_curr)/len(sel_prob_curr)=}")
                print(f"{loss=}")
            loss.backward(retain_graph=True)
            optimizer.step()

        # Saves trained model
        torch.save(
            {
                "epoch": epoch,
                "rl_model_state_dict": self.selection_predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"tmp/{self.checkpoint_file_name}",
        )

        # Trains DVRL predictor
        # Generate data values
        final_data_value_weights = self.selection_predictor(
            self.x_train, self.y_train_onehot, self.y_pred_diff
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
                batch_size=batch_size,
                epochs=pred_epochs,
                verbose=False,
            )
        else:
            self.final_model.fit(self.x_train, self.y_train, final_data_value_weights)
        return final_data_value_weights

    def predict_data_value(self, x: torch.tensor, y: torch.tensor):
        """Returns data values using the data valuator model.

        :param torch.tensor x: x input to find data value
        :param torch.tensor y: y labels to find data value
        :return torch.tensor: Predicted data values/selection poportions for every index of inputs
        """
        # One-hot encoded labels
        if self.problem == "classification":
            y_onehot = F.one_hot(
                y.long(), num_classes=self.label_dim
            ).to(torch.float32)
            y_valid_pred = self.val_model.predict_proba(x)

        elif self.problem == "regression":
            y_onehot = torch.reshape(y, [len(y), 1])
            y_valid_pred = torch.reshape(self.val_model.predict(x), [-1, 1])

        # Generates y_train_hat
        if self.problem == "classification":
            y_train_hat = torch.abs(y_onehot - y_valid_pred)

        elif self.problem == "regression":
            y_train_hat = (
                torch.abs(y_onehot - y_valid_pred) / y_onehot
            )

        # Estimates data value
        est_data_value = self.data_value_evaluator()

        est_data_value.load_state_dict(
            torch.load(f"tmp/{self.checkpoint_file_name}")["rl_model_state_dict"]
        )

        final_data_value = est_data_value(x, y_valid_pred, y_train_hat)[:, 0]

        return final_data_value

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

        :param int x_dim: x variable dims
        :param int y_dim: y variable dims
        :param int hidden_dim: number of dims per hidden layer
        :param int layer_number: number of hidden layers
        :param int comb_dim: number of layers after combining with y_hat_pred
        :param callable (torch.tensor -> torch.tensor) act_fn: activation function between hidden layers
    """
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        layer_number: int,
        comb_dim: int,
        act_fn: callable,
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
        yhat_combine["out_acti"] = nn.Sigmoid()  # Sigmoid because binary selection
        self.yhat_comb = nn.Sequential(yhat_combine)

    def forward(self, x: torch.tensor, y_input: torch.tensor, y_hat_input: torch.tensor):
        """Forward pass through Dvrl

        :param torch.tensor x: _description_
        :param torch.tensor y_input: _description_
        :param torch.tensor y_hat_input: _description_
        :return _type_: _description_
        """
        x = torch.concat((x, y_input), axis=1)
        x = self.mlp(x)
        x = torch.cat((x, y_hat_input), axis=1)
        x = self.yhat_comb(x)
        return x

class DveLoss(nn.Module):
    """_summary_  TODO

        :param float threshold: Threshold selection porportion which encourages deeper searches,
        gradient might get stuck above `threshold` or below `1-threshhold`, defaults to .9
        :param _type_ epsilon: Added epsilon value to prevent log overflow, defaults to 1e-8
        :param float search_weight: value to multiply to improve search, TODO figure out proper naem, defaults to 1e3
    """
    def __init__(self, threshold: float=.9, epsilon=1e-8, search_weight: float=1e3):
        super(DveLoss, self).__init__()
        self.threshold = threshold
        self.epsilon = epsilon
        self.search_weight = search_weight


    def forward(self, predicted_data_val: torch.tensor, selector_input: torch.tensor,  reward_input: float):
        """_summary_

        :param torch.tensor predicted_data_val: _description_
        :param torch.tensor selector_input: _description_
        :param float reward_input: _description_
        """
        likelyhood = (  # TODO figure out what this actually is cause you've seen this before
            selector_input * torch.log(predicted_data_val + self.epsilon) +
            (1 - selector_input) * torch.log(1 - predicted_data_val + self.epsilon)
        )
        reward_loss = -reward_input * torch.sum(likelyhood)
        search_loss = self.search_weight = (
            F.relu(torch.mean(predicted_data_val) - self.threshold) +
            F.relu((1 - self.threshold) - torch.mean(predicted_data_val))
        )
        return reward_loss + self.search_weight * search_loss


