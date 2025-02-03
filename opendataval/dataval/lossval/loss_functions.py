"""Implements the loss functions that are needed for LossVal."""

import torch
from geomloss import SamplesLoss


def LossVal_mse(
    train_X: torch.Tensor,
    train_y_true: torch.Tensor,
    train_y_pred: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    sample_ids: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """LossVal for regression using mean squared error.
    Give the indices of the samples in the batch to the function!
    This is necessary to select the correct subset of the weights.

    :param train_X: training data
    :param train_y_true: true labels of the training data
    :param train_y_pred: predicted labels of the training data
    :param val_X: validation data
    :param val_y: true labels of the validation data
    :param sample_ids: indices of the samples that are used in this batch
    :param weights: a vector containing a weight for each instance
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical
        instability
    :return: the LossVal loss
    """
    weights = weights.index_select(
        0, sample_ids
    )  # Select the weights corresponding to the sample_ids

    # Step 1: Compute the weighted mse loss
    loss = torch.sum((train_y_true - train_y_pred) ** 2, dim=1)
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    # Step 2: Compute the Sinkhorn distance between the training and validation
    # distributions Dynamically chose a good value for the diameter parameter
    dist_matrix = torch.cdist(train_X, val_X, p=2)
    max_distance = dist_matrix.max().item()

    sinkhorn_distance = SamplesLoss(loss="sinkhorn", diameter=max_distance * 1.1)
    dist_loss = sinkhorn_distance(
        weights,
        train_X,
        torch.ones(val_X.shape[0], requires_grad=True).to(device),
        val_X,
    )

    # Step 3: Combine cross entropy and Sinkhorn distance
    return weighted_loss * dist_loss**2


def LossVal_cross_entropy(
    train_X: torch.Tensor,
    train_y_true: torch.Tensor,
    train_y_pred: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    sample_ids: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    epsilon_for_log=1e-8,
) -> torch.Tensor:
    """LossVal for classification using cross-entropy loss.
    Give the indices of the samples in the batch to the function!
    This is necessary to select the correct subset of the weights.

    :param train_X: training data
    :param train_y_true: true labels of the training data
    :param train_y_pred: predicted labels of the training data
    :param val_X: validation data
    :param val_y: true labels of the validation data
    :param sample_ids: indices of the samples that are used in this batch
    :param weights: a vector containing a weight for each instance
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical
        instability
    :return: the LossVal loss
    """
    weights = weights.index_select(
        0, sample_ids
    )  # Select the weights corresponding to the sample_ids

    # Step 1: Compute the weighted cross-entropy loss; targets are already
    # one-hot encoded!
    loss = -torch.sum(train_y_true * torch.log(train_y_pred + epsilon_for_log), dim=1)
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    # Step 2: Compute the Sinkhorn distance between the training and validation
    # distributions
    # Dynamically chose a good value for the diameter parameter
    dist_matrix = torch.cdist(train_X, val_X, p=2)
    max_distance = dist_matrix.max().item()

    sinkhorn_distance = SamplesLoss(loss="sinkhorn", diameter=max_distance * 1.1)
    dist_loss = sinkhorn_distance(
        weights,
        train_X,
        torch.ones(val_X.shape[0], requires_grad=True).to(device),
        val_X,
    )

    # Step 3: Combine cross entropy and Sinkhorn distance
    return weighted_loss * dist_loss**2
