"""Experiments to test :py:class:`~dataoob.dataval.api.DataEvaluator`.

Experiments to pass into :py:meth:`~dataoob.evaluator.api.ExperimentMediator.evaluate`
and :py:meth:`~dataoob.evaluator.api.ExperimentMediator.plot` evaluate performance of
one :py:class:`~dataoob.dataval.api.DataEvaluator` at a a time.
"""
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch.utils.data import Subset

from dataoob.dataloader import DataLoader
from dataoob.dataval import DataEvaluator


def noisy_detection(evaluator: DataEvaluator, loader: DataLoader) -> dict[str, float]:
    """Evaluate ability to identify noisy indices.

    Compute recall and F1 score (of 2NN classifier) of the data evaluator
    on the noisy indices.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    loader : DataLoader
        DataLoader containing noisy indices

    Returns
    -------
    dict[str, float]
        'recall', 'kmeans_f1' (F1 score of 2-means classifier) of the DataEvaluator
        in detecting noisy indices
    """
    data_values = evaluator.evaluate_data_values()
    noisy_indices = loader.noisy_indices

    num_points = len(data_values)
    num_noisy = len(noisy_indices)

    sorted_indices = np.argsort(data_values)
    recall = len(np.intersect1d(sorted_indices[:num_noisy], noisy_indices)) / num_noisy

    # Computes F1 of a KMeans(k=2) classifier of the data values
    kmeans = KMeans(n_clusters=2, n_init="auto").fit(data_values.reshape(-1, 1))

    # Because of the convexity of KMeans classification, the least valuable data point
    # will always belong to the lower class on a number line, and vice-versa
    validation = np.full((num_points,), kmeans.labels_[sorted_indices[-1]])
    validation[noisy_indices] = kmeans.labels_[sorted_indices[0]]

    f1_kmeans_label = f1_score(kmeans.labels_, validation)

    return {"recall": recall, "kmeans_f1": f1_kmeans_label}


def point_addition(
    evaluator: DataEvaluator,
    loader: DataLoader,
    order: Literal["random", "ascending", "descending"] = "random",
    percentile: float = 0.05,
    plot: Axes = None,
    metric_name: str = "accuracy",
    train_kwargs: dict[str, Any] = None,
) -> dict[str, list[float]]:
    """Evaluate performance after adding points according to `order`.

    Repeatedly adds `percentile` of data points and trains/evaluates performance.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    loader : DataLoader
        DataLoader containing training and valid data points
    order : Literal["random", "ascending";, "descending";], optional
        Order which data points will be added, by default "random"
    percentile : float, optional
        Percentile of data points to add each iteration, by default 0.05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric_name : str, optional
        Name of DataEvaluator defined performance metric, by default assumed "accuracy"
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None

    Returns
    -------
    dict[str, list[float]]
        dict containing performance list after adding ``(i * percentile)`` data points
    """
    x_train, y_train, x_valid, y_valid = loader.datapoints
    data_values = evaluator.evaluate_data_values()
    curr_model = evaluator.pred_model.clone()

    num_sample = len(data_values)
    num_period = max(round(num_sample * percentile), 5)  # Add at least 5 per bin
    num_bins = int(num_sample // num_period)

    match order:
        case "ascending":
            sorted_value_list = np.argsort(data_values)
        case "descending":
            sorted_value_list = np.argsort(-data_values)
        case "random":
            sorted_value_list = np.random.permutation(num_sample)

    metric_list = []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for bin_index in range(0, num_sample, num_period):

        sorted_value_coalition = sorted_value_list[:bin_index]

        new_model = curr_model.clone()
        new_model.fit(
            Subset(x_train, sorted_value_coalition),
            Subset(y_train, sorted_value_coalition),
            **train_kwargs,
        )
        y_hat_valid = new_model.predict(x_valid)
        model_score = evaluator.evaluate(y_valid, y_hat_valid)

        metric_list.append(model_score)

    x_axis = [i * (1.0 / num_bins) for i in range(num_bins)]
    eval_results = {f"{order}_add_{metric_name}": metric_list, "axis": x_axis}

    if plot:
        plot.plot(x_axis, metric_list[:num_bins], "o-")

        plot.set_xlabel("Fraction Added")
        plot.set_ylabel(metric_name)
        plot.set_title(
            evaluator.plot_title
        )  # Figure out a better way to find instance variable

    return eval_results


def remove_high_low(
    evaluator: DataEvaluator,
    loader: DataLoader,
    percentile: float = 0.05,
    plot: Axes = None,
    metric_name: str = "accuracy",
    train_kwargs: dict[str, Any] = None,
) -> dict[str, list[float]]:
    """Evaluate performance after removing high/low points determined by data valuator.

    Repeatedly removes ``percentile`` of most valuable/least valuable data points
    and computes the performance of the metric.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    loader : DataLoader
        DataLoader containing training and valid data points
    percentile : float, optional
        Percentile of data points to add remove iteration, by default 0.05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric_name : str, optional
        Name of DataEvaluator defined performance metric, by default assumed "accuracy"
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None

    Returns
    -------
    dict[str, list[float]]
        dict containing list of the performance of the DataEvaluator
        ``(i * percentile)`` valuable/most valuable data points are removed
    """
    x_train, y_train, x_valid, y_valid = loader.datapoints
    data_values = evaluator.evaluate_data_values()
    curr_model = evaluator.pred_model.clone()

    num_sample = len(x_train)
    num_period = max(round(num_sample * percentile), 5)  # Add at least 5/bin
    num_bins = int(num_sample // num_period) + 1
    sorted_value_list = np.argsort(data_values)

    valuable_list, unvaluable_list = [], []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for bin_index in range(0, num_sample + num_period, num_period):

        # Removing least valuable samples first
        most_valuable_indices = sorted_value_list[bin_index:]

        # Fitting on valuable subset
        valuable_model = curr_model.clone()
        valuable_model.fit(
            Subset(x_train, most_valuable_indices),
            Subset(y_train, most_valuable_indices),
            **train_kwargs,
        )
        y_hat_valid = valuable_model.predict(x_valid)
        valuable_score = evaluator.evaluate(y_valid, y_hat_valid)
        valuable_list.append(valuable_score)

        # Removing most valuable samples first
        least_valuable_indices = sorted_value_list[: max(num_sample - bin_index, 0)]

        # Fitting on unvaluable subset
        unvaluable_model = curr_model.clone()
        unvaluable_model.fit(
            Subset(x_train, least_valuable_indices),
            Subset(y_train, least_valuable_indices),
            **train_kwargs,
        )
        iy_hat_valid = unvaluable_model.predict(x_valid)
        unvaluable_score = evaluator.evaluate(y_valid, iy_hat_valid)
        unvaluable_list.append(unvaluable_score)

    x_axis = [a * (1.0 / num_bins) for a in range(num_bins)]

    eval_results = {
        f"remove_mostval_{metric_name}": valuable_list,
        f"remove_leastval_{metric_name}": unvaluable_list,
        "axis": x_axis,
    }

    # Plot graphs
    if plot:
        # Prediction performances after removing high or low values
        plot.plot(x_axis, valuable_list[:num_bins], "o-")
        plot.plot(x_axis, unvaluable_list[:num_bins], "x-")

        plot.set_xlabel("Fraction Removed")
        plot.set_ylabel(metric_name)
        plot.legend(["Removing low value data", "Removing high value data"])

        plot.set_title(evaluator.plot_title)

    return eval_results


def discover_corrupted_sample(
    evaluator: DataEvaluator,
    loader: DataLoader,
    percentile: float = 0.05,
    plot: Axes = None,
) -> dict[str, list[float]]:
    """Evaluate discovery of noisy indices in low data value points.

    Repeatedly explores ``percentile`` of the data values and determines
    if within that total percentile, what proportion of the noisy indices are found.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    loader : DataLoader
        DataLoader containing noisy indices
    percentile : float, optional
        Percentile of data points to additionally search per iteration, by default .05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None, by default None

    Returns
    -------
    Dict[str, list[float]]
        dict containing list of the proportion of noisy indices found after exploring
        the ``(i * percentile)`` least valuable data points. If plot is not None,
        also returns optimal and random search performances as lists
    """
    x_train, *_ = loader.datapoints
    noisy_indices = loader.noisy_indices
    data_values = evaluator.evaluate_data_values()

    num_sample = len(x_train)
    num_period = max(round(num_sample * percentile), 5)  # Add at least 5 per bin
    num_bins = int(num_sample // num_period) + 1

    sorted_value_list = np.argsort(data_values, kind="stable")  # Order descending
    noise_rate = len(noisy_indices) / len(data_values)

    # Output initialization
    found_rates = []

    # For each bin
    for bin_index in range(0, num_sample + num_period, num_period):
        # from low to high data values
        found_rates.append(
            len(np.intersect1d(sorted_value_list[:bin_index], noisy_indices))
            / len(noisy_indices)
        )

    x_axis = [a * (1.0 / num_bins) for a in range(num_bins)]
    eval_results = {"corrupt_found": found_rates, "axis": x_axis}

    # Plot corrupted label discovery graphs
    if plot is not None:
        # Corrupted label discovery results (dvrl, optimal, random)
        y_dv = found_rates[:num_bins]
        y_opt = [min((a * (1.0 / num_bins / noise_rate), 1.0)) for a in range(num_bins)]
        y_random = x_axis

        eval_results["optimal"] = y_opt
        eval_results["random"] = y_random

        plot.plot(x_axis, y_dv, "o-")
        plot.plot(x_axis, y_opt, "--")
        plot.plot(x_axis, y_random, ":")
        plot.set_xlabel("Prop of data inspected")
        plot.set_ylabel("Prop of discovered corrupted samples")
        plot.legend(["Evaluator", "Optimal", "Random"])

        plot.set_title(evaluator.plot_title)

    # Returns True Positive Rate of corrupted label discovery
    return eval_results
