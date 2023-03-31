import copy  # TODO return a dict and inside this dict have an option for plot

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch.utils.data import Subset

from dataoob.dataval import DataEvaluator


def noisy_detection(
    evaluator: DataEvaluator, noisy_index: np.ndarray
) -> tuple[float, float]:
    """Computes recall and F1 score (of 2NN classifier) of the data evaluator
    on the noisy indices

    Parameters
    ----------
    evaluator : DataEvaluator
        Data Evaluator
    noisy_index : np.ndarray
        Indices with noise added to them from DataLoader.

    Returns
    -------
    tuple[float, float]
        Recall, F1 Score (Kmeans) for data evaluator
    """
    data_values = evaluator.evaluate_data_values()

    num_points = len(data_values)
    num_noisy = len(noisy_index)

    sorted_indices = np.argsort(data_values)
    recall = len(np.intersect1d(sorted_indices[:num_noisy], noisy_index)) / num_noisy

    # Computes F1 of a KMeans(k=2) classifier of the data values
    kmeans = KMeans(n_clusters=2).fit(data_values.reshape(-1, 1))

    # Because of the convexity of KMeans classification, the least valuable data point
    # will always belong to the lower class on a number line, and vice-versa
    validation = np.full((num_points,), kmeans.labels_[sorted_indices[-1]])
    validation[noisy_index] = kmeans.labels_[sorted_indices[0]]

    f1_kmeans_label = f1_score(kmeans.labels_, validation)

    return recall, f1_kmeans_label


def point_removal(  # TODO consider just passing in the x_values
    evaluator: DataEvaluator,
    order: str = "random",
    percentile_increment: float = 0.05,
    batch_size: int = 32,
    epochs: int = 1,
    plot: bool = True,
    metric_name: str = "Accuracy",
) -> list[float]:
    """Repeatedly add `percentile_increment` of data points and evaluates performance

    Parameters
    ----------
    evaluator : DataEvaluator
        Data Evaluator
    order : str, optional
        Order which data points will be added, must be 'ascending',
        'descending', otherwise defaults to 'random', by default "random"
    percentile_increment : float, optional
        Percentile of data points to add each iteration, by default .05
    batch_size : int, optional
        Training batch size, by default 32
    epochs : int, optional
        Number of training epochs, by default 1
    plot : bool, optional
        Whether to plot the data, by default True
    metric_name : str, optional
        Name of the passed in performance metric, by default "Accuracy"

    Returns
    -------
    list[float]
        List of the performance metric at each interval
    """
    (x_train, y_train), (x_valid, y_valid) = evaluator.get_data_points()
    data_values = evaluator.evaluate_data_values()
    curr_model = copy.deepcopy(evaluator.pred_model)

    num_sample = len(data_values)
    num_period = max(round(num_sample * percentile_increment), 5)  # Add at least 5/bin
    num_bins = int(num_sample // num_period)

    if order == "ascending":
        sorted_value_list = np.argsort(data_values)
    elif order == "descending":
        sorted_value_list = np.argsort(-data_values)
    else:
        sorted_value_list = np.random.permutation(num_sample)

    metric_list = []

    for bin_index in range(0, num_sample, num_period):

        sorted_value_coalition = sorted_value_list[:bin_index]

        new_model = copy.deepcopy(curr_model)
        new_model.fit(
            Subset(x_train, sorted_value_coalition),
            Subset(y_train, sorted_value_coalition),
            batch_size=batch_size,
            epochs=epochs,
        )
        y_hat_valid = new_model.predict(x_valid)
        model_score = evaluator.evaluate(y_valid, y_hat_valid)

        metric_list.append(model_score)

    if plot:
        x_axis = [a * (1.0 / num_bins) for a in range(num_bins)]
        plt.figure(figsize=(6, 7.5))
        plt.plot(x_axis, metric_list[:num_bins], "o-")

        plt.xlabel("Fraction of Removed Samples", size=16)
        plt.ylabel(metric_name, size=16)
        plt.title(f"Removing value {order}", size=16)
        plt.show()

    return metric_list


def remove_high_low(
    evaluator: DataEvaluator,
    percentile_increment: float = 0.05,
    batch_size: int = 32,
    epochs: int = 1,
    plot: bool = True,
    metric_name: str = "Accuracy",
) -> tuple[list[float], list[float]]:
    """Repeatedly removes ``percentile_increment`` of most valuable/least valuable data
    points and computes the change in the measurement metric as a result

    Parameters
    ----------
    evaluator : DataEvaluator
        Data Evaluator
    percentile_increment : float, optional
        Percentile of data points to add each iteration, by default .05
    batch_size : int, optional
        Training batch size, by default 32
    epochs : int, optional
        Number of training epochs, by default 1
    plot : bool, optional
        Whether to plot the data, by default True
    metric_name : str, optional
        Name of the passed in performance metric, by default "Accuracy"

    Returns
    -------
    tuple[list[float], list[float]]
        List of the performance metric the Data Evaluator
        for each bin when the least valuable/most valuable are incrementally removed.
    """
    (x_train, y_train), (x_valid, y_valid) = evaluator.get_data_points()
    data_values = evaluator.evaluate_data_values()
    curr_model = copy.deepcopy(evaluator.pred_model)

    num_sample = len(x_train)
    num_period = max(round(num_sample * percentile_increment), 5)  # Add at least 5/bin
    num_bins = int(num_sample // num_period) + 1
    sorted_value_list = np.argsort(data_values)

    valuable_list, unvaluable_list = [], []

    for bin_index in range(0, num_sample + num_period, num_period):

        # Removing least valuable samples first
        most_valuable_indices = sorted_value_list[bin_index:]

        # Fitting on valuable subset
        valuable_model = copy.deepcopy(curr_model)
        valuable_model.fit(
            Subset(x_train, most_valuable_indices),
            Subset(y_train, most_valuable_indices),
            batch_size=batch_size,
            epochs=epochs,
        )
        y_hat_valid = valuable_model.predict(x_valid)
        valuable_score = evaluator.evaluate(y_valid, y_hat_valid)
        valuable_list.append(valuable_score)

        # Removing most valuable samples first
        least_valuable_indices = sorted_value_list[: max(num_sample - bin_index, 0)]

        # Fitting on unvaluable subset
        unvaluable_model = copy.deepcopy(curr_model)
        unvaluable_model.fit(
            Subset(x_train, least_valuable_indices),
            Subset(y_train, least_valuable_indices),
            batch_size=batch_size,
            epochs=epochs,
        )
        iy_hat_valid = unvaluable_model.predict(x_valid)
        unvaluable_score = evaluator.evaluate(y_valid, iy_hat_valid)
        unvaluable_list.append(unvaluable_score)

    # Plot graphs
    if plot:
        x_axis = [a * (1.0 / num_bins) for a in range(num_bins)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x_axis, valuable_list[:num_bins], "o-")
        plt.plot(x_axis, unvaluable_list[:num_bins], "x-")

        plt.xlabel("Fraction of Removed Samples", size=16)
        plt.ylabel(metric_name, size=16)
        plt.legend(
            ["Removing low value data", "Removing high value data"], prop={"size": 16}
        )
        plt.title("Remove High/Low Valued Samples", size=16)

        plt.show()

    return valuable_list, unvaluable_list


def discover_corrupted_sample(
    evaluator: DataEvaluator,
    noisy_index: np.ndarray,
    percentile_increment: float = 0.05,
    plot: bool = True,
):
    """Repeatedly explores ``percentile_increment`` of the data values and determines
    if within that total percentile, what proportion of the noisy indices are found.

    Parameters
    ----------
    evaluator : DataEvaluator
        Data Evaluator
    noisy_index : np.ndarray
        Indices with noise added to them from DataLoader.
    percentile_increment : float, optional
        Percentile of data points to add each iteration, by default .05
    plot : bool, optional
        Whether to plot the data, by default True, by default True

    Returns
    -------
    list[float]
        Proportion of noisy indices found at each interval
    """
    (x_train, y_train), (x_valid, y_valid) = evaluator.get_data_points()
    data_values = evaluator.evaluate_data_values()

    num_sample = len(x_train)
    num_period = max(round(num_sample * percentile_increment), 5)  # Add at least 5/bin
    num_bins = int(num_sample // num_period) + 1

    sorted_value_list = np.argsort(data_values)  # Order descending
    noise_rate = len(noisy_index) / len(data_values)

    # Output initialization
    found_rates = []

    # For each bin
    for bin_index in range(0, num_sample + num_period, num_period):
        # from low to high data values
        found_rates.append(
            len(np.intersect1d(sorted_value_list[:bin_index], noisy_index))
            / len(noisy_index)
        )

    # Plot corrupted label discovery graphs
    if plot:
        x_axis = [a * (1.0 / num_bins) for a in range(num_bins)]

        # Corrupted label discovery results (dvrl, optimal, random)
        y_dv = found_rates[:num_bins]
        y_opt = [min((a * (1.0 / num_bins / noise_rate), 1.0)) for a in range(num_bins)]
        y_random = x_axis

        plt.figure(figsize=(6, 7.5))
        plt.plot(x_axis, y_dv, "o-")
        plt.plot(x_axis, y_opt, "--")
        plt.plot(x_axis, y_random, ":")
        plt.xlabel("Fraction of data Inspected", size=16)
        plt.ylabel("Fraction of discovered corrupted samples", size=16)
        plt.legend(["Evaluator", "Optimal", "Random"], prop={"size": 16})
        plt.title("Corrupted Sample Discovery", size=16)
        plt.show()

    # Returns True Positive Rate of corrupted label discovery
    return found_rates