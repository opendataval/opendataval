import numpy as np
import copy

from torch.utils.data import Subset
from sklearn.cluster import KMeans
from dataoob.dataval import DataEvaluator
from sklearn.metrics import f1_score, roc_auc_score

from matplotlib import pyplot as plt

"""
TODO clean up the code here, quite messy rn, shouldn't be teribly difficult just
TODO create pipeline for Evaluator
not something I can do rn
noisy detection task
"""

def noisy_detection(
    evaluator: DataEvaluator, noisy_index: np.ndarray, num_classes: int=2
) -> tuple[float, float]:
    data_values = evaluator.evaluate_data_values()

    n_points = len(data_values)
    n_noisy = len(noisy_index)
    index_of_small_values = np.argsort(data_values)[: n_noisy]
    recall = len(np.intersect1d(index_of_small_values, noisy_index)) / n_noisy

    # using kmeans label
    kmeans = KMeans(n_clusters=num_classes).fit(data_values.reshape(-1, 1))
    validation = np.zeros((n_points,))
    validation[noisy_index] = 1
    f1_kmeans_label = f1_score(kmeans.labels_, validation)

    return recall, f1_kmeans_label


def point_removal(  # TODO consider just passing in the x_values
    evaluator: DataEvaluator,
    order: str = "ascending",
    percentile_increment: float=.05,
    batch_size: int = 32,
    epochs: int = 1,
    plot: bool = False,
    metric_name: str = "Accuracy",
):
    """We repeatedly remove 5% of entire data points at each step.
    The data points whose value belongs to the lowest group are removed first.
    The larger, the better

    :param bool ascending: _description_, defaults to True
    :param float percentile_increment: _description_, defaults to .05
    :param int batch_size: _description_, defaults to 32
    :param int epochs: _description_, defaults to 1
    :param bool plot: _description_, defaults to False
    :return _type_: _description_
    """
    (x_train, y_train), (x_valid, y_valid) = evaluator.get_data_points()
    data_values = evaluator.evaluate_data_values()
    curr_model = copy.deepcopy(evaluator.pred_model)

    if order == "ascending":  # ascending order. low to high.
        sorted_value_list = np.argsort(data_values)
    elif order == "descending":  # descending order. high to low.
        sorted_value_list = np.argsort(data_values)[::-1]
    elif order == "random":
        sorted_value_list = np.random.permutation(len(data_values))

    n_sample = len(x_train)
    n_period = max(round(n_sample * percentile_increment), 5)  # Adding five min change
    num_bins = int(n_sample//n_period//2)

    metric_list = []
    for bin_index in range(0, n_sample, n_period):

        sorted_value_coalition = sorted_value_list[bin_index:]

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
        # Defines x-axis
        x = [a*(1.0/num_bins) for a in range(num_bins)]
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, metric_list[:num_bins], 'o-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel(metric_name, size=16)
        plt.title(
            f'Removing value {order}', size=16
        )
        plt.show()

    return metric_list


def remove_high_low(
    evaluator: DataEvaluator,
    percentile_increment: float=.05,
    batch_size: int = 32,
    epochs: int = 1,
    plot: bool=True,
    metric_name: str = "Accuracy",
):

    """Evaluates performance after removing a portion of high/low valued samples.
    Args:
        dve_out: data values
        eval_model: evaluation model (object)
        x_train: training features
        y_train: training labels
        x_valid: validation features
        y_valid: validation labels
        x_test: testing features
        y_test: testing labels
        perf_metric: 'auc', 'accuracy', or 'rmspe'
        plot: print plot or not
    Returns:
        output_perf: Prediction performances after removing a portion of high
                    or low valued samples.
    """
    (x_train, y_train), (x_valid, y_valid) = evaluator.get_data_points()
    data_values = evaluator.evaluate_data_values()
    curr_model = copy.deepcopy(evaluator.pred_model)

    # Sorts samples by data values
    sorted_value_list = np.argsort(data_values)

    n_sample = len(x_train)
    n_period = max(round(n_sample * percentile_increment), 5)  # Adding five min change
    num_bins = int(n_sample//n_period//2)

    valuable_list, unvaluable_list = [], []
    for bin_index in range(0, n_sample//2, n_period):

        # Remove least valuable samples first
        most_valuable_indices = sorted_value_list[bin_index:]
        # Remove most valuable samples first
        least_valuable_indices = sorted_value_list[bin_index:]

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

        # Defines x-axis
        x = [a*(1.0/num_bins) for a in range(num_bins)]

        # Prediction performances after removing high or low values
        plt.figure(figsize=(6, 7.5))
        plt.plot(x, valuable_list[:num_bins], 'o-')
        plt.plot(x, unvaluable_list[:num_bins], 'x-')

        plt.xlabel('Fraction of Removed Samples', size=16)
        plt.ylabel(metric_name, size=16)
        plt.legend(['Removing low value data', 'Removing high value data'],
                prop={'size': 16})
        plt.title('Remove High/Low Valued Samples', size=16)

        plt.show()

    return valuable_list, unvaluable_list


def discover_corrupted_sample(
    evaluator: DataEvaluator,
    noise_idx,
    percentile_increment: float=.05,
    plot=True
):
    """Reports True Positive Rate (TPR) of corrupted label discovery.
    Args:
        dve_out: data values
        noise_idx: noise index
        noise_rate: the ratio of noisy samples
        plot: print plot or not
    Returns:
        output_perf: True positive rate (TPR) of corrupted label discovery
                    (per 5 percentiles)
    """

    (x_train, y_train), (x_valid, y_valid) = evaluator.get_data_points()
    data_values = evaluator.evaluate_data_values()
    noise_rate = len(data_values) / len(noise_idx)
    # Sorts samples by data values
    n_sample = len(x_train)
    n_period = max(round(n_sample * percentile_increment), 5)  # Adding five min change
    num_bins = int(n_sample//n_period)

    sort_idx = np.argsort(data_values)
    # Output initialization
    output_perf = np.zeros((num_bins,))

    # For each percentile
    for itt in range(num_bins):
        # from low to high data values
        output_perf[itt] = len(np.intersect1d(sort_idx[:int((itt+1)*n_period)], noise_idx)) / len(noise_idx)

    # Plot corrupted label discovery graphs
    if plot:
        # Defines x-axis
        x = [a*(1.0/num_bins) for a in range(num_bins)]

        # Corrupted label discovery results (dvrl, optimal, random)
        y_dv = np.concatenate((np.zeros(1), output_perf[:(num_bins-1)]))
        y_opt = [min([a*((1.0/num_bins)/noise_rate), 1]) for a in range(num_bins)]
        y_random = x

        plt.figure(figsize=(6, 7.5))
        plt.plot(x, y_dv, 'o-')
        plt.plot(x, y_opt, '--')
        plt.plot(x, y_random, ':')
        plt.xlabel('Fraction of data Inspected', size=16)
        plt.ylabel('Fraction of discovered corrupted samples', size=16)
        plt.legend(['Evaluator', 'Linear', 'Random'], prop={'size': 16})
        plt.title('Corrupted Sample Discovery', size=16)
        plt.show()

    # Returns True Positive Rate of corrupted label discovery
    return output_perf