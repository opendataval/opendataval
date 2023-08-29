"""Experiments to test :py:class:`~opendataval.dataval.api.DataEvaluator`.

Experiments pass into :py:meth:`~opendataval.experiment.api.ExperimentMediator.evaluate`
and :py:meth:`~opendataval.experiment.api.ExperimentMediator.plot` evaluate performance
of one :py:class:`~opendataval.dataval.api.DataEvaluator` at a time.
"""
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.utils.data import Subset

from opendataval.dataloader import DataFetcher
from opendataval.dataval import DataEvaluator
from opendataval.experiment.util import f1_score, oned_twonn_clustering
from opendataval.metrics import Metrics
from opendataval.model import Model
from opendataval.util import get_name


def noisy_detection(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    indices: Optional[list[int]] = None,
) -> dict[str, float]:
    """Evaluate ability to identify noisy indices.

    Compute F1 score (of 2NN classifier) of the data evaluator
    on the noisy indices. KMeans labels are random, but because of the convexity of
    KMeans, the highest data point and lowest data point have different labels and
    belong to the most valuable/least valuable group. Thus, the least valuable group
    will be in one group and most valuable to zero for the F1 score.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing noisy indices
    indices : list[int], optional
        Alternatively, pass in noisy indices instead of DataFetcher, by default None

    Returns
    -------
    dict[str, float]

        - **"kmeans_f1"** -- F1 score performance of a 1D KNN binary classifier
            of the data points. Classifies the lower data value data points as
            corrupted, and the higher value data points as correct.
    """
    data_values = evaluator.data_values
    noisy_train_indices = (
        fetcher.noisy_train_indices if isinstance(fetcher, DataFetcher) else indices
    )

    unvaluable, _ = oned_twonn_clustering(data_values.flatten())
    f1_kmeans_label = f1_score(unvaluable, noisy_train_indices, len(data_values))

    return {"kmeans_f1": f1_kmeans_label}


def remove_high_low(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    percentile: float = 0.05,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[float]]:
    """Evaluate performance after removing high/low points determined by data valuator.

    Repeatedly removes ``percentile`` of most valuable/least valuable data points
    and computes the performance of the metric.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing training and testing data points, by default None
    model : Model, optional
        Model which performance will be evaluated, if not defined,
        uses evaluator's model to evaluate performance if evaluator uses a model
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
        - **"y_train"** Training labels
        - **"x_test"** Testing covariates
        - **"y_test"** Testing labels
    percentile : float, optional
        Percentile of data points to remove per iteration, by default 0.05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric : Metrics | Callable[[Tensor, Tensor], float], optional
        Name of DataEvaluator defined performance metric which is one of the defined
        metrics or a Callable[[Tensor, Tensor], float], by default accuracy
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None

    Returns
    -------
    dict[str, list[float]]
        dict containing list of the performance of the DataEvaluator
        ``(i * percentile)`` valuable/most valuable data points are removed

        - **"axis"** -- Proportion of data values removed currently
        - **f"remove_least_influential_first_{metric}"** -- Performance of model
            after removing a proportion of the data points with the lowest data values
        - **"f"remove_most_influential_first_{metric}""** -- Performance of model
            after removing a proportion of the data points with the highest data values
    """
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()

    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)  # Add at least 5/bin
    num_bins = int(num_points // num_period)
    sorted_value_list = np.argsort(data_values)

    valuable_list, unvaluable_list = [], []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for bin_index in range(0, num_points, num_period):
        # Removing least valuable samples first
        most_valuable_indices = sorted_value_list[bin_index:]

        # Fitting on valuable subset
        valuable_model = curr_model.clone()
        valuable_model.fit(
            Subset(x_train, most_valuable_indices),
            Subset(y_train, most_valuable_indices),
            **train_kwargs,
        )
        y_hat_valid = valuable_model.predict(x_test)
        valuable_score = metric(y_test, y_hat_valid)
        valuable_list.append(valuable_score)

        # Removing most valuable samples first
        least_valuable_indices = sorted_value_list[: num_points - bin_index]

        # Fitting on unvaluable subset
        unvaluable_model = curr_model.clone()
        unvaluable_model.fit(
            Subset(x_train, least_valuable_indices),
            Subset(y_train, least_valuable_indices),
            **train_kwargs,
        )
        iy_hat_valid = unvaluable_model.predict(x_test)
        unvaluable_score = metric(y_test, iy_hat_valid)
        unvaluable_list.append(unvaluable_score)

    x_axis = [i / num_bins for i in range(num_bins)]

    eval_results = {
        f"remove_least_influential_first_{get_name(metric)}": valuable_list,
        f"remove_most_influential_first_{get_name(metric)}": unvaluable_list,
        "axis": x_axis,
    }

    # Plot graphs
    if plot is not None:
        # Prediction performances after removing high or low values
        plot.plot(x_axis, valuable_list[:num_bins], "o-")
        plot.plot(x_axis, unvaluable_list[:num_bins], "x-")

        plot.set_xlabel("Fraction Removed")
        plot.set_ylabel(get_name(metric))
        plot.legend(["Removing low value data", "Removing high value data"])

        plot.set_title(str(evaluator))

    return eval_results


def discover_corrupted_sample(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    data: Optional[dict[str, Any]] = None,
    percentile: float = 0.05,
    plot: Optional[Axes] = None,
) -> dict[str, list[float]]:
    """Evaluate discovery of noisy indices in low data value points.

    Repeatedly explores ``percentile`` of the data values and determines
    if within that total percentile, what proportion of the noisy indices are found.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing noisy indices, by default None
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
    percentile : float, optional
        Percentile of data points to additionally search per iteration, by default .05
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None

    Returns
    -------
    Dict[str, list[float]]
        dict containing list of the proportion of noisy indices found after exploring
        the ``(i * percentile)`` least valuable data points. If plot is not None,
        also returns optimal and random search performances as lists

        - **"axis"** -- Proportion of data values explored currently.
        - **"corrupt_found"** -- Proportion of corrupted data values found currently
        - **"optimal"** -- Optimal proportion of corrupted values found currently
            meaning if the inspected **only** contained corrupted samples until
            the number of corrupted samples are completely exhausted.
        - **"random"** -- Random proportion of corrupted samples found, meaning
            if the data points were explored randomly, we'd expect to find
            corrupted_samples in proportion to the number of corruption in the data set.
    """
    if isinstance(fetcher, DataFetcher):
        x_train, *_ = fetcher.datapoints
    else:
        x_train = data["x_train"]
    noisy_train_indices = fetcher.noisy_train_indices
    data_values = evaluator.data_values

    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)  # Add at least 5 per bin
    num_bins = int(num_points // num_period) + 1

    sorted_value_list = np.argsort(data_values, kind="stable")  # Order descending
    noise_rate = len(noisy_train_indices) / len(data_values)

    # Output initialization
    found_rates = []

    # For each bin
    for bin_index in range(0, num_points + num_period, num_period):
        # from low to high data values
        found_rates.append(
            len(np.intersect1d(sorted_value_list[:bin_index], noisy_train_indices))
            / len(noisy_train_indices)
        )

    x_axis = [i / num_bins for i in range(len(found_rates))]
    eval_results = {"corrupt_found": found_rates, "axis": x_axis}

    # Plot corrupted label discovery graphs
    if plot is not None:
        # Corrupted label discovery results (dvrl, optimal, random)
        y_dv = found_rates[:num_bins]
        y_opt = [min((i / num_bins / noise_rate, 1.0)) for i in range(len(found_rates))]
        y_random = x_axis

        eval_results["optimal"] = y_opt
        eval_results["random"] = y_random

        plot.plot(x_axis, y_dv, "o-")
        plot.plot(x_axis, y_opt, "--")
        plot.plot(x_axis, y_random, ":")
        plot.set_xlabel("Prop of data inspected")
        plot.set_ylabel("Prop of discovered corrupted samples")
        plot.legend(["Evaluator", "Optimal", "Random"])

        plot.set_title(str(evaluator))

    # Returns True Positive Rate of corrupted label discovery
    return eval_results


def save_dataval(
    evaluator: DataEvaluator,
    fetcher: DataFetcher = None,
    indices: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
):
    """Save the indices and the respective data values of the DataEvaluator."""
    train_indices = (
        fetcher.train_indices if isinstance(fetcher, DataFetcher) else indices
    )
    data_values = evaluator.data_values

    data = {"indices": train_indices, "data_values": data_values}

    if output_path:
        df_data = {str(evaluator): data}
        df = pd.DataFrame.from_dict(df_data, "index")
        df.explode(list(df.columns)).to_csv(output_path)

    return data


def increasing_bin_removal(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    bin_size: int = 1,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[float]]:
    """Evaluate accuracy after removing data points with data values above threshold.

    For each subplot, displays the proportion of the data set with data values less
    than the specified data value (x-axis) and the performance of the model when all
    data values greater than the specified data value is removed. This implementation
    was inspired by V. Feldman and C. Zhang in their paper [1] where the same principle
    was applied to memorization functions.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing training and valid data points, by default None
    model : Model, optional
        Model which performance will be evaluated, if not defined,
        uses evaluator's model to evaluate performance if evaluator uses a model
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
        - **"y_train"** Training labels
        - **"x_test"** Testing covariates
        - **"y_test"** Testing labels
    bin_size : float, optional
        We look at bins of equal size and find the data values cutoffs for the x-axis,
        by default 1
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric : Metrics | Callable[[Tensor, Tensor], float], optional
        Name of DataEvaluator defined performance metric which is one of the defined
        metrics or a Callable[[Tensor, Tensor], float], by default accuracy
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None

    Returns
    -------
    Dict[str, list[float]]
        dict containing the thresholds of data values examined, proportion of training
        data points removed, and performance after those data points were removed.

        - **"axis"** -- Thresholds of data values examined. For a given threshold,
            considers the subset of data points with data values below.
        - **"frac_datapoints_explored"** -- Proportion of data points with data values
            below the specified threshold
        - **f"{metric}_at_datavalues"** -- Performance metric when data values
            above the specified threshold are removed
    """
    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    num_points = len(data_values)

    # Starts with 10 data points
    bins_indices = [*range(5, num_points - 1, bin_size), num_points - 1]
    frac_datapoints_explored = [(i + 1) / num_points for i in bins_indices]

    sorted_indices = np.argsort(data_values)
    x_axis = data_values[sorted_indices[bins_indices]] / np.max(data_values)

    perf = []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for bin_end in bins_indices:
        coalition = sorted_indices[:bin_end]

        new_model = curr_model.clone()
        new_model.fit(
            Subset(x_train, coalition),
            Subset(y_train, coalition),
            **train_kwargs,
        )
        y_hat = new_model.predict(x_test)
        perf.append(metric(y_hat, y_test))

    eval_results = {
        "frac_datapoints_explored": frac_datapoints_explored,
        f"{get_name(metric)}_at_datavalues": perf,
        "axis": x_axis,
    }

    if plot is not None:  # Removing everything above this threshold
        plot.plot(x_axis, perf)

        plot.set_xticks([])
        plot.set_ylabel(get_name(metric))
        plot.set_title(str(evaluator))

        divider = make_axes_locatable(plot)
        frac_inspected_plot = divider.append_axes("bottom", size="40%", pad="5%")

        frac_inspected_plot.fill_between(x_axis, frac_datapoints_explored)
        frac_inspected_plot.set_xlabel("Data Values Threshold")
        frac_inspected_plot.set_ylabel("Trainset Fraction")

    return eval_results
