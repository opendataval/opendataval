import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import RandomState
from sklearn.utils import check_random_state

from dataoob.dataloader import DataLoader, mix_labels
from dataoob.dataval import DataEvaluator
from dataoob.model import Model


def accuracy_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute accuracy of two one-hot encoding tensors."""
    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()


metrics_dict = {  # TODO add metrics and change this implementation
    "accuracy": accuracy_metric,
    "l2": lambda a, b: torch.square(a - b).sum().sqrt().item(),
    "mse": lambda a, b: F.mse_loss(a, b).item(),
}


@dataclass
class DataLoaderArgs:
    """DataLoaderArgs dataclass for easier creation of ExperimentMediator."""

    dataset: str
    force_download: bool = False
    device: torch.device = torch.device("cpu")
    random_state: RandomState = None

    train_count: int | float = 0.7  # 70-20-10 split is relatively standard
    valid_count: int | float = 0.2
    test_count: int | float = 0.1

    noise_kwargs: dict[str, Any] = field(default_factory=dict)
    add_noise_func: Callable[[DataLoader, Any, ...], dict[str, Any]] = mix_labels


@dataclass
class DataEvaluatorArgs:
    """DataLoaderArgs dataclass for easier creation of ExperimentMediator."""

    pred_model: Model
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    metric_name: str = "accuracy"


@dataclass
class DataEvaluatorFactoryArgs:
    """DataEvaluatorArgs dataclass for ExperimentMediator if input/output dim known."""

    pred_model_factory: Callable[[int, int, torch.device], Model]
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    metric_name: str = "accuracy"
    device: torch.device = torch.device("cpu")


class ExperimentMediator:
    """Set up an experiment to compare a group of DataEvaluators.

    Parameters
    ----------
    loader : DataLoader
        DataLoader for the data set used for the experiment. All `exper_func` take a
        DataLoader as an argument to have access to all data points and noisy indices.
    data_evaluators : list[DataEvaluator]
        List of DataEvaluators to be tested by `exper_func`
    pred_model : Model
        Prediction model for the DataEvaluators
    metric_name : str, optional
        Name of the performance metric used to evaluate the performance of the
        prediction model, must be string for better labeling, by default "accuracy"
    train_kwargs : dict[str, Any], optional
        Training key word arguments for the prediction model, by default None
    """

    def __init__(
        self,
        loader: DataLoader,
        data_evaluators: list[DataEvaluator],
        pred_model: Model,
        train_kwargs: dict[str, Any] = None,
        metric_name: str = "accuracy",
    ):
        self.loader = loader
        self.train_kwargs = {} if train_kwargs is None else train_kwargs
        self.metric_name = metric_name
        self.data_evaluators = []

        for data_val in data_evaluators:
            try:
                self.data_evaluators.append(
                    data_val.input_model_metric(pred_model, metrics_dict[metric_name])
                    .input_dataloader(loader)
                    .train_data_values(**self.train_kwargs)
                )

            except Exception as ex:
                import warnings

                warnings.warn(
                    f"""
                    An error occured during training, however training all evaluators
                    takes a long time, so we will be ignoring the evaluator:
                    {data_val.plot_title} and proceeding.

                    The error is as follows: {str(ex)}
                    """,
                    stacklevel=10,
                )

        self.num_data_eval = len(self.data_evaluators)

    @staticmethod
    def create_dataloader(
        dataset: str,
        force_download: bool = False,
        train_count: int | float = 0,
        valid_count: int | float = 0,
        test_count: int | float = 0,
        noise_kwargs: dict[str, Any] = None,
        add_noise_func: Callable[[DataLoader, Any, ...], dict[str, Any]] = mix_labels,
        device: torch.device = torch.device("cpu"),
        random_state: RandomState = None,
        pred_model: Model = None,
        train_kwargs: dict[str, Any] = None,
        metric_name: str = "accuracy",
        data_evaluators: list[DataEvaluator] = None,
    ):
        """Create a DataLoader from args and passes it into the init."""
        random_state = check_random_state(random_state)
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        loader = (
            DataLoader(dataset, force_download, device, random_state)
            .split_dataset(train_count, valid_count, test_count)
            .noisify(add_noise_func, **noise_kwargs)
        )

        return ExperimentMediator(
            loader=loader,
            data_evaluators=data_evaluators,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
        )

    @staticmethod
    def setup(
        loader_args: DataLoaderArgs,
        data_evaluator_args: DataEvaluatorArgs,
        data_evaluators: list[DataEvaluator] = None,
    ):
        """Create ExperimentMediator from dataclass arg wrappers."""
        return ExperimentMediator.create_dataloader(
            data_evaluators=data_evaluators,
            **(asdict(loader_args) | asdict(data_evaluator_args)),
        )

    @staticmethod
    def preset_setup(
        loader_args: DataLoaderArgs,
        de_factory_args: DataEvaluatorFactoryArgs,
        data_evaluators: list[DataEvaluator] = None,
    ):
        """Create ExperimentMediator from presets, infers input/output dimensions."""
        rs = check_random_state(loader_args.random_state)

        if loader_args.device != de_factory_args.device:
            raise Exception("All tensors must be on same device")
        device = loader_args.device

        train_count = loader_args.train_count
        valid_count = loader_args.valid_count
        test_count = loader_args.test_count

        loader = (
            DataLoader(loader_args.dataset, loader_args.force_download, device, rs)
            .split_dataset(train_count, valid_count, test_count)
            .noisify(loader_args.add_noise_func, **loader_args.noise_kwargs)
        )

        covar_dim = len(loader.x_train[0])
        label_dim = loader.y_train.shape[1] if loader.y_train.ndim == 2 else 1
        pred_model = de_factory_args.pred_model_factory(covar_dim, label_dim, device)

        return ExperimentMediator(
            loader=loader,
            data_evaluators=data_evaluators,
            pred_model=pred_model,
            train_kwargs=de_factory_args.train_kwargs,
            metric_name=de_factory_args.metric_name,
        )

    def evaluate(
        self,
        exper_func: Callable[[DataEvaluator, DataLoader, ...], dict[str, Any]],
        include_train: bool = False,
        **exper_kwargs,
    ) -> pd.DataFrame:
        """Evaluate `exper_func` on each DataEvaluator.

        Runs an experiment on a list of pre-train DataEvaluators and their
        corresponding dataset and returns a DataFrame of the results.

        Parameters
        ----------
        exper_func : Callable[[DataEvaluator, DataLoader, ...], dict[str, Any]]
            Experiment function, runs an experiment on a DataEvaluator and the data of
            the DataLoader associated. Output must be a dict with results of the
            experiment. NOTE, the results must all be <= 1 dimensional but does not
            need to be the same length.
        include_train : bool, optional
            Whether to pass to exper_func the training kwargs defined for the
            ExperimentMediator. If True, also passes in metric_name, by default False
        eval_kwargs : dict[str, Any], optional
            Additional key word arguments to be passed to the exper_func


        Returns
        -------
        pd.DataFrame
            DataFrame containing the results for each DataEvaluator experiment.
            DataFrame is indexed: [result_title, DataEvaluator.plot_title]
            Any 1-D experiment result is expanded into columns: list(range(len(result)))

            To get the results by result_title, df.loc[result_title]
            To get the results by DataEvaluator, use df.ax(plot_title, level=1)
        """
        data_eval_perf = {}
        if include_train:
            # All methods that train the underlying model track the model performance
            exper_kwargs["train_kwargs"] = self.train_kwargs
            exper_kwargs["metric_name"] = self.metric_name

        for data_val in self.data_evaluators:
            eval_resp = exper_func(data_val, self.loader, **exper_kwargs)
            data_eval_perf[data_val.plot_title] = eval_resp

        # index=[result_title, plot_title] columns=[range(len(axis))]
        return pd.DataFrame.from_dict(data_eval_perf).stack().apply(pd.Series)

    def plot(
        self,
        exper_func: Callable[[DataEvaluator, DataLoader, Axes, ...], dict[str, Any]],
        figure: Figure = None,
        row: int = None,
        col: int = 2,
        include_train: bool = False,
        **exper_kwargs,
    ) -> tuple[pd.DataFrame, Figure]:
        """Evaluate `exper_func` on each DataEvaluator and plots result in `fig`.

        Run an experiment on a list of pre-train DataEvaluators and their
        corresponding dataset and plots the result.

        Parameters
        ----------
        exper_func : Callable[[DataEvaluator, DataLoader, Axes, ...], dict[str, Any]]
            Experiment function, runs an experiment on a DataEvaluator and the data of
            the DataLoader associated. Output must be a dict with results of the
            experiment. NOTE, the results must all be <= 1 dimensional but does not
            need to be the same length.
        fig : Figure, optional
            MatPlotLib Figure which each experiment result is plotted, by default None
        row : int, optional
            Number of rows of subplots in the plot, by default set to num_evaluators/col
        col : int, optional
            Number of columns of subplots in the plot, by default 2
        include_train : bool, optional
            Whether to pass to exper_func the training kwargs defined for the
            ExperimentMediator. If True, passes in metric_name, by default False
        eval_kwargs : dict[str, Any], optional
            Additional key word arguments to be passed to the exper_func

        Returns
        -------
        tuple[pd.DataFrame, Figure]
            DataFrame containing the results for each DataEvaluator experiment.
            DataFrame is indexed: [result_title, DataEvaluator.plot_title]
            Any 1-D experiment result is expanded into columns: list(range(len(result)))

            To get the results by result_title, df.loc[result_title]
            To get the results by DataEvaluator, use df.ax(plot_title, level=1)

            Figure is a plotted version of the results dict.
        """
        if figure is None:
            figure = plt.figure(figsize=(15, 15))

        if not row:
            row = math.ceil(self.num_data_eval / col)

        data_eval_perf = {}
        if include_train:
            # All methods that train the underlying model track the model performance
            exper_kwargs["train_kwargs"] = self.train_kwargs
            exper_kwargs["metric_name"] = self.metric_name

        for i, data_val in enumerate(self.data_evaluators, start=1):
            plot = figure.add_subplot(row, col, i)
            eval_resp = exper_func(data_val, self.loader, plot=plot, **exper_kwargs)

            data_eval_perf[data_val.plot_title] = eval_resp

        # index=[result_title, plot_title] columns=[range(len(axis))]
        return pd.DataFrame.from_dict(data_eval_perf).stack().apply(pd.Series), figure
