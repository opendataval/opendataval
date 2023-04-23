import math
from functools import partial
from typing import Any, Callable, Union

import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import RandomState
from sklearn.utils import check_random_state

from dataoob.dataloader import DataFetcher, mix_labels
from dataoob.dataval import DataEvaluator
from dataoob.model.api import Model

# Models
from dataoob.model.logistic_regression import LogisticRegression
from dataoob.model.mlp import ClassifierMLP, RegressionMLP


def accuracy_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute accuracy of two one-hot encoding tensors."""
    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()


metrics_dict = {  # TODO add metrics and change this implementation
    "accuracy": accuracy_metric,
    # Metrics should be the higher the better
    "l2": lambda a, b: -torch.square(a - b).sum().sqrt().item(),
    "mse": lambda a, b: -F.mse_loss(a, b).item(),
}


def model_factory(
    model_name: str,
    covar_dim: tuple[int, ...],
    label_dim: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
):
    if model_name == "logreg":
        return LogisticRegression(*covar_dim, *label_dim).to(device=device)
    elif model_name == "mlpclass":
        return ClassifierMLP(*covar_dim, *label_dim).to(device=device)
    elif model_name == "mlpregress":
        return RegressionMLP(*covar_dim, *label_dim).to(device=device)
    elif model_name == "bert":
        # Temporary fix while I figure out a better way for model factory
        from dataoob.model.bert import BertClassifier

        return BertClassifier(num_classes=label_dim[0]).to(device=device)
    else:
        raise ValueError(f"{model_name} is not a valid predefined model")


class ExperimentMediator:
    """Set up an experiment to compare a group of DataEvaluators.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher for the data set used for the experiment. All `exper_func` take a
        DataFetcher as an argument to have access to all data points and noisy indices.
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
        fetcher: DataFetcher,
        data_evaluators: list[DataEvaluator],
        pred_model: Model,
        train_kwargs: dict[str, Any] = None,
        metric_name: str = "accuracy",
    ):
        self.fetcher = fetcher
        self.train_kwargs = {} if train_kwargs is None else train_kwargs
        self.metric_name = metric_name
        self.data_evaluators = []

        for data_val in data_evaluators:
            try:

                self.data_evaluators.append(
                    data_val.input_model_metric(pred_model, metrics_dict[metric_name])
                    .input_fetcher(fetcher)
                    .train_data_values(**self.train_kwargs)
                )

            except Exception as ex:
                import warnings

                warnings.warn(
                    f"""
                    An error occured during training, however training all evaluators
                    takes a long time, so we will be ignoring the evaluator:
                    {str(data_val)} and proceeding.

                    The error is as follows: {str(ex)}
                    """,
                    stacklevel=10,
                )

        self.num_data_eval = len(self.data_evaluators)

    @classmethod
    def setup(
        cls,
        dataset_name: str,
        force_download: bool = False,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
        add_noise_func: Callable[[DataFetcher, Any, ...], dict[str, Any]] = mix_labels,
        noise_kwargs: dict[str, Any] = None,
        random_state: RandomState = None,
        pred_model: Model = None,
        train_kwargs: dict[str, Any] = None,
        metric_name: str = "accuracy",
        data_evaluators: list[DataEvaluator] = None,
    ):
        """Create a DataFetcher from args and passes it into the init."""
        random_state = check_random_state(random_state)
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            force_download=force_download,
            random_state=random_state,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise_func=add_noise_func,
            noise_kwargs=noise_kwargs,
        )

        return cls(
            fetcher=fetcher,
            data_evaluators=data_evaluators,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
        )

    @classmethod
    def partial_setup(
        cls,
        dataset_name: str,
        force_download: bool = False,
        train_count: Union[int, float] = 0,
        valid_count: Union[int, float] = 0,
        test_count: Union[int, float] = 0,
        add_noise_func: Callable[[DataFetcher, Any, ...], dict[str, Any]] = mix_labels,
        noise_kwargs: dict[str, Any] = None,
        random_state: RandomState = None,
        model_name: "str" = None,
        device: torch.device = torch.device("cpu"),
        train_kwargs: dict[str, Any] = None,
        metric_name: str = "accuracy",
    ) -> partial:
        """Set up ExperimentMediator without inputting the DataEvaluators

        Return a partial[ExperimentMediator] initialized with the

        Parameters
        ----------
        dataset_name : str
            Name of the data set, must be registered with
            :py:class:`~dataoob.dataloader.Register`
        force_download : bool, optional
            Forces download from source URL, by default False
        train_count : Union[int, float]
            Number/proportion training points
        valid_count : Union[int, float]
            Number/proportion validation points
        test_count : Union[int, float]
            Number/proportion test points
        add_noise_func : Callable
            If None, no changes are made. Takes as argument required arguments
            DataFetcher and adds noise to those the data points of DataFetcher as
            needed. Returns dict[str, np.ndarray] that has the updated np.ndarray in a
            dict to update the data loader with the following keys:

            - **"x_train"** -- Updated training covariates with noise, optional
            - **"y_train"** -- Updated training labels with noise, optional
            - **"x_valid"** -- Updated validation covariates with noise, optional
            - **"y_valid"** -- Updated validation labels with noise, optional
            - **"x_test"** -- Updated testing covariates with noise, optional
            - **"y_test"** -- Updated testing labels with noise, optional
            - **"noisy_train_indices"** -- Indices of training data set with noise
        noise_kwargs : dict[str, Any], optional
            Key word arguments passed to ``add_noise_func``, by default None
        random_state : RandomState, optional
            Random initial state, by default None
        model_name : str, optional
            Name of the preset model, check :py:func:`model_factory` for preset models,
            by default None
        device : torch.device, optional
            Tensor device for acceleration, by default torch.device("cpu")
        metric_name : str, optional
            Name of the performance metric used to evaluate the performance of the
            prediction model, must be string for better labeling, by default "accuracy"
        train_kwargs : dict[str, Any], optional
            Training key word arguments for the prediction model, by default None

        Returns
        -------
        partial[ExperimentMediator]
            Partially initialized ExperimentMediator. When called, pass in the
            list[:py:class:`~dataoob.dataval.DataEvaluator`] to run the experiment.
        """
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            force_download=force_download,
            random_state=random_state,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise_func=add_noise_func,
            noise_kwargs=noise_kwargs,
        )

        pred_model = model_factory(
            model_name=model_name,
            covar_dim=fetcher.covar_dim,
            label_dim=fetcher.label_dim,
            device=device,
        )

        # Prints base line performance
        model = pred_model.clone()
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints

        model.fit(x_train, y_train, **train_kwargs)
        perf = metrics_dict[metric_name](y_test, model.predict(x_test).cpu())
        print(f"Base line model {metric_name}: {perf}")

        return partial(
            cls,
            fetcher=fetcher,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
        )

    def evaluate(
        self,
        exper_func: Callable[[DataEvaluator, DataFetcher, ...], dict[str, Any]],
        include_train: bool = False,
        **exper_kwargs,
    ) -> pd.DataFrame:
        """Evaluate `exper_func` on each DataEvaluator.

        Runs an experiment on a list of pre-train DataEvaluators and their
        corresponding dataset and returns a DataFrame of the results.

        Parameters
        ----------
        exper_func : Callable[[DataEvaluator, DataFetcher, ...], dict[str, Any]]
            Experiment function, runs an experiment on a DataEvaluator and the data of
            the DataFetcher associated. Output must be a dict with results of the
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
            DataFrame is indexed: [result_title, DataEvaluator]
            Any 1-D experiment result is expanded into columns: list(range(len(result)))

            To get the results by result_title, df.loc[result_title]
            To get the results by DataEvaluator, use df.ax(DataEvaluator, level=1)
        """
        data_eval_perf = {}
        if include_train:
            # All methods that train the underlying model track the model performance
            exper_kwargs["train_kwargs"] = self.train_kwargs
            exper_kwargs["metric_name"] = self.metric_name

        for data_val in self.data_evaluators:
            eval_resp = exper_func(data_val, self.fetcher, **exper_kwargs)
            data_eval_perf[str(data_val)] = eval_resp

        # index=[result_title, DataEvaluator] columns=[range(len(axis))]
        return pd.DataFrame.from_dict(data_eval_perf).stack().apply(pd.Series)

    def plot(
        self,
        exper_func: Callable[[DataEvaluator, DataFetcher, Axes, ...], dict[str, Any]],
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
        exper_func : Callable[[DataEvaluator, DataFetcher, Axes, ...], dict[str, Any]]
            Experiment function, runs an experiment on a DataEvaluator and the data of
            the DataFetcher associated. Output must be a dict with results of the
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
            DataFrame is indexed: [result_title, DataEvaluator.DataEvaluator]
            Any 1-D experiment result is expanded into columns: list(range(len(result)))

            To get the results by result_title, df.loc[result_title]
            To get the results by DataEvaluator, use df.ax(DataEvaluator, level=1)

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
            eval_resp = exper_func(data_val, self.fetcher, plot=plot, **exper_kwargs)

            data_eval_perf[str(data_val)] = eval_resp

        # index=[result_title, DataEvaluator] columns=[range(len(axis))]
        return pd.DataFrame.from_dict(data_eval_perf).stack().apply(pd.Series), figure
