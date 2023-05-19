import math
import os
import warnings
from typing import Any, Callable, Union

import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataloader import DataFetcher, mix_labels
from opendataval.dataval import DataEvaluator
from opendataval.model import Model, ModelFactory


def accuracy_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute accuracy of two one-hot encoding tensors."""
    return (a.argmax(dim=1) == b.argmax(dim=1)).float().mean().item()


metrics_dict = {  # TODO add metrics and change this implementation
    "accuracy": accuracy_metric,
    # Metrics should be the higher the better
    "l2": lambda a, b: -torch.square(a - b).sum().sqrt().item(),
    "mse": lambda a, b: -F.mse_loss(a, b).item(),
}


class ExperimentMediator:
    """Set up an experiment to compare a group of DataEvaluators.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher for the data set used for the experiment. All `exper_func` take a
        DataFetcher as an argument to have access to all data points and noisy indices.
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
        pred_model: Model,
        train_kwargs: dict[str, Any] = None,
        metric_name: str = "accuracy",
    ):
        self.fetcher = fetcher
        self.pred_model = pred_model
        self.train_kwargs = {} if train_kwargs is None else train_kwargs

        self.metric_name = metric_name
        self.metric = metrics_dict[self.metric_name]
        self.data_evaluators = []

    @classmethod
    def setup(
        cls,
        dataset_name: str,
        cache_dir: str = None,
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
    ):
        """Create a DataFetcher from args and passes it into the init."""
        random_state = check_random_state(random_state)
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
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
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
        )

    @classmethod
    def model_factory_setup(
        cls,
        dataset_name: str,
        cache_dir: str = None,
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
    ):
        """Set up ExperimentMediator from ModelFactory using an input string.

        Return a ExperimentMediator initialized with
        py:function`~opendataval.model.ModelFactory`

        Parameters
        ----------
        dataset_name : str
            Name of the data set, must be registered with
            :py:class:`~opendataval.dataloader.Register`
        cache_dir : str, optional
            Directory of where to cache the loaded data, by default None which uses
            :py:attr:`Register.CACHE_DIR`
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
        ExperimentMediator
            ExperimentMediator created from ModelFactory defaults
        """
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            force_download=force_download,
            random_state=random_state,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise_func=add_noise_func,
            noise_kwargs=noise_kwargs,
        )

        pred_model = ModelFactory(
            model_name=model_name,
            fetcher=fetcher,
            device=device,
        )

        # Prints base line performance
        model = pred_model.clone()
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints

        model.fit(x_train, y_train, **train_kwargs)
        perf = metrics_dict[metric_name](y_test, model.predict(x_test).cpu())
        print(f"Base line model {metric_name=}: {perf=}")

        return cls(
            fetcher=fetcher,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
        )

    def compute_data_values(
        self, data_evaluators: list[DataEvaluator], *args, **kwargs
    ):
        """Computes the data values for the input data evaluators.

        Parameters
        ----------
        data_evaluators : list[DataEvaluator]
            List of DataEvaluators to be tested by `exper_func`
        """
        kwargs = {**kwargs, **self.train_kwargs}
        for data_val in data_evaluators:
            try:
                self.data_evaluators.append(
                    data_val.train(
                        self.fetcher, self.pred_model, self.metric, *args, **kwargs
                    )
                )

            except Exception as ex:
                warnings.warn(
                    f"""
                    An error occured during training, however training all evaluators
                    takes a long time, so we will be ignoring the evaluator:
                    {data_val!s} and proceeding.

                    The error is as follows: {ex!s}
                    """,
                    stacklevel=10,
                )

        self.num_data_eval = len(self.data_evaluators)
        return self

    def evaluate(
        self,
        exper_func: Callable[[DataEvaluator, DataFetcher, ...], dict[str, Any]],
        include_train: bool = False,
        save_output: bool = False,
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
        save_output : bool, optional
            Wether to save the outputs to ``self.output_dir``, by default False
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
        df_resp = pd.DataFrame.from_dict(data_eval_perf).stack().apply(pd.Series)
        if save_output:
            self.save_output(f"{exper_func.__name__}.csv", df_resp)
        return df_resp

    def plot(
        self,
        exper_func: Callable[[DataEvaluator, DataFetcher, Axes, ...], dict[str, Any]],
        figure: Figure = None,
        row: int = None,
        col: int = 2,
        include_train: bool = False,
        save_output: bool = False,
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
        save_output : bool, optional
            Wether to save the outputs to ``self.output_dir``, by default False
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
        df_resp = pd.DataFrame.from_dict(data_eval_perf).stack().apply(pd.Series)

        if save_output:
            self.save_output(f"{exper_func.__name__}.csv", df_resp)
        return df_resp, figure

    def set_output_directory(self, output_directory: str):
        """Set directory to save output of experiment."""
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.output_directory = output_directory
        return self

    def save_output(self, file_name: str, df: pd.DataFrame):
        """Saves the output of the DataFrame to f"{self.output_directory}/{file_name}".

        Parameters
        ----------
        file_name : str
            Name of the file to save the DataFrame to.
        df : pd.DataFrame
            Output DataFrame from an experiment run by ExperimentMediator
        """
        if not hasattr(self, "output_directory"):
            warnings.warn("Output directory not set, output has not been saved")
            return

        file_path = os.path.join(self.output_directory, file_name)
        df.to_csv(file_path)
