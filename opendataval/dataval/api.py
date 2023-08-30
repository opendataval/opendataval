from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from opendataval.dataloader import DataFetcher
from opendataval.metrics import accuracy, neg_mse
from opendataval.model import Model
from opendataval.util import ReprMixin

Self = TypeVar("Self")


class DataEvaluator(ABC, ReprMixin):
    """Abstract class of Data Evaluators. Facilitates Data Evaluation computation.

    The following is an example of how the api would work:
    ::
        dataval = (
            DataEvaluator(*args, **kwargs)
            .input_data(x_train, y_train, x_valid, y_valid)
            .train_data_values(batch_size, epochs)
            .evaluate_data_values()
        )

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    args : tuple[Any]
        DavaEvaluator positional arguments
    kwargs : Dict[str, Any]
        DavaEvaluator key word arguments

    Attributes
    ----------
    pred_model : Model
        Prediction model to find how much each training datum contributes towards it.
    data_values: np.array
        Cached data values, used by :py:mod:`opendataval.experiment.exper_methods`
    """

    Evaluators: ClassVar[dict[str, Self]] = {}

    def __init__(self, random_state: Optional[RandomState] = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def __init_subclass__(cls, *args, **kwargs):
        """Registers DataEvaluator types, used as part of the CLI."""
        super().__init_subclass__(*args, **kwargs)
        cls.Evaluators[cls.__name__.lower()] = cls

    def input_data(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: torch.Tensor,
        x_valid: Union[torch.Tensor, Dataset],
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for DataEvaluator.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        return self

    def setup(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    ):
        """Inputs model, metric and data into Data Evaluator.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model Less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.input_fetcher(fetcher)

        if isinstance(self, ModelMixin):
            if metric is None:
                metric = accuracy if fetcher.one_hot else neg_mse
            self.input_model(pred_model).input_metric(metric)
        return self

    def train(
        self,
        fetcher: DataFetcher,
        pred_model: Model,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        *args,
        **kwargs,
    ):
        """Store and transform data, then train model to predict data values.

        Trains the Data Evaluator and the underlying prediction model. Wrapper for
        ``self.input_data`` and ``self.train_data_values`` under one method.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model
            Prediction model
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.setup(fetcher, pred_model, metric)
        self.train_data_values(*args, **kwargs)

        return self

    @abstractmethod
    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a trained Data Evaluator.
        """
        return self

    @abstractmethod
    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """

    @cached_property
    def data_values(self) -> np.ndarray:
        """Cached data values."""
        return self.evaluate_data_values()

    def input_fetcher(self, fetcher: DataFetcher):
        """Input data from a DataFetcher object. Alternative way of adding data."""
        x_train, y_train, x_valid, y_valid, *_ = fetcher.datapoints
        return self.input_data(x_train, y_train, x_valid, y_valid)


class ModelMixin:
    """Mixin for data evaluators that require a model"""

    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor):
        """Evaluate performance of the specified metric between label and predictions.

        Moves input tensors to cpu because of certain bugs/errors that arise when the
        tensors are not on the same device

        Parameters
        ----------
        y : torch.Tensor
            Labels to be evaluate performance of predictions
        y_hat : torch.Tensor
            Predictions of labels

        Returns
        -------
        float
            Performance metric
        """
        return self.metric(y.cpu(), y_hat.cpu())

    def input_model(self, pred_model: Model):
        """Input the prediction model.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        """
        self.pred_model = pred_model.clone()
        return self

    def input_metric(self, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        """Input the evaluation metric.

        Parameters
        ----------
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance
        """
        self.metric = metric
        return self

    def input_model_metric(
        self, pred_model: Model, metric: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        """Input the prediction model and the evaluation metric.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        return self.input_model(pred_model).input_metric(metric)


class EmbeddingMixin:
    """Mixin for DataEvaluators with embeddings.

    Using embeddings is most frequently used on model-less DataEvaluators. When
    comparing performance for a specific model, if we want to use an embedding, we'd
    either want the Model itself to apply the transformation or the data to already
    be transformed. For model-less data evaluators, the ``embedding_model`` allows us
    to still use those embeddings.
    The Ruoxi Jia Group with their KNN Shapley and LAVA data evaluators use embeddings.

    Attributes
    ----------
    embedding_model : Model
        Embedding model used by model-less DataEvaluator to compute the data values for
        the embeddings and not the raw input.
    """

    def embeddings(
        self, *tensors: tuple[Union[Dataset, torch.Tensor], ...]
    ) -> tuple[torch.Tensor, ...]:
        """Returns Embeddings for the input tensors

        Returns
        -------
        tuple[torch.Tensor, ...]
            Returns tupple of tensors equal to the number of tensors input
        """
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            return tuple(self.embedding_model.predict(tensor) for tensor in tensors)

        # No embedding is used
        return tensors
