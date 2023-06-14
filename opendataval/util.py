import itertools
import time
from datetime import timedelta
from enum import Enum
from functools import update_wrapper
from typing import Callable, Generic, TypeVar

import numpy as np
import pandas as pd
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state


def load_mediator_output(filepath: str):
    """Loads output of Pandas DataFrame csv generated by ExperimentMediator."""
    return pd.read_csv(filepath, index_col=[0])


def set_random_state(random_state: RandomState = None) -> RandomState:
    """Set the random state of opendataval, useful for recreation of results."""
    print(f"Initial random seed is: {random_state}.")
    torch.manual_seed(check_random_state(random_state).tomaxint())
    random_state = check_random_state(random_state)
    return random_state


class StrEnum(str, Enum):
    """StrEnum is not implemented in Python3.9."""

    def __new__(cls, val):
        "Values must already be of convertable to type `str`"
        member = str.__new__(cls, str(val))
        member._value_ = val
        return member

    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()


X, Y = TypeVar("X"), TypeVar("Y")


class wrapper(str, Generic[X, Y]):
    def __new__(cls, function: Callable[[X, ...], Y], name: str = None):
        """Wrapper is a walks and talks like a str but can be called with the func."""
        out = str.__new__(cls, function.__name__ if name is None else name)
        out.function = function
        update_wrapper(out, function)
        return out

    def __call__(self, *args, **kwargs) -> Y:
        return self.function(*args, **kwargs)

    def __repr__(self):
        return self


class FuncEnum(StrEnum):
    """Creating a Enum of functions identifiable by a string."""

    @staticmethod
    def wrap(function: Callable[[X, ...], Y], name: str = None) -> wrapper[X, Y]:
        """Function wrapper: class functions are seen as methods and str conversion."""
        return wrapper(function, name)

    def __call__(self, *args, **kwargs) -> Y:
        """Redirecting the function call."""
        return self.value(*args, **kwargs)


class MeanStdTime:
    """Formats Mean and standard time."""

    def __init__(self, input_data: list[float], elapsed_time: float = 0.0):
        self.mean = np.mean(input_data)
        self.std = np.std(input_data, ddof=1)
        self.avg_time = elapsed_time / len(input_data)

    def __repr__(self):
        """1e5 since it's rough Order of Magnitude # of fittings for Data Evaluators."""
        return (
            f"mean: {self.mean} | std: {self.std} | "
            f"average_time: {timedelta(seconds=self.avg_time)} | "
            f"1e5 time: {timedelta(seconds=1e5*self.avg_time)}"
        )


class ParamSweep:
    def __init__(self, pred_model, evaluator, fetcher, samples: int = 10):
        self.model = pred_model
        self.x_train, self.y_train, self.x_valid, self.y_valid, *_ = fetcher.datapoints
        self.evaluator = evaluator
        self.samples = samples

    def sweep(self, **kwargs_list) -> dict[str, MeanStdTime]:
        self.result = {}
        for kwargs in self._param_product(**kwargs_list):
            perf_list = []
            start_time = time.perf_counter()

            for _ in tqdm.trange(self.samples):
                curr_model = self.model.clone()
                curr_model.fit(self.x_train, self.y_train, **kwargs)
                yhat = curr_model.predict(self.x_valid).cpu()
                perf = self.evaluator(yhat, self.y_valid)
                perf_list.append(perf)

            end_time = time.perf_counter()
            self.result[str(kwargs)] = MeanStdTime(perf_list, end_time - start_time)
        return self.result

    @staticmethod
    def _param_product(**kwarg_list):
        keys = kwarg_list.keys()
        for instance in itertools.product(*kwarg_list.values()):
            yield dict(zip(keys, instance))
