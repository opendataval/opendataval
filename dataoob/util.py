import pandas as pd
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state


def load_mediator_output(file_path: str):
    return pd.read_csv(file_path, index_col=[0, 1])


def set_random_state(random_state: RandomState = None) -> RandomState:
    """Set the random state of dataoob, useful for recreation of results."""
    print(f"Initial random seed is: {random_state}.")
    random_state = check_random_state(random_state)
    torch.manual_seed(random_state.tomaxint())
    return random_state
