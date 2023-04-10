import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state


def set_random_state(random_state: RandomState = None) -> RandomState:
    """Sets the random state of dataoob, useful for recreation of results."""
    print(f"Initial random seed is: {random_state}.")
    random_state = check_random_state(random_state)
    torch.manual_seed(random_state.tomaxint())
    return random_state
