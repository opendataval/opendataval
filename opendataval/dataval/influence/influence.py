import numpy as np
import torch
import tqdm
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Subset

from opendataval.dataval.api import DataEvaluator


class InfluenceFunctionEval(DataEvaluator):
    """Influence Function Data Evaluation implementation.

    Compute influence of each training example on the accuracy at each test example
    through closely-related subsampled influence.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    samples : int, optional
        Number of models to fit to take to find data values, by default 1000
    proportion : float, optional
        Proportion of data points to be in each sample, cardinality of each subset is
        :math:`(p)(num_points)`, by default 0.7 as specified by V. Feldman and C. Zhang
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        num_models: int = 1000,
        proportion: float = 0.7,
        random_state: RandomState = None,
    ):
        self.num_models = num_models
        self.proportion = proportion
        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Store and transform input data for Influence Function Data Valuation.

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
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        # [:, 1] represents included, [:, 0] represents excluded for following arrays
        self.influence_matrix = np.zeros(shape=(self.num_points, 2))
        self.sample_counts = np.zeros(shape=(self.num_points, 2))
        return self

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Trains the Influence Function Data Valuator by sampling from subsets of
        :math:`(p)(num_points)` cardinality and computing the performance with the
        :math:`i` data point and without the :math:`i` data point. The form of sampling
        is similar to the shapely value when :math:`p` is :math:`0.5: (V. Feldman).
        Likewise, if we sample not from the subsets of a specific cardinality but the
        uniform across all subsets, it is similar to the Banzhaf value.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        for i in tqdm.tqdm(range(self.num_models)):
            subset = self.random_state.choice(
                self.num_points, round(self.proportion * self.num_points), replace=False
            )  # Random subset of cardinality `round(self.proportion * self.num_points)`

            curr_model = self.pred_model.clone()
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                *args,
                **kwargs
            )
            y_valid_hat = curr_model.predict(self.x_valid)
            curr_perf = self.evaluate(self.y_valid, y_valid_hat)

            included = (np.bincount(subset, minlength=self.num_points) != 0).astype(int)
            self.influence_matrix[range(self.num_points), included] += curr_perf
            self.sample_counts[range(self.num_points), included] += 1

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using the Influence Function data valuator. Finds
        the difference of average performance of all sets including data point minus
        not-including.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        msr = np.divide(
            self.influence_matrix,
            self.sample_counts,
            out=np.zeros_like(self.influence_matrix),
            where=self.sample_counts != 0,
        )
        return msr[:, 1] - msr[:, 0]  # Diff of subsets including/excluding i data point
