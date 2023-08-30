from typing import Optional

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state

from opendataval.dataval.api import DataEvaluator, EmbeddingMixin
from opendataval.dataval.lava.otdd import DatasetDistance, FeatureCost
from opendataval.model import Model


def macos_fix():
    """Geomloss package has a bug on MacOS remedied as follows.

    `Link to similar bug: https://github.com/NVlabs/stylegan3/issues/75`_.
    """
    import os
    import sys

    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class LavaEvaluator(DataEvaluator, EmbeddingMixin):
    """Data valuation using LAVA implementation.

    References
    ----------
    .. [1] H. A. Just, F. Kang, T. Wang, Y. Zeng, M. Ko, M. Jin, and R. Jia,
        LAVA: Data Valuation without Pre-Specified Learning Algorithms,
        2023. Available: https://openreview.net/forum?id=JJuP86nBl4q

    Parameters
    ----------
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")
    random_state: RandomState, optional
        Random initial state, by default None

    Mixins
    ------
    EmbeddingMixin
        Mixin for a data evaluator to possibly use an embedding model.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        embedding_model: Optional[Model] = None,
        random_state: RandomState = None,
    ):
        macos_fix()
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.embedding_model = embedding_model
        self.device = device

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Computes the class-wise Wasserstein distance between the training and the
        validation set.

        References
        ----------
        .. [1] H. A. Just, F. Kang, T. Wang, Y. Zeng, M. Ko, M. Jin, and R. Jia,
            LAVA: Data Valuation without Pre-Specified Learning Algorithms,
            2023. Available: https://openreview.net/forum?id=JJuP86nBl4q
        .. [2] D. Alvarez-Melis and N. Fusi,
            Geometric Dataset Distances via Optimal Transport,
            arXiv.org, 2020. Available: https://arxiv.org/abs/2002.02923.
        .. [3] D. Alvarez-Melis and N. Fusi,
            Dataset Dynamics via Gradient Flows in Probability Space,
            arXiv.org, 2020. Available: https://arxiv.org/abs/2010.12760.
        """
        feature_cost = None

        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            resize = 32
            feature_cost = FeatureCost(
                src_embedding=self.embedding_model,
                src_dim=(3, resize, resize),
                tgt_embedding=self.embedding_model,
                tgt_dim=(3, resize, resize),
                p=2,
                device=self.device.type,
            )

        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)
        dist = DatasetDistance(
            x_train=x_train,
            y_train=self.y_train,
            x_valid=x_valid,
            y_valid=self.y_valid,
            feature_cost=feature_cost if feature_cost else "euclidean",
            lam_x=1.0,
            lam_y=1.0,
            p=2,
            entreg=1e-1,
            device=self.device,
        )
        self.dual_sol = dist.dual_sol()

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Gets the calibrated gradient of the dual solution, which can be interpreted as
        the data values.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        f1k = self.dual_sol[0].squeeze()
        num_points = len(f1k) - 1
        train_gradient = f1k * (1 + 1 / (num_points)) - f1k.sum() / num_points

        # We multiply -1 to align LAVA with other data valuation algorithms
        # Low values should indicate detrimental data points
        train_gradient = -1 * train_gradient
        return train_gradient.numpy(force=True)
