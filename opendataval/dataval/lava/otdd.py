"""Main module for computing exact wasserstein distance between two data sets.

`OTDD Repository <https://github.com/microsoft/otdd>`_.

References
----------
    .. [1] D. Alvarez-Melis and N. Fusi,
        Geometric Dataset Distances via Optimal Transport,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2002.02923.
    .. [2] D. Alvarez-Melis and N. Fusi,
        Dataset Dynamics via Gradient Flows in Probability Space,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2010.12760.
    .. [3] `OTDD repo <https://github.com/microsoft/otdd>`_.
        The following implementation was taken from this repository. It is intended as a
        strict subset of the options provided in the repository, only computing the
        class-wise Wasserstein as needed by the LAVA Paper by H.A. Just et al.

Legacy notation:
    X1, X2: feature tensors of the two datasets
    Y1, Y2: label tensors of the two datasets
    N1, N2 (or N,M): number of samples in datasets
    D1, D2: (feature) dimension of the datasets
    C1, C2: number of classes in the datasets
"""
## Local Imports
import itertools
from functools import partial
from typing import Callable, Literal, Optional, Union

import geomloss
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from opendataval.dataloader import CatDataset

cost_routines = {
    1: geomloss.utils.distances,
    2: lambda x, y: geomloss.utils.squared_distances(x, y) / 2,
}


class DatasetDistance:
    """The main class for the Optimal Transport Dataset Distance.

    An object of this class is instantiated with two datasets (the source and
    target), which are stored in it, and various arguments determining how the
    exact Wasserstein distance is to be computed.

    Parameters
    ----------
    x_train : torch.Tensor
        Covariates of the first distribution
    y_train : torch.Tensor
        Labels of the first distribution
    x_valid : torch.Tensor
        Covariates of the second/validation distribution
    y_valid : torch.Tensor
        Labels of the second/validation distribution
    feature_cost : Literal["euclidean"] | Callable, optional
        If not 'euclidean', must be a callable that implements a cost function
        between feature vectors, by default "euclidean"
    p : int, optional
        The coefficient in the OT cost (i.e., the p in p-Wasserstein), by default 2
    entreg : float, optional
        The strength of entropy regularization for sinkhorn, by default 0.1
    lam_x : float, optional
        Weight parameter for feature component of distance, by default 1.0
    lam_y : float, optional
        Weight parameter for label component of distance.=, by default 1.0
    inner_ot_loss : str, optional
        Loss type to exact OT problem, by default "sinkhorn"
    inner_ot_debiased : bool, optional
        Whether to use the debiased version of sinkhorn in the inner OT problem,
        by default False
    inner_ot_p : int, optional
        The coefficient in the inner OT cost., by default 2
    inner_ot_entreg : float, optional
        The strength of entropy regularization for sinkhorn in the inner OT problem,
        by default 0.1
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")
    """

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
        feature_cost: Union[
            Literal["euclidean"], Callable[..., torch.Tensor]
        ] = "euclidean",
        p: int = 2,
        entreg: float = 0.1,
        lam_x: float = 1.0,
        lam_y: float = 1.0,
        ## Inner OT (label to label) problem arguments
        inner_ot_loss: str = "sinkhorn",
        inner_ot_debiased: bool = False,
        inner_ot_p: int = 2,
        inner_ot_entreg: float = 0.1,
        ## Misc
        device: torch.device = torch.device("cpu"),
    ):
        self.feature_cost = feature_cost
        self.inner_ot_loss = inner_ot_loss
        ## For outer OT problem
        self.p = p
        self.entreg = entreg
        self.lam_x = lam_x
        self.lam_y = lam_y
        ## For inner (label) OT problem - only used if gaussian approx is False
        self.inner_ot_p = inner_ot_p
        self.inner_ot_entreg = inner_ot_entreg
        self.inner_ot_debiased = inner_ot_debiased

        self.device = device

        [*self.covar_dim] = x_train[0].shape  # Syntax for unpacking tensor shapes
        [*self.label_dim] = (1,) if y_valid.ndim == 1 else y_train.shape[1:]
        self.label_distances = None

        self.x_train, self.y_train = extract_dataset(x_train, y_train)
        self.x_valid, self.y_valid = extract_dataset(
            x_valid, y_valid, reindex_start=np.prod(*self.label_dim)
        )
        self.num_train, self.num_valid = len(y_train), len(y_valid)

    def _get_label_distances(self) -> torch.Tensor:
        """Precompute label-to-label distances.

        Returns tensor of size nclasses_1 x nclasses_2. DISTANCE BETWEEN LABEL IN D1 AND
        LABEL IN D2!

        Returns
        -------
        torch.Tensor
            tensor of size (C1, C2) with pairwise label-to-label distances across
            the two datasets.
        """
        ## Check if already computed
        if self.label_distances is not None:
            return self.label_distances

        # exact way of computing
        # We just define a function ahead, before loading real data
        # pwdist_exact From Geomloss defined function
        pwdist = partial(
            pwdist_exact,
            symmetric=False,
            p=self.inner_ot_p,
            loss=self.inner_ot_loss,
            debias=self.inner_ot_debiased,
            entreg=self.inner_ot_entreg,
            cost_function=self.feature_cost,
            device=self.device,
        )

        ## Then we also need within-collection label distances
        DYY1 = pwdist(self.x_train, self.y_train)
        DYY2 = pwdist(self.x_valid, self.y_valid)
        DYY12 = pwdist(self.x_train, self.y_train, self.x_valid, self.y_valid)

        D = torch.cat([torch.cat([DYY1, DYY12], 1), torch.cat([DYY12.t(), DYY2], 1)])

        ## Collect and save
        self.label_distances = D

        return self.label_distances

    def dual_sol(self) -> tuple[float, torch.Tensor]:
        """Compute dataset distance.

        Note:
            Currently requires fully loading dataset into memory, this can probably be
            avoided, e.g., via subsampling.

        Returns
        -------
        tuple[float, torch.Tensor]
            dist (float): the optimal transport dataset distance value.
            pi (tensor, optional): the optimal transport coupling.
        """
        wasserstein = self._get_label_distances().to(self.device)

        ## This one leverages precomputed pairwise label distances
        cost_geomloss = partial(
            batch_augmented_cost,
            W=wasserstein,
            lam_x=self.lam_x,
            lam_y=self.lam_y,
            feature_cost=self.feature_cost,
        )

        loss = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=self.p,
            cost=cost_geomloss,
            debias=True,
            blur=self.entreg ** (1 / self.p),
            backend="tensorized",
        )

        Z1 = torch.cat((self.x_train, self.y_train.float().unsqueeze(dim=1)), -1)
        Z2 = torch.cat((self.x_valid, self.y_valid.float().unsqueeze(dim=1)), -1)

        with torch.no_grad():
            loss.debias = False
            loss.potentials = True
            torch.cuda.empty_cache()

            F_i, G_j = loss(Z1.to(self.device), Z2.to(self.device))
            pi = [F_i, G_j]

        del Z1, Z2
        torch.cuda.empty_cache()

        return pi


class FeatureCost:
    """Class implementing a cost (or distance) between feature vectors.

    Arguments:
        p (int): the coefficient in the OT cost (i.e., the p in p-Wasserstein).
        src_embedding (callable, optional): if provided, source data will be
            embedded using this function prior to distance computation.
        tgt_embedding (callable, optional): if provided, target data will be
            embedded using this function prior to distance computation.

    """

    def __init__(
        self,
        src_embedding=None,
        tgt_embedding=None,
        src_dim=None,
        tgt_dim=None,
        p=2,
        device="cpu",
    ):
        assert (src_embedding is None) or (src_dim is not None)
        assert (tgt_embedding is None) or (tgt_dim is not None)
        self.src_emb = src_embedding
        self.tgt_emb = tgt_embedding
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.p = p
        self.device = device

    def _get_batch_shape(self, b):
        if b.ndim == 3:
            return b.shape
        elif b.ndim == 2:
            return (1, *b.shape)
        elif b.ndim == 1:
            return (1, 1, b.shape[0])

    def _batchify_computation(self, X, side="x", slices=20):
        embed = self.src_emb if side == "x" else self.tgt_emb
        out = torch.cat(embed(b).to(self.device) for b in torch.chunk(X, slices, dim=0))
        return out.to(X.device)

    def __call__(self, X1, X2):
        if self.src_emb is not None:
            B1, N1, _ = self._get_batch_shape(X1)
            self.src_emb = self.src_emb.to(self.device)
            X1 = self.src_emb(X1.view(-1, *self.src_dim)).reshape(B1, N1, -1)
        if self.tgt_emb is not None:
            B2, N2, _ = self._get_batch_shape(X2)
            self.tgt_emb = self.tgt_emb.to(self.device)
            X2 = self.tgt_emb(X2.view(-1, *self.tgt_dim)).reshape(B2, N2, -1)

        return cost_routines[self.p](X1, X2)


def extract_dataset(
    x_input: torch.Tensor,
    y_input: torch.Tensor,
    batch_size: int = 256,
    reindex_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads full dataset into memory and reindexes the labels.

    Parameters
    ----------
    x_input : Dataset | torch.Tensor
        Covariate Dataset/tensor to be loaded
    y_input : Dataset | torch.Tensor
        Label Dataset/tensor to be loaded
    batch_size : int, optional
        Batch size of data to be loaded at a time, by default 256
    reindex_start : int, optional
        How much to offset the labels by, useful when comparing different
        data sets so that the data have different labels, by default 0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        **x_tensor** Covariates stacked along first dimension
        **y_tensor** Labels, no longer in one-hot-encoding and offset by reindex_start
    """
    loader = DataLoader(CatDataset(x_input, y_input), batch_size=batch_size)

    x_list = []
    y_list = []

    for x, y in tqdm.tqdm(loader, leave=False):
        x_list.append(x.squeeze().view(x.shape[0], -1))
        y_list.append(y.argmax(dim=1).squeeze())

    x_tensor = torch.cat(x_list)
    y_tensor = torch.cat(y_list)

    return x_tensor, y_tensor + reindex_start


def pwdist_exact(
    X1: torch.Tensor,
    Y1: torch.Tensor,
    X2: torch.Tensor = None,
    Y2: torch.Tensor = None,
    symmetric: bool = False,
    loss: str = "sinkhorn",
    cost_function: Union[
        Literal["euclidean"], Callable[..., torch.Tensor]
    ] = "euclidean",
    p: int = 2,
    debias: bool = True,
    entreg: float = 1e-1,
    device: torch.device = torch.device("cpu"),
):
    """Computation of pairwise Wasserstein distances.

    Efficient computation of pairwise label-to-label Wasserstein distances
    between multiple distributions, without using Gaussian assumption.

    Parameters
    ----------
    X1 : torch.Tensor
        Covariates of first distribution
    Y1 : torch.Tensor
        Labels of first distribution
    X2 : torch.Tensor, optional
        Covariates of second distribution, if None distributions are treated as same,
        by default None
    Y2 : torch.Tensor, optional
        Labels of second distribution, iif None distributions are treated as same,
        by default None
    symmetric : bool, optional
        Whether X1/Y1 and X2/Y2 are to be treated as the same dataset, by default False
    loss : str, optional
        The loss function to compute.  Sinkhorn divergence, which interpolates between
        (blur=0) and kernel (blur= :math:`+\infty` ) distances., by default "sinkhorn"
    cost_function : : Literal["euclidean"] | Callable[..., torch.Tensor], optional
        Cost function that should be used instead of :math:`\\tfrac{1}{p}\|x-y\|^p`,
        by default "euclidean"
    p : int, optional
        power of the cost (i.e. order of p-Wasserstein distance), by default 2
    debias : bool, optional
        If true, uses debiased sinkhorn divergence., by default True
    entreg : float, optional
        The strength of entropy regularization for sinkhorn., by default 1e-1
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")

    Returns
    -------
    torch.Tensor
        Computed Wasserstein distance
    """
    if X2 is None:  # If not specified, assume symmetric
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    ## We account for the possibility that labels are shifted (c1[0]!=0), see below
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are asymmetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    if cost_function == "euclidean":
        cost_function = cost_routines[p]

    distance = geomloss.SamplesLoss(
        loss=loss,
        p=p,
        cost=cost_function,
        debias=debias,
        blur=entreg ** (1 / p),
    )

    D = torch.zeros((n1, n2), device=device, dtype=X1.dtype)
    for i, j in tqdm.tqdm(pairs, leave=False, desc="Computing label-to-label distance"):
        m1 = X1[Y1 == c1[i]].to(device)
        m2 = X2[Y2 == c2[j]].to(device)

        D[i, j] = distance(m1, m2).item()

        if symmetric:
            D[j, i] = D[i, j]

    return D


def batch_augmented_cost(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    W: torch.Tensor = None,
    feature_cost: Optional[str] = None,
    p: int = 2,
    lam_x: float = 1.0,
    lam_y: float = 1.0,
):
    """Batch ground cost computation on augmented datasets.

    Parameters
    ----------
    Z1 : torch.Tensor
        Tensor of size (B,N,D1), where last position in last dim corresponds to label Y.
    Z2 : torch.Tensor
        Tensor of size (B,M,D2), where last position in last dim corresponds to label Y.
    W : torch.Tensor, optional
        Tensor of size (V1,V2) of precomputed pairwise label distances for all labels
        V1,V2 and returns a batched cost matrix as a (B,N,M) Tensor. W is expected to be
        congruent with p. I.e, if p=2, W[i,j] should be squared Wasserstein distance.,
        by default None
    feature_cost : str, optional
        if None or 'euclidean', uses euclidean distances as feature metric,
        otherwise uses this function as metric., by default None
    p : int, optional
        Power of the cost (i.e. order of p-Wasserstein distance), by default 2
    lam_x : float, optional
        Weight parameter for feature component of distance, by default 1.0
    lam_y : float, optional
        Weight parameter for label component of distance, by default 1.0

    Returns
    -------
    torch.Tensor
        torch Tensor of size (B,N,M)

    Raises
    ------
    ValueError
        If W is not provided
    """
    _, _, D1 = Z1.shape
    _, M, D2 = Z2.shape
    assert (D1 == D2) or (feature_cost is not None)

    Y1 = Z1[:, :, -1].long()
    Y2 = Z2[:, :, -1].long()

    if feature_cost is None or feature_cost == "euclidean":  # default is euclidean
        C1 = cost_routines[p](Z1[:, :, :-1], Z2[:, :, :-1])  # Get from GeomLoss
    else:
        C1 = feature_cost(Z1[:, :, :-1], Z2[:, :, :-1])  # Feature Embedding

    # Label Distances
    if W is not None:
        ## Label-to-label distances have been precomputed and passed
        ## Stores flattened index corresponoding to label pairs
        M = W.shape[1] * Y1[:, :, None] + Y2[:, None, :]
        C2 = W.flatten()[M.flatten(start_dim=1)].reshape(-1, Y1.shape[1], Y2.shape[1])
    else:
        raise ValueError("Must provide either label distances or Means+Covs")

    assert C1.shape == C2.shape

    ## NOTE: geomloss's cost_routines as defined above already divide by p. We do
    ## so here too for consistency. But as a consequence, need to divide C2 by p too.
    D = lam_x * C1 + lam_y * (C2 / p)

    return D
