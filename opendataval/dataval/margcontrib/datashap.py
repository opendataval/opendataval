from opendataval.dataval.margcontrib.shap import ShapEvaluator


class DataShapley(ShapEvaluator):
    """Data Shapley implementation.

    References
    ----------
    .. [1] A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    Parameters
    ----------
    sampler : Sampler, optional
        Sampler used to compute the marginal contributions. Can be found in
        :py:mod:`~opendataval.margcontrib.sampler`, by default uses *args, **kwargs for
        :py:class:`~opendataval.dataval.margcontrib.sampler.GrTMCSampler`.
    """

    def compute_weight(self) -> float:
        """Compute weights (uniform) for each cardinality of training set.

        Shapley values take a simple average of the marginal contributions across
        all different cardinalities.

        Returns
        -------
        np.ndarray
            Weights by cardinality of subset
        """
        return 1 / self.num_points
