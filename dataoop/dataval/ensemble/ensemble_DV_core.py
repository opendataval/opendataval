"""
This files is built off on sklearn https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef/sklearn/ensemble/_forest.py
"""
import numbers
from warnings import catch_warnings, simplefilter, warn
import threading
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel
from sklearn.base import is_classifier
from sklearn.base import (
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, compute_sample_weight, deprecated
from sklearn.exceptions import DataConversionWarning

# from ._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    _check_feature_names_in,
)
from sklearn.utils.validation import _num_samples
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.ensemble import GradientBoostingClassifier


def _get_n_samples_bootstrap(n_samples, max_samples):
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples <= 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return round(n_samples * max_samples)

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)

        return (tree, curr_sample_weight)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

        return (tree, None)


def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices


class RandomForestClassifierDV(RandomForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X, y, sample_weight=None):
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        # Check parameters
        self._validate_estimator()
        # TODO(1.2): Remove "mse" and "mae"
        if isinstance(self, (RandomForestRegressor, ExtraTreesRegressor)):
            if self.criterion == "mse":
                warn(
                    "Criterion 'mse' was deprecated in v1.0 and will be "
                    "removed in version 1.2. Use `criterion='squared_error'` "
                    "which is equivalent.",
                    FutureWarning,
                )
            elif self.criterion == "mae":
                warn(
                    "Criterion 'mae' was deprecated in v1.0 and will be "
                    "removed in version 1.2. Use `criterion='absolute_error'` "
                    "which is equivalent.",
                    FutureWarning,
                )

            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features=1.0` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestRegressors and ExtraTreesRegressors.",
                    FutureWarning,
                )
        elif isinstance(self, (RandomForestClassifier, ExtraTreesClassifier)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features='sqrt'` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestClassifiers and ExtraTreesClassifiers.",
                    FutureWarning,
                )

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees_all = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            trees = [trees_tmp for (trees_tmp, bincounts_tmp) in trees_all]
            bincounts = [bincounts_tmp for (trees_tmp, bincounts_tmp) in trees_all]

            # Collect newly grown trees
            self.estimators_.extend(trees)

            # Create DV data
            self._ensemble_X = bincounts

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def evaluate_importance(self, X_val, y_val, score="acc"):
        """
        With the held-out data (X_val, y_val), the performance of a model trained on a bootstrapped dataset is evaluated
        """
        assert (
            len(self.estimators_) != 0
        ), "Run fit first. self.estimators_ is not defined"

        self._ensemble_y = []
        if score == "acc":
            for weak_learner in self.estimators_:
                y_val_pred = weak_learner.predict(X_val)
                self._ensemble_y.append(accuracy_score(y_val, y_val_pred))
        elif score == "logit":
            for weak_learner in self.estimators_:
                y_val_pred = np.log(weak_learner.predict_proba(X_val) + 1e-16)
                logit_list = (y_val_pred[:, 1] - y_val_pred[:, 0]) * (2 * y_val - 1)
                self._ensemble_y.append(np.mean(logit_list))
        elif score == "pred":
            for weak_learner in self.estimators_:
                y_val_pred = weak_learner.predict_proba(X_val)
                pred_list = y_val_pred[:, 1]
                self._ensemble_y.append(np.mean(pred_list))
        else:
            raise NotImplementedError("Check a score parameter")

        return np.array(self._ensemble_X), np.array(self._ensemble_y)

    def evaluate_oob_accuracy(self, X, y):
        import pandas as pd

        assert (
            len(self.estimators_) != 0
        ), "Run fit first. self.estimators_ is not defined"
        assert (
            len(self._ensemble_X) != 0
        ), "Run evaluate_importance first. self._ensemble_X is not defined"

        oob_performance = []
        for i, weak_learner in enumerate(self.estimators_):
            oob_ind = np.where(self._ensemble_X[i] == 0)[0]
            oob_acc = (weak_learner.predict(X[oob_ind]) == y[oob_ind]).astype(float)
            oob_performance.append({oob_ind[ind]: j for ind, j in enumerate(oob_acc)})
        df_oob = pd.DataFrame(oob_performance)[np.arange(len(X))]

        return np.mean(df_oob, axis=0)


class RandomForestRegressorDV(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X, y, sample_weight=None):
        self._validate_params()

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._validate_estimator()
        if isinstance(self, (RandomForestRegressor, ExtraTreesRegressor)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features=1.0` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestRegressors and ExtraTreesRegressors.",
                    FutureWarning,
                )
        elif isinstance(self, (RandomForestClassifier, ExtraTreesClassifier)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features='sqrt'` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestClassifiers and ExtraTreesClassifiers.",
                    FutureWarning,
                )

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees_all = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            trees = [trees_tmp for (trees_tmp, bincounts_tmp) in trees_all]
            bincounts = [bincounts_tmp for (trees_tmp, bincounts_tmp) in trees_all]

            # Collect newly grown trees
            self.estimators_.extend(trees)

            # Create DV data
            self._ensemble_X = bincounts

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def evaluate_oob_accuracy(self, X, y):
        import pandas as pd

        assert (
            len(self.estimators_) != 0
        ), "Run fit first. self.estimators_ is not defined"
        assert (
            len(self._ensemble_X) != 0
        ), "Run evaluate_importance first. self._ensemble_X is not defined"

        oob_performance = []
        for i, weak_learner in enumerate(self.estimators_):
            oob_ind = np.where(self._ensemble_X[i] == 0)[0]
            oob_acc = -((weak_learner.predict(X[oob_ind]) - y[oob_ind]) ** 2)
            oob_performance.append({oob_ind[ind]: j for ind, j in enumerate(oob_acc)})
        df_oob = pd.DataFrame(oob_performance)[np.arange(len(X))]

        return np.mean(df_oob, axis=0)
