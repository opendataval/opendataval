"""
This files is built off on sklearn https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef/sklearn/ensemble/_bagging.py
"""
import itertools
from warnings import  warn
import numpy as np
from joblib import Parallel
from sklearn.utils.parallel import delayed
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state

from sklearn.linear_model import LassoCV
import torch
import copy
from dataoob.model import Model
from dataoob.dataval import DataEvaluator
from torch.utils.data import Subset
MAX_INT = np.iinfo(np.int32).max



class AME:
    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        num_models: int=10,
        n_jobs=None,
        estimators: list=None,
        random_state: np.random.RandomState=None,
    ):
        self.bagging_kwargs = {  # TODO clean up the AME thing we got going
            'pred_model': pred_model,
            'metric': metric,
            'num_models': num_models,
            'n_jobs':n_jobs,
            'estimators': estimators,
            'random_state': random_state,
        }
    def input_data(
        self,
        *args, **kwargs
    ):
        self.inpargs = args
        self.inp = {**kwargs}

    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        select, dve = [], []
        for max_sample in [0.2, 0.4, 0.6, 0.8]:
            ame = BaggingEvaluator(max_samples=max_sample, **self.bagging_kwargs)
            ame.input_data(*self.inpargs, **self.inp)
            ame.train_data_values(batch_size, epochs)
            estimates, dv = ame.evaluate_data_values()

            select.append(estimates)
            dve.append(dv)


        self.select = np.vstack(select)
        self.dve = np.vstack(dve).reshape(-1)
        print(self.select)

    def evaluate_data_values(self):
        # normalize X and y
        self.select = (  # This is likely incrrect, will probs have to [0-1] it
            (self.select.T - np.mean(self.select, axis=1))
            / (np.mean(self.select, axis=1) * (1 - np.mean(self.select, axis=1)))
        ).T
        self.dve = self.dve - np.mean(self.dve)

        dv_ame = LassoCV()
        dv_ame.fit(X=self.select, y=self.dve)
        return dv_ame.coef_





class BaggingEvaluator(DataEvaluator):
    def __init__(
        self,
        pred_model: Model,
        metric: callable,
        num_models: int=10,
        max_samples=1.0,
        n_jobs=None,
        estimators: list=None,
        random_state: np.random.RandomState=None,
    ):
        self.pred_model = copy.deepcopy(pred_model)
        self.metric = metric

        self.num_models = num_models
        self.max_samples = max_samples
        self.n_jobs = n_jobs

        self.warm_start = False
        self.estimators = estimators if estimators is not None else None

        self.random_state = check_random_state(random_state)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        """Stores and transforms input data for DVRL

        :param torch.Tensor x_train: Data covariates
        :param torch.Tensor y_train: Data labels
        :param torch.Tensor x_valid: Test+Held-out covariates
        :param torch.Tensor y_valid: Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_samples = len(x_train)
        self.max_features=108
        self.max_samples = self.num_samples


    def train_data_values(self, batch_size: int = 32, epochs: int = 1):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.
        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        check_input : bool, default=True
            Override value used when fitting base estimator. Only supported
            if the base estimator has a check_input parameter for fit function.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Free allocated memory, if any
        if not self.warm_start:
            self.estimators = []
            self.selected_datapoints = []
            self.estimators_features = []

        n_more_estimators = self.num_models - len(self.estimators)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.num_models, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop can easily remove, not absolutely necessary
        n_jobs, num_estimators, starts = _partition_estimators(n_more_estimators, self.n_jobs)
        total_n_estimators = sum(num_estimators)

        seeds = self.random_state.randint(MAX_INT, size=n_more_estimators)

        parallel_pipeline = Parallel(n_jobs=n_jobs, verbose=True)
        scheduled_jobs = (
            delayed(self._parallel_build_estimators)(  # Necessary cause of 2 year old bug
                num_estimators[i],
                seeds[starts[i] : starts[i + 1]],
                total_n_estimators,
                epochs,
                batch_size
            ) for i in range(n_jobs)
        )
        all_results = parallel_pipeline(scheduled_jobs)

        self.estimators += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )

        self.selected_datapoints += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        return self

    def evaluate_data_values(self):
        """
        # TODO talk to prof about how we can reformat this to be a little nicer
        With the held-out data (X_val, y_val), the performance of a model trained on a bootstrapped dataset is evaluated
        """

        self._ensemble_y = []
        for weak_learner in self.estimators:
            y_val_pred = weak_learner.predict(self.x_valid)
            self._ensemble_y.append(self.evaluate(self.y_valid, y_val_pred))

        return np.array(self.selected_datapoints), np.array(self._ensemble_y)



    def _parallel_build_estimators(
        self,
        n_estimators,
        seeds,
        total_n_estimators,
        batch_size,
        epochs,
    ):
        """Private function used to build a batch of estimators within a job."""
        estimators, selected_datapoints = [], []
        for i in range(n_estimators):
            print(
                "Building estimator %d of %d for this parallel run (total %d)..."
                % (i + 1, n_estimators, total_n_estimators)
            )

            # Draw random feature, sample indices
            subset = check_random_state(seeds[i]).choice(self.max_samples, (self.num_samples,))

            # Draw samples, using sample weights, and then fit
            curr_model = copy.deepcopy(self.pred_model)
            curr_model.fit(
                Subset(self.x_train, indices=subset),
                Subset(self.y_train, indices=subset),
                batch_size=batch_size,
                epochs=epochs,
            )

            estimators.append(curr_model)
            selected = np.zeros(self.num_samples)
            selected[subset] = 1
            selected_datapoints.append(selected)

        return estimators, selected_datapoints
