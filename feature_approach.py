import warnings
from collections import defaultdict
from time import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dataoop.data_val.bagging.bagging_DV_core import (
    BaggingClassifierDV,
    BaggingRegressorDV,
)
from dataoop.data_val.ensemble.ensemble_DV_core import (
    RandomForestClassifierDV,
    RandomForestRegressorDV,
)

warnings.filterwarnings("ignore")


class FeatureApproach(object):
    def __init__(self, X, y, X_val, y_val, problem, model_family, n_trees):
        """
        Args:
            (X,y): (inputs,outputs) to be valued.
            (X_val,y_val): (inputs,outputs) to be used for utility evaluation.
            problem: "clf"
            model_family: The model family used for learning algorithm
            GR_threshold: Gelman-Rubin threshold for convergence criteria
            max_iters: maximum number of iterations (for a fixed cardinality)
        """
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.problem = problem
        self.model_family = model_family
        self.n_trees = n_trees
        self._initialize_instance()

    def _initialize_instance(self):
        # create placeholders
        self.data_value_dict = defaultdict(list)
        self.time_dict = defaultdict(list)

    def run(
        self,
        AME_run=True,
        lasso_run=True,
        boosting_run=True,
        treeshap_run=True,
        bagging_run=True,
        simple_run=False,
    ):
        self._calculate_proposed()
        if AME_run is True:
            self._calculate_AME()

    def _calculate_AME(self):
        print(f"Start: AME computation")
        # fit AME model
        time_init = time()
        X_dv_ame_list, y_dv_ame_list = [], []
        N_to_be_valued = len(self.y)
        if self.problem == "clf":
            for max_sample in [0.2, 0.4, 0.6, 0.8]:
                AME_clf = BaggingClassifierDV(
                    n_estimators=(self.n_trees // 4),
                    estimator=DecisionTreeClassifier(),
                    max_samples=max_sample,
                    bootstrap=False,
                    n_jobs=-1,
                )
                AME_clf.fit(self.X, self.y)

                # create the data_valuation dataset
                X_dv_ame, y_dv_ame = AME_clf.evaluate_importance(self.X_val, self.y_val)
                X_dv_ame_list.append(X_dv_ame)
                y_dv_ame_list.append(y_dv_ame)
        else:
            for max_sample in [0.2, 0.4, 0.6, 0.8]:
                AME_model = BaggingRegressorDV(
                    n_estimators=(self.n_trees // 4),
                    estimator=DecisionTreeRegressor(),
                    max_samples=max_sample,
                    bootstrap=False,
                    n_jobs=-1,
                )
                AME_model.fit(self.X, self.y)

                # create the data_valuation dataset
                X_dv_ame, y_dv_ame = AME_model.evaluate_importance(
                    self.X_val, self.y_val
                )
                X_dv_ame_list.append(X_dv_ame)
                y_dv_ame_list.append(y_dv_ame)

        X_dv_ame_list = np.vstack(X_dv_ame_list)
        y_dv_ame_list = np.vstack(y_dv_ame_list).reshape(-1)

        # normalize X and y
        X_dv_ame_list = (
            (X_dv_ame_list.T - np.mean(X_dv_ame_list, axis=1))
            / (np.mean(X_dv_ame_list, axis=1) * (1 - np.mean(X_dv_ame_list, axis=1)))
        ).T
        y_dv_ame_list = y_dv_ame_list - np.mean(y_dv_ame_list)

        dv_ame = LassoCV()
        dv_ame.fit(X=X_dv_ame_list, y=y_dv_ame_list)
        self.data_value_dict["AME"] = dv_ame.coef_
        self.time_dict["AME"] = time() - time_init
        print(f"Done: AME computation")

    def _calculate_proposed(self):
        print(f"Start: OOB computation")
        # fit a random forest model
        time_init = time()
        if self.problem == "clf":
            self.rf_model = RandomForestClassifierDV(
                n_estimators=self.n_trees, n_jobs=-1
            )
        else:
            self.rf_model = RandomForestRegressorDV(
                n_estimators=self.n_trees, n_jobs=-1
            )
        self.rf_model.fit(self.X, self.y)
        self.time_dict["RF_fitting"] = time() - time_init
        self.data_value_dict["OOB"] = (
            self.rf_model.evaluate_oob_accuracy(self.X, self.y)
        ).to_numpy()
        self.time_dict["OOB"] = time() - time_init
        print(f"Done: OOB computation")
