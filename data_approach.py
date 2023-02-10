import numpy as np
from time import time
from collections import defaultdict


class DataApproach(object):
    def __init__(
        self, X, y, X_val, y_val, problem, model_family, GR_threshold=1.05, max_iters=50
    ):
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
        self.GR_threshold = GR_threshold
        self.max_iters = max_iters

        self._initialize_instance()
        self.random_score = self.init_score()
        self.model = return_model(self.model_family, problem=self.problem)

    def _initialize_instance(self):
        if self.problem == "clf":
            # classification
            self.metric = "accuracy"  # utility is fixed to classification accuracy
        else:
            # regression
            self.metric = "r2"  # utility is fixed to r2

        # Initialize sources
        self.n_points = len(self.X)
        self.sources = {i: np.array([i]) for i in range(self.n_points)}

        # create placeholders.
        self.data_value_dict = defaultdict(list)
        self.time_dict = defaultdict(list)

    def init_score(self):
        """
        Gives the utility of a random guessing model. (Best constant prediction)
        We suppose that the higher the score is, the better it is.
        """
        if self.problem == "clf":
            hist = np.bincount(self.y_val) / len(self.y_val)
            return np.max(hist)
        elif self.problem == "reg":
            # base r2 is zero
            return 0
        else:
            raise NotImplementedError("Check problem")

    def compute_utility(self, X=None, y=None):
        """
        Computes the utility of the given model
        """
        if X is None:
            X = self.X_val
        if y is None:
            y = self.y_val

        return self.model.score(X, y)

    def run(self, loo_run=True, betashap_run=True):
        """
        Calculates data values.
        Args:
            loo_run: If True, computes and saves leave-one-out (LOO) scores.
        """
        if len(self.X) <= 500000:
            self._calculate_knn()
        if loo_run is True:
            self._calculate_loo()
        if betashap_run is True and len(self.X) <= 1000:
            self._calculate_betashap()

    def _calculate_loo(self):
        """
        calculate Leave-one-out scores
        """
        print(f"Start: LOO computation")
        time_init = time()
        self.model.fit(self.X, self.y)
        baseline_value = self.compute_utility()
        vals_lld = np.zeros(self.n_points)
        for i in self.sources.keys():
            X_batch = np.delete(self.X, self.sources[i], axis=0)
            y_batch = np.delete(self.y, self.sources[i], axis=0)
            self.model.fit(X_batch, y_batch)
            removed_value = self.compute_utility()
            vals_lld[i] = baseline_value - removed_value
        self.data_value_dict["LOO_last"] = vals_lld
        self.time_dict["LOO_last"] = time() - time_init
        print(f"Done: LOO computation")

    def _calculate_knn(self, n_neighbors=None):
        """
        calculate the KNN_Shapley
        from https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py
        """
        print(f"Start: KNN_Shapley computation")
        n_neighbors = self.n_points // 10
        time_init = time()
        if self.problem == "clf":
            # classification
            n_val = len(self.X_val)
            # knn_mat=np.zeros((self.n_points, n_val))
            knn_vec = np.zeros(self.n_points)
            for i, (X_val_sample, y_val_sample) in enumerate(
                zip(self.X_val, self.y_val)
            ):
                knn_vec_tmp = np.zeros(self.n_points)
                diff = (self.X - X_val_sample).reshape(self.n_points, -1)
                dist = np.einsum("ij, ij->i", diff, diff)
                idx = np.argsort(dist)
                ans = self.y[idx]
                # knn_mat[idx[self.n_points - 1]][i] = float(ans[self.n_points - 1] == y_val_sample) / self.n_points
                knn_vec_tmp[idx[self.n_points - 1]] = (
                    float(ans[self.n_points - 1] == y_val_sample) / self.n_points
                )
                cur = self.n_points - 2
                for _ in range(self.n_points - 1):
                    const_factor = min(cur + 1, n_neighbors) / (cur + 1)
                    # knn_mat[idx[cur]][i] = knn_mat[idx[cur + 1]][i] + float(int(ans[cur] == y_val_sample) - int(ans[cur + 1] == y_val_sample)) / n_neighbors * const_factor
                    knn_vec_tmp[idx[cur]] = (
                        knn_vec_tmp[idx[cur + 1]]
                        + float(
                            int(ans[cur] == y_val_sample)
                            - int(ans[cur + 1] == y_val_sample)
                        )
                        / n_neighbors
                        * const_factor
                    )
                    cur -= 1
                knn_vec = (knn_vec * i + knn_vec_tmp) / (i + 1)
        else:
            # regression
            def generate_weight_vector(i, n, k):
                weighte_vector = np.ones(n)
                for l in range(i - 1):
                    weighte_vector[l] = min(k - 1, i - 1) / i - 1
                for l in range(i + 1, n):
                    weighte_vector[l] = (
                        min(k, l) * min(k - 1, l - 1) * i / (l * (l - 1) * min(k, i))
                    )
                return weighte_vector

            n_val = len(self.X_val)
            knn_vec = np.zeros(self.n_points)
            for i, (X_val_sample, y_val_sample) in enumerate(
                zip(self.X_val, self.y_val)
            ):
                knn_vec_tmp = np.zeros(self.n_points)
                diff = (self.X - X_val_sample).reshape(self.n_points, -1)
                dist = np.einsum("ij, ij->i", diff, diff)
                idx = np.argsort(dist)
                ans = self.y[idx]
                knn_vec_tmp[idx[self.n_points - 1]] = (
                    -((n_neighbors - 1) / (self.n_points * n_neighbors))
                    * (ans[self.n_points - 1])
                    * (
                        ans[self.n_points - 1] / n_neighbors
                        - 2 * y_val_sample
                        + (np.sum(ans) - ans[self.n_points - 1]) / (self.n_points - 1)
                    )
                    - ((ans[self.n_points - 1] / n_neighbors - y_val_sample) ** 2)
                    / self.n_points
                )
                cur = self.n_points - 2
                for _ in range(self.n_points - 1):
                    const_factor = (
                        ((ans[cur + 1] - ans[cur]) / n_neighbors)
                        * min(cur + 1, n_neighbors)
                        / (cur + 1)
                    )
                    weight_vector = generate_weight_vector(
                        cur + 1, self.n_points, n_neighbors
                    )
                    sum_part = (
                        np.sum(weight_vector * ans) / n_neighbors - 2 * y_val_sample
                    )
                    knn_vec_tmp[idx[cur]] = (
                        knn_vec_tmp[idx[cur + 1]] + sum_part * const_factor
                    )
                    cur -= 1
                knn_vec = (knn_vec * i + knn_vec_tmp) / (i + 1)
        self.data_value_dict["KNN_Shapley"] = knn_vec
        self.time_dict["KNN_Shapley"] = time() - time_init
        print(f"Done: KNN_Shapley computation")

    def _calculate_betashap(self):
        print(f"Start: Beta_Shapley computation")
        time_init = time()
        self._calculate_marginal_contributions()

        self.weight_list = self.compute_weight_list(
            N_total=self.n_points, alpha_param=1, beta_param=1
        )
        self.data_value_dict["Data_Shapley"] = np.sum(
            self.marginal_contribution * self.weight_list, axis=1
        )
        self.time_dict["Data_Shapley"] = time() - time_init

        # put more weights on marginal contributions based on small cardinalities.
        self.weight_list = self.compute_weight_list(
            N_total=self.n_points, alpha_param=16, beta_param=1
        )
        self.data_value_dict["Beta_Shapley(16,1)"] = np.sum(
            self.marginal_contribution * self.weight_list, axis=1
        )
        self.time_dict["Beta_Shapley(16,1)"] = self.time_dict["Data_Shapley"]

        self.weight_list = self.compute_weight_list(
            N_total=self.n_points, alpha_param=4, beta_param=1
        )
        self.data_value_dict["Beta_Shapley"] = np.sum(
            self.marginal_contribution * self.weight_list, axis=1
        )
        self.time_dict["Beta_Shapley"] = self.time_dict["Data_Shapley"]
        print(f"Done: Beta_Shapley computation")

    def _calculate_marginal_contributions(self):
        """
        calculate marginal contributions.
        marginal_increment_array_stack : an array of marginal increments when one data point (idx) is added.
         Average of this value is Shapley as we consider a random permutation.
        """
        print(f"Start: marginal contribution computation", flush=True)
        self.marginal_contrib_sum = np.zeros((self.n_points, self.n_points))
        self.marginal_contrib_count = np.zeros((self.n_points, self.n_points))
        self.marginal_increment_array_stack = np.zeros((0, self.n_points))
        self.GR_dict = dict()

        for iters in range(self.max_iters):
            # we check the convergence every 100 random sets.
            # we terminate iteration if Shapley value is converged.
            self.GR_dict[iters] = self.compute_GR_statistics(
                self.marginal_increment_array_stack
            )
            if self.GR_dict[iters] < self.GR_threshold:
                break
            else:
                marginal_increment_array = self._calculate_marginal_contributions_core()
                self.marginal_increment_array_stack = np.concatenate(
                    [self.marginal_increment_array_stack, marginal_increment_array],
                    axis=0,
                )

        self.marginal_contribution = self.marginal_contrib_sum / (
            self.marginal_contrib_count + 1e-6
        )
        print(f"Done: marginal contribution computation", flush=True)

    def _calculate_marginal_contributions_core(self, min_cardinality=5):
        """
        Compute marginal contribution for Beta Shapley.
        marginal_increment_array : an array of marginal increments when one data point (idx) is added.
        marginal_increment : a marginal increment when one data point (idx) is added. Average of this value is Shapley as we consider a random permutation.
        """
        marginal_increment_array = np.zeros((0, self.n_points))
        for _ in range(100):
            # for each iteration, we use random permutation of indices.
            idxs = np.random.permutation(self.n_points)
            marginal_increment = np.zeros(self.n_points)
            X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
            y_batch = np.zeros(0, int)
            truncation_counter = 0
            for n, idx in enumerate(idxs):
                X_batch = np.concatenate([X_batch, self.X[self.sources[idx]]])
                y_batch = np.concatenate([y_batch, self.y[self.sources[idx]]])

                # At this point, y_batch has (n+1) samples
                if n < (min_cardinality - 1):
                    continue

                # When n == (min_cardinality-1), we compute performance based on 'min_cardinality' samples
                if n == (min_cardinality - 1):
                    # first old score is based on 'min_cardinality' samples
                    try:
                        self.model.fit(X_batch, y_batch)
                        old_score = self.compute_utility()
                    except:
                        old_score = self.random_score
                    continue

                try:
                    # 'new_score' is the performance with 'idx' sample.
                    # Baseline model ('old_score') is based on 'n' samples
                    self.model.fit(X_batch, y_batch)
                    new_score = self.compute_utility()
                except:
                    new_score = self.random_score
                marginal_increment[idx] = new_score - old_score

                # When the cardinality of random set is 'n',
                self.marginal_contrib_sum[idx, n] += marginal_increment[idx]
                self.marginal_contrib_count[idx, n] += 1

                # if a new increment is not large enough, we terminate the valuation.
                distance_to_full_score = np.abs(
                    marginal_increment[idx] / (sum(marginal_increment) + 1e-12)
                )  # np.abs((new_score-old_score)/(new_score+1e-12))
                # If updates are too small then we assume it contributes 0.
                if distance_to_full_score < 1e-8:
                    truncation_counter += 1
                    if truncation_counter == 10:
                        # print(f'Among {self.n_points}, {n} samples are observed!', flush=True)
                        break
                else:
                    truncation_counter = 0
                # update score
                old_score = new_score
            marginal_increment_array = np.concatenate(
                [marginal_increment_array, marginal_increment.reshape(1, -1)], axis=0
            )

        return marginal_increment_array

    def compute_weight_list(self, N_total, alpha_param=1, beta_param=1):
        """
        We denote Equation (5) of https://arxiv.org/pdf/2110.14049.pdf by $w^{(N_total)}(j)$ for $j$ in ${1, ..., N_total}$.
        Then from Equation (3), what we want to compute is an array of $w^{(N_total)}(j)*binom{N_total-1}{j-1}/N_total$.
        Since
        w^{(N_total)}(j)*binom{N_total-1}{j-1}/N_total
        = Beta(j+beta_param-1, N_total-j+alpha_param)*binom{N_total-1}{j-1}/Beta(alpha_param, beta_param)
        = Constant*Beta(j+beta_param-1, N_total-j+alpha_param)/Beta(j, N_total-j+1)
        where $Constant = 1/(N_total*Beta(alpha_param, beta_param))$.
        """
        weight_list = [
            beta(j + beta_param - 1, N_total - j + alpha_param)
            / beta(j, N_total - j + 1)
            for j in range(1, N_total + 1)
        ]
        weight_list = np.array(weight_list)
        return weight_list / np.sum(weight_list)

    def compute_GR_statistics(self, mem, n_chains=10):
        """
        Compute Gelman-Rubin statistic
        Ref. https://arxiv.org/pdf/1812.09384.pdf (p.7, Eq.4)
        """
        if len(mem) < 1000:
            return 100

        (N, n_to_be_valued) = mem.shape
        if (N % n_chains) == 0:
            n_MC_sample, offset = N // n_chains, 0
        else:
            n_MC_sample, offset = N // n_chains, (N % n_chains)
        mem = mem[offset:]

        mem_tmp = mem.reshape(n_chains, n_MC_sample, n_to_be_valued)
        GR_list = []
        for j in range(n_to_be_valued):
            mem_tmp_j_original = mem_tmp[
                :, :, j
            ].T  # now we have (n_MC_sample, n_chains)
            mem_tmp_j = mem_tmp_j_original  # /IQR_contstant
            mem_tmp_j_mean = np.mean(mem_tmp_j, axis=0)
            s_term = np.sum((mem_tmp_j - mem_tmp_j_mean) ** 2) / (
                n_chains * (n_MC_sample - 1)
            )  # + 1e-16 this could lead to wrong estimator

            mu_hat_j = np.mean(mem_tmp_j)
            B_term = (
                n_MC_sample * np.sum((mem_tmp_j_mean - mu_hat_j) ** 2) / (n_chains - 1)
            )

            GR_stat = np.sqrt(
                (n_MC_sample - 1) / n_MC_sample + B_term / (s_term * n_MC_sample)
            )
            GR_list.append(GR_stat)
        GR_stat = np.max(GR_list)
        print(
            f"Total number of random sets: {len(mem)}, GR_stat: {GR_stat}", flush=True
        )
        return GR_stat


import inspect
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.special import beta


def return_model(model_type, problem="clf", **kwargs):
    if inspect.isclass(model_type):
        assert (
            getattr(model_type, "fit", None) is not None
        ), "Custom model family should have a fit() method"
        model = model_type(**kwargs)
    elif model_type == "logistic":
        solver = kwargs.get("solver", "liblinear")
        n_jobs = kwargs.get("n_jobs", -1)
        C = kwargs.get("C", 0.05)  # 1.
        max_iter = kwargs.get("max_iter", 5000)
        model = LogisticRegression(
            solver=solver, n_jobs=n_jobs, C=C, max_iter=max_iter, random_state=666
        )
    elif model_type == "linear":
        n_jobs = kwargs.get("n_jobs", -1)
        model = LinearRegression(n_jobs=n_jobs)
    elif model_type == "ridge":
        alpha = kwargs.get("alpha", 1.0)
        model = Ridge(alpha=alpha, random_state=666)
    elif model_type == "Tree":
        if problem == "clf":
            model = DecisionTreeClassifier(random_state=666)
        else:
            model = DecisionTreeRegressor(random_state=666)
    elif model_type == "RandomForest":
        n_estimators = kwargs.get("n_estimators", 5)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=666)
    elif model_type == "GB":
        n_estimators = kwargs.get("n_estimators", 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=666)
    elif model_type == "AdaBoost":
        n_estimators = kwargs.get("n_estimators", 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=666)
    elif model_type == "SVC":
        kernel = kwargs.get("kernel", "rbf")
        C = kwargs.get("C", 0.05)  # 1.
        max_iter = kwargs.get("max_iter", 5000)
        model = SVC(kernel=kernel, max_iter=max_iter, C=C, random_state=666)
    elif model_type == "LinearSVC":
        C = kwargs.get("C", 0.05)  # 1.
        max_iter = kwargs.get("max_iter", 5000)
        model = LinearSVC(loss="hinge", max_iter=max_iter, C=C, random_state=666)
    elif model_type == "GP":
        model = GaussianProcessClassifier(random_state=666)
    elif model_type == "KNN":
        n_neighbors = kwargs.get("n_neighbors", 5)
        n_jobs = kwargs.get("n_jobs", -1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    elif model_type == "NB":
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model_type!")
    return model
