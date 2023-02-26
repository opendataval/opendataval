import torch

from dataoob.dataval import Evaluator, Model


class DataShap(Evaluator):
    def __init__(
        self, pred_model: Model, metric: callable, GR_threshold=1.05, max_iters=50
    ):
        self.pred_model = pred_model
        self.metric = metric

        self.GR_threshold = GR_threshold
        self.max_iters = max_iters
        # create placeholders  TODO evaluate and decide if we need this
        self.data_value_dict = defaultdict(list)
        self.time_dict = defaultdict(list)
        self.noisy_detect_dict = defaultdict(list)
        self.removal_dict = defaultdict(list)

    def input_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_valid: torch.Tensor,
        y_valid: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def _dict_update(self, engine):
        self.data_value_dict.update(engine.data_value_dict)
        self.time_dict.update(engine.time_dict)

    def compute_data_shap(self, loo_run=True, betashap_run=True):
        """
        This function computes regular Data-Valuation methods
        """
        self.data_shap_engine = DataApproach(
            X=self.X,
            y=self.y,
            X_val=self.X_val,
            y_val=self.y_val,
            problem=self.problem,
            model_family=self.model_family,
        )
        self.data_shap_engine.run(loo_run=loo_run, betashap_run=betashap_run)
        self._dict_update(self.data_shap_engine)

    def evaluate_data_values(self, noisy_index, X_test, y_test, removal_run=True):
        time_init = time()
        if self.dargs["is_noisy"] > 0:
            self.noisy_detect_dict = utils_eval.noisy_detection_experiment(
                self.data_value_dict, noisy_index
            )
            self.time_dict["Eval:noisy"] = time() - time_init

        if removal_run is True:
            time_init = time()
            self.removal_dict = utils_eval.point_removal_experiment(
                self.data_value_dict,
                self.X,
                self.y,
                X_test,
                y_test,
                problem=self.problem,
            )
            self.time_dict["Eval:removal"] = time() - time_init

    def save_results(self, runpath, dataset, dargs_ind, noisy_index):
        self.sparsity_dict = defaultdict(list)
        for key in self.data_value_dict:
            self.sparsity_dict[key] = np.mean(self.data_value_dict[key] == 0)

        print("-" * 50)
        print("Save results")
        print("-" * 50)
        result_dict = {
            "data_value": self.data_value_dict,
            "sparse": self.sparsity_dict,
            "time": self.time_dict,
            "noisy": self.noisy_detect_dict,
            "removal": self.removal_dict,
            "dargs": self.dargs,
            "dataset": dataset,
            "input_dim": self.X.shape[1],
            "model_name": self.model_name,
            "noisy_index": noisy_index,
            "baseline_score": self.baseline_score_dict,
        }

        with open(runpath + f"/run_id{self.run_id}_{dargs_ind}.pkl", "wb") as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done! path: {runpath}, run_id: {self.run_id}.", flush=True)
