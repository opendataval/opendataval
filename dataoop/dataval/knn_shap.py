from dataoop.data_val import Evaluator, Model
from KNN_PVLDB.LSH_sp import get_contrast, find_best_r_normalize, g_normalize, f_h


class KNNShapley(Evaluator):
    def __init__(
        self,
        model: Model,
    ):
        super().__init__(model)
