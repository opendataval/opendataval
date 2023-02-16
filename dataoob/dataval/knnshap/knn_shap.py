from KNN_PVLDB.LSH_sp import f_h, find_best_r_normalize, g_normalize, get_contrast

from dataoob.data_val import Evaluator, Model


class KNNShapley(Evaluator):
    def __init__(
        self,
        model: Model,
    ):
        super().__init__(model)
