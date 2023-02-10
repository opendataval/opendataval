import torch
import torch.nn as nn

from dataoob.dataloader import data_loading
from dataoob.dataval.dvrl.dvrl import DVRL
from dataoob.model.ann import ANN

device = torch.device("mps")
def cast(*args):
    return [torch.from_numpy(arg).to(dtype=torch.float32, device=device) for arg in args]


if __name__ == "__main__":
    # Data loading
    dict_no = dict()
    dict_no["train"] = 1000
    dict_no["valid"] = 400
    # _ = data_loading.load_tabular_data('adult', dict_no, 0.0)
    print("LOADING DATA")
    parameters = {}
    parameters["hidden_dim"] = 100
    parameters["comb_dim"] = 10
    parameters["iterations"] = 100
    parameters["activation"] = nn.ReLU()
    parameters["inner_iterations"] = 100
    parameters["layer_number"] = 5
    parameters["learning_rate"] = 0.01
    parameters["batch_size"] = 256
    parameters["batch_size_predictor"] = 256
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        col_names,
    ) = data_loading.preprocess_data("standard", "train.csv", "valid.csv", "test.csv")

    print("Finished data preprocess.")
    pred_model = LogisticRegression(x_train.shape[1])
    # Flags for using stochastic gradient descent / pre-trained model
    flags = {"sgd": True, "pretrain": False}
    d = DVRL(pred_model, parameters, flags)
    print(d.data_value_evaluator(*cast(x_train, y_train, x_valid, y_valid)))
