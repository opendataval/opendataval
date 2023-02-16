import sklearn.metrics as metrics
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from dataoob.dataloader import data_loading, utils
from dataoob.dataval.dvrl.dvrl import DVRL
from dataoob.model import ClassifierSkLearnWrapper

# from dataoob.model.logistic_regression import LogisticRegression



device = torch.device("mps")


def cast(*args):
    return [
        torch.from_numpy(arg).to(dtype=torch.float32) for arg in args
    ]


if __name__ == "__main__":
    # Data loading
    dict_no = dict()
    dict_no["train"] = 1000
    dict_no["valid"] = 400
    # noise_idx = data_loading.load_tabular_data('adult', dict_no, 0.1)
    print("LOADING DATA")
    parameters = {}
    parameters["hidden_dim"] = 100
    parameters["comb_dim"] = 10
    parameters["act_fn"] = nn.ReLU()
    parameters["layer_number"] = 5
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        col_names,
    ) = data_loading.preprocess_data("minmax", "train.csv", "valid.csv", "test.csv")

    print("Finished data preprocess.")
    pred_model = LogisticRegression()  # Work on unifying the return type, god these values are dumb
    pred_model = ClassifierSkLearnWrapper(pred_model)
    # Flags for using stochastic gradient descent / pre-trained model

    d = DVRL(
        x_dim=x_train.shape[1],
        y_dim=2,
        pred_model=pred_model,
        metric = lambda x, y: metrics.roc_auc_score(x.detach().cpu(), y.detach().cpu()),
        **parameters
    )
    x1, y1, x2, y2 = cast(x_train, y_train, x_valid, y_valid)
    y1, y2 = utils.one_hot_encode(y1, y2)
    d.input_data(x1, y1, x2, y2)
    # d.evaluate_baseline_models(pre_train=False, batch_size=256, epochs=100)
    d.train_data_values(batch_size=128, rl_epochs=2000)
    val = d.evaluate_data_values(x1, y1)

    print(val)
    print(val[noise_idx])
    sorted_idx = torch.argsort(-val.detach().cpu())

    print(torch.mean(val))
    print(torch.mean(val.detach().cpu()[noise_idx]))
