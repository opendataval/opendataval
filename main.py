import sklearn.metrics as metrics
import torch
import torch.nn as nn

from dataoob.dataloader.data_loader import DataLoader
from dataoob.dataval.dvrl.dvrl import DVRL
from dataoob.dataval.shap.shap import ShapEvaluator
from dataoob.model import ClassifierSkLearnWrapper
from dataoob.model.logistic_regression import LogisticRegression

# from sklearn.linear_model import LogisticRegression


device = torch.device("cpu")

if __name__ == "__main__":
    # Data loading
    dict_no = dict()
    dict_no["train"] = 100
    dict_no["valid"] = 400
    (x_train, y_train), (x_valid, y_valid), (xt, yt), noisy_indices = DataLoader('adult', False, 200, 400, categorical=True, device=device)


    model = LogisticRegression(109)
    metric =  lambda a, b: metrics.roc_auc_score(a.detach().cpu(), b.detach().cpu())
    dvrl = ShapEvaluator(
        model,
        metric,
        1.1

    )

    dvrl.input_data(x_train, y_train, x_valid, y_valid)
    dvrl.train_data_values(batch_size=128, epochs=20)
    e = dvrl.evaluate_data_values()
    print(e)