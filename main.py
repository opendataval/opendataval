import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from dataoob.dataloader.data_loader import DataLoader
from dataoob.dataval.dvrl.dvrl import DVRL
from dataoob.dataval.knnshap.knnshap import KNNShapley
from dataoob.dataval.shap.shap import ShapEvaluator
from dataoob.model import ClassifierSkLearnWrapper, ClassifierUnweightedSkLearnWrapper
from dataoob.model.logistic_regression import LogisticRegression as LR

device = torch.device("cpu")

if __name__ == "__main__":
    # Data loading
    (x_train, y_train), (x_valid, y_valid), (xt, yt), noisy_indices = DataLoader('adult', False, 200, 50, categorical=True, device=device, noise_rate=.15)


    model = LR(109).to(device)
    metric =  lambda a, b: metrics.roc_auc_score(a.detach().cpu(), b.detach().cpu())
    dvrl = KNNShapley(
        model,
        metric,
        2
    )
    dvrl.input_data(x_train, y_train, x_valid, y_valid)
    dvrl.train_data_values(batch_size=128, epochs=20)
    e = dvrl.evaluate_data_values()

    print(f"normal={torch.mean(np.delete(e.cpu(), noisy_indices, axis=0))}")
    print(f"noisy ={torch.mean(e[noisy_indices])}")

    dvrl = ShapEvaluator(
        model,
        metric,
        1.05,
        model_name="tim"
    )

    dvrl.input_data(x_train, y_train, x_valid, y_valid)
    dvrl.train_data_values(batch_size=128, epochs=10)
    e = dvrl.evaluate_data_values()
    print(np.mean(np.delete(e.cpu(), noisy_indices, axis=0)))
    print(np.mean(e[noisy_indices]))

    dvrl = ShapEvaluator(
        model,
        metric,
        1.05,
        model_name="tim"
    )

    dvrl.input_data(x_train, y_train, x_valid, y_valid)
    dvrl.train_data_values(batch_size=128, epochs=10)
    e = dvrl.evaluate_data_values()
    print(np.mean(np.delete(e.cpu(), noisy_indices, axis=0)))
    print(np.mean(e[noisy_indices]))