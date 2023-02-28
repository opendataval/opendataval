from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from collections import namedtuple

CACHE_DIR = "data_files"

dataset_directory = {}
"""Creates a directory for all registred/downloadable datset functions"""

def load_dataset(
    dataset_name: str,
    device: int = torch.device("cpu")
) -> tuple[torch.Tensor | Dataset, torch.Tensor]:
    if dataset_name not in dataset_directory:
        raise Exception("Must register Dataset in register_dataset")

    covariates, labels =  dataset_directory[dataset_name]()  # Pass in force download and device

    if not isinstance(covariates, Dataset):
        covariates = torch.tensor(covariates).to(dtype=torch.float32, device=device)
    labels = torch.tensor(labels).to(dtype=torch.float32, device=device)

    return covariates, labels


def register_dataset(dataset_name: str, register_type="both"):
    def cache_download(func: callable):
        if register_type=="both":
            dataset_directory[dataset_name] = func
        else:
            if dataset_name not in dataset_directory:
                dataset_directory[dataset_name] = CovLabelWrapper()
            dataset_directory[dataset_name].add_func(func, register_type)
        return func
    return cache_download

class CovLabelWrapper:
    def __init__(self):
        self.cov_func, self.label_func = None, None
    def add_func(self, func: callable, register_type: str):
        if register_type == "label":
            self.cov_func = func
        elif register_type == "label":
            self.label_func = func
        else:
            raise Exception()
    def __call__(self):
        assert self.cov_func is not None and self.label_func is not None
        return self.cov_func(), self.label_func()


@register_dataset(dataset_name="gaussian_classifier", register_type="both")
def gaussian_classifier(n=10000, input_dim=10):
    covariates = np.random.normal(size=(n, input_dim))
    beta_true = np.random.normal(size=input_dim).reshape(input_dim,1)
    p_true = np.exp(covariates.dot(beta_true))/(1.+np.exp(covariates.dot(beta_true)))
    labels = np.random.binomial(n=1, p=p_true).reshape(-1)
    return covariates, labels

@register_dataset(dataset_name="adult", register_type="both")
def download_adult():
    uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    train_url = uci_base_url + "adult/adult.data"
    test_url = uci_base_url + "adult/adult.test"

    data_train = pd.read_csv(train_url, header=None)
    data_test = pd.read_csv(test_url, skiprows=1, header=None)

    df = pd.concat((data_train, data_test), axis=0)

    # Column names
    df.columns = [
        "Age",
        "WorkClass",
        "fnlwgt",
        "Education",
        "EducationNum",
        "MaritalStatus",
        "Occupation",
        "Relationship",
        "Race",
        "Gender",
        "CapitalGain",
        "CapitalLoss",
        "HoursPerWeek",
        "NativeCountry",
        "Income",
    ]

    # Creates binary labels
    df["Income"] = df["Income"].map(
        {" <=50K": 0, " >50K": 1, " <=50K.": 0, " >50K.": 1}
    )

    # Changes string to float
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.CapitalGain = df.CapitalGain.astype(float)
    df.CapitalLoss = df.CapitalLoss.astype(float)

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=[
            "WorkClass",
            "Education",
            "MaritalStatus",
            "Occupation",
            "Relationship",
            "Race",
            "Gender",
            "NativeCountry",
        ],
    )

    # Sets label name as Y
    df = df.rename(columns={"Income": "Income"})
    df["Income"] = df["Income"].astype(int)

    # Resets index
    df = df.reset_index()
    df = df.drop(columns=["index"])
    return df.drop("Income", axis=1), df["Income"]

@register_dataset("imagsset", register_type="covariates")
class imageset(Dataset):
    def __init__(self):
        self.lables
    def __getitem__(self, index):
        return self.covariate[index]
@register_dataset("imagsset", register_type="labels")
def imagesetlabels():
    return ...