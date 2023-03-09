import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CACHE_DIR = "data_files"

DatasetDirectory = {}
"""Creates a directory for all registered/downloadable dataset functions"""


def register_dataset(dataset_name: str, register_type="both"):
    # TODO clean up this decorator, I'm not entirely happy with this API that exists
    def wrap_func_outputs(func: callable):
        if dataset_name not in DatasetDirectory:
            DatasetDirectory[dataset_name] = CovLabelWrapper(dataset_name)
        DatasetDirectory[dataset_name].add_func(func, register_type)
        return func

    return wrap_func_outputs


class CovLabelWrapper:
    def __init__(self, dataset_name, categorical: bool = False):
        self.dataset_name = dataset_name
        self.func = None
        self.cov_func, self.label_func = None, None

    def add_func(self, func: callable, register_type: str):
        if register_type == "label":
            self.cov_func = func
        elif register_type == "label":
            self.label_func = func
        elif register_type == "both":
            self.func = func
        else:
            raise Exception()

    def __call__(self, force_download: bool):
        if self.func is not None:
            covariates, labels = self.func(self.dataset_name)
        elif self.cov_func is not None and self.label_func is not None:
            covariates, labels = self.cov_func(self.dataset_name), self.label_func(
                self.dataset_name
            )
        else:
            raise Exception()
        # TODO APPLY TRANSFORMS HERE
        return covariates, labels


def cache(url: str, dataset_name: str, file_name: str = None):
    """Loads a file from the URL and caches it locally."""
    if file_name is None:
        file_name = os.path.basename(url)

    data_dir = os.path.join(os.getcwd(), f"{CACHE_DIR}/{dataset_name}")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path


@register_dataset(dataset_name="gaussian_classifier", register_type="both")
def gaussian_classifier(dataset_name: str, n=10000, input_dim=10):
    covar = np.random.normal(size=(n, input_dim))

    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))

    labels = np.random.binomial(n=1, p=p_true).reshape(-1)

    return covar, labels


@register_dataset(dataset_name="adult", register_type="both")
def download_adult(dataset_name: str):
    uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    train_url = cache(uci_base_url + "adult/adult.data", dataset_name, "train.csv")
    test_url = cache(uci_base_url + "adult/adult.test", dataset_name, "test.csv")

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


# @register_dataset("imageset", register_type="covariates")
# class imageset(Dataset):
#     def __init__(self):
#         self.lables
#     def __getitem__(self, index):
#         return self.covariate[index]
# @register_dataset("imageset", register_type="labels")
# def imagesetlabels():
#     return ...
