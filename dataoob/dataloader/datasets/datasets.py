"""Default data sets."""
import os

import numpy as np
import opendatasets as od
import pandas as pd
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler, minmax_scale

from dataoob.dataloader.register import Register, cache


@Register("gaussian_classifier", categorical=True)
def gaussian_classifier(n: int = 10000, input_dim: int = 10):
    """Binary category data set registered as ``"gaussian_classifier"``.

    Artificially generated gaussian noise data set.
    """
    covar = np.random.normal(size=(n, input_dim))

    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))

    labels = np.random.binomial(n=1, p=p_true).reshape(-1)

    return covar, labels


adult_dataset = Register("adult", categorical=True, cacheable=True)


@adult_dataset.add_covar_transform(StandardScaler().fit_transform)
def download_adult(cache_dir: str, force_download: bool = False):
    """Binary category data set registered as ``"adult"``. Adult Income data set.

    Implementation from DVRL repository.

    References
    ----------
    DVRL paper: https://arxiv.org/abs/1909.11671.
    UCI Adult data link: https://archive.ics.uci.edu/ml/datasets/Adult
    """
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    train_url = cache(uci_url + "/adult.data", cache_dir, "train.csv", force_download)
    test_url = cache(uci_url + "/adult.test", cache_dir, "test.csv", force_download)

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
    return df.drop("Income", axis=1).values, df["Income"].values


@Register("iris", categorical=True)
def download_iris():
    """Categorical data set registered as ``"iris"``."""
    return ds.load_iris(return_X_y=True)


@Register("digits", categorical=True)
def download_digits():
    """Categorical data set registered as ``"digits"``."""
    return ds.load_digits(return_X_y=True)


@Register("breast_cancer", True).add_covar_transform(minmax_scale)
def download_breast_cancer():
    """Categorical data set registered as ``"digits"``."""
    return ds.load_breast_cancer(return_X_y=True)


@Register("election", categorical=True, cacheable=True)
def download_election(cache_dir: str, force_download: bool):
    """Categorical data set registered as ``"election"``.

    Presidential election results by state since 1976 courtesy of Bojan Tunguz.

    References
    ----------
    Bojan Tunguz: https://www.kaggle.com/datasets/tunguz/us-elections-dataset
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    dataset_url = "https://www.kaggle.com/tunguz/us-elections-dataset"
    od.download(dataset_url, data_dir=cache_dir, force=force_download)

    df = pd.read_csv(f"{cache_dir}/us-elections-dataset/1976-2020-president.csv")
    df = df.drop(
        ["notes", "party_detailed", "candidate", "state_po", "version", "office"],
        axis=1,
    )
    df = pd.get_dummies(df, columns=["state"])

    covar = df.drop("party_simplified", axis=1).astype("float").values
    labels = df["party_simplified"].astype("category").cat.codes.values
    return covar, labels


# Alternative registration methods, should only be used on ad-hoc basis
Register(
    "gaussian_classifier_high_dim", categorical=True, dataset_kwargs={"input_dim": 100}
).from_covar_label_func(gaussian_classifier)
"""Registers gaussian classifier, but the input_dim is changed."""

# Regression data sets.
@Register("diabetes")
def download_diabetes():
    """Regression data set registered as ``"diabetes"``."""
    return ds.load_diabetes(return_X_y=True)


@Register("linnerud")
def download_linnerud():
    """Regression data set registered as ``"linnerud"``."""
    return ds.load_linnerud(return_X_y=True)