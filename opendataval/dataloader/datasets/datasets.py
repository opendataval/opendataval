"""Default data sets."""
import os

import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, minmax_scale

from opendataval.dataloader.register import Register, cache


def load_openml(data_id: int):
    """load openml datasets.

    A help function to load openml datasets with OpenML ID.
    """
    dataset = fetch_openml(data_id=data_id, as_frame=False)
    category_list = list(dataset["categories"].keys())
    if len(category_list) > 0:
        category_indices = [dataset["feature_names"].index(x) for x in category_list]
        noncategory_indices = [
            i for i in range(len(dataset["feature_names"])) if i not in category_indices
        ]
        X, y = dataset["data"][:, noncategory_indices], dataset["target"]
    else:
        X, y = dataset["data"], dataset["target"]
    list_of_classes, y = np.unique(y, return_inverse=True)
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)  # standardization
    return X, y


@Register("gaussian_classifier", one_hot=True)
def gaussian_classifier(n: int = 10000, input_dim: int = 10):
    """Binary category data set registered as ``"gaussian_classifier"``.

    Artificially generated gaussian noise data set.
    """
    covar = np.random.normal(size=(n, input_dim))

    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))

    labels = np.random.binomial(n=1, p=p_true).reshape(-1)

    return covar, labels


adult_dataset = Register("adult", one_hot=True, cacheable=True)


@adult_dataset.add_covar_transform(StandardScaler().fit_transform)
def download_adult(cache_dir: str, force_download: bool = False):
    """Binary category data set registered as ``"adult"``. Adult Income data set.

    Implementation from DVRL repository.

    References
    ----------
    .. [1] R. Kohavi, Scaling Up the Accuracy of
        Naive-Bayes Classifiers: a Decision-Tree Hybrid,
        Proceedings of the Second International Conference on Knowledge Discovery
        and Data Mining, 1996
    .. [2] J. Yoon, Arik, Sercan O, and T. Pfister,
        Data Valuation using Reinforcement Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1909.11671.
    """
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    train_path = cache(uci_url + "/adult.data", cache_dir, "train.csv", force_download)
    test_path = cache(uci_url + "/adult.test", cache_dir, "test.csv", force_download)

    data_train = pd.read_csv(train_path, header=None)
    data_test = pd.read_csv(test_path, skiprows=1, header=None)

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


@Register("iris", one_hot=True)
def download_iris():
    """Categorical data set registered as ``"iris"``."""
    return ds.load_iris(return_X_y=True)


@Register("digits", one_hot=True)
def download_digits():
    """Categorical data set registered as ``"digits"``."""
    return ds.load_digits(return_X_y=True)


@Register("breast_cancer", True).add_covar_transform(minmax_scale)
def download_breast_cancer():
    """Categorical data set registered as ``"digits"``."""
    return ds.load_breast_cancer(return_X_y=True)


@Register("election", one_hot=True, cacheable=True)
def download_election(cache_dir: str, force_download: bool):
    """Categorical data set registered as ``"election"``.

    Presidential election results by MIT Election Data and Science Lab.

    References
    ----------
    .. [1] M. E. Data and S. Lab,
        U.S. President 1976-2020.
        Harvard Dataverse, 2017. doi: 10.7910/DVN/42MVDX.
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    url = "https://dataverse.harvard.edu/api/access/datafile/4299753?gbrecs=false"
    filepath = cache(url, cache_dir, "1976-2020-president.tab", force_download)

    df = pd.read_csv(filepath, delimiter="\t")
    drop_col = [
        "notes",
        "party_detailed",
        "candidate",
        "version",
        "state_po",
        "writein",
        "office",
    ]

    df = df.drop(drop_col, axis=1)
    df = pd.get_dummies(df, columns=["state"])

    covar = df.drop("party_simplified", axis=1).astype("float").values
    labels = df["party_simplified"].astype("category").cat.codes.values

    return covar, labels


# Alternative registration methods, should only be used on ad-hoc basis
Register("gaussian_classifier_high_dim", one_hot=True).from_covar_label_func(
    gaussian_classifier, input_dim=100
)
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


# OpenML Datasets
@Register("2dplanes", one_hot=True)
def download_2dplanes():
    """Categorical data set registered as ``"2dplanes"``."""
    return load_openml(data_id=727)


@Register("electricity", one_hot=True)
def download_electricity():
    """Categorical data set registered as ``"electricity"``."""
    return load_openml(data_id=44080)


@Register("MiniBooNE", one_hot=True)
def download_MiniBooNE():
    """Categorical data set registered as ``"MiniBooNE"``."""
    return load_openml(data_id=43974)


@Register("pol", one_hot=True)
def download_pol():
    """Categorical data set registered as ``"pol"``."""
    return load_openml(data_id=722)


@Register("fried", one_hot=True)
def download_fried():
    """Categorical data set registered as ``"fried"``."""
    return load_openml(data_id=901)


@Register("nomao", one_hot=True)
def download_nomao():
    """Categorical data set registered as ``"nomao"``."""
    return load_openml(data_id=1486)


@Register("creditcard", one_hot=True)
def download_creditcard():
    """Categorical data set registered as ``"creditcard"``."""
    return load_openml(data_id=42477)

