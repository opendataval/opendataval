import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sklearn.datasets as ds


from typing import Any


def one_hot_encode(data: np.ndarray) -> np.ndarray:
    num_values = np.max(data) + 1
    return np.eye(num_values)[data]


def cache(
    url: str, cache_dir: str, file_name: str = "", force_redownload: bool = False
):
    """Loads a file from the URL and caches it locally."""
    if file_name is None:
        file_name = os.path.basename(url)

    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    file_path = os.path.join(cache_dir, file_name)
    if not os.path.isfile(file_path) or force_redownload:
        urlretrieve(url, file_path)

    return file_path


class Register:
    CACHE_DIR = "data_files"

    Datasets = {}
    """Creates a directory for all registered/downloadable dataset functions"""

    def __init__(
        self,
        dataset_name: str,
        categorical: bool = False,
        cacheable: bool = False,
        dataset_kwargs: dict[str, Any] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs else {}

        self.covariate_transform = None
        self.label_transform = None
        if categorical:
            self.label_transform = one_hot_encode

        if cacheable:
            self.download_dir = os.path.join(
                os.getcwd(), f"{Register.CACHE_DIR}/{dataset_name}"
            )

        Register.Datasets[dataset_name] = self

    def add_both(self, func: callable) -> callable:
        self.cov_label_func = func
        return func

    def add_covariates(self, func: callable) -> callable:
        self.cov_func = func
        return func

    def add_labels(self, func: callable) -> callable:
        self.label_func = func
        return func

    def add_covariate_transform(self, transform: callable):
        self.covariate_transform = transform
        return self

    def add_label_transform(self, transform: callable):
        self.label_transform = transform
        return self

    def load_data(self, force_redownload=False):  # Consider caching higher up the chain
        if hasattr(self, "download_dir"):
            self.dataset_kwargs["cache_dir"] = self.download_dir
            self.dataset_kwargs["force_redownload"] = force_redownload

        if hasattr(self, "cov_label_func"):
            covariates, labels = self.cov_label_func(**self.dataset_kwargs)
        else:
            covariates = self.cov_func(**self.dataset_kwargs)
            labels = self.label_func(**self.dataset_kwargs)

        if self.covariate_transform:
            if isinstance(covariates, Dataset):
                covariates.transform = self.covariate_transform
            else:
                covariates = self.covariate_transform(covariates)

        if self.label_transform:
            labels = self.label_transform(labels)

        return covariates, labels


@Register("gaussian_classifier", categorical=True).add_both
def gaussian_classifier(n=10000, input_dim=10):
    covar = np.random.normal(size=(n, input_dim))

    beta_true = np.random.normal(size=input_dim).reshape(input_dim, 1)
    p_true = np.exp(covar.dot(beta_true)) / (1.0 + np.exp(covar.dot(beta_true)))

    labels = np.random.binomial(n=1, p=p_true).reshape(-1)

    return covar, labels


Register(  # Register different versions of datasets
    "gaussian_classifier_high_dim", categorical=True, dataset_kwargs={"input_dim": 100}
).add_both(gaussian_classifier)


@Register("adult", categorical=True, cacheable=True).add_both
def download_adult(cache_dir: str, force_redownload=False):
    uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    train_url = cache(uci_base_url + "/adult.data", cache_dir, "train.csv", force_redownload)
    test_url = cache(uci_base_url + "/adult.test", cache_dir, "test.csv", force_redownload)

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

@Register("iris", categorical=True).add_both
def download_iris():
    return ds.load_iris(return_X_y=True)


@Register("diabetes", categorical=True).add_both
def download_iris():
    return ds.load_diabetes(return_X_y=True)


@Register("digits", categorical=True).add_both
def download_iris():
    return ds.load_digits(return_X_y=True)


@Register("breast_cancer", categorical=True).add_both
def download_iris():
    return ds.load_breast_cancer(return_X_y=True)
