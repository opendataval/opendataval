import os

import numpy as np
import pandas as pd

CACHE_DIR = "data_files"

dataset_directory = {}
"""Creates a directory for all registred/downloadable datset functions"""


def download_dataset(dataset_name: str, force_redownload: bool):
    if dataset_name in dataset_directory:
        x_raw, y_raw = dataset_directory[dataset_name](force_redownload)
        return x_raw, y_raw
    else:
        raise KeyError(
            f"`{dataset_name}` dataset not supported yet. \n"
            "Please include and register a function with the format\n"
            f"`def download_{dataset_name}()`"
        )


def register_dataset(func: callable):
    """Registers a function in the datset directory,
    must have the following format
    ```def download_{dataset_name}()```

    :param callable func: A function that will download a dataset
    :return callable: Modified function that appends a data_files path
    to the registered function
    """
    dataset_name = func.__name__.replace("download_", "")

    def check_cache(force_redownload: bool = False):
        if not os.path.exists(f"{CACHE_DIR}"):
            os.makedirs(f"{CACHE_DIR}")

        if not os.path.exists(f"{CACHE_DIR}/{dataset_name}.csv") or force_redownload:
            x_raw, y_raw = func()

            if isinstance(x_raw, pd.DataFrame) and isinstance(y_raw, pd.DataFrame):
                y_raw = y_raw.rename(columns=lambda name: f"Y_{name}")
                pd.concat((x_raw, y_raw), axis=1).to_csv(
                    f"{CACHE_DIR}/{dataset_name}.csv", index=False
                )

            return x_raw, y_raw
        else:
            # Loads dataset from cache, seperates the response variables
            dataset = pd.read_csv(f"{CACHE_DIR}/{dataset_name}.csv")
            y_col = dataset.columns.str.startswith("Y_")
            return dataset.loc[:, ~y_col], dataset.loc[:, y_col]

    dataset_directory[dataset_name] = check_cache
    return check_cache


@register_dataset
def download_gaussian():
    n, input_dim = 10000, 10
    X_raw = np.random.normal(size=(n, input_dim))
    beta = np.random.normal(size=(input_dim, 1))
    error_raw = np.random.normal(size=(n, 1))
    Y_raw = X_raw.dot(beta) + error_raw
    return X_raw, Y_raw


@register_dataset
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
