import unittest

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.datasets import make_classification

from dataoob.dataloader import DataLoader, Register, mix_labels

data = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_classes=2,
    random_state=123,
)

Register("dummy", categorical=True).from_covar_label_func(lambda: data)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data = data
        self.loader = DataLoader("dummy", random_state=RandomState(123))

    def test_datasets(self):
        datasets = DataLoader.datasets_available()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        n = len(datasets)
        Register("dummy2", categorical=True)
        self.assertTrue(n + 1 == len(DataLoader.datasets_available()))

    def test_split_dataset(self):
        self.loader.split_dataset(train_count=0.8, valid_count=0.2)
        x_train, y_train, x_valid, y_valid = self.loader.datapoints
        self.assertIsInstance(x_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(x_valid, torch.Tensor)
        self.assertIsInstance(y_valid, torch.Tensor)
        self.assertEqual(x_train.shape[0], 80)
        self.assertEqual(x_valid.shape[0], 20)

    def test_noisify(self):
        # Test with no noise
        self.loader.split_dataset(train_count=80, valid_count=20)
        x_train = self.loader.x_train.copy()
        y_train = self.loader.y_train.copy()
        self.loader.noisify(mix_labels, noise_rate=0.0)
        self.assertTrue(np.array_equal(self.loader.x_train, x_train))
        self.assertTrue(np.array_equal(self.loader.y_train, y_train))
        self.assertTrue(not self.loader.noisy_indices.any())

        # Test with noise
        self.loader.split_dataset(train_count=80, valid_count=20)
        x_train = self.loader.x_train.copy()
        y_train = self.loader.y_train.copy()
        self.loader.noisify(mix_labels, noise_rate=0.5)
        self.assertTrue(np.array_equal(self.loader.x_train, x_train))
        self.assertFalse(np.array_equal(self.loader.y_train, y_train))
        self.assertTrue(self.loader.noisy_indices.any())

    def test_invalid_split(self):
        self.assertRaises(KeyError, DataLoader, dataset_name="nonexistent")
        self.assertRaises(
            ValueError, self.loader.split_dataset, train_count=80, valid_count=100
        )
        self.assertRaises(
            ValueError, self.loader.split_dataset, train_count=0.8, valid_count=0.3
        )
        self.loader.covar = np.array([1])
        self.assertRaises(
            ValueError, self.loader.split_dataset, train_count=0.8, valid_count=0.0
        )


if __name__ == "__main__":
    unittest.main()
