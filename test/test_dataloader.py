import unittest

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.datasets import make_classification

from dataoob.dataloader import DataLoader, Register, add_gauss_noise, mix_labels

Data = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_classes=2,
    random_state=123,
)

Register("dummy", categorical=True).from_covar_label_func(lambda: Data)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data = Data
        self.loader = DataLoader("dummy", random_state=RandomState(123))

    def test_datasets(self):
        datasets = DataLoader.datasets_available()
        self.assertIsInstance(datasets, list)
        self.assertTrue(len(datasets) > 0)
        n = len(datasets)
        Register("dummy2", categorical=True)
        self.assertTrue(n + 1 == len(DataLoader.datasets_available()))

    def test_split_dataset(self):
        self.loader.split_dataset(train_count=0.7, valid_count=0.2, test_count=0.1)
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.loader.datapoints
        self.assertIsInstance(x_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(x_valid, torch.Tensor)
        self.assertIsInstance(y_valid, torch.Tensor)
        self.assertIsInstance(x_test, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)
        self.assertEqual(x_train.shape[0], 70)
        self.assertEqual(y_train.ndim, 2)
        self.assertEqual(x_valid.shape[0], 20)
        self.assertEqual(y_valid.ndim, 2)
        self.assertEqual(x_test.shape[0], 10)
        self.assertEqual(y_test.ndim, 2)

    def test_split_dataset_by_indices(self):
        self.loader.split_dataset_by_indices(range(10), None, range(11, 20))
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.loader.datapoints
        self.assertIsInstance(x_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(x_valid, torch.Tensor)
        self.assertIsInstance(y_valid, torch.Tensor)
        self.assertIsInstance(x_test, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)

        train_idx, test_idx = list(range(10)), list(range(11, 20))
        self.assertTrue(np.array_equal(self.data[0][train_idx], self.loader.x_train))
        self.assertTrue(
            np.array_equal(self.data[1][train_idx], self.loader.y_train.argmax(axis=1))
        )
        self.assertEqual(x_valid.shape[0], 0)
        self.assertEqual(y_valid.ndim, 2)
        self.assertTrue(np.array_equal(self.data[0][test_idx], self.loader.x_test))
        self.assertTrue(
            np.array_equal(self.data[1][test_idx], self.loader.y_test.argmax(axis=1))
        )

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

        # Test with gauss noise
        self.loader.split_dataset(train_count=80, valid_count=20)
        x_train = self.loader.x_train.copy()
        y_train = self.loader.y_train.copy()
        self.loader.noisify(add_gauss_noise, mu=10, sigma=10, noise_rate=0.5)
        self.assertFalse(np.array_equal(self.loader.x_train, x_train))
        self.assertTrue(np.array_equal(self.loader.y_train, y_train))
        self.assertTrue(self.loader.noisy_indices.any())

    def test_invalid_dataset(self):
        self.assertRaises(KeyError, DataLoader, dataset_name="nonexistent")
        Register("dummy3", categorical=True).from_covar_label_func(
            lambda: (np.array([1]), Data[1])
        )

        self.assertRaises(ValueError, DataLoader, "dummy3")

    def test_invalid_split(self):
        self.assertRaises(KeyError, DataLoader, dataset_name="nonexistent")
        self.assertRaises(
            ValueError,
            self.loader.split_dataset,
            train_count=80,
            valid_count=100,
            test_count=100,
        )
        self.assertRaises(
            ValueError,
            self.loader.split_dataset,
            train_count=0.8,
            valid_count=0.3,
            test_count=0.1,
        )

        self.assertRaises(
            ValueError, self.loader.split_dataset_by_indices, range(10), range(10)
        )

        self.assertRaises(
            ValueError, self.loader.split_dataset_by_indices, range(100), range(10)
        )


if __name__ == "__main__":
    unittest.main()
