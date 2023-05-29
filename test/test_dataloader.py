import unittest

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.datasets import make_classification

from opendataval.dataloader import DataFetcher, Register, add_gauss_noise, mix_labels

Data = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_classes=2,
    random_state=123,
)

Register("dummy", one_hot=True).from_covar_label_func(lambda: Data)


class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.data = Data
        self.fetcher = DataFetcher("dummy", random_state=RandomState(123))

    def test_datasets(self):
        datasets = DataFetcher.datasets_available()
        self.assertIsInstance(datasets, set)
        self.assertTrue(len(datasets) > 0)
        n = len(datasets)
        Register("dummy2", one_hot=True)
        self.assertTrue(n + 1 == len(DataFetcher.datasets_available()))

    def test_split_dataset_prop(self):
        self.fetcher.split_dataset_by_prop(
            train_prop=0.7, valid_prop=0.2, test_prop=0.1
        )
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.fetcher.datapoints
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
        self.fetcher.split_dataset_by_indices(range(10), None, range(11, 20))
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.fetcher.datapoints
        self.assertIsInstance(x_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(x_valid, torch.Tensor)
        self.assertIsInstance(y_valid, torch.Tensor)
        self.assertIsInstance(x_test, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)

        train_idx, test_idx = list(range(10)), list(range(11, 20))
        self.assertTrue(np.array_equal(self.data[0][train_idx], self.fetcher.x_train))
        self.assertTrue(
            np.array_equal(self.data[1][train_idx], self.fetcher.y_train.argmax(axis=1))
        )
        self.assertEqual(x_valid.shape[0], 0)
        self.assertEqual(y_valid.ndim, 2)
        self.assertTrue(np.array_equal(self.data[0][test_idx], self.fetcher.x_test))
        self.assertTrue(
            np.array_equal(self.data[1][test_idx], self.fetcher.y_test.argmax(axis=1))
        )

    def test_noisify(self):
        # Test with no noise
        self.fetcher.split_dataset_by_count(train_count=80, valid_count=20)
        x_train = self.fetcher.x_train.copy()
        y_train = self.fetcher.y_train.copy()
        self.fetcher.noisify(mix_labels, noise_rate=0.0)
        self.assertTrue(np.array_equal(self.fetcher.x_train, x_train))
        self.assertTrue(np.array_equal(self.fetcher.y_train, y_train))
        self.assertTrue(not self.fetcher.noisy_train_indices.any())

        # Test with noise
        self.fetcher.split_dataset_by_count(train_count=80, valid_count=20)
        x_train = self.fetcher.x_train.copy()
        y_train = self.fetcher.y_train.copy()
        self.fetcher.noisify(mix_labels, noise_rate=0.5)
        x_train_noise = self.fetcher.x_train
        y_train_noise = self.fetcher.y_train
        indices = self.fetcher.noisy_train_indices
        self.assertTrue(np.array_equal(x_train_noise, x_train))
        self.assertFalse(np.array_equal(y_train_noise, y_train))

        self.assertFalse(
            np.equal(y_train[indices], y_train_noise[indices]).all(axis=1).any()
        )
        self.assertTrue(self.fetcher.noisy_train_indices.any())

        # Test with gauss noise
        self.fetcher.split_dataset_by_count(train_count=80, valid_count=20)
        x_train = self.fetcher.x_train.copy()
        y_train = self.fetcher.y_train.copy()
        self.fetcher.noisify(add_gauss_noise, mu=10, sigma=10, noise_rate=0.5)
        self.assertFalse(np.array_equal(self.fetcher.x_train, x_train))
        self.assertTrue(np.array_equal(self.fetcher.y_train, y_train))
        self.assertTrue(self.fetcher.noisy_train_indices.any())

    def test_invalid_dataset(self):
        self.assertRaises(KeyError, DataFetcher, dataset_name="nonexistent")
        Register("dummy3", one_hot=True).from_covar_label_func(
            lambda: (np.array([1]), Data[1])
        )

        self.assertRaises(ValueError, DataFetcher, "dummy3")

    def test_invalid_split(self):
        self.assertRaises(KeyError, DataFetcher, dataset_name="nonexistent")
        self.assertRaises(
            ValueError,
            self.fetcher.split_dataset_by_count,
            train_count=80,
            valid_count=100,
            test_count=100,
        )
        self.assertRaises(
            ValueError,
            self.fetcher.split_dataset_by_prop,
            train_prop=0.8,
            valid_prop=0.3,
            test_prop=0.1,
        )

        self.assertRaises(
            ValueError, self.fetcher.split_dataset_by_indices, range(10), range(10)
        )

        self.assertRaises(
            ValueError, self.fetcher.split_dataset_by_indices, range(100), range(10)
        )


if __name__ == "__main__":
    unittest.main()
