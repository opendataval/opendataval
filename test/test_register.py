import unittest

import numpy as np
import pandas as pd

from opendataval.dataloader import Register


class TestRegister(unittest.TestCase):
    def test_from_pandas(self):
        reg = Register("test_pandas")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "label": [0, 1, 1]})
        label_columns = "label"
        result = reg.from_pandas(df, label_columns).load_data()
        self.assertTrue(np.array_equal(result[0], df.drop("label", axis=1).values))
        self.assertTrue(np.array_equal(result[1], df["label"].values))

    def test_from_numpy(self):
        reg = Register("test_numpy")
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        label_columns = 1
        result = reg.from_numpy(arr, label_columns).load_data()
        self.assertTrue(np.array_equal(result[0], arr[:, [0]]))
        self.assertTrue(np.array_equal(result[1], arr[:, [1]]))

    def test_from_data(self):
        reg = Register("test_from_data")
        arr = np.array([[1, 0], [3, 1], [5, 2]])
        result = reg.from_data(arr[:, [0]], arr[:, [1]], True).load_data()
        self.assertTrue(np.array_equal(result[0], arr[:, [0]]))
        # Defined labels to be the identity matrix
        self.assertTrue(np.array_equal(result[1], np.identity(3)))

    def test_from_covar_label_func(self):
        reg = Register("test_covar_label")
        a, b = np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1])
        reg.from_covar_label_func(lambda: (a.copy(), b.copy()))
        result = reg.load_data()
        self.assertTrue(np.array_equal(result[0], a))
        self.assertTrue(np.array_equal(result[1], b))

    def test_from_multi_func(self):
        reg = Register("test_covar")
        a, b = np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1])
        reg.from_covar_func(lambda: a.copy())
        reg.from_label_func(lambda: b.copy())
        result = reg.load_data()
        self.assertTrue(np.array_equal(result[0], a))
        self.assertTrue(np.array_equal(result[1], b))

    def test_add_covar_transform(self):
        reg = Register("test_add_covar_transform")
        a, b = np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1])
        reg.from_covar_label_func(lambda: (a.copy(), b.copy()))
        result = reg.add_covar_transform(lambda x: x * 2).load_data()
        self.assertTrue(np.array_equal(result[0], a * 2))
        self.assertTrue(np.array_equal(result[1], b))

    def test_add_label_transform(self):
        reg = Register("test_add_label_transform")
        a, b = np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 1])
        reg.from_covar_label_func(lambda: (a.copy(), b.copy()))
        result = reg.add_label_transform(lambda x: x + 1).load_data()
        self.assertTrue(np.array_equal(result[0], a))
        self.assertTrue(np.array_equal(result[1], b + 1))

    def test_repeat_register(self):
        n = len(Register.Datasets)
        Register("repeat1")
        self.assertWarns(Warning, Register, "repeat1")
        self.assertEqual(n + 1, len(Register.Datasets))


if __name__ == "__main__":
    unittest.main()
