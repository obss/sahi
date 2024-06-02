import unittest

import torch

from sahi.postprocess.utils import ObjectPredictionList


class TestPostprocessUtils(unittest.TestCase):
    def setUp(self):
        self.test_input = [ObjectPredictionList([1, 2, 3, 4])]

    def test_get_item_int(self):
        obj = self.test_input[0]
        self.assertEqual(obj[0].tolist(), 1)

    def test_len(self):
        obj = self.test_input[0]
        self.assertEqual(len(obj), 4)

    def test_extend(self):
        obj = self.test_input[0]
        obj.extend(ObjectPredictionList([torch.randn(1, 2, 3, 4)]))
        self.assertEqual(len(obj), 5)

    def test_tostring(self):
        obj = self.test_input[0]
        self.assertEqual(str(obj), str([1, 2, 3, 4]))


if __name__ == "__main__":
    unittest.main()
