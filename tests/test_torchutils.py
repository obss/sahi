import unittest

import numpy as np
import torch

from sahi.utils.torch import empty_cuda_cache, to_float_tensor, torch_to_numpy


class TestTorchUtils(unittest.TestCase):
    def test_empty_cuda_cache(self):
        if torch.cuda.is_available():
            self.assertIsNone(empty_cuda_cache())

    def test_to_float_tensor(self):

        img = to_float_tensor(np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))
        self.assertEqual(img.shape, (3, 10, 10))

    def test_torch_to_numpy(self):
        img_t = torch.tensor(np.random.rand(3, 10, 10))
        img = torch_to_numpy(img_t)
        self.assertEqual(img.shape, (10, 10, 3))


if __name__ == "__main__":
    unittest.main()
