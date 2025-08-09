# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2025.

import numpy as np

from sahi.utils.torch_utils import empty_cuda_cache, to_float_tensor, torch_to_numpy


class TestTorchUtils:
    def test_empty_cuda_cache(self):
        import torch

        if torch.cuda.is_available():
            assert empty_cuda_cache() is None

    def test_to_float_tensor(self):
        img = to_float_tensor(np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))
        assert img.shape == (3, 10, 10)

    def test_torch_to_numpy(self):
        import torch

        img_t = torch.tensor(np.random.rand(3, 10, 10))
        img = torch_to_numpy(img_t)
        assert img.shape == (10, 10, 3)
