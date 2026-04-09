"""Tests for PyTorch utility functions."""

from __future__ import annotations

import numpy as np

from sahi.utils.torch_utils import empty_cuda_cache, to_float_tensor, torch_to_numpy


class TestTorchUtils:
    """Test PyTorch utility functions."""

    def test_empty_cuda_cache(self) -> None:
        """Test CUDA cache clearing."""
        import torch

        if torch.cuda.is_available():
            empty_cuda_cache()  # should not raise

    def test_to_float_tensor(self) -> None:
        """Test converting NumPy array to float tensor."""
        img = to_float_tensor(np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))
        assert img.shape == (3, 10, 10)

    def test_torch_to_numpy(self) -> None:
        """Test converting PyTorch tensor to NumPy array."""
        import torch

        img_t = torch.tensor(np.random.rand(3, 10, 10))
        img = torch_to_numpy(img_t)
        assert img.shape == (10, 10, 3)
