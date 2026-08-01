"""Tests for PyTorch utility functions."""

from __future__ import annotations

import os
import warnings
from typing import Iterator

import numpy as np
import pytest
import torch

from sahi.utils.torch_utils import empty_cuda_cache, select_device, to_float_tensor, torch_to_numpy

MPS_AVAILABLE = torch.backends.mps.is_available()


@pytest.fixture
def keep_cuda_visible_devices() -> Iterator[None]:
    """Restore CUDA_VISIBLE_DEVICES, which select_device sets as a side effect."""
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    yield
    if original is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original


def patch_availability(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    """Pretend the current machine has (or lacks) CUDA and MPS."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


class TestTorchUtils:
    """Test PyTorch utility functions."""

    def test_empty_cuda_cache(self) -> None:
        """Test CUDA cache clearing."""
        if torch.cuda.is_available():
            empty_cuda_cache()  # should not raise

    def test_to_float_tensor(self) -> None:
        """Test converting NumPy array to float tensor."""
        img = to_float_tensor(np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))
        assert img.shape == (3, 10, 10)

    def test_torch_to_numpy(self) -> None:
        """Test converting PyTorch tensor to NumPy array."""
        img_t = torch.tensor(np.random.rand(3, 10, 10))
        img = torch_to_numpy(img_t)
        assert img.shape == (10, 10, 3)


@pytest.mark.usefixtures("keep_cuda_visible_devices")
class TestSelectDevice:
    """Test automatic device selection."""

    def test_auto_prefers_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA wins over MPS when both are available."""
        patch_availability(monkeypatch, cuda=True, mps=True)
        assert select_device().type == "cuda"

    def test_auto_falls_back_to_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MPS is picked when CUDA is missing."""
        patch_availability(monkeypatch, cuda=False, mps=True)
        assert select_device().type == "mps"

    def test_auto_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CPU is the last resort."""
        patch_availability(monkeypatch, cuda=False, mps=False)
        assert select_device().type == "cpu"

    def test_explicit_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit request is never overridden by auto-detection."""
        patch_availability(monkeypatch, cuda=True, mps=True)
        assert select_device("cpu").type == "cpu"

    def test_explicit_mps_without_mps_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asking for MPS on a machine without it returns CPU rather than raising."""
        patch_availability(monkeypatch, cuda=False, mps=False)
        assert select_device("mps").type == "cpu"

    def test_does_not_use_deprecated_has_mps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch.has_mps warns on every access and is gone in newer torch."""
        patch_availability(monkeypatch, cuda=False, mps=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            select_device("mps")
        assert not [w for w in caught if "has_mps" in str(w.message)]


@pytest.mark.skipif(not MPS_AVAILABLE, reason="requires an Apple Silicon machine with MPS")
@pytest.mark.usefixtures("keep_cuda_visible_devices")
class TestMPS:
    """Tests that need real MPS hardware."""

    def test_auto_selects_mps(self) -> None:
        """On a Mac without CUDA, auto-detection lands on MPS."""
        assert select_device().type == "mps"

    def test_explicit_mps(self) -> None:
        """An explicit "mps" request is honoured."""
        assert select_device("mps").type == "mps"

    def test_tensor_roundtrip_on_mps(self) -> None:
        """Tensors actually run on the selected device."""
        device = select_device("mps")
        tensor = torch.ones(4, 4, device=device)
        assert tensor.device.type == "mps"
        assert torch.equal((tensor * 2).cpu(), torch.full((4, 4), 2.0))
