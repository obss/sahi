"""Tests for postprocessing utilities."""

from __future__ import annotations

from sahi.postprocess.utils import ObjectPredictionList


class TestPostprocessUtils:
    """Test postprocessing utility functions."""

    def setup_method(self) -> None:
        """Initialize test fixtures."""
        self.test_input = [ObjectPredictionList([1, 2, 3, 4])]

    def test_get_item_int(self) -> None:
        """Test getting item by integer index."""
        obj = self.test_input[0]
        assert obj[0].tolist() == 1

    def test_len(self) -> None:
        """Test length of ObjectPredictionList."""
        obj = self.test_input[0]
        assert len(obj) == 4

    def test_extend(self) -> None:
        """Test extending ObjectPredictionList."""
        import torch

        obj = self.test_input[0]
        obj.extend(ObjectPredictionList([torch.randn(1, 2, 3, 4)]))
        assert len(obj) == 5

    def test_tostring(self) -> None:
        """Test string representation of ObjectPredictionList."""
        obj = self.test_input[0]
        assert str(obj) == str([1, 2, 3, 4])
