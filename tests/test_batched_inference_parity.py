from __future__ import annotations
import numpy as np
import pytest
from typing import Any, List

from sahi.models.base import DetectionModel


def test_perform_inference_batch_fallback():
    """Test that perform_inference_batch falls back to sequential calls."""
    
    class DummyModel(DetectionModel):
        def __init__(self):
            super().__init__()
            self.model = "dummy"
            self.call_count = 0

        def load_model(self):
            pass

        def perform_inference(self, image: Any, **kwargs: Any) -> Any:
            self.call_count += 1
            return f"result_{self.call_count}"

    model = DummyModel()
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    
    # Test fallback behavior
    results = model.perform_inference_batch(images)
    
    # Should have called perform_inference 3 times
    assert model.call_count == 3
    assert len(results) == 3
    assert results == ["result_1", "result_2", "result_3"]


def test_perform_inference_batch_override():
    """Test that subclasses can override perform_inference_batch."""
    
    class BatchedModel(DetectionModel):
        def __init__(self):
            super().__init__()
            self.model = "dummy"
            self.batch_call_count = 0

        def load_model(self):
            pass

        def perform_inference(self, image: Any, **kwargs: Any) -> Any:
            return "single_result"

        def perform_inference_batch(self, images: List[Any], **kwargs: Any) -> List[Any]:
            self.batch_call_count += 1
            return [f"batch_result_{i}" for i in range(len(images))]

    model = BatchedModel()
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    
    # Test override behavior
    results = model.perform_inference_batch(images)
    
    # Should have called perform_inference_batch once, not perform_inference
    assert model.batch_call_count == 1
    assert len(results) == 3
    assert results == ["batch_result_0", "batch_result_1", "batch_result_2"]
