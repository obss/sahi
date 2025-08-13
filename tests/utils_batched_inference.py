"""
Utility functions for batched inference tests.
"""

# Standard library imports
from unittest.mock import Mock

# Third-party imports
import numpy as np
from PIL import Image


def create_test_image(width=100, height=100, color="red"):
    """Create a test image with specified dimensions and color."""
    return Image.new("RGB", (width, height), color=color)


def create_mock_detection_model():
    """Create a mock detection model for testing."""
    import torch

    model = Mock()
    model.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model = Mock()

    # Mock inference results
    mock_result = Mock()
    mock_result.boxes = Mock()
    mock_result.boxes.xyxy = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])
    mock_result.boxes.conf = torch.tensor([0.8, 0.9])
    mock_result.boxes.cls = torch.tensor([0, 1])

    model.model.return_value = [mock_result]
    model.perform_inference.return_value = mock_result

    return model


def create_mock_slice_result(slice_images, offsets):
    """Create mock slice result data structure."""
    slices = []
    for img, offset in zip(slice_images, offsets):
        slice_data = Mock()
        slice_data.image = img
        slice_data.starting_pixel = offset
        slices.append(slice_data)
    return slices


def create_test_slices(num_slices=4, slice_size=256):
    """Create test slice images and offsets."""
    slices = []
    offsets = []

    for i in range(num_slices):
        for j in range(num_slices):
            slice_img = Image.new("RGB", (slice_size, slice_size), color="blue")
            slices.append(slice_img)
            offsets.append((i * 200, j * 200))  # With overlap

    return slices, offsets


def assert_detection_results_equal(results1, results2, tolerance=1e-6):
    """Assert that two detection results are equal within tolerance."""
    assert len(results1) == len(results2), f"Different number of detections: {len(results1)} vs {len(results2)}"

    for r1, r2 in zip(results1, results2):
        assert np.allclose(r1["bbox"], r2["bbox"], atol=tolerance), f"Bbox mismatch: {r1['bbox']} vs {r2['bbox']}"
        assert np.allclose(r1["score"], r2["score"], atol=tolerance), f"Score mismatch: {r1['score']} vs {r2['score']}"
        assert r1["class_id"] == r2["class_id"], f"Class mismatch: {r1['class_id']} vs {r2['class_id']}"


def create_benchmark_data(image_sizes=[(512, 512), (1024, 1024), (2048, 2048)]):
    """Create benchmark test data with different image sizes."""
    benchmark_data = []

    for width, height in image_sizes:
        image = Image.new("RGB", (width, height), color="green")
        slices, offsets = create_test_slices(num_slices=4, slice_size=min(width, height) // 2)
        benchmark_data.append({"image": image, "slices": slices, "offsets": offsets, "size": (width, height)})

    return benchmark_data
