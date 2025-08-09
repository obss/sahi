"""
Tests for SAHI Batched Inference functionality.

Tests the new batched_inference parameter in get_sliced_prediction
and the BatchedSAHIInference class.
"""

import unittest
from unittest.mock import Mock, patch
from PIL import Image

from sahi.models.batched_inference import BatchedSAHIInference
from sahi.predict import get_sliced_prediction
from tests.utils_batched_inference import (
    create_mock_detection_model,
    create_test_image,
    create_mock_slice_result
)


class TestBatchedInference(unittest.TestCase):
    """Test cases for batched inference functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock detection model using utility function
        self.mock_model = create_mock_detection_model(
            device='cpu',
            confidence_threshold=0.25,
            image_size=640
        )
        
        # Create a test image using utility function
        self.test_image = create_test_image(1024, 1024, 'red')

    def test_batched_inference_parameter_exists(self):
        """Test that batched_inference parameter is accepted."""
        with patch('sahi.predict.BatchedSAHIInference') as mock_batched:
            with patch('sahi.predict.slice_image') as mock_slice:
                # Mock slice_image to return minimal result
                mock_slice_result = Mock()
                mock_slice_result.images = [self.test_image]
                mock_slice_result.starting_pixels = [(0, 0)]
                mock_slice_result.original_image_height = 1024
                mock_slice_result.original_image_width = 1024
                mock_slice_result.__len__ = Mock(return_value=1)  # Mock len() method
                mock_slice.return_value = mock_slice_result
                
                # Mock batched inference
                mock_batched_instance = Mock()
                mock_batched_instance.batched_slice_inference.return_value = []
                mock_batched.return_value = mock_batched_instance
                
                # Test that function accepts batched_inference parameter
                try:
                    result = get_sliced_prediction(
                        image=self.test_image,
                        detection_model=self.mock_model,
                        batched_inference=True,
                        batch_size=8
                    )
                    self.assertIsNotNone(result)
                except TypeError as e:
                    self.fail(f"get_sliced_prediction should accept batched_inference parameter: {e}")

    def test_backward_compatibility(self):
        """Test that existing code without batched_inference still works."""
        with patch('sahi.predict.slice_image') as mock_slice:
            with patch('sahi.predict.get_prediction') as mock_pred:
                # Mock slice_image
                mock_slice_result = Mock()
                mock_slice_result.images = [self.test_image]
                mock_slice_result.starting_pixels = [(0, 0)]
                mock_slice_result.original_image_height = 1024
                mock_slice_result.original_image_width = 1024
                mock_slice_result.__len__ = Mock(return_value=1)  # Mock len() method
                mock_slice.return_value = mock_slice_result
                
                # Mock get_prediction
                mock_pred.return_value.object_prediction_list = []
                
                # Test without batched_inference parameter (should work)
                result = get_sliced_prediction(
                    image=self.test_image,
                    detection_model=self.mock_model
                )
                self.assertIsNotNone(result)

    @patch('sahi.models.batched_inference.torch')
    def test_batched_inference_class_initialization(self, mock_torch):
        """Test BatchedSAHIInference class initialization."""
        mock_torch.cuda.is_available.return_value = True
        
        # Test initialization
        batched_inferencer = BatchedSAHIInference(
            model=self.mock_model,
            device='cuda',
            batch_size=12,
            image_size=640
        )
        
        self.assertEqual(batched_inferencer.device, 'cuda')
        self.assertEqual(batched_inferencer.batch_size, 12)
        self.assertEqual(batched_inferencer.image_size, 640)

    def test_fallback_to_standard_inference(self):
        """Test that system falls back to standard inference if batched fails."""
        with patch('sahi.predict.BatchedSAHIInference') as mock_batched:
            with patch('sahi.predict.slice_image') as mock_slice:
                with patch('sahi.predict.get_prediction') as mock_pred:
                    # Setup mocks
                    mock_slice_result = Mock()
                    mock_slice_result.images = [self.test_image, self.test_image]
                    mock_slice_result.starting_pixels = [(0, 0), (512, 0)]
                    mock_slice_result.original_image_height = 1024
                    mock_slice_result.original_image_width = 1024
                    mock_slice_result.__len__ = Mock(return_value=2)  # Mock len() method
                    mock_slice.return_value = mock_slice_result
                    
                    # Make batched inference fail
                    mock_batched.side_effect = Exception("Batched inference failed")
                    
                    # Mock standard inference
                    mock_pred.return_value.object_prediction_list = []
                    
                    # Should not raise exception, should fall back
                    result = get_sliced_prediction(
                        image=self.test_image,
                        detection_model=self.mock_model,
                        batched_inference=True,
                        verbose=0  # Suppress warning messages
                    )
                    self.assertIsNotNone(result)

    def test_single_slice_uses_standard_inference(self):
        """Test that single slice doesn't use batched inference."""
        with patch('sahi.predict.slice_image') as mock_slice:
            with patch('sahi.predict.get_prediction') as mock_pred:
                # Mock single slice
                mock_slice_result = Mock()
                mock_slice_result.images = [self.test_image]  # Only one slice
                mock_slice_result.starting_pixels = [(0, 0)]
                mock_slice_result.original_image_height = 1024
                mock_slice_result.original_image_width = 1024
                mock_slice_result.__len__ = Mock(return_value=1)  # Mock len() method
                mock_slice.return_value = mock_slice_result
                
                mock_pred.return_value.object_prediction_list = []
                
                # Even with batched_inference=True, should use standard for single slice
                result = get_sliced_prediction(
                    image=self.test_image,
                    detection_model=self.mock_model,
                    batched_inference=True
                )
                self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
