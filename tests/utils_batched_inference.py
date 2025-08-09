"""
Test utilities for batched inference functionality.

This module contains mock functions and test helpers that were moved
from the main batched_inference.py module as requested by reviewers.
"""

from typing import List, Union
from PIL import Image


def mock_slice_image(image, slice_height, slice_width, overlap_h_ratio, overlap_w_ratio):
    """
    Mock slice_image function for testing purposes.
    
    This is a simplified version for testing - real SAHI slice_image is more sophisticated.
    
    Args:
        image: PIL Image to slice
        slice_height: Height of each slice
        slice_width: Width of each slice  
        overlap_h_ratio: Height overlap ratio
        overlap_w_ratio: Width overlap ratio
        
    Returns:
        List of mock slice data objects
    """
    w, h = image.size
    slices = []
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Create slice
            slice_img = image.crop((x, y, min(x + slice_width, w), min(y + slice_height, h)))
            
            # Mock slice data object
            slice_data = type('SliceData', (), {
                'image': slice_img,
                'starting_pixel': (x, y)
            })()
            
            slices.append(slice_data)
            
            # Move to next x position
            x += int(slice_width * (1 - overlap_w_ratio))
            if x >= w:
                break
        
        # Move to next y position
        y += int(slice_height * (1 - overlap_h_ratio))
        if y >= h:
            break
    
    return slices


def mock_standard_sahi_inference(slice_images, slice_offsets, detection_model, conf_th, image_size):
    """
    Mock standard SAHI inference for testing backward compatibility.
    
    Args:
        slice_images: List of PIL Image slices
        slice_offsets: List of (x, y) offset tuples
        detection_model: Mock detection model
        conf_th: Confidence threshold
        image_size: Model input size
        
    Returns:
        Empty list (placeholder for actual SAHI inference)
    """
    object_prediction_list = []
    
    for slice_img, offset in zip(slice_images, slice_offsets):
        # This would call the original SAHI inference method
        # For testing, return empty list as placeholder
        pass
    
    return object_prediction_list


def create_mock_detection_model(device='cpu', confidence_threshold=0.25, image_size=640):
    """
    Create a mock detection model for testing.
    
    Args:
        device: Device string ('cpu' or 'cuda')
        confidence_threshold: Confidence threshold
        image_size: Model input size
        
    Returns:
        Mock model object with required attributes
    """
    from unittest.mock import Mock
    
    mock_model = Mock()
    mock_model.device = device
    mock_model.confidence_threshold = confidence_threshold
    mock_model.image_size = image_size
    mock_model.model = Mock()  # Inner model for frameworks like Ultralytics
    
    return mock_model


def create_test_image(width=1024, height=1024, color='red'):
    """
    Create a test PIL Image for testing.
    
    Args:
        width: Image width
        height: Image height
        color: Background color
        
    Returns:
        PIL Image object
    """
    return Image.new('RGB', (width, height), color=color)


def create_mock_batch_results(num_results=2, detections_per_result=3):
    """
    Create mock batch results for testing.
    
    Args:
        num_results: Number of batch results
        detections_per_result: Number of detections per result
        
    Returns:
        List of mock result dictionaries
    """
    results = []
    
    for i in range(num_results):
        detections = []
        for j in range(detections_per_result):
            detection = {
                'bbox': [10 + j*50, 10 + i*50, 60 + j*50, 60 + i*50],
                'score': 0.8 - (j * 0.1),
                'class_id': j % 3,
                'class_name': f'class_{j % 3}'
            }
            detections.append(detection)
        
        result = {
            'detections': detections,
            'slice_offset': (i * 100, 0),
            'num_detections': len(detections)
        }
        results.append(result)
    
    return results


def create_mock_slice_result(num_slices=4, image_size=(512, 512)):
    """
    Create mock slice result for testing.
    
    Args:
        num_slices: Number of slices to create
        image_size: Size of each slice image
        
    Returns:
        Mock slice result object
    """
    from unittest.mock import Mock
    
    # Create mock slice images
    slice_images = [create_test_image(image_size[0], image_size[1]) for _ in range(num_slices)]
    
    # Create mock starting pixels
    starting_pixels = [(i * 100, 0) for i in range(num_slices)]
    
    # Create mock slice result
    mock_result = Mock()
    mock_result.images = slice_images
    mock_result.starting_pixels = starting_pixels
    mock_result.original_image_height = 1024
    mock_result.original_image_width = 1024
    mock_result.__len__ = Mock(return_value=num_slices)
    
    return mock_result


class MockPerformanceStats:
    """Mock performance statistics for testing."""
    
    def __init__(self):
        self.stats = {
            'total_inference_time': 0.045,
            'avg_batch_time': 0.012,
            'total_slices': 12,
            'slices_per_second': 42.0,
            'estimated_fps_improvement': 3.5
        }
    
    def get_stats(self):
        return self.stats
    
    def log_batch(self, inference_time, batch_size):
        pass
    
    def reset(self):
        pass


def validate_batched_inference_result(result):
    """
    Validate that batched inference result has correct structure.
    
    Args:
        result: Result from get_sliced_prediction with batched_inference=True
        
    Returns:
        bool: True if result structure is valid
    """
    required_keys = ['object_prediction_list', 'durations_in_seconds']
    
    if not isinstance(result, dict):
        return False
    
    for key in required_keys:
        if key not in result:
            return False
    
    # Check object_prediction_list is a list
    if not isinstance(result['object_prediction_list'], list):
        return False
    
    # Check durations_in_seconds is a dict
    if not isinstance(result['durations_in_seconds'], dict):
        return False
    
    return True


def compare_performance_results(standard_result, batched_result, tolerance=0.1):
    """
    Compare performance between standard and batched inference results.
    
    Args:
        standard_result: Result from standard inference
        batched_result: Result from batched inference
        tolerance: Tolerance for comparing detection counts
        
    Returns:
        dict: Performance comparison metrics
    """
    standard_detections = len(standard_result.get('object_prediction_list', []))
    batched_detections = len(batched_result.get('object_prediction_list', []))
    
    standard_time = standard_result.get('durations_in_seconds', {}).get('prediction', 0)
    batched_time = batched_result.get('durations_in_seconds', {}).get('prediction', 0)
    
    speedup = standard_time / batched_time if batched_time > 0 else 0
    detection_diff = abs(standard_detections - batched_detections) / max(standard_detections, 1)
    
    return {
        'speedup': speedup,
        'detection_accuracy': 1.0 - detection_diff,
        'within_tolerance': detection_diff <= tolerance,
        'standard_detections': standard_detections,
        'batched_detections': batched_detections,
        'standard_time': standard_time,
        'batched_time': batched_time
    }
