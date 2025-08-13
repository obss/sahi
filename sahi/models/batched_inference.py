"""
SAHI Batched Inference Implementation
====================================

This file contains the actual implementation that can be directly
integrated into the SAHI repository for batched GPU inference.
"""

# Standard library imports
from typing import Any, List, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Local imports
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction


def get_sliced_prediction_batched(
    image: Union[Image.Image, str, np.ndarray],
    detection_model: DetectionModel,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    conf_th: float = 0.25,
    image_size: int = 640,
    batched_inference: bool = True,
    batch_size: int = 12,
    **kwargs,
) -> Any:
    """
    Perform sliced prediction with optional batched inference for improved performance.

    Args:
        image: Input image
        detection_model: SAHI detection model
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Height overlap ratio between slices
        overlap_width_ratio: Width overlap ratio between slices
        conf_th: Confidence threshold
        image_size: Model input size
        batched_inference: Whether to use batched inference (default: True)
        batch_size: Number of slices to process simultaneously
        **kwargs: Additional arguments

    Returns:
        PredictionResult object with detected objects
    """
    # Local imports to avoid circular dependencies
    from sahi.postprocess.combine import PredictionResult
    from sahi.slicing import slice_image

    # Convert image to PIL if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Slice the image
    slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    # Prepare slices and offsets
    slice_images = []
    slice_offsets = []

    for slice_data in slice_image_result:
        if hasattr(slice_data, "image"):
            slice_images.append(slice_data.image)
            slice_offsets.append(slice_data.starting_pixel)
        elif isinstance(slice_data, dict):
            slice_images.append(slice_data["image"])
            slice_offsets.append(slice_data["starting_pixel"])

    # Perform inference
    if batched_inference and len(slice_images) > 1:
        # Use batched inference for better performance
        object_prediction_list = _perform_batched_inference(
            slice_images=slice_images,
            slice_offsets=slice_offsets,
            detection_model=detection_model,
            batch_size=batch_size,
            conf_th=conf_th,
            image_size=image_size,
        )
    else:
        # Fallback to standard inference
        object_prediction_list = _perform_standard_inference(
            slice_images=slice_images,
            slice_offsets=slice_offsets,
            detection_model=detection_model,
            conf_th=conf_th,
            image_size=image_size,
        )

    # Create prediction result
    prediction_result = PredictionResult(object_prediction_list=object_prediction_list, image=image)

    return prediction_result


def _perform_batched_inference(
    slice_images: List[Image.Image],
    slice_offsets: List[Tuple[int, int]],
    detection_model: DetectionModel,
    batch_size: int,
    conf_th: float,
    image_size: int,
) -> List[ObjectPrediction]:
    """
    Perform batched inference on image slices.

    Args:
        slice_images: List of PIL image slices
        slice_offsets: List of slice offsets
        detection_model: SAHI detection model
        batch_size: Batch size for inference
        conf_th: Confidence threshold
        image_size: Model input size

    Returns:
        List of ObjectPrediction objects
    """
    all_predictions = []
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    # Process in batches
    for i in range(0, len(slice_images), batch_size):
        batch_slices = slice_images[i : i + batch_size]
        batch_offsets = slice_offsets[i : i + batch_size]

        # Prepare batch tensors
        batch_tensors = []
        for slice_img in batch_slices:
            tensor = transform(slice_img).unsqueeze(0)
            batch_tensors.append(tensor)

        if batch_tensors:
            # Stack into batch and move to device
            batch = torch.cat(batch_tensors, dim=0)
            if hasattr(detection_model, "device"):
                batch = batch.to(detection_model.device)

            # Batch inference
            with torch.no_grad():
                if hasattr(detection_model.model, "__call__"):
                    # For YOLO models
                    batch_results = detection_model.model(batch, verbose=False)
                else:
                    # For other models
                    batch_results = detection_model.perform_inference(batch)

            # Process batch results
            for j, (result, offset) in enumerate(zip(batch_results, batch_offsets)):
                slice_predictions = _extract_predictions_from_result(
                    result=result,
                    offset=offset,
                    conf_th=conf_th,
                    slice_width=slice_images[i + j].width,
                    slice_height=slice_images[i + j].height,
                    image_size=image_size,
                )
                all_predictions.extend(slice_predictions)

    return all_predictions


def _perform_standard_inference(
    slice_images: List[Image.Image],
    slice_offsets: List[Tuple[int, int]],
    detection_model: DetectionModel,
    conf_th: float,
    image_size: int,
) -> List[ObjectPrediction]:
    """
    Perform standard (sequential) inference on image slices.

    Fallback method for compatibility.
    """
    all_predictions = []

    for slice_img, offset in zip(slice_images, slice_offsets):
        # Standard SAHI inference
        result = detection_model.perform_inference(slice_img, image_size)

        slice_predictions = _extract_predictions_from_result(
            result=result,
            offset=offset,
            conf_th=conf_th,
            slice_width=slice_img.width,
            slice_height=slice_img.height,
            image_size=image_size,
        )
        all_predictions.extend(slice_predictions)

    return all_predictions


def _extract_predictions_from_result(
    result: Any, offset: Tuple[int, int], conf_th: float, slice_width: int, slice_height: int, image_size: int
) -> List[ObjectPrediction]:
    """
    Extract ObjectPrediction objects from model result.

    Args:
        result: Model inference result
        offset: Slice offset in original image
        conf_th: Confidence threshold
        slice_width: Width of the slice
        slice_height: Height of the slice
        image_size: Model input size

    Returns:
        List of ObjectPrediction objects
    """
    predictions = []
    offset_x, offset_y = offset

    # Scale factors from model input to slice size
    scale_x = slice_width / image_size
    scale_y = slice_height / image_size

    if hasattr(result, "boxes") and result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= conf_th:
                x1, y1, x2, y2 = box

                # Scale coordinates back to slice size
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                # Adjust to original image coordinates
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                # Create ObjectPrediction
                prediction = ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    score=score,
                    category_id=int(class_id),
                    category_name=str(int(class_id)),  # Can be mapped to actual names
                )
                predictions.append(prediction)

    return predictions


# Performance monitoring
class BatchedInferenceProfiler:
    """Profiler for batched inference performance."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.inference_times = []
        self.batch_sizes = []
        self.total_slices = 0

    def log_batch(self, inference_time: float, batch_size: int):
        self.inference_times.append(inference_time)
        self.batch_sizes.append(batch_size)
        self.total_slices += batch_size

    def get_stats(self) -> dict:
        if not self.inference_times:
            return {}

        return {
            "total_inference_time": sum(self.inference_times),
            "avg_batch_time": np.mean(self.inference_times),
            "total_slices": self.total_slices,
            "avg_batch_size": np.mean(self.batch_sizes),
            "slices_per_second": self.total_slices / sum(self.inference_times) if sum(self.inference_times) > 0 else 0,
        }


class BatchedSAHIInference:
    """Main class for batched SAHI inference."""
    
    def __init__(self, detection_model: DetectionModel, batch_size: int = 12):
        self.detection_model = detection_model
        self.batch_size = batch_size
        self.profiler = BatchedInferenceProfiler()
    
    def batched_slice_inference(
        self,
        slice_images: List[Image.Image],
        slice_offsets: List[Tuple[int, int]],
        conf_th: float = 0.25,
        image_size: int = 640,
    ) -> List[ObjectPrediction]:
        """Perform batched inference on image slices."""
        return _perform_batched_inference(
            slice_images=slice_images,
            slice_offsets=slice_offsets,
            detection_model=self.detection_model,
            batch_size=self.batch_size,
            conf_th=conf_th,
            image_size=image_size,
        )
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return self.profiler.get_stats()
