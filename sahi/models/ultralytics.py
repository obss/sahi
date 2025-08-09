# OBSS SAHI Tool
# Code written by Fatih Cagatay Akyon, 2025.

from typing import Any, List, Optional

import cv2
import numpy as np
import torch

from sahi.logger import logger
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements


class UltralyticsDetectionModel(DetectionModel):
    """
    Detection model for Ultralytics YOLO models.

    Supports both PyTorch (.pt) and ONNX (.onnx) models.
    """

    def __init__(self, *args, **kwargs):
        self.required_packages = list(getattr(self, "required_packages", [])) + ["ultralytics"]
        super().__init__(*args, **kwargs)

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        Supports both PyTorch (.pt) and ONNX (.onnx) models.
        """

        from ultralytics import YOLO

        if self.model_path and ".onnx" in self.model_path:
            check_requirements(["onnx", "onnxruntime"])

        try:
            model = YOLO(self.model_path)
            # Only call .to(device) for PyTorch models, not ONNX
            if self.model_path and not self.model_path.endswith(".onnx"):
                model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid Ultralytics model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying Ultralytics model.
        Args:
            model: Any
                A Ultralytics model
        """

        self.model = model
        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device}

        if self.image_size is not None:
            kwargs = {"imgsz": self.image_size, **kwargs}

        prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLO expects numpy arrays to have BGR

        # Handle different result types for PyTorch vs ONNX models
        # ONNX models might return results in a different format
        if self.has_mask:
            from ultralytics.engine.results import Masks

            if not prediction_result[0].masks:
                # Create empty masks if none exist
                if hasattr(self.model, "device"):
                    device = self.model.device
                else:
                    device = "cpu"  # Default for ONNX models
                prediction_result[0].masks = Masks(
                    torch.tensor([], device=device), prediction_result[0].boxes.orig_shape
                )

            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [
                (
                    result.boxes.data,
                    result.masks.data,
                )
                for result in prediction_result
            ]
        elif self.is_obb:
            # For OBB task, get OBB points in xyxyxyxy format
            device = getattr(self.model, "device", "cpu")
            prediction_result = [
                (
                    # Get OBB data: xyxy, conf, cls
                    torch.cat(
                        [
                            result.obb.xyxy,  # box coordinates
                            result.obb.conf.unsqueeze(-1),  # confidence scores
                            result.obb.cls.unsqueeze(-1),  # class ids
                        ],
                        dim=1,
                    )
                    if result.obb is not None
                    else torch.empty((0, 6), device=device),
                    # Get OBB points in (N, 4, 2) format
                    result.obb.xyxyxyxy if result.obb is not None else torch.empty((0, 4, 2), device=device),
                )
                for result in prediction_result
            ]
        else:  # If model doesn't do segmentation or OBB then no need to check masks
            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [result.boxes.data for result in prediction_result]

        self._original_predictions = prediction_result
        self._original_shape = image.shape

    @property
    def category_names(self):
        # For ONNX models, names might not be available, use category_mapping
        if hasattr(self.model, "names") and self.model.names:
            return self.model.names.values()
        elif self.category_mapping:
            return list(self.category_mapping.values())
        else:
            raise ValueError("Category names not available. Please provide category_mapping for ONNX models.")

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        if hasattr(self.model, "names") and self.model.names:
            return len(self.model.names)
        elif self.category_mapping:
            return len(self.category_mapping)
        else:
            raise ValueError("Cannot determine number of categories. Please provide category_mapping for ONNX models.")

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        # Check if model has 'task' attribute (for both .pt and .onnx models)
        if hasattr(self.model, "overrides") and "task" in self.model.overrides:
            return self.model.overrides["task"] == "segment"
        # For ONNX models, task might be stored differently
        elif hasattr(self.model, "task"):
            return self.model.task == "segment"
        # For ONNX models without task info, check model path
        elif self.model_path and isinstance(self.model_path, str):
            return "seg" in self.model_path.lower()
        return False

    @property
    def is_obb(self):
        """
        Returns if model output contains oriented bounding boxes
        """
        # Check if model has 'task' attribute (for both .pt and .onnx models)
        if hasattr(self.model, "overrides") and "task" in self.model.overrides:
            return self.model.overrides["task"] == "obb"
        # For ONNX models, task might be stored differently
        elif hasattr(self.model, "task"):
            return self.model.task == "obb"
        # For ONNX models without task info, check model path
        elif self.model_path and isinstance(self.model_path, str):
            return "obb" in self.model_path.lower()
        return False

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatibility for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []

        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # Extract boxes and optional masks/obb
            if self.has_mask or self.is_obb:
                boxes = image_predictions[0].cpu().detach().numpy()
                masks_or_points = image_predictions[1].cpu().detach().numpy()
            else:
                boxes = image_predictions.data.cpu().detach().numpy()
                masks_or_points = None

            # Process each prediction
            for pred_ind, prediction in enumerate(boxes):
                # Get bbox coordinates
                bbox = prediction[:4].tolist()
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # Fix box coordinates
                bbox = [max(0, coord) for coord in bbox]
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # Ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                # Get segmentation or OBB points
                segmentation = None
                if masks_or_points is not None:
                    if self.has_mask:
                        bool_mask = masks_or_points[pred_ind]
                        # Resize mask to original image size
                        bool_mask = cv2.resize(
                            bool_mask.astype(np.uint8), (self._original_shape[1], self._original_shape[0])
                        )
                        segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                    else:  # is_obb
                        obb_points = masks_or_points[pred_ind]  # Get OBB points for this prediction
                        segmentation = [obb_points.reshape(-1).tolist()]

                    if len(segmentation) == 0:
                        continue

                # Create and append object prediction
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=self._original_shape[:2] if full_shape is None else full_shape,  # (height, width)
                )
                object_prediction_list.append(object_prediction)

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
