# OBSS SAHI Tool
# Code written by AnNT, 2023.

import logging
from typing import Any, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements


class UltralyticsDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        from ultralytics import YOLO

        try:
            model = YOLO(self.model_path)
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
        If predictions have masks, each prediction is a tuple like (boxes, masks).
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.

        """

        from ultralytics.engine.results import Masks

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device}

        if self.image_size is not None:
            kwargs = {"imgsz": self.image_size, **kwargs}

        prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLOv8 expects numpy arrays to have BGR

        if self.has_mask:
            if not prediction_result[0].masks:
                prediction_result[0].masks = Masks(
                    torch.tensor([], device=self.model.device), prediction_result[0].boxes.orig_shape
                )

            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [
                (
                    result.boxes.data,
                    result.masks.data,
                )
                for result in prediction_result
            ]

        else:  # If model doesn't do segmentation then no need to check masks
            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [result.boxes.data for result in prediction_result]

        self._original_predictions = prediction_result
        self._original_shape = image.shape

    @property
    def category_names(self):
        return self.model.names.values()

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return self.model.overrides["task"] == "segment"

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

            # Extract boxes and optional masks
            if self.has_mask:
                boxes = image_predictions[0].cpu().detach().numpy()
                masks = image_predictions[1].cpu().detach().numpy()
            else:
                boxes = image_predictions.data.cpu().detach().numpy()
                masks = None

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

                # Get segmentation if available
                segmentation = None
                if masks is not None:
                    bool_mask = masks[pred_ind]
                    orig_width = self._original_shape[1]
                    orig_height = self._original_shape[0]
                    bool_mask = cv2.resize(bool_mask.astype(np.uint8), (orig_width, orig_height))
                    segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
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
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image