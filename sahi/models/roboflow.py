"""Roboflow detection model wrapper for SAHI.

Provides integration with Roboflow's inference SDK for object detection and
instance segmentation models.
"""

from __future__ import annotations

from itertools import chain, zip_longest
from typing import Any

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask


class RoboflowDetectionModel(DetectionModel):
    """Roboflow object detection model.

    Supports both Roboflow Universe models (API-based) and local RF-DETR models.
    """

    def __init__(
        self,
        model: object | None = None,
        model_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = True,
        image_size: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the RoboflowDetectionModel with the given parameters.

        Args:
            model: object
                Either a Roboflow model string identifier or an RF-DETR model class.
            api_key: str
                Roboflow API key for authentication.
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: Torch device, "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initialization
            image_size: int
                Inference input size.
        """
        self._use_universe = model and isinstance(model, str)
        self._model = model
        self._device = device
        self._api_key = api_key

        if self._use_universe:
            existing_packages = getattr(self, "required_packages", None) or []
            self.required_packages = [*list(existing_packages), "inference"]
        else:
            existing_packages = getattr(self, "required_packages", None) or []
            self.required_packages = [*list(existing_packages), "rfdetr"]

        super().__init__(
            model=model,
            model_path=model_path,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=False,
            image_size=image_size,
        )

        if load_at_init:
            self.load_model()

    def set_model(self, model: Any, **kwargs: Any) -> None:
        """Set the detection model.

        Args:
            model: Any
                Loaded model.
            **kwargs: Additional keyword arguments.
        """
        self.model = model

    def load_model(self) -> None:
        """Load detection model from Roboflow.

        This function initializes detection model and sets to self.model.
        Uses self.model_path, self.config_path, and self.device.
        """
        if self._use_universe:
            from inference import get_model
            from inference.core.env import API_KEY
            from inference.core.exceptions import RoboflowAPINotAuthorizedError

            api_key = self._api_key or API_KEY

            try:
                model = get_model(self._model, api_key=api_key)
            except RoboflowAPINotAuthorizedError as e:
                raise ValueError(
                    "Authorization failed. Please pass a valid API key with "
                    "the `api_key` parameter or set the `ROBOFLOW_API_KEY` environment variable."
                ) from e

            assert model.task_type in ["object-detection", "instance-segmentation"], (
                "Roboflow model must be an object detection model or an instance segmentation model."
            )

        else:
            from rfdetr.detr import (
                RFDETRBase,
                RFDETRLarge,
                RFDETRMedium,
                RFDETRNano,
                RFDETRSeg2XLarge,
                RFDETRSegLarge,
                RFDETRSegMedium,
                RFDETRSegNano,
                RFDETRSegSmall,
                RFDETRSegXLarge,
                RFDETRSmall,
            )

            model, model_path = self._model, self.model_path
            model_names = (
                "RFDETRBase",
                "RFDETRNano",
                "RFDETRSmall",
                "RFDETRMedium",
                "RFDETRLarge",
                "RFDETRSegNano",
                "RFDETRSegSmall",
                "RFDETRSegMedium",
                "RFDETRSegLarge",
                "RFDETRSegXLarge",
                "RFDETRSeg2XLarge",
            )
            model_types = (
                RFDETRBase,
                RFDETRNano,
                RFDETRSmall,
                RFDETRMedium,
                RFDETRLarge,
                RFDETRSegNano,
                RFDETRSegSmall,
                RFDETRSegMedium,
                RFDETRSegLarge,
                RFDETRSegXLarge,
                RFDETRSeg2XLarge,
            )
            if hasattr(model, "__name__") and model.__name__ in model_names:
                model_params = dict(
                    device=self._device,
                    num_classes=len(self.category_mapping.keys()) if self.category_mapping else None,
                )
                if model_path:
                    model_params["pretrain_weights"] = model_path
                    if self.image_size:
                        model_params["resolution"] = int(self.image_size)

                model = model(**model_params)  # type: ignore[operator]
            elif isinstance(model, model_types):
                model = model
            else:
                raise ValueError(
                    f"Model must be a Roboflow model string or one of {model_names} models, got {self.model}."
                )

        self.set_model(model)

    def perform_inference(
        self,
        image: np.ndarray,
    ) -> None:
        """Run inference on image and store predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        if self._use_universe:
            self._original_predictions = self.model.infer(image, confidence=self.confidence_threshold)
        else:
            self._original_predictions = [self.model.predict(image, threshold=self.confidence_threshold)]

    @property
    def has_mask(self) -> bool:
        """Returns if model output contains segmentation mask."""
        if self._use_universe:
            return self.model.task_type == "instance-segmentation"
        else:
            return "seg" in self.model.size

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        """Convert predictions to ObjectPrediction list.

        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image. self.mask_threshold can also be utilized.

        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        # compatibility for sahi v0.8.15
        shift_amount_list_typed: list[list[int | float]] = fix_shift_amount_list(shift_amount_list)
        full_shape_list_typed: list[list[int | float]] | None = fix_full_shape_list(full_shape_list)

        object_prediction_list: list[ObjectPrediction] = []

        if self._use_universe:
            try:
                from pycocotools import mask as mask_utils

                can_decode_rle = True
            except ImportError:
                can_decode_rle = False

            original_reponses = self._original_predictions

            assert len(original_reponses) == len(shift_amount_list_typed) == len(full_shape_list_typed or []), (
                "Length mismatch between original responses, shift amounts, and full shapes."
            )

            for original_reponse, shift_amount, full_shape in zip(
                original_reponses,
                shift_amount_list_typed,
                full_shape_list_typed or [],
            ):
                for prediction in original_reponse.predictions:
                    bbox = [
                        prediction.x - prediction.width / 2,
                        prediction.y - prediction.height / 2,
                        prediction.x + prediction.width / 2,
                        prediction.y + prediction.height / 2,
                    ]

                    segmentation: list[list[float]] | None = None
                    if self.has_mask:
                        if hasattr(prediction, "points"):
                            segmentation = [list(chain(*[[pt.x, pt.y] for pt in prediction.points]))]
                        elif hasattr(prediction, "rle"):
                            if can_decode_rle:
                                bool_mask = mask_utils.decode(prediction.rle)
                            else:
                                raise ValueError(
                                    "Can not decode rle mask. Please install pycocotools. ex: 'pip install pycocotools'"
                                )
                            # check if mask is valid
                            if get_bbox_from_bool_mask(bool_mask) is None:
                                continue
                            segmentation = get_coco_segmentation_from_bool_mask(bool_mask)

                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        segmentation=segmentation,
                        category_id=prediction.class_id,
                        category_name=prediction.class_name,
                        score=prediction.confidence,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)

        else:
            from supervision.detection.core import Detections

            original_detections: list[Detections] = self._original_predictions

            assert len(original_detections) == len(shift_amount_list_typed) == len(full_shape_list_typed or []), (
                "Length mismatch between original responses, shift amounts, and full shapes."
            )

            for original_detection, shift_amount, full_shape in zip(
                original_detections,
                shift_amount_list_typed,
                full_shape_list_typed or [],
            ):
                for xyxy, mask, confidence, class_id in zip_longest(
                    original_detection.xyxy,
                    original_detection.mask if original_detection.mask is not None else [],
                    original_detection.confidence,
                    original_detection.class_id,
                ):
                    segmentation = get_coco_segmentation_from_bool_mask(mask) if mask is not None else None

                    object_prediction = ObjectPrediction(
                        bbox=xyxy,
                        segmentation=segmentation,
                        category_id=int(class_id),
                        category_name=self.category_mapping.get(int(class_id), None) if self.category_mapping else None,
                        score=float(confidence),
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)

        object_prediction_list_per_image = [object_prediction_list]
        self._object_prediction_list_per_image = object_prediction_list_per_image
