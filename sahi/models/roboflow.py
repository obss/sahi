from __future__ import annotations

from typing import Any

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


class RoboflowDetectionModel(DetectionModel):
    def __init__(
        self,
        model: Any | None = None,
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
    ):
        """Initialize the RoboflowDetectionModel with the given parameters.

        Args:
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
            self.required_packages = [*list(getattr(self, "required_packages", [])), "inference"]
        else:
            self.required_packages = [*list(getattr(self, "required_packages", [])), "rfdetr"]

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

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        self.model = model

    def load_model(self):
        """This function should be implemented in a way that detection model should be initialized and set to
        self.model.

        (self.model_path, self.config_path, and self.device should be utilized)
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

            assert model.task_type == "object-detection", "Roboflow model must be an object detection model."

        else:
            from rfdetr.detr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

            model, model_path = self._model, self.model_path
            model_names = ("RFDETRBase", "RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge")
            if hasattr(model, "__name__") and model.__name__ in model_names:
                model_params = dict(
                    resolution=int(self.image_size) if self.image_size else 560,
                    device=self._device,
                    num_classes=len(self.category_mapping.keys()) if self.category_mapping else None,
                )
                if model_path:
                    model_params["pretrain_weights"] = model_path

                model = model(**model_params)
            elif isinstance(model, (RFDETRBase, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge)):
                model = model
            else:
                raise ValueError(
                    f"Model must be a Roboflow model string or one of {model_names} models, got {self.model}."
                )

        self.set_model(model)

    def perform_inference(
        self,
        image: np.ndarray,
    ):
        """This function should be implemented in a way that prediction should be performed using self.model and the
        prediction result should be set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        if self._use_universe:
            self._original_predictions = self.model.infer(image, confidence=self.confidence_threshold)
        else:
            self._original_predictions = [self.model.predict(image, threshold=self.confidence_threshold)]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        """This function should be implemented in a way that self._original_predictions should be converted to a list of
        prediction.ObjectPrediction and set to self._object_prediction_list.

        self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        # compatibility for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list: list[ObjectPrediction] = []

        if self._use_universe:
            from inference.core.entities.responses.inference import (
                ObjectDetectionInferenceResponse as InferenceObjectDetectionInferenceResponse,
            )
            from inference.core.entities.responses.inference import (
                ObjectDetectionPrediction as InferenceObjectDetectionPrediction,
            )

            original_reponses: list[InferenceObjectDetectionInferenceResponse] = self._original_predictions

            assert len(original_reponses) == len(shift_amount_list) == len(full_shape_list), (
                "Length mismatch between original responses, shift amounts, and full shapes."
            )

            for original_reponse, shift_amount, full_shape in zip(
                original_reponses,
                shift_amount_list,
                full_shape_list,
            ):
                for prediction in original_reponse.predictions:
                    prediction: InferenceObjectDetectionPrediction
                    bbox = [
                        prediction.x - prediction.width / 2,
                        prediction.y - prediction.height / 2,
                        prediction.x + prediction.width / 2,
                        prediction.y + prediction.height / 2,
                    ]
                    object_prediction = ObjectPrediction(
                        bbox=bbox,
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

            assert len(original_detections) == len(shift_amount_list) == len(full_shape_list), (
                "Length mismatch between original responses, shift amounts, and full shapes."
            )

            for original_detection, shift_amount, full_shape in zip(
                original_detections,
                shift_amount_list,
                full_shape_list,
            ):
                for xyxy, confidence, class_id in zip(
                    original_detection.xyxy,
                    original_detection.confidence,
                    original_detection.class_id,
                ):
                    object_prediction = ObjectPrediction(
                        bbox=xyxy,
                        category_id=int(class_id),
                        category_name=self.category_mapping.get(int(class_id), None),
                        score=float(confidence),
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)

        object_prediction_list_per_image = [object_prediction_list]
        self._object_prediction_list_per_image = object_prediction_list_per_image
