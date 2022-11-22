# OBSS SAHI Tool
# Code written by Fatih C Akyon and Devrim Cavusoglu, 2022.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybboxes.functional as pbf
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements, ensure_package_minimum_version

logger = logging.getLogger(__name__)


class HuggingfaceDetectionModel(DetectionModel):
    import torch

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):

        self._feature_extractor = feature_extractor
        self._image_shapes = []
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

    def check_dependencies(self):
        check_requirements(["torch", "transformers"])
        ensure_package_minimum_version("transformers", "4.24.0")

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def image_shapes(self):
        return self._image_shapes

    @property
    def num_categories(self) -> int:
        """
        Returns number of categories
        """
        return self.model.config.num_labels

    def load_model(self):

        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

        model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        if self.image_size is not None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_path, size=self.image_size, do_resize=True
            )
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.set_model(model, feature_extractor)

    def set_model(self, model: Any, feature_extractor: Any = None):
        feature_extractor = feature_extractor or self.feature_extractor
        if feature_extractor is None:
            raise ValueError(f"'feature_extractor' is required to be set, got {feature_extractor}.")
        elif (
            "ObjectDetection" not in model.__class__.__name__
            or "FeatureExtractor" not in feature_extractor.__class__.__name__
        ):
            raise ValueError(
                "Given 'model' is not an ObjectDetectionModel or 'feature_extractor' is not a valid FeatureExtractor."
            )
        self.model = model
        self.model.to(self.device)
        self._feature_extractor = feature_extractor
        self.category_mapping = self.model.config.id2label

    def perform_inference(self, images: List):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            images: List[np.ndarray, PIL.Image.Image]
                A numpy array that contains a list of images to be predicted. 3 channel image should be in RGB order.
        """
        import torch

        if not isinstance(images, list):
            images = [images]

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        # save image shapes
        for image in images:
            if isinstance(image, np.ndarray):
                self._image_shapes.append(image.shape[:2])
            elif isinstance(image, Image.Image):
                self._image_shapes.append(image.size[::-1])
            else:
                raise ValueError(f"Unsupported image type {type(image)}")

        with torch.no_grad():
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            inputs["pixel_values"] = inputs.pixel_values.to(self.device)
            if hasattr(inputs, "pixel_mask"):
                inputs["pixel_mask"] = inputs.pixel_mask.to(self.device)
            outputs = self.model(**inputs)

        self._original_predictions = outputs

    def get_valid_predictions(
        self, logits: torch.Tensor, pred_boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch

        probs = logits.softmax(-1)
        scores = probs.max(-1).values
        cat_ids = probs.argmax(-1)
        valid_detections = torch.where(cat_ids < self.num_categories, 1, 0)
        valid_confidences = torch.where(scores >= self.confidence_threshold, 1, 0)
        valid_mask = valid_detections.logical_and(valid_confidences)
        scores = scores[valid_mask]
        cat_ids = cat_ids[valid_mask]
        boxes = pred_boxes[valid_mask]
        return scores, cat_ids, boxes

    def _create_object_predictions_from_original_predictions(
        self,
        shift_amounts: Optional[List[List[int]]] = [[0, 0]],
        full_shapes: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_predictions_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        n_image = original_predictions.logits.shape[0]
        object_predictions_per_image = []
        for image_ind in range(n_image):
            image_height, image_width = self.image_shapes[image_ind]
            scores, cat_ids, boxes = self.get_valid_predictions(
                logits=original_predictions.logits[image_ind], pred_boxes=original_predictions.pred_boxes[image_ind]
            )

            # create object_predictions
            object_predictions = []

            shift_amount = shift_amounts[image_ind]
            full_shape = None if full_shapes is None else full_shapes[image_ind]

            for ind in range(len(boxes)):
                category_id = cat_ids[ind].item()
                yolo_bbox = boxes[ind].tolist()
                bbox = list(
                    pbf.convert_bbox(
                        yolo_bbox,
                        from_type="yolo",
                        to_type="voc",
                        image_size=(image_width, image_height),
                        return_values=True,
                        strict=False,
                    )
                )

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(bbox[2], image_width)
                bbox[3] = min(bbox[3], image_height)

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    bool_mask=None,
                    category_id=category_id,
                    category_name=self.category_mapping[category_id],
                    shift_amount=shift_amount,
                    score=scores[ind].item(),
                    full_shape=full_shape,
                )
                object_predictions.append(object_prediction)
            object_predictions_per_image.append(object_predictions)

        self._object_predictions_per_image = object_predictions_per_image
