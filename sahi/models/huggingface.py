"""HuggingFace Transformers detection model wrapper for SAHI.

Provides integration with Hugging Face Transformers library for object detection
and instance segmentation models like DETR variants.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import ensure_package_minimum_version

_TRANSFORMERS_MIN_VERSION = "4.42.0"
_TRANSFORMERS_ZERO_SHOT_MIN_VERSION = "4.49.0"


class HuggingfaceDetectionModel(DetectionModel):
    """HuggingFace Transformers object detection model.

    Supports DETR-style object detection models and GroundingDINO-style zero-shot detection models.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model: object | None = None,
        processor: object | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = True,
        image_size: int | None = None,
        token: str | None = None,
        text_prompt: str | None = None,
        text_labels: list[str] | None = None,
        text_threshold: float = 0.25,
    ) -> None:
        """Initialize HuggingFace detection model."""
        self._processor = processor
        self._original_shapes: list[tuple[int, ...]] = []
        self._token = token
        self.text_prompt = text_prompt
        self.text_labels = text_labels
        self.text_threshold = text_threshold
        self._original_input_ids: Any | None = None
        self._is_zero_shot_model = False
        self._category_name_to_id: dict[str, int] = {}
        existing_packages = getattr(self, "required_packages", None) or []
        self.required_packages = [*list(existing_packages), "torch", "transformers"]
        ensure_package_minimum_version("transformers", _TRANSFORMERS_MIN_VERSION)
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

    @property
    def processor(self) -> Any:
        """Return the image processor."""
        return self._processor

    @property
    def image_shapes(self) -> list:
        """Return original image shapes."""
        # TODO: remove this property in a future release; use _original_shapes directly
        return self._original_shapes

    @property
    def num_categories(self) -> int:
        """Returns number of categories."""
        if self._is_zero_shot_model:
            return len(self.category_mapping)
        return self.model.config.num_labels  # type: ignore[attr-defined]

    def load_model(self) -> None:
        """Load model from HuggingFace."""
        from transformers import AutoConfig, AutoModelForObjectDetection, AutoProcessor

        hf_token = os.getenv("HF_TOKEN", self._token)
        assert self.model_path is not None, "model_path must be provided for HuggingFace models"
        config = AutoConfig.from_pretrained(self.model_path, token=hf_token)
        if self._is_zero_shot(config):
            ensure_package_minimum_version("transformers", _TRANSFORMERS_ZERO_SHOT_MIN_VERSION)
            from transformers import AutoModelForZeroShotObjectDetection

            model_class: Any = AutoModelForZeroShotObjectDetection
        else:
            model_class = AutoModelForObjectDetection
        model = model_class.from_pretrained(self.model_path, token=hf_token)
        if self.image_size is not None:
            # RT-DETR family expects explicit height/width; other models use shortest_edge
            if model.__class__.__name__.startswith("RTDetr"):
                size: dict[str, int | None] = {"height": self.image_size, "width": self.image_size}
            else:
                size = {"shortest_edge": self.image_size, "longest_edge": None}
            # use_fast=True raises error: AttributeError: 'SizeDict' object has no attribute 'keys'
            processor = AutoProcessor.from_pretrained(
                self.model_path, size=size, do_resize=True, use_fast=False, token=hf_token
            )
        else:
            processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False, token=hf_token)
        self.set_model(model, processor)

    def set_model(self, model: Any, processor: Any | None = None, **kwargs: Any) -> None:
        """Set the detection model and processor."""
        processor = processor or self.processor
        if processor is None:
            raise ValueError(f"'processor' is required to be set, got {processor}.")
        self._is_zero_shot_model = self._is_zero_shot(model)
        valid_processor = "ImageProcessor" in processor.__class__.__name__ or self._is_zero_shot(processor)
        if "ObjectDetection" not in model.__class__.__name__ or not valid_processor:
            raise ValueError(
                "Given 'model' is not an ObjectDetectionModel or 'processor' is not a valid ImageProcessor."
            )
        self.model = model
        self.model.to(self.device)  # type: ignore[attr-defined]
        self._processor = processor
        if self._is_zero_shot_model:
            self.category_mapping = {i: name for i, name in enumerate(self.text_labels or [])}
            self._category_name_to_id = {name: i for i, name in self.category_mapping.items()}
        else:
            self.category_mapping = self.model.config.id2label  # type: ignore[attr-defined]

    def perform_inference(self, image: list | np.ndarray) -> None:
        """Prediction is performed using self.model and the prediction result is set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        import torch

        # Confirm model is loaded
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        with torch.no_grad():
            if self._is_zero_shot_model:
                text = self._get_zero_shot_text_input(len(image) if isinstance(image, list) else 1)
                inputs = self.processor(images=image, text=text, return_tensors="pt")
            else:
                inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            outputs = self.model(**inputs)
        self._original_input_ids = inputs.get("input_ids")

        images = image if isinstance(image, list) else [image]
        self._original_shapes = [img.shape for img in images]
        self._original_predictions = outputs

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        """Native batch inference: process all images in a single processor + model call.

        Unlike the base-class default (which runs images sequentially), this
        feeds the entire list to the HuggingFace processor at once and executes
        one batched forward pass.  The processor pads images to a uniform size
        internally, so images of different resolutions are handled correctly.

        This avoids setting ``_batch_images`` so
        ``convert_original_predictions`` uses the standard multi-image path
        rather than the sequential fallback.

        Args:
            images: List of numpy arrays (H, W, C) in RGB order.
        """
        self.perform_inference(images)

    # Models using per-class sigmoid (no background class in logits)
    _SIGMOID_CLS_PREFIXES = ("RTDetr", "ConditionalDetr", "DeformableDetr", "Deta", "GroundingDino")

    @property
    def _uses_sigmoid_cls(self) -> bool:
        """True for models that use per-class sigmoid instead of softmax+background."""
        cls_name = self.model.__class__.__name__
        return any(cls_name.startswith(p) for p in self._SIGMOID_CLS_PREFIXES)

    @staticmethod
    def _is_zero_shot(obj: Any) -> bool:
        """Return whether a HuggingFace config/model/processor is a GroundingDINO-style zero-shot detector."""
        if hasattr(obj, "post_process_grounded_object_detection"):
            return True
        return obj.__class__.__name__.startswith("GroundingDino") or getattr(obj, "model_type", "") == "grounding-dino"

    def _get_zero_shot_text_input(self, num_images: int) -> list:
        """Return per-image text input for the HuggingFace zero-shot processor."""
        prompt = self.text_labels or self.text_prompt
        if not prompt:
            raise ValueError("'text_labels' or 'text_prompt' is required for zero-shot HuggingFace detection models.")
        return [prompt] * num_images

    @staticmethod
    def _clamp_bbox(bbox: list, image_width: int, image_height: int) -> list:
        """Clamp a [x1, y1, x2, y2] box to image bounds."""
        x1, y1, x2, y2 = bbox
        return [max(0, x1), max(0, y1), min(x2, image_width), min(y2, image_height)]

    @staticmethod
    def _shift_and_full_shape(
        shift_amount_list: list[list[int | float]],
        full_shape_list: list[list[int | float]] | None,
        image_ind: int,
    ) -> tuple[list[int], list[int] | None]:
        """Return the int-cast shift amount and full shape for a single image."""
        shift_amount = [int(x) for x in shift_amount_list[image_ind]]
        full_shape = None if full_shape_list is None else [int(x) for x in full_shape_list[image_ind]]
        return shift_amount, full_shape

    def _get_zero_shot_category_id(self, category_name: str) -> int:
        """Return a stable category id for a zero-shot label, assigning a new one for unseen phrases."""
        if category_name not in self._category_name_to_id:
            new_id = len(self.category_mapping)
            self._category_name_to_id[category_name] = new_id
            self.category_mapping[new_id] = category_name
        return self._category_name_to_id[category_name]

    def get_valid_predictions(self, logits: Any, pred_boxes: Any) -> tuple:
        """Get predictions above confidence threshold.

        Args:
            logits: torch.Tensor
            pred_boxes: torch.Tensor

        Returns:
            scores: torch.Tensor
            cat_ids: torch.Tensor
            boxes: torch.Tensor
        """
        import torch

        if self._uses_sigmoid_cls:
            # RT-DETR family: per-class sigmoid, logits shape (Q, num_classes) — no background class
            probs = logits.sigmoid()
            scores, cat_ids = probs.max(-1)
            valid_mask = scores >= self.confidence_threshold
        else:
            # DETR family: softmax over (num_classes + 1), last index is no-object/background
            probs = logits.softmax(-1)
            scores = probs.max(-1).values
            cat_ids = probs.argmax(-1)
            valid_detections = torch.where(cat_ids < self.num_categories, 1, 0)
            valid_confidences = torch.where(scores >= self.confidence_threshold, 1, 0)
            valid_mask = valid_detections.logical_and(valid_confidences).bool()

        scores = scores[valid_mask]
        cat_ids = cat_ids[valid_mask]
        boxes = pred_boxes[valid_mask]
        return scores, cat_ids, boxes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        """Convert predictions to ObjectPrediction list.

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
        assert self._original_predictions is not None
        original_predictions: Any = self._original_predictions

        # compatibility for sahi v0.8.15
        shift_amount_list_typed: list[list[int | float]] = fix_shift_amount_list(shift_amount_list)
        full_shape_list_typed: list[list[int | float]] | None = fix_full_shape_list(full_shape_list)

        if self._is_zero_shot_model:
            self._create_object_prediction_list_from_zero_shot_predictions(
                original_predictions=original_predictions,
                shift_amount_list=shift_amount_list_typed,
                full_shape_list=full_shape_list_typed,
            )
            return

        from sahi.utils.cv import yolo_bbox_to_voc_bbox

        n_image = original_predictions.logits.shape[0]
        object_prediction_list_per_image = []
        for image_ind in range(n_image):
            image_height, image_width, _ = self.image_shapes[image_ind]
            scores, cat_ids, boxes = self.get_valid_predictions(
                logits=original_predictions.logits[image_ind], pred_boxes=original_predictions.pred_boxes[image_ind]
            )

            # create object_prediction_list
            object_prediction_list = []

            shift_amount, full_shape = self._shift_and_full_shape(
                shift_amount_list_typed, full_shape_list_typed, image_ind
            )

            for ind in range(len(boxes)):
                category_id = cat_ids[ind].item()
                bbox = yolo_bbox_to_voc_bbox(boxes[ind].tolist(), image_width=image_width, image_height=image_height)
                bbox = self._clamp_bbox(bbox, image_width, image_height)

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    segmentation=None,
                    category_id=category_id,
                    category_name=self.category_mapping[category_id] if self.category_mapping else "",  # type: ignore[index]
                    shift_amount=shift_amount,
                    score=scores[ind].item(),
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    def _create_object_prediction_list_from_zero_shot_predictions(
        self,
        original_predictions: Any,
        shift_amount_list: list[list[int | float]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        """Convert HuggingFace zero-shot detection output to ObjectPrediction objects."""
        if self._original_input_ids is None:
            raise RuntimeError("Zero-shot text input ids are missing. Run .perform_inference() before conversion.")

        target_sizes = [(image_shape[0], image_shape[1]) for image_shape in self.image_shapes]
        results = self.processor.post_process_grounded_object_detection(
            original_predictions,
            input_ids=self._original_input_ids,
            threshold=self.confidence_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(results):
            image_height, image_width, _ = self.image_shapes[image_ind]
            shift_amount, full_shape = self._shift_and_full_shape(shift_amount_list, full_shape_list, image_ind)
            labels = image_predictions.get("text_labels") or image_predictions.get("labels", [])

            object_prediction_list = [
                ObjectPrediction(
                    bbox=self._clamp_bbox(bbox.tolist(), image_width, image_height),
                    segmentation=None,
                    category_id=self._get_zero_shot_category_id(str(name)),
                    category_name=str(name),
                    shift_amount=shift_amount,
                    score=float(score),
                    full_shape=full_shape,
                )
                for score, bbox, name in zip(image_predictions["scores"], image_predictions["boxes"], labels)
                # when fixed text_labels are given, drop combined phrases (e.g. "car truck")
                if not self.text_labels or str(name) in self.text_labels
            ]
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
