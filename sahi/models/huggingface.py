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
            return len(self.category_mapping or {})
        return self.model.config.num_labels  # type: ignore[attr-defined]

    def load_model(self) -> None:
        """Load model from HuggingFace."""
        from transformers import AutoConfig, AutoModelForObjectDetection, AutoProcessor

        hf_token = os.getenv("HF_TOKEN", self._token)
        assert self.model_path is not None, "model_path must be provided for HuggingFace models"
        config = AutoConfig.from_pretrained(self.model_path, token=hf_token)
        if self._is_zero_shot_config(config):
            ensure_package_minimum_version("transformers", _TRANSFORMERS_ZERO_SHOT_MIN_VERSION)
            from transformers import AutoModelForZeroShotObjectDetection

            model_class = AutoModelForZeroShotObjectDetection
        else:
            model_class = AutoModelForObjectDetection
        model = model_class.from_pretrained(self.model_path, token=hf_token)
        if self.image_size is not None:
            size = self._get_processor_size(model, self.image_size)
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
        self._is_zero_shot_model = self._is_zero_shot_model_instance(model)
        valid_processor = "ImageProcessor" in processor.__class__.__name__ or self._is_zero_shot_processor(processor)
        if "ObjectDetection" not in model.__class__.__name__ or not valid_processor:
            raise ValueError(
                "Given 'model' is not an ObjectDetectionModel or 'processor' is not a valid ImageProcessor."
            )
        self.model = model
        self.model.to(self.device)  # type: ignore[attr-defined]
        self._processor = processor
        if self._is_zero_shot_model:
            self._set_zero_shot_category_mapping()
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

        if isinstance(image, list):
            self._original_shapes = [img.shape for img in image]
        else:
            self._original_shapes = [image.shape]
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
    def _is_zero_shot_config(config: Any) -> bool:
        """Return whether a HuggingFace config belongs to a zero-shot detector."""
        model_type = getattr(config, "model_type", "")
        architectures = getattr(config, "architectures", []) or []
        return model_type == "grounding-dino" or any(str(arch).startswith("GroundingDino") for arch in architectures)

    @staticmethod
    def _is_zero_shot_model_instance(model: Any) -> bool:
        """Return whether a HuggingFace model instance needs text-conditioned inference."""
        return model.__class__.__name__.startswith("GroundingDino")

    @staticmethod
    def _is_zero_shot_processor(processor: Any) -> bool:
        """Return whether the processor exposes zero-shot post-processing."""
        return hasattr(processor, "post_process_grounded_object_detection")

    @staticmethod
    def _get_processor_size(model: Any, image_size: int) -> dict[str, int]:
        """Return HuggingFace processor resize arguments."""
        if model.__class__.__name__.startswith("RTDetr"):
            return {"height": image_size, "width": image_size}
        return {"shortest_edge": image_size, "longest_edge": image_size}

    def _set_zero_shot_category_mapping(self) -> None:
        """Initialize deterministic category ids for prompt labels."""
        if self.category_mapping is None:
            self.category_mapping = {}
        self.category_mapping = {int(category_id): name for category_id, name in self.category_mapping.items()}
        self._category_name_to_id = {name: category_id for category_id, name in self.category_mapping.items()}
        if self.text_labels:
            for category_name in self.text_labels:
                self._get_zero_shot_category_id(category_name)

    def _get_zero_shot_text_input(self, num_images: int) -> str | list[str] | list[list[str]]:
        """Return text input for the HuggingFace zero-shot processor."""
        if self.text_labels:
            return [self.text_labels for _ in range(num_images)]
        if self.text_prompt:
            return self.text_prompt if num_images == 1 else [self.text_prompt for _ in range(num_images)]
        if self.category_mapping:
            text_labels = [self.category_mapping[key] for key in sorted(self.category_mapping)]
            self.text_labels = text_labels
            self._category_name_to_id = {name: category_id for category_id, name in self.category_mapping.items()}
            return [text_labels for _ in range(num_images)]
        raise ValueError("'text_labels' or 'text_prompt' is required for zero-shot HuggingFace detection models.")

    def _get_zero_shot_category_id(self, category_name: str) -> int:
        """Return a stable category id for a zero-shot label."""
        if category_name not in self._category_name_to_id:
            category_id = max(self.category_mapping.keys(), default=-1) + 1 if self.category_mapping else 0
            self._category_name_to_id[category_name] = category_id
            self.category_mapping[category_id] = category_name
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

        n_image = original_predictions.logits.shape[0]
        object_prediction_list_per_image = []
        for image_ind in range(n_image):
            image_height, image_width, _ = self.image_shapes[image_ind]
            scores, cat_ids, boxes = self.get_valid_predictions(
                logits=original_predictions.logits[image_ind], pred_boxes=original_predictions.pred_boxes[image_ind]
            )

            # create object_prediction_list
            object_prediction_list = []

            shift_amount = [int(x) for x in shift_amount_list_typed[image_ind]]
            full_shape = None if full_shape_list_typed is None else [int(x) for x in full_shape_list_typed[image_ind]]

            for ind in range(len(boxes)):
                category_id = cat_ids[ind].item()
                from sahi.utils.cv import yolo_bbox_to_voc_bbox

                yolo_bbox = boxes[ind].tolist()
                bbox = yolo_bbox_to_voc_bbox(
                    yolo_bbox,
                    image_width=image_width,
                    image_height=image_height,
                )

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(bbox[2], image_width)
                bbox[3] = min(bbox[3], image_height)

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
        if self.processor is None:
            raise RuntimeError("Processor is not loaded, load it by calling .load_model()")
        if self._original_input_ids is None:
            raise RuntimeError("GroundingDINO input ids are missing. Run .perform_inference() before conversion.")

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
            shift_amount = [int(x) for x in shift_amount_list[image_ind]]
            full_shape = None if full_shape_list is None else [int(x) for x in full_shape_list[image_ind]]
            if "text_labels" in image_predictions:
                labels = image_predictions["text_labels"]
            else:
                labels = image_predictions.get("labels", [])

            object_prediction_list = []
            for score, bbox, category_name in zip(image_predictions["scores"], image_predictions["boxes"], labels):
                if self.text_labels and category_name not in self.text_labels:
                    continue
                bbox_list = bbox.tolist()
                bbox_list[0] = max(0, bbox_list[0])
                bbox_list[1] = max(0, bbox_list[1])
                bbox_list[2] = min(bbox_list[2], image_width)
                bbox_list[3] = min(bbox_list[3], image_height)
                category_id = self._get_zero_shot_category_id(str(category_name))

                object_prediction = ObjectPrediction(
                    bbox=bbox_list,
                    segmentation=None,
                    category_id=category_id,
                    category_name=str(category_name),
                    shift_amount=shift_amount,
                    score=score.item() if hasattr(score, "item") else float(score),
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
