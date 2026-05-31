"""HuggingFace segmentation model wrapper for SAHI.

Supports MaskFormer, Mask2Former, and OneFormer for instance, semantic, and
panoptic segmentation via Hugging Face Transformers.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

import numpy as np

from sahi.models.huggingface import HuggingfaceDetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

# {model_class_name: processor_class_name} — strings to avoid eager transformers import
_SUPPORTED_MODELS: dict[str, str] = {
    "Mask2FormerForUniversalSegmentation": "Mask2FormerImageProcessor",
    "MaskFormerForInstanceSegmentation": "MaskFormerImageProcessor",
    "OneFormerForUniversalSegmentation": "OneFormerProcessor",
}


class SegmentationType(Enum):
    INSTANCE_SEGMENTATION = "instance"
    SEMANTIC_SEGMENTATION = "semantic"
    PANOPTIC_SEGMENTATION = "panoptic"


class HuggingfaceSegmentationModel(HuggingfaceDetectionModel):
    """HuggingFace segmentation model.

    Subclasses :class:`HuggingfaceDetectionModel`, reusing its processor,
    ``num_categories``, token handling, and dependency checks. Supports
    MaskFormer, Mask2Former, and OneFormer for instance, semantic, and
    panoptic segmentation.

    Args:
        overlap_mask_area_threshold: Overlap mask area threshold to merge or
            discard small disconnected parts within each binary instance mask.
        label_ids_to_fuse: Label ids whose instances will be fused together
            (panoptic only). E.g. sky can be a single segment per image.
        min_segment_area: Segments below this contour area are dropped.
        segmentation_type: Which segmentation head to use. Params that do not
            apply to the chosen type are ignored.
    """

    def __init__(
        self,
        *args: Any,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: list[int] | None = None,
        min_segment_area: int = 100,
        segmentation_type: SegmentationType = SegmentationType.INSTANCE_SEGMENTATION,
        **kwargs: Any,
    ) -> None:
        self.segmentation_type = segmentation_type
        self.overlap_mask_area_threshold = overlap_mask_area_threshold
        self.label_ids_to_fuse = label_ids_to_fuse
        self.min_segment_area = min_segment_area
        # Segmentation models default to a stricter threshold than detection (0.3).
        kwargs.setdefault("confidence_threshold", 0.5)
        super().__init__(*args, **kwargs)

    def load_model(self) -> None:
        """Load model and processor from HuggingFace."""
        from transformers import AutoModelForUniversalSegmentation, AutoProcessor

        if self.model_path is None:
            raise ValueError("model_path must be provided for HuggingFace models")

        hf_token = os.getenv("HF_TOKEN", self._token)
        model = AutoModelForUniversalSegmentation.from_pretrained(self.model_path, token=hf_token)

        processor_kwargs: dict[str, Any] = {"use_fast": False, "token": hf_token}
        if self.image_size is not None:
            processor_kwargs |= {
                "size": {"height": self.image_size, "width": self.image_size},
                "do_resize": True,
            }
        # use_fast=True raises: AttributeError: 'SizeDict' object has no attribute 'keys'
        processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)

        self.set_model(model, processor)

    def set_model(self, model: Any, processor: Any = None, **kwargs: Any) -> None:
        processor = processor or self.processor
        if processor is None:
            raise ValueError("'processor' is required to be set.")

        model_name = type(model).__name__
        processor_name = type(processor).__name__
        expected_processor_prefix = _SUPPORTED_MODELS.get(model_name)
        # Newer transformers append backend suffixes (e.g. Mask2FormerImageProcessorPil), so match by prefix.
        if expected_processor_prefix is None or not processor_name.startswith(expected_processor_prefix):
            raise ValueError(
                f"Invalid model/processor pair: {model_name} + {processor_name}. Supported: {_SUPPORTED_MODELS}"
            )

        self.model = model
        self.model.to(self.device)
        self._processor = processor
        self.category_mapping = self.model.config.id2label

    def perform_inference(self, image: list[np.ndarray] | np.ndarray) -> None:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded; call .load_model() first.")

        inputs = self._pre_process(image)
        with torch.no_grad():
            outputs = self.model(**inputs)

        images = image if isinstance(image, list) else [image]
        self._original_shapes = [(int(img.shape[0]), int(img.shape[1])) for img in images]
        self._original_predictions = outputs

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        self.perform_inference(images)

    def _is_oneformer(self) -> bool:
        return type(self.model).__name__ == "OneFormerForUniversalSegmentation"

    def _pre_process(self, image: list[np.ndarray] | np.ndarray) -> Any:
        kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if self._is_oneformer():
            seg_value = self.segmentation_type.value
            n = len(image) if isinstance(image, list) else 1
            kwargs["task_inputs"] = [seg_value] * n

        inputs = self.processor(**kwargs)
        for key in ("pixel_values", "pixel_mask", "task_inputs"):
            if key in inputs:
                inputs[key] = inputs[key].to(self.device)
        return inputs

    def _post_process(self, predictions: Any, target_sizes: list) -> list[dict]:
        processor = self.processor
        seg_type = self.segmentation_type
        common: dict[str, Any] = {
            "threshold": self.confidence_threshold,
            "mask_threshold": self.mask_threshold,
            "overlap_mask_area_threshold": self.overlap_mask_area_threshold,
            "target_sizes": target_sizes,
        }

        if seg_type is SegmentationType.SEMANTIC_SEGMENTATION:
            outputs = processor.post_process_semantic_segmentation(predictions, target_sizes)
            return _convert_semantic_to_binary_masks(outputs)

        if seg_type is SegmentationType.PANOPTIC_SEGMENTATION:
            outputs = processor.post_process_panoptic_segmentation(
                predictions, label_ids_to_fuse=self.label_ids_to_fuse, **common
            )
            return _convert_panoptic_to_binary_masks(outputs)

        # OneFormer's instance output matches panoptic format and needs conversion;
        # MaskFormer/Mask2Former can emit binary maps directly.
        if self._is_oneformer():
            outputs = processor.post_process_instance_segmentation(predictions, **common)
            return _convert_panoptic_to_binary_masks(outputs)
        return processor.post_process_instance_segmentation(predictions, return_binary_maps=True, **common)

    def _extract_segments(self, post_processed_output: dict) -> tuple[list, list, list]:
        """Convert per-segment binary masks to (scores, category_ids, coco_segmentations).

        Each mask yields at most one COCO multi-polygon entry; masks smaller
        than ``min_segment_area`` pixels (or yielding no valid polygons) are dropped.
        """
        scores: list = []
        category_ids: list = []
        coco_segmentations: list[list[list[float]]] = []

        segments = post_processed_output["segmentation"]
        segments_info = post_processed_output["segments_info"]
        if segments is None or not segments_info:
            return scores, category_ids, coco_segmentations

        for segment, segment_info in zip(segments, segments_info):
            bool_mask = segment.cpu().numpy().astype(bool)
            if bool_mask.sum() < self.min_segment_area:
                continue
            coco_segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
            if not coco_segmentation:
                continue
            coco_segmentations.append(coco_segmentation)
            scores.append(segment_info["score"])
            category_ids.append(segment_info["label_id"])

        return scores, category_ids, coco_segmentations

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int | float]] | None = [[0, 0]],
        full_shape_list: list[list[int | float]] | None = None,
    ) -> None:
        target_sizes = self._original_shapes or []
        post_processed_outputs = self._post_process(self._original_predictions, target_sizes)

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image: list[list[ObjectPrediction]] = []
        for image_ind, output in enumerate(post_processed_outputs):
            scores, category_ids, segments = self._extract_segments(output)
            shift_amount = shift_amount_list[image_ind]
            full_shape = list(target_sizes[image_ind]) if full_shape_list is None else full_shape_list[image_ind]

            object_prediction_list = [
                ObjectPrediction(
                    bbox=None,
                    segmentation=segment,
                    category_id=category_id,
                    category_name=self.category_mapping[category_id],
                    shift_amount=shift_amount,
                    score=score,
                    full_shape=full_shape,
                )
                for category_id, segment, score in zip(category_ids, segments, scores)
            ]
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


def _convert_semantic_to_binary_masks(class_masks: list) -> list[dict]:
    """Split per-pixel class id tensors into one binary mask per label."""
    import torch

    outputs: list[dict] = []
    for class_mask in class_masks:
        output: dict = {"segmentation": [], "segments_info": []}
        for label_id in torch.unique(class_mask):
            output["segmentation"].append((class_mask == label_id).to(torch.uint8))
            output["segments_info"].append({"score": 1.0, "label_id": label_id.item()})
        outputs.append(output)
    return outputs


def _convert_panoptic_to_binary_masks(post_processed_outputs: list[dict]) -> list[dict]:
    """Split panoptic id-tensors into one binary mask per segment, dropping background ids."""
    import torch

    outputs: list[dict] = []
    for post_processed_output in post_processed_outputs:
        segmentation = post_processed_output["segmentation"]
        segments_info = post_processed_output["segments_info"]
        if segmentation is None or not segments_info:
            continue

        segments_info_map = {info["id"]: info for info in segments_info}
        output: dict = {"segmentation": [], "segments_info": []}
        for segment_id in torch.unique(segmentation).tolist():
            info = segments_info_map.get(segment_id)
            if info is not None:
                output["segmentation"].append((segmentation == segment_id).to(torch.uint8))
                output["segments_info"].append(info)
        outputs.append(output)
    return outputs
