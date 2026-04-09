"""Prediction classes for object detection results."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
from PIL import Image

from sahi.annotation import ObjectAnnotation
from sahi.utils.coco import CocoPrediction
from sahi.utils.cv import read_image_as_pil, visualize_object_predictions
from sahi.utils.file import Path


class PredictionScore:
    """Wrapper around a numeric prediction confidence score.

    Provides comparison operators and conversion from numpy scalars to
    native Python floats for serialization safety.
    """

    value: float

    def __init__(self, value: float | np.ndarray) -> None:
        """Initialize PredictionScore.

        Args:
            value: prediction score between 0 and 1.
        """
        # if score is a numpy object, convert it to python variable
        if isinstance(value, np.ndarray):
            value = copy.deepcopy(value).tolist()
        # set score
        self.value: float = value  # type: ignore[assignment]

    def is_greater_than_threshold(self, threshold: float) -> bool:
        """Check if score is greater than threshold."""
        return self.value > threshold

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        """Check equality with another value."""
        if isinstance(other, (float, int)):
            return self.value == other
        return NotImplemented

    def __gt__(self, other: object) -> bool:  # type: ignore[override]
        """Check if greater than another value."""
        if isinstance(other, (float, int)):
            return self.value > other
        return NotImplemented

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        """Check if less than another value."""
        if isinstance(other, (float, int)):
            return self.value < other
        return NotImplemented

    def __repr__(self) -> str:
        """Return string representation of prediction score."""
        return f"PredictionScore: <value: {self.value}>"


class ObjectPrediction(ObjectAnnotation):
    """Class for handling detection model predictions."""

    def __init__(
        self,
        bbox: list[float] | None = None,
        category_id: int | None = None,
        category_name: str | None = None,
        segmentation: list[list[float]] | None = None,
        score: float = 0.0,
        shift_amount: list[int] | list[int | float] | None = None,
        full_shape: list[int] | list[int | float] | None = None,
    ) -> None:
        """Initialize ObjectPrediction from bbox, score, category_id, category_name, segmentation.

        Args:
            bbox: list
                [minx, miny, maxx, maxy]
            score: float
                Prediction score between 0 and 1
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            segmentation: List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            shift_amount: list
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        self.score = PredictionScore(score)
        super().__init__(
            bbox=bbox,
            category_id=category_id,
            segmentation=segmentation,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def get_shifted_object_prediction(self) -> ObjectPrediction:
        """Get shifted version of ObjectPrediction.

        Shifts bbox and mask coords. Used for mapping sliced predictions over full image.
        """
        if self.mask:
            shifted_mask = self.mask.get_shifted_mask()
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                score=self.score.value,
                segmentation=shifted_mask.segmentation,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=shifted_mask.full_shape,
            )
        else:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                score=self.score.value,
                segmentation=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def to_coco_prediction(self, image_id: int | None = None) -> CocoPrediction:
        """Convert to sahi.utils.coco.CocoPrediction representation."""
        bbox_xywh = self.bbox.to_xywh()
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(  # type: ignore[arg-type]
                segmentation=self.mask.segmentation,
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=bbox_xywh,  # type: ignore[arg-type]
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        return coco_prediction

    def to_fiftyone_detection(self, image_height: int, image_width: int) -> object:
        """Convert to fiftyone.Detection representation."""
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError('Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.')

        x1, y1, x2, y2 = self.bbox.to_xyxy()
        rel_box = [x1 / image_width, y1 / image_height, (x2 - x1) / image_width, (y2 - y1) / image_height]
        fiftyone_detection = fo.Detection(label=self.category.name, bounding_box=rel_box, confidence=self.score.value)
        return fiftyone_detection

    def __repr__(self) -> str:
        """Return string representation of ObjectPrediction."""
        return f"""ObjectPrediction<
    bbox: {self.bbox},
    mask: {self.mask},
    score: {self.score},
    category: {self.category}>"""


class PredictionResult:
    """Container for detection results on a single image.

    Holds the list of ``ObjectPrediction`` instances together with the
    source image and optional profiling durations. Provides helpers for
    exporting results to COCO, FiftyOne, and visual formats.
    """

    def __init__(
        self,
        object_prediction_list: list[ObjectPrediction],
        image: Image.Image | str | np.ndarray,
        durations_in_seconds: dict[str, Any] = dict(),
    ) -> None:
        """Initialize a PredictionResult.

        Args:
            object_prediction_list: list[ObjectPrediction]
                Detected objects for this image.
            image: Image.Image or str or np.ndarray
                The source image as a PIL Image, file path, or numpy array.
            durations_in_seconds: dict[str, Any]
                Elapsed times for profiling (e.g. inference, postprocess).
        """
        self.image: Image.Image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.object_prediction_list: list[ObjectPrediction] = object_prediction_list
        self.durations_in_seconds = durations_in_seconds

    def export_visuals(
        self,
        export_dir: str,
        text_size: float | None = None,
        rect_th: int | None = None,
        hide_labels: bool = False,
        hide_conf: bool = False,
        file_name: str = "prediction_visual",
    ) -> None:
        """Export prediction visualizations to directory.

        Args:
            export_dir: directory for resulting visualization to be exported.
            text_size: size of the category name over box.
            rect_th: rectangle thickness.
            hide_labels: hide labels.
            hide_conf: hide confidence.
            file_name: saving name.
        """
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        visualize_object_predictions(
            image=np.ascontiguousarray(self.image),
            object_prediction_list=self.object_prediction_list,
            rect_th=rect_th,
            text_size=text_size,
            text_th=None,
            color=None,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            output_dir=export_dir,
            file_name=file_name,
            export_format="png",
        )

    def to_coco_annotations(self) -> list:
        """Convert predictions to COCO annotation format."""
        coco_annotation_list = []
        for object_prediction in self.object_prediction_list:
            coco_annotation_list.append(object_prediction.to_coco_prediction().json)
        return coco_annotation_list

    def to_coco_predictions(self, image_id: int | None = None) -> list:
        """Convert predictions to COCO prediction format."""
        coco_prediction_list = []
        for object_prediction in self.object_prediction_list:
            coco_prediction_list.append(object_prediction.to_coco_prediction(image_id=image_id).json)
        return coco_prediction_list

    def to_imantics_annotations(self) -> list:
        """Convert predictions to imantics annotation format."""
        imantics_annotation_list = []
        for object_prediction in self.object_prediction_list:
            imantics_annotation_list.append(object_prediction.to_imantics_annotation())
        return imantics_annotation_list

    def to_fiftyone_detections(self) -> list:
        """Convert predictions to FiftyOne detection format."""
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError('Please run "uv pip install -U fiftyone" to install fiftyone for conversion.')

        fiftyone_detection_list: list[fo.Detection] = []
        for object_prediction in self.object_prediction_list:
            fiftyone_detection_list.append(
                object_prediction.to_fiftyone_detection(image_height=self.image_height, image_width=self.image_width)
            )
        return fiftyone_detection_list
