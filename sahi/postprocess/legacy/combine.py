"""Legacy postprocessing implementations for object prediction merging."""

from __future__ import annotations

import copy

import numpy as np

from sahi.annotation import BoundingBox, Category, Mask
from sahi.postprocess.utils import calculate_area, calculate_box_union, calculate_intersection_area
from sahi.prediction import ObjectPrediction


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions."""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ) -> None:
        """Initialize the postprocessor with matching configuration.

        Args:
            match_threshold: Minimum overlap value to consider predictions matching.
            match_metric: Metric for overlap computation, "IOU" or "IOS".
            class_agnostic: If True, apply postprocessing across all categories.
        """
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        if match_metric == "IOU":
            self.calculate_match = self.calculate_bbox_iou
        elif match_metric == "IOS":
            self.calculate_match = self.calculate_bbox_ios
        else:
            raise ValueError(f"'match_metric' should be one of ['IOU', 'IOS'] but given as {match_metric}")

    def _has_match(self, pred1: ObjectPrediction, pred2: ObjectPrediction) -> bool:
        threshold_condition = self.calculate_match(pred1, pred2) > self.match_threshold
        category_condition = self.has_same_category_id(pred1, pred2) or self.class_agnostic
        return threshold_condition and category_condition

    @staticmethod
    def get_score_func(object_prediction: ObjectPrediction) -> float:
        """Used for sorting predictions."""
        return object_prediction.score.value

    @staticmethod
    def has_same_category_id(pred1: ObjectPrediction, pred2: ObjectPrediction) -> bool:
        """Check if two predictions belong to the same category.

        Args:
            pred1: First ObjectPrediction instance.
            pred2: Second ObjectPrediction instance.

        Returns:
            True if both predictions have the same category ID.
        """
        return pred1.category.id == pred2.category.id

    @staticmethod
    def calculate_bbox_iou(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
        """Returns the ratio of intersection area to the union."""
        box1 = np.array(pred1.bbox.to_xyxy())
        box2 = np.array(pred2.bbox.to_xyxy())
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        return intersect / (area1 + area2 - intersect)

    @staticmethod
    def calculate_bbox_ios(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
        """Returns the ratio of intersection area to the smaller box's area."""
        box1 = np.array(pred1.bbox.to_xyxy())
        box2 = np.array(pred2.bbox.to_xyxy())
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        smaller_area = np.minimum(area1, area2)
        return intersect / smaller_area

    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ) -> list[ObjectPrediction]:
        """Apply postprocessing to object predictions.

        Args:
            object_predictions: List of object predictions to postprocess.

        Returns:
            List of postprocessed object predictions.
        """
        raise NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    """Non-Maximum Suppression postprocessor for legacy usage."""

    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ) -> list[ObjectPrediction]:
        """Apply NMS to object predictions."""
        source_object_predictions: list[ObjectPrediction] = copy.deepcopy(object_predictions)
        selected_object_predictions: list[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            source_object_predictions.sort(reverse=True, key=self.get_score_func)
            selected_object_prediction = source_object_predictions[0]
            # remove selected prediction from source list
            del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it
            new_source_object_predictions: list[ObjectPrediction] = []
            for candidate_object_prediction in source_object_predictions:
                if self._has_match(selected_object_prediction, candidate_object_prediction):
                    pass
                else:
                    new_source_object_predictions.append(candidate_object_prediction)
            source_object_predictions = new_source_object_predictions
            # append selected prediction to selected list
            selected_object_predictions.append(selected_object_prediction)
        return selected_object_predictions


class UnionMergePostprocess(PostprocessPredictions):
    """Union merging postprocessor for overlapping predictions."""

    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ) -> list[ObjectPrediction]:
        """Apply union merging to overlapping object predictions."""
        source_object_predictions: list[ObjectPrediction] = copy.deepcopy(object_predictions)
        selected_object_predictions: list[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            source_object_predictions.sort(reverse=True, key=self.get_score_func)
            selected_object_prediction = source_object_predictions[0]
            # remove selected prediction from source list
            del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it and merge with selected prediction
            new_source_object_predictions: list[ObjectPrediction] = []
            for ind, candidate_object_prediction in enumerate(source_object_predictions):
                if self._has_match(selected_object_prediction, candidate_object_prediction):
                    selected_object_prediction = self._merge_object_prediction_pair(
                        selected_object_prediction, candidate_object_prediction
                    )
                else:
                    new_source_object_predictions.append(candidate_object_prediction)
            source_object_predictions = new_source_object_predictions
            # append selected prediction to selected list
            selected_object_predictions.append(selected_object_prediction)
        return selected_object_predictions

    def _merge_object_prediction_pair(
        self,
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> ObjectPrediction:
        shift_amount = list(pred1.bbox.shift_amount)
        merged_bbox: BoundingBox = self._get_merged_bbox(pred1, pred2)
        merged_score: float = self._get_merged_score(pred1, pred2)
        merged_category: Category = self._get_merged_category(pred1, pred2)
        if pred1.mask and pred2.mask:
            merged_mask: Mask = self._get_merged_mask(pred1, pred2)
            segmentation = merged_mask.segmentation
            full_shape = merged_mask.full_shape
        else:
            segmentation = None
            full_shape = None
        return ObjectPrediction(
            bbox=merged_bbox.to_xyxy(),
            score=merged_score,
            category_id=merged_category.id,
            category_name=merged_category.name,
            segmentation=segmentation,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @staticmethod
    def _get_merged_category(pred1: ObjectPrediction, pred2: ObjectPrediction) -> Category:
        if pred1.score.value > pred2.score.value:
            return pred1.category
        else:
            return pred2.category

    @staticmethod
    def _get_merged_bbox(pred1: ObjectPrediction, pred2: ObjectPrediction) -> BoundingBox:
        box1: list[float] = pred1.bbox.to_xyxy()
        box2: list[float] = pred2.bbox.to_xyxy()
        bbox = BoundingBox(box=calculate_box_union(box1, box2))
        return bbox

    @staticmethod
    def _get_merged_score(
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> float:
        scores: list[float] = [pred.score.value for pred in (pred1, pred2)]
        return max(scores)

    @staticmethod
    def _get_merged_mask(pred1: ObjectPrediction, pred2: ObjectPrediction) -> Mask:
        mask1 = pred1.mask
        mask2 = pred2.mask
        if mask1 is None or mask2 is None:
            raise ValueError("Both predictions must have masks to merge them")
        union_mask = np.logical_or(mask1.bool_mask, mask2.bool_mask)
        from sahi.utils.cv import get_coco_segmentation_from_bool_mask

        return Mask(
            segmentation=get_coco_segmentation_from_bool_mask(union_mask),
            full_shape=mask1.full_shape,
            shift_amount=mask1.shift_amount,
        )
