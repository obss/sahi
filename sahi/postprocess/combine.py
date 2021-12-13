# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2021.

import copy
from typing import List, Union

import numpy as np

from sahi.annotation import BoundingBox, Category, Mask
from sahi.prediction import ObjectPrediction


def calculate_area(box: Union[List[int], np.ndarray]) -> float:
    """
    Args:
        box (List[int]): [x1, y1, x2, y2]
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Args:
        box1 (np.ndarray): np.array([x1, y1, x2, y2])
        box2 (np.ndarray): np.array([x1, y1, x2, y2])
    """
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    width_height = (right_bottom - left_top).clip(min=0)
    return width_height[0] * width_height[1]


def calculate_box_union(box1: Union[List[int], np.ndarray], box2: Union[List[int], np.ndarray]) -> List[int]:
    """
    Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    left_top = np.minimum(box1[:2], box2[:2])
    right_bottom = np.maximum(box1[2:], box2[2:])
    return list(np.concatenate((left_top, right_bottom)))


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
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
    def get_score_func(object_prediction: ObjectPrediction):
        """Used for sorting predictions"""
        return object_prediction.score.value

    @staticmethod
    def has_same_category_id(pred1: ObjectPrediction, pred2: ObjectPrediction) -> bool:
        return pred1.category.id == pred2.category.id

    @staticmethod
    def calculate_bbox_iou(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
        """Returns the ratio of intersection area to the union"""
        box1 = np.array(pred1.bbox.to_voc_bbox())
        box2 = np.array(pred2.bbox.to_voc_bbox())
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        return intersect / (area1 + area2 - intersect)

    @staticmethod
    def calculate_bbox_ios(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
        """Returns the ratio of intersection area to the smaller box's area"""
        box1 = np.array(pred1.bbox.to_voc_bbox())
        box2 = np.array(pred2.bbox.to_voc_bbox())
        area1 = calculate_area(box1)
        area2 = calculate_area(box2)
        intersect = calculate_intersection_area(box1, box2)
        smaller_area = np.minimum(area1, area2)
        return intersect / smaller_area

    def __call__(self):
        NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        source_object_predictions: List[ObjectPrediction] = copy.deepcopy(object_predictions)
        selected_object_predictions: List[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            source_object_predictions.sort(reverse=True, key=self.get_score_func)
            selected_object_prediction = source_object_predictions[0]
            # remove selected prediction from source list
            del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it
            new_source_object_predictions: List[ObjectPrediction] = []
            for candidate_object_prediction in source_object_predictions:
                if self._has_match(selected_object_prediction, candidate_object_prediction):
                    new_source_object_predictions.append(candidate_object_prediction)
            source_object_predictions = new_source_object_predictions
            # append selected prediction to selected list
            selected_object_predictions.append(selected_object_prediction)
        return selected_object_predictions


class UnionMergePostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        source_object_predictions: List[ObjectPrediction] = copy.deepcopy(object_predictions)
        selected_object_predictions: List[ObjectPrediction] = []
        while len(source_object_predictions) > 0:
            # select object prediction with highest score
            source_object_predictions.sort(reverse=True, key=self.get_score_func)
            selected_object_prediction = source_object_predictions[0]
            # remove selected prediction from source list
            del source_object_predictions[0]
            # if any element from remaining source prediction list matches, remove it and merge with selected prediction
            new_source_object_predictions: List[ObjectPrediction] = []
            for candidate_object_prediction in source_object_predictions:
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
        shift_amount = pred1.bbox.shift_amount
        merged_bbox: BoundingBox = self._get_merged_bbox(pred1, pred2)
        merged_score: float = self._get_merged_score(pred1, pred2)
        merged_category: Category = self._get_merged_category(pred1, pred2)
        if pred1.mask and pred2.mask:
            merged_mask: Mask = self._get_merged_mask(pred1, pred2)
            bool_mask = merged_mask.bool_mask
            full_shape = merged_mask.full_shape
        else:
            bool_mask = None
            full_shape = None
        return ObjectPrediction(
            bbox=merged_bbox.to_voc_bbox(),
            score=merged_score,
            category_id=merged_category.id,
            category_name=merged_category.name,
            bool_mask=bool_mask,
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
        box1: List[int] = pred1.bbox.to_voc_bbox()
        box2: List[int] = pred2.bbox.to_voc_bbox()
        bbox = BoundingBox(box=calculate_box_union(box1, box2))
        return bbox

    @staticmethod
    def _get_merged_score(
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> float:
        scores: List[float] = [pred.score.value for pred in (pred1, pred2)]
        return max(scores)

    @staticmethod
    def _get_merged_mask(pred1: ObjectPrediction, pred2: ObjectPrediction) -> Mask:
        mask1 = pred1.mask
        mask2 = pred2.mask
        union_mask = np.logical_or(mask1.bool_mask, mask2.bool_mask)
        return Mask(
            bool_mask=union_mask,
            full_shape=mask1.full_shape,
            shift_amount=mask1.shift_amount,
        )


class LSNMSPostprocess(PostprocessPredictions):
    # https://github.com/remydubois/lsnms/blob/10b8165893db5bfea4a7cb23e268a502b35883cf/lsnms/nms.py#L62
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        try:
            from lsnms import nms
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please run "pip install lsnms>0.3.1" to install lsnms first for lsnms utilities.'
            )

        class_ids = np.array([object_prediction.category.id for object_prediction in object_predictions])
        boxes = np.array([object_prediction.bbox.to_voc_bbox() for object_prediction in object_predictions])
        scores = np.array([object_prediction.score.value for object_prediction in object_predictions])
        keep = nms(
            boxes, scores, iou_threshold=self.match_threshold, class_ids=None if self.class_agnostic else class_ids
        )

        # https://www.kite.com/python/answers/how-to-access-multiple-indices-of-a-list-in-python
        accessed_mapping = map(object_predictions.__getitem__, keep.tolist())
        selected_object_predictions = list(accessed_mapping)

        return selected_object_predictions
