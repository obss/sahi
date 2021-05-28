# OBSS SAHI Tool
# Code written by Cemil Cengiz, 2020.
# Modified by Fatih C Akyon, 2020.

from enum import Enum
from typing import Callable, List

import numpy as np
from sahi.annotation import Mask
from sahi.postprocess.match import PredictionList, PredictionMatcher
from sahi.postprocess.ops import (
    BoxArray,
    box_union,
    box_intersection,
    calculate_area,
)
from sahi.prediction import ObjectPrediction


class ScoreMergingPolicy(Enum):
    """
    Denotes the policy used to determine the score of the resulting segment
    after merging two segments.
    """

    SMALLER_SCORE = 0
    LARGER_SCORE = 1
    AVERAGE = 2
    SMALLER_BOX = 3
    LARGER_BOX = 4
    WEIGHTED_AVERAGE = 5


class PredictionMerger:
    """
    Determines if two segmentation predictions are matches by comparing
    the score of the overlapping box pairs with a threshold.

    Args:
        score_merging : The policy used to determine the score of the
            resulting segment after merging
        box_merger : A function that can merge two boxes. Must be an element
            of BOX_MERGERS.
    """

    BOX_MERGERS = {"UNION", "INTERSECTION"}

    def __init__(
        self,
        score_merging: ScoreMergingPolicy = ScoreMergingPolicy.WEIGHTED_AVERAGE,
        box_merger: str = box_union, # INTERSECTION or UNION
    ):
        self._score_merging_method = score_merging
        self._box_merger = box_merger
        if box_merger == "UNION":
            self._box_merger: Callable = box_union
        elif box_merger == "INTERSECTION":
            self._box_merger: Callable = box_intersection
        else:
            raise ValueError(str(box_merger) + " is not inside " + str(self.BOX_MERGERS))

    def merge_batch(
        self,
        matcher: PredictionMatcher,
        predictions: PredictionList,
        merge_type: str = "merge",
        ignore_class_label: bool = True,
    ) -> PredictionList:
        """
        Merges the predictions that can be matched with each other and
        returns the result in a new list.

        Args:
            matcher :
            predictions :
            merge_type : a string from ["merge", "ensemble"]. If "ensemble"
                is chosen, the `num_models` field will be field. The `merged`
                field will be filled regardless of the merge_type choice.
            ignore_class_label : Set True, to allow merging the predictions
                that have different labels
        """
        if merge_type not in ["merge", "ensemble"]:
            raise ValueError(
                'Unknown merge type. Supported types are ["merge", "ensemble"], got type: ',
                merge_type,
            )

        unions = matcher.find_matched_predictions(predictions, ignore_class_label)
        return self._merge_predictions(unions, predictions, merge_type)

    def _merge_predictions(
        self, unions: List[List[int]], preds: PredictionList, merge_type: str
    ) -> PredictionList:
        results = []
        for inds in unions:
            count = len(inds)
            current = preds[inds[0]]
            for i in inds[1:]:
                current = self._merge_pair(current, preds[i])
            if merge_type == "ensemble":
                current.model_names = self._combine_model_names(
                    [preds[i] for i in inds]
                )

            self._store_merging_info(count, current, merge_type)
            results.append(current)

        return results

    @staticmethod
    def _combine_model_names(preds: PredictionList):
        model_names = set()
        for pred in preds:
            model_names.update(pred.model_names)
        return model_names

    @staticmethod
    def _store_merging_info(count, prediction, merge_type):
        if merge_type == "ensemble":
            prediction.num_models = count

        if count > 1:
            prediction.merged = True
        else:
            prediction.merged = False

    def _merge_pair(
        self,
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> ObjectPrediction:
        box1 = pred1.bbox.to_voc_bbox()
        box2 = pred2.bbox.to_voc_bbox()
        merged_box = list(self._merge_box(box1, box2))
        score = self._merge_score(pred1, pred2)
        shift_amount = pred1.bbox.shift_amount
        if pred1.mask and pred2.mask:
            mask = self._merge_mask(pred1, pred2)
            bool_mask = mask.bool_mask
            full_shape = mask.full_shape
        else:
            bool_mask = None
            full_shape = None
        category = self._merge_label(pred1, pred2)
        return ObjectPrediction(
            bbox=merged_box,
            score=score,
            category_id=category.id,
            category_name=category.name,
            bool_mask=bool_mask,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @staticmethod
    def _merge_label(pred1: ObjectPrediction, pred2: ObjectPrediction):
        if pred1.score.score > pred2.score.score:
            return pred1.category
        else:
            return pred2.category

    def _merge_box(self, box1: BoxArray, box2: BoxArray) -> BoxArray:
        return self._box_merger(box1, box2)

    def _merge_score(
        self,
        pred1: ObjectPrediction,
        pred2: ObjectPrediction,
    ) -> float:
        scores = [pred.score.score for pred in (pred1, pred2)]
        policy = self._score_merging_method
        if policy == ScoreMergingPolicy.SMALLER_SCORE:
            return min(scores)
        elif policy == ScoreMergingPolicy.LARGER_SCORE:
            return max(scores)
        elif policy == ScoreMergingPolicy.AVERAGE:
            return (scores[0] + scores[1]) / 2
        areas = np.array([calculate_area(pred.bbox.to_voc_bbox()) for pred in (pred1, pred2)])
        if policy == ScoreMergingPolicy.SMALLER_BOX:
            return scores[areas.argmin()]
        elif policy == ScoreMergingPolicy.LARGER_BOX:
            return scores[areas.argmax()]
        elif policy == ScoreMergingPolicy.WEIGHTED_AVERAGE:
            return (scores * areas).sum() / areas.sum()

    @staticmethod
    def _merge_mask(pred1: ObjectPrediction, pred2: ObjectPrediction) -> Mask:
        mask1 = pred1.mask
        mask2 = pred2.mask
        union_mask = np.logical_or(mask1.bool_mask, mask2.bool_mask)
        return Mask(
            bool_mask=union_mask,
            full_shape=mask1.full_shape,
            shift_amount=mask1.shift_amount,
        )
