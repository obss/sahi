# OBSS SAHI Tool
# Code written by Cemil Cengiz, 2020.
# Modified by Fatih C Akyon, 2020.

from typing import Callable, List

from sahi.postprocess.legacy.ops import BoxArray, box_ios, extract_box, have_same_class
from sahi.prediction import ObjectPrediction

PredictionList = List[ObjectPrediction]


class UnionFind:
    def __init__(self, length):
        self._length = length
        self._ids = list(range(length))

    def unify(self, ind1: int, ind2: int):
        id1 = self._ids[ind1]
        id2 = self._ids[ind2]
        for i in range(self._length):
            if self._ids[i] == id1:
                self._ids[i] = id2

    def unions(self) -> List[List[int]]:
        results = []
        for id in range(self._length):
            inds = [ind for ind in range(self._length) if self._ids[ind] == id]
            if len(inds) > 0:
                results.append(inds)

        return results


class PredictionMatcher:
    BOX_SCORERS = {"box_ios", "box_iou"}
    """
    Determines if two segmentation prediction are matches by comparing
    the score of the box pair with a threshold.

    Args:
        threshold : Minimum score for the boxes to determine two
            prediction matches
        scorer : The function that returns a match score for a box pair. 
            Must be an element of BOX_SCORERS.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        scorer: Callable[[BoxArray, BoxArray], float] = box_ios,
    ):
        self._threshold = threshold
        self._scorer = scorer

    def _validate_scorer(self, scorer: Callable):
        if scorer.__name__ not in self.BOX_SCORERS:
            raise ValueError(str(scorer) + " is not inside " + str(self.BOX_SCORERS))

    def score(self, box1: BoxArray, box2: BoxArray) -> float:
        return self._scorer(box1, box2)

    def exceeds_threshold(self, score: float):
        return score > self._threshold

    def can_match(self, pred1: ObjectPrediction, pred2: ObjectPrediction) -> bool:
        box1 = extract_box(pred1)
        box2 = extract_box(pred2)
        score = self._scorer(box1, box2)
        return score > self._threshold

    def find_matched_predictions(self, predictions: PredictionList, ignore_class_label: bool = True) -> List[List[int]]:
        """
        predictions :
        ignore_class_label : Set True, to allow matching the predictions
            that have different labels
        """
        size = len(predictions)
        union_find = UnionFind(size)
        for i in range(size):
            for j in range(i + 1, size):
                label_condition = ignore_class_label or have_same_class(predictions[i], predictions[j])
                if label_condition and self.can_match(predictions[i], predictions[j]):
                    union_find.unify(i, j)

        return union_find.unions()
