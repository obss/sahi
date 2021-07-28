# OBSS SAHI Tool
# Code written by Cemil Cengiz, 2020.

import pickle
from copy import deepcopy
from math import floor
from typing import List

import cv2
import pytest
from sahi.annotation import BoundingBox
from sahi.postprocess.legacy.match import PredictionList, PredictionMatcher
from sahi.postprocess.legacy.merge import PredictionMerger, ScoreMergingPolicy
from sahi.postprocess.legacy.ops import box_ios, box_union
from sahi.prediction import ObjectPrediction


@pytest.fixture(scope="module")
def matcher():
    return PredictionMatcher(threshold=0.5, scorer=box_ios)


@pytest.fixture(scope="module")
def merger():
    return PredictionMerger(score_merging=ScoreMergingPolicy.LARGER_SCORE, box_merger=box_union)


@pytest.fixture(scope="function")
def image():
    return cv2.imread("../outputs/openTest88_merged/raw-images/openTest_88.png")


@pytest.fixture(scope="function")
def pickle_file():
    return "../outputs/openTest88_merged/pickle-outputs/openTest_88.pickle"


@pytest.fixture(scope="function")
def raw_predictions(pickle_file):
    with open(pickle_file, "rb") as f:
        predictions = pickle.load(f)
    for pred in predictions:
        pred.num_models = None
        pred.merged = None
    return predictions


@pytest.fixture(scope="function")
def single_predictions(raw_predictions, merger, matcher):
    return merger.merge_batch(matcher, raw_predictions, merge_type="merge")


@pytest.mark.skip()
def test_reading_raw_pred_from_pickle(raw_predictions):
    assert len(raw_predictions) == 7


def _perturb(box: BoundingBox):
    minx = floor((box.minx + box.maxx) * 2 / 5)
    maxx = floor(box.maxx * 10 / 9)

    miny = floor((box.miny + box.maxy) * 2 / 5)
    maxy = floor(box.maxy * 10 / 9)

    return BoundingBox(box=[minx, miny, maxx, maxy], shift_amount=box.shift_amount)


def perturb_boxes(preds: List[ObjectPrediction],) -> List[ObjectPrediction]:
    preds = deepcopy(preds)
    for i in range(len(preds)):
        if i % 2 == 0:
            preds[i].box = _perturb(preds[i].box)
    return preds


def _are_equal(box1: BoundingBox, box2: BoundingBox):
    for attr in ("minx", "maxx", "miny", "maxy"):
        if getattr(box1, attr) != getattr(box2, attr):
            return False
    return True


def _contains_box(preds: PredictionList, box: BoundingBox):
    for pred in preds:
        if _are_equal(pred.box, box):
            return True
    return False


def _merge_boxes_sequentially(preds1: PredictionList, preds2: PredictionList) -> List[BoundingBox]:
    return [
        BoundingBox(list(box_union(pred1.bbox.to_voc_bbox(), pred2.bbox.to_voc_bbox())))
        for pred1, pred2 in zip(preds1, preds2)
    ]


def _assert_boxes_are_equal(boxes1: List[BoundingBox], boxes2: List[BoundingBox]):
    box_arrays1 = set((box.minx, box.miny, box.maxx, box.maxy) for box in boxes1)

    box_arrays2 = set((box.minx, box.miny, box.maxx, box.maxy) for box in boxes2)

    difference = box_arrays1.difference(box_arrays2)
    assert len(difference) == 0
