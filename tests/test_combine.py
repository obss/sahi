import time
import math

import torch

from sahi.postprocess.combine import (
    nms,
    batched_nms,
    nmm,
    batched_nmm,
    greedy_nmm,
    batched_greedy_nmm,
)


def make_pred(x1, y1, x2, y2, score, cid):
    return [x1, y1, x2, y2, score, cid]


def test_nms_basic():
    # two overlapping boxes, same class -> keep highest scored and the distant box
    preds = torch.tensor(
        [
            make_pred(0, 0, 10, 10, 0.9, 1),
            make_pred(1, 1, 9, 9, 0.8, 1),
            make_pred(100, 100, 110, 110, 0.7, 1),
        ],
        dtype=torch.float32,
    )

    keep = nms(preds, match_metric="IOU", match_threshold=0.5)
    # highest scored box (index 0) should be kept and far box (index 2)
    assert 0 in keep and 2 in keep
    assert 1 not in keep


def test_batched_nms_class_aware():
    # overlapping boxes but different classes -> both kept
    preds = torch.tensor(
        [
            make_pred(0, 0, 10, 10, 0.9, 1),
            make_pred(1, 1, 9, 9, 0.8, 2),
            make_pred(0, 0, 10, 10, 0.85, 1),
        ],
        dtype=torch.float32,
    )

    keep = batched_nms(preds, match_metric="IOU", match_threshold=0.5)
    # for class 1 there are two boxes, the highest (0) should survive, for class 2 (index 1) also survives
    assert 0 in keep
    assert 1 in keep
    assert 2 not in keep


def test_nmm_merge_mapping():
    # create one high-score box overlapping two lower-score boxes
    preds = torch.tensor(
        [
            make_pred(0, 0, 20, 20, 0.95, 1),
            make_pred(2, 2, 10, 10, 0.5, 1),
            make_pred(5, 5, 15, 15, 0.4, 1),
        ],
        dtype=torch.float32,
    )

    keep_to_merge = nmm(preds, match_metric="IOU", match_threshold=0.1)
    # highest scored box should be the keeper and map to the others
    # key should be the index of keeper (0) and include 1 and 2
    assert isinstance(keep_to_merge, dict)
    assert 0 in keep_to_merge
    merged = set(keep_to_merge[0])
    assert 1 in merged and 2 in merged


def test_greedy_nmm_and_batched():
    # similar to nmm but greedy variant should also map the two low-score boxes to keeper
    preds = torch.tensor(
        [
            make_pred(0, 0, 20, 20, 0.95, 1),
            make_pred(2, 2, 10, 10, 0.5, 1),
            make_pred(5, 5, 15, 15, 0.4, 1),
        ],
        dtype=torch.float32,
    )

    greedy_map = greedy_nmm(preds, match_metric="IOU", match_threshold=0.1)
    assert 0 in greedy_map
    assert set(greedy_map[0]) == {1, 2}

    # batched_greedy_nmm should behave the same when class-aware splitting is trivial
    batched = batched_greedy_nmm(preds, match_metric="IOU", match_threshold=0.1)
    # batched_greedy_nmm returns mapping keyed by global indices; check keeper maps
    assert any(len(v) > 0 for v in batched.values())


def test_batched_nmm_class_aware():
    # create two classes, ensure merging works per-class
    preds = torch.tensor(
        [
            make_pred(0, 0, 20, 20, 0.95, 1),
            make_pred(2, 2, 10, 10, 0.5, 2),
            make_pred(3, 3, 11, 11, 0.4, 2),
        ],
        dtype=torch.float32,
    )

    keep_to_merge = batched_nmm(preds, match_metric="IOU", match_threshold=0.1)
    # class 1 has no merges, class 2 should have one keeper mapping to the overlapping lower box
    assert 0 in keep_to_merge or 1 in keep_to_merge or 2 in keep_to_merge
    # find any mapping that contains index 2 (the low-score class2 box) as merged
    merged_any = any(2 in v for v in keep_to_merge.values())
    assert merged_any
