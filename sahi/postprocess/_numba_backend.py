"""Numba JIT-compiled postprocessing backend.

Uses @njit(cache=True) to compile NMS/NMM loops to machine code.
First call incurs ~0.5-1s JIT compilation; subsequent calls are fast.
Compiled code is cached on disk across sessions.

For NMS and greedy NMM, the entire loop is JIT-compiled.
For NMM (transitive merge), the metric matrix is JIT-computed but the
merge logic runs in Python via the shared nmm_from_matrix function.
"""

from __future__ import annotations

import numba
import numpy as np

from sahi.postprocess._numpy_backend import (
    _score_tiebreak_order,
    nmm_from_matrix,
)


@numba.njit(cache=True)
def _compute_intersection(x1_a, y1_a, x2_a, y2_a, x1_b, y1_b, x2_b, y2_b):
    """Compute intersection area of two axis-aligned boxes."""
    ix1 = max(x1_a, x1_b)
    iy1 = max(y1_a, y1_b)
    ix2 = min(x2_a, x2_b)
    iy2 = min(y2_a, y2_b)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


@numba.njit(cache=True)
def _compute_metric(x1_a, y1_a, x2_a, y2_a, area_a, x1_b, y1_b, x2_b, y2_b, area_b, use_iou):
    """Compute IoU or IoS between two boxes."""
    inter = _compute_intersection(x1_a, y1_a, x2_a, y2_a, x1_b, y1_b, x2_b, y2_b)
    if use_iou:
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    else:
        smaller = min(area_a, area_b)
        return inter / smaller if smaller > 0 else 0.0


@numba.njit(cache=True)
def _argsort_descending(scores, x1, y1, x2, y2):
    """Sort indices by score descending with deterministic coordinate tie-breaking."""
    n = len(scores)
    indices = np.arange(n)
    for i in range(1, n):
        key = indices[i]
        j = i - 1
        while j >= 0:
            jj = indices[j]
            if scores[jj] > scores[key]:
                break
            if scores[jj] == scores[key]:
                if (x1[jj], y1[jj], x2[jj], y2[jj]) <= (x1[key], y1[key], x2[key], y2[key]):
                    break
            indices[j + 1] = indices[j]
            j -= 1
        indices[j + 1] = key
    return indices


@numba.njit(cache=True)
def _nms_numba_inner(x1, y1, x2, y2, scores, areas, match_threshold, use_iou):
    """Core NMS loop — fully JIT-compiled."""
    n = len(scores)
    sorted_idxs = _argsort_descending(scores, x1, y1, x2, y2)
    suppressed = np.zeros(n, dtype=numba.boolean)
    keep = []

    for i in range(n):
        idx = sorted_idxs[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        for j in range(i + 1, n):
            cand = sorted_idxs[j]
            if suppressed[cand]:
                continue
            metric = _compute_metric(
                x1[idx],
                y1[idx],
                x2[idx],
                y2[idx],
                areas[idx],
                x1[cand],
                y1[cand],
                x2[cand],
                y2[cand],
                areas[cand],
                use_iou,
            )
            if metric >= match_threshold:
                suppressed[cand] = True

    return keep


@numba.njit(cache=True)
def _greedy_nmm_numba_inner(x1, y1, x2, y2, scores, areas, match_threshold, use_iou):
    """Core greedy NMM loop — fully JIT-compiled.

    Returns (keep_order, keeper_of) where keeper_of[i] = keeper index for box i, or -1 if keeper.
    """
    n = len(scores)
    sorted_idxs = _argsort_descending(scores, x1, y1, x2, y2)
    suppressed = np.zeros(n, dtype=numba.boolean)
    keeper_of = np.full(n, -1, dtype=numba.int64)
    keep_order = []

    for i in range(n):
        idx = sorted_idxs[i]
        if suppressed[idx]:
            continue
        keep_order.append(idx)

        for j in range(i + 1, n):
            cand = sorted_idxs[j]
            if suppressed[cand]:
                continue
            metric = _compute_metric(
                x1[idx],
                y1[idx],
                x2[idx],
                y2[idx],
                areas[idx],
                x1[cand],
                y1[cand],
                x2[cand],
                y2[cand],
                areas[cand],
                use_iou,
            )
            if metric >= match_threshold:
                suppressed[cand] = True
                keeper_of[cand] = idx

    return keep_order, keeper_of


@numba.njit(cache=True)
def _compute_metric_matrix_numba(boxes, areas, use_iou):
    """Compute full metric matrix with numba — symmetric, only compute upper triangle."""
    n = len(boxes)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            metric = _compute_metric(
                boxes[i, 0],
                boxes[i, 1],
                boxes[i, 2],
                boxes[i, 3],
                areas[i],
                boxes[j, 0],
                boxes[j, 1],
                boxes[j, 2],
                boxes[j, 3],
                areas[j],
                use_iou,
            )
            matrix[i, j] = metric
            matrix[j, i] = metric
    return matrix


# ---------------------------------------------------------------------------
# Public numba backend functions
# ---------------------------------------------------------------------------


def nms_numba(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
    """Non-maximum suppression using numba JIT compilation."""
    if len(predictions) == 0:
        return []

    preds = predictions.astype(np.float64)
    x1, y1, x2, y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4]
    areas = (x2 - x1) * (y2 - y1)

    return _nms_numba_inner(x1, y1, x2, y2, scores, areas, match_threshold, match_metric == "IOU")


def greedy_nmm_numba(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Greedy non-maximum merging using numba JIT compilation."""
    preds = predictions.astype(np.float64)
    x1, y1, x2, y2 = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    scores = preds[:, 4]
    areas = (x2 - x1) * (y2 - y1)

    keep_order, keeper_of = _greedy_nmm_numba_inner(
        x1, y1, x2, y2, scores, areas, match_threshold, match_metric == "IOU"
    )

    # Convert parallel arrays to dict
    keep_to_merge: dict[int, list[int]] = {}
    for k in keep_order:
        keep_to_merge[int(k)] = []
    for i in range(len(keeper_of)):
        if keeper_of[i] >= 0:
            keep_to_merge[int(keeper_of[i])].append(int(i))

    return keep_to_merge


def nmm_numba(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """NMM using numba-computed metric matrix + shared Python merge logic."""
    preds = predictions.astype(np.float64)
    boxes = preds[:, :4]
    scores = preds[:, 4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    matrix = _compute_metric_matrix_numba(boxes, areas, match_metric == "IOU").astype(np.float32)
    sorted_idxs = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)

    return nmm_from_matrix(matrix, sorted_idxs, scores, boxes, match_threshold)
