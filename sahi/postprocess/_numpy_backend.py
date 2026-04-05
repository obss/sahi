"""Pure numpy postprocessing backend — no shapely dependency.

Replaces shapely STRtree + intersection with vectorized numpy broadcasting
for axis-aligned bounding box IoU/IoS computation.
"""

from __future__ import annotations

import numpy as np

# Maximum N for full NxN matrix computation. Above this, compute row-batches.
_MAX_FULL_MATRIX = 8000


# ---------------------------------------------------------------------------
# Shared utilities — used by all backends via precomputed metric matrix
# ---------------------------------------------------------------------------


def compute_metric_matrix(boxes: np.ndarray, areas: np.ndarray, match_metric: str) -> np.ndarray:
    """Compute pairwise IoU or IoS matrix for axis-aligned boxes.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2]
        areas: (N,) precomputed areas
        match_metric: "IOU" or "IOS"

    Returns:
        (N, N) float32 matrix of pairwise metric values
    """
    n = len(boxes)
    if n <= _MAX_FULL_MATRIX:
        return _compute_metric_matrix_full(boxes, areas, match_metric)

    # Batched computation for large N to limit memory
    matrix = np.zeros((n, n), dtype=np.float32)
    batch = _MAX_FULL_MATRIX
    for i in range(0, n, batch):
        ie = min(i + batch, n)
        inter_x1 = np.maximum(boxes[i:ie, None, 0], boxes[None, :, 0])
        inter_y1 = np.maximum(boxes[i:ie, None, 1], boxes[None, :, 1])
        inter_x2 = np.minimum(boxes[i:ie, None, 2], boxes[None, :, 2])
        inter_y2 = np.minimum(boxes[i:ie, None, 3], boxes[None, :, 3])
        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        if match_metric == "IOU":
            union = areas[i:ie, None] + areas[None, :] - inter
            matrix[i:ie] = np.where(union > 0, inter / union, 0)
        else:  # IOS
            smaller = np.minimum(areas[i:ie, None], areas[None, :])
            matrix[i:ie] = np.where(smaller > 0, inter / smaller, 0)
    return matrix


def _compute_metric_matrix_full(boxes: np.ndarray, areas: np.ndarray, match_metric: str) -> np.ndarray:
    """Compute the full N-by-N pairwise metric matrix in one vectorized pass.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        areas: Precomputed areas of shape (N,).
        match_metric: "IOU" for Intersection over Union, or "IOS" for
            Intersection over Smaller.

    Returns:
        A float32 array of shape (N, N) with pairwise metric values.
    """
    inter_x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    if match_metric == "IOU":
        union = areas[:, None] + areas[None, :] - inter
        return np.where(union > 0, inter / union, 0).astype(np.float32)
    else:  # IOS
        smaller = np.minimum(areas[:, None], areas[None, :])
        return np.where(smaller > 0, inter / smaller, 0).astype(np.float32)


def _score_tiebreak_order(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Return indices sorted by score descending, with deterministic tie-breaking.

    When scores are equal, ties are broken by box coordinates
    (y2, x2, y1, x1) to ensure reproducible ordering.

    Args:
        x1: Array of left x-coordinates.
        y1: Array of top y-coordinates.
        x2: Array of right x-coordinates.
        y2: Array of bottom y-coordinates.
        scores: Array of confidence scores.

    Returns:
        Array of indices that would sort the predictions by score
        descending with deterministic tie-breaking.
    """
    order = np.lexsort((y2, x2, y1, x1, -scores))
    return order


def _prepare_matrix(predictions: np.ndarray, match_metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract boxes, scores, and areas, then compute the pairwise metric matrix.

    Shared setup step used by nms_numpy, greedy_nmm_numpy, and nmm_numpy.

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".

    Returns:
        A tuple of (matrix, sorted_idxs) where matrix is the (N, N)
        pairwise metric array and sorted_idxs is an array of indices
        sorted by score descending.
    """
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    matrix = compute_metric_matrix(boxes, areas, match_metric)
    sorted_idxs = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)
    return matrix, sorted_idxs


# ---------------------------------------------------------------------------
# Shared greedy loop functions — operate on precomputed matrix
# ---------------------------------------------------------------------------


def nms_from_matrix(matrix: np.ndarray, sorted_idxs: np.ndarray, match_threshold: float) -> list[int]:
    """NMS using a precomputed metric matrix. Used by numpy and torchvision backends.

    Args:
        matrix: (N, N) pairwise metric array.
        sorted_idxs: Indices sorted by score descending.
        match_threshold: Minimum metric value to suppress a candidate.

    Returns:
        List of kept indices sorted by score descending.
    """
    keep = []
    suppressed = np.zeros(matrix.shape[0], dtype=bool)

    for idx in sorted_idxs:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        mask = matrix[idx] >= match_threshold
        suppressed |= mask

    return keep


def greedy_nmm_from_matrix(matrix: np.ndarray, sorted_idxs: np.ndarray, match_threshold: float) -> dict[int, list[int]]:
    """Greedy NMM using a precomputed metric matrix. Used by numpy, numba, and torchvision backends.

    Args:
        matrix: (N, N) pairwise metric array.
        sorted_idxs: Indices sorted by score descending.
        match_threshold: Minimum metric value to merge a candidate.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    keep_to_merge_list: dict[int, list[int]] = {}
    suppressed = np.zeros(matrix.shape[0], dtype=bool)

    for i, idx in enumerate(sorted_idxs):
        if suppressed[idx]:
            continue

        remaining = sorted_idxs[i + 1 :]
        if len(remaining) == 0:
            keep_to_merge_list[int(idx)] = []
            continue

        not_suppressed = ~suppressed[remaining]
        candidates = remaining[not_suppressed]

        if len(candidates) == 0:
            keep_to_merge_list[int(idx)] = []
            continue

        metrics = matrix[idx, candidates]
        above_threshold = metrics >= match_threshold
        merge_indices = candidates[above_threshold]

        suppressed[merge_indices] = True
        keep_to_merge_list[int(idx)] = merge_indices.tolist()

    return keep_to_merge_list


def nmm_from_matrix(
    matrix: np.ndarray,
    sorted_idxs: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    match_threshold: float,
) -> dict[int, list[int]]:
    """NMM (non-greedy, transitive merge) using a precomputed metric matrix.

    Used by numpy, numba, and torchvision backends.

    The inner candidate search is vectorized with numpy; only the
    transitive merge bookkeeping remains sequential.
    """
    n = len(sorted_idxs)
    if n == 0:
        return {}

    # Precompute a boolean dominance matrix: dominates[i, j] = True means
    # box i can claim box j as a merge candidate (j has lower score, or
    # equal score with j's coordinates <= i's coordinates).
    s = scores  # (N,)
    cand_lower_score = s[:, None] > s[None, :]  # s[i] > s[j]: j has strictly lower score
    score_equal = s[:, None] == s[None, :]

    # Lexicographic comparison: cur_lt_cand[i, j] = coords(i) < coords(j).
    # We need: coords(j) <= coords(i), which is NOT (coords(j) > coords(i))
    # = NOT (coords(i) < coords(j)) = ~cur_lt_cand[i, j].
    b = boxes  # (N, 4)
    cur_lt_cand = np.zeros((n, n), dtype=bool)
    still_eq = np.ones((n, n), dtype=bool)
    for col in range(4):
        col_lt = b[:, col][:, None] < b[:, col][None, :]
        col_eq = b[:, col][:, None] == b[:, col][None, :]
        cur_lt_cand |= still_eq & col_lt
        still_eq &= col_eq

    # dominates[i, j]: i can merge j (j is a valid candidate for i)
    eye_mask = np.eye(n, dtype=bool)
    dominates = (~eye_mask) & (cand_lower_score | (score_equal & ~cur_lt_cand))

    # Threshold mask from the metric matrix
    above_thresh = matrix >= match_threshold

    # For each box, precompute the set of candidates it dominates AND exceeds threshold
    # candidates_of[i] is the array of indices j that i can merge
    candidates_of = dominates & above_thresh

    # Sequential transitive merge bookkeeping
    keep_to_merge_list: dict[int, list[int]] = {}
    merge_to_keep = np.full(n, -1, dtype=np.intp)

    for idx_pos in range(n):
        current_idx = int(sorted_idxs[idx_pos])
        matched = np.where(candidates_of[current_idx])[0]

        if merge_to_keep[current_idx] < 0:
            # current_idx is a keeper
            keep_to_merge_list[current_idx] = []
            for m in matched:
                m_int = int(m)
                if merge_to_keep[m_int] < 0:
                    keep_to_merge_list[current_idx].append(m_int)
                    merge_to_keep[m_int] = current_idx
        else:
            # current_idx was already merged into a keeper
            keep_idx = int(merge_to_keep[current_idx])
            merge_list = keep_to_merge_list.get(keep_idx, [])
            if keep_idx not in keep_to_merge_list:
                keep_to_merge_list[keep_idx] = merge_list
            for m in matched:
                m_int = int(m)
                if m_int not in merge_list and merge_to_keep[m_int] < 0:
                    merge_list.append(m_int)
                    merge_to_keep[m_int] = keep_idx

    return keep_to_merge_list


# ---------------------------------------------------------------------------
# Public numpy backend functions
# ---------------------------------------------------------------------------


def nms_numpy(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
    """Non-maximum suppression using vectorized numpy metric matrix."""
    if len(predictions) == 0:
        return []
    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    return nms_from_matrix(matrix, sorted_idxs, match_threshold)


def greedy_nmm_numpy(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Greedy non-maximum merging using vectorized numpy metric matrix."""
    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    return greedy_nmm_from_matrix(matrix, sorted_idxs, match_threshold)


def nmm_numpy(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Non-maximum merging using vectorized numpy metric matrix."""
    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    return nmm_from_matrix(matrix, sorted_idxs, predictions[:, 4], predictions[:, :4], match_threshold)
