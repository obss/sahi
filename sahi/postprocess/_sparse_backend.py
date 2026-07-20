"""Sparse postprocessing path for large prediction counts.

The greedy loops only need ``metric >= threshold``, never the metric itself, so
this module skips the dense ``N x N`` matrix and builds the thresholded
adjacency directly from the box pairs a shapely STRtree reports as
intersecting. The adjacency is CSR: row ``i`` is
``indices[indptr[i]:indptr[i + 1]]``, sorted ascending.
"""

from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely import box as shapely_box

# Below this many boxes the dense path wins: the STRtree build costs more than
# the N x N matrix it saves.
SPARSE_MIN_BOXES = 2000


def should_use_sparse(n: int, match_threshold: float) -> bool:
    """Return whether the sparse path applies to this input.

    A non-positive threshold matches every pair, including disjoint ones, which
    the intersection-based prefilter would never enumerate. Those inputs must
    stay on the dense path to preserve behaviour.

    Args:
        n: Number of predictions.
        match_threshold: Overlap threshold used for matching.

    Returns:
        True when the sparse path is both applicable and worthwhile.
    """
    return n >= SPARSE_MIN_BOXES and match_threshold > 0


def build_sparse_matches(
    boxes: np.ndarray,
    areas: np.ndarray,
    match_metric: str,
    match_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the thresholded match adjacency without materializing an N x N matrix.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        areas: Precomputed areas of shape (N,).
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum metric value counted as a match.

    Returns:
        A tuple of (indptr, indices) describing the CSR adjacency. Columns
        within each row are sorted ascending and never include the row itself.
    """
    n = len(boxes)
    geoms = shapely_box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    tree = STRtree(geoms)

    # Only intersecting pairs can clear a positive threshold. Returns both
    # (i, j) and (j, i), so the resulting adjacency is symmetric like the
    # dense matrix. Both IOU and IOS are symmetric metrics.
    rows, cols = tree.query(geoms, predicate="intersects")

    self_pair = rows == cols
    if self_pair.any():
        rows, cols = rows[~self_pair], cols[~self_pair]

    inter_x1 = np.maximum(boxes[rows, 0], boxes[cols, 0])
    inter_y1 = np.maximum(boxes[rows, 1], boxes[cols, 1])
    inter_x2 = np.minimum(boxes[rows, 2], boxes[cols, 2])
    inter_y2 = np.minimum(boxes[rows, 3], boxes[cols, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    if match_metric == "IOU":
        denom = areas[rows] + areas[cols] - inter
    else:  # IOS
        denom = np.minimum(areas[rows], areas[cols])
    metric = np.where(denom > 0, inter / denom, 0)

    matched = metric >= match_threshold
    rows, cols = rows[matched], cols[matched]

    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]

    indptr = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols.astype(np.intp)


def nms_sparse(indptr: np.ndarray, indices: np.ndarray, sorted_idxs: np.ndarray) -> list[int]:
    """NMS over a CSR match adjacency. Mirrors ``nms_from_matrix``.

    Args:
        indptr: CSR row pointers of length N + 1.
        indices: CSR column indices.
        sorted_idxs: Indices sorted by score descending.

    Returns:
        List of kept indices sorted by score descending.
    """
    keep: list[int] = []
    suppressed = np.zeros(len(indptr) - 1, dtype=bool)

    for idx in sorted_idxs:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        suppressed[indices[indptr[idx] : indptr[idx + 1]]] = True

    return keep


def greedy_nmm_sparse(
    indptr: np.ndarray,
    indices: np.ndarray,
    sorted_idxs: np.ndarray,
) -> dict[int, list[int]]:
    """Greedy NMM over a CSR match adjacency. Mirrors ``greedy_nmm_from_matrix``.

    Args:
        indptr: CSR row pointers of length N + 1.
        indices: CSR column indices.
        sorted_idxs: Indices sorted by score descending.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    n = len(indptr) - 1
    suppressed = np.zeros(n, dtype=bool)

    # The dense loop only considers candidates that come later in score order,
    # and emits them in that order.
    rank = np.empty(n, dtype=np.intp)
    rank[sorted_idxs] = np.arange(n)

    keep_to_merge_list: dict[int, list[int]] = {}
    for position, idx in enumerate(sorted_idxs):
        if suppressed[idx]:
            continue

        neighbours = indices[indptr[idx] : indptr[idx + 1]]
        merge_indices = neighbours[(rank[neighbours] > position) & ~suppressed[neighbours]]
        merge_indices = merge_indices[np.argsort(rank[merge_indices])]

        suppressed[merge_indices] = True
        keep_to_merge_list[int(idx)] = merge_indices.tolist()

    return keep_to_merge_list


def _dominates(current: int, neighbours: np.ndarray, scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return which neighbours ``current`` may claim as merge candidates.

    A box claims another when the other scores lower, or scores equal and does
    not sort before it lexicographically by coordinates. Same rule as the dense
    ``dominates`` matrix, evaluated only for the given neighbours.

    Args:
        current: Index of the claiming box.
        neighbours: Indices of the candidate boxes.
        scores: Prediction scores of shape (N,).
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].

    Returns:
        Boolean array aligned with ``neighbours``, True where it may be claimed.
    """
    lower_score = scores[current] > scores[neighbours]
    score_equal = scores[current] == scores[neighbours]

    current_lt = np.zeros(len(neighbours), dtype=bool)
    still_equal = np.ones(len(neighbours), dtype=bool)
    for col in range(4):
        col_lt = boxes[current, col] < boxes[neighbours, col]
        col_eq = boxes[current, col] == boxes[neighbours, col]
        current_lt |= still_equal & col_lt
        still_equal &= col_eq

    return lower_score | (score_equal & ~current_lt)


def nmm_sparse(
    indptr: np.ndarray,
    indices: np.ndarray,
    sorted_idxs: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
) -> dict[int, list[int]]:
    """NMM (non-greedy, transitive merge) over a CSR match adjacency.

    Mirrors ``nmm_from_matrix``.

    Args:
        indptr: CSR row pointers of length N + 1.
        indices: CSR column indices.
        sorted_idxs: Indices sorted by score descending.
        scores: Prediction scores of shape (N,).
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    n = len(sorted_idxs)
    if n == 0:
        return {}

    keep_to_merge_list: dict[int, list[int]] = {}
    merge_to_keep = np.full(n, -1, dtype=np.intp)

    for idx_pos in range(n):
        current_idx = int(sorted_idxs[idx_pos])
        neighbours = indices[indptr[current_idx] : indptr[current_idx + 1]]
        matched = neighbours[_dominates(current_idx, neighbours, scores, boxes)]

        if merge_to_keep[current_idx] < 0:
            keep_to_merge_list[current_idx] = []
            for m in matched:
                m_int = int(m)
                if merge_to_keep[m_int] < 0:
                    keep_to_merge_list[current_idx].append(m_int)
                    merge_to_keep[m_int] = current_idx
        else:
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
