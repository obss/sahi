"""Sparse postprocessing path for large prediction counts.

The greedy loops only need ``metric >= threshold``, never the metric itself, so
this module skips the dense ``N x N`` matrix and builds the thresholded
adjacency directly from the box pairs a shapely STRtree reports as
intersecting. The adjacency is CSR: row ``i`` is
``indices[indptr[i]:indptr[i + 1]]``, sorted ascending.

Storing every pair is only a saving while boxes are spread out. Boxes piled on
top of each other intersect nearly everything, so the pair count approaches
``N ^ 2`` and the CSR gets as expensive as the matrix it replaced. The loops
read one row at a time and NMS and greedy NMM only ever read rows of boxes that
survived, so ``MatchQuery`` answers rows from the tree on demand instead, which
holds peak memory at ``O(N + max_degree)`` whatever the layout.
"""

from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely import box as shapely_box

# Below this many boxes the dense path wins: the STRtree build costs more than
# the N x N matrix it saves.
SPARSE_MIN_BOXES = 2000

# NMM reads a row per box rather than per survivor, so answering its rows from
# the tree costs it roughly an order of magnitude in time. It keeps the stored
# pair list until that list would get large, measured at about 110 bytes per
# pair once the intermediates are counted.
NMM_MAX_PAIRS = 2_000_000


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide where the denominator is positive, yielding zero elsewhere.

    Degenerate boxes have zero area, so the denominator can be zero. Masking the
    division rather than the result keeps numpy from evaluating ``0 / 0`` and
    warning about it.
    """
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.broadcast(numerator, denominator).shape, dtype=np.result_type(numerator, denominator)),
        where=denominator > 0,
    )


def should_stream_nmm(boxes: np.ndarray) -> bool:
    """Return whether NMM should answer rows from the tree instead of storing pairs.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].

    Returns:
        True when the pair list is projected to exceed ``NMM_MAX_PAIRS``.
    """
    return estimate_mean_degree(boxes) * len(boxes) > NMM_MAX_PAIRS


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


class MatchQuery:
    """Answers 'which boxes match box i' from an STRtree, without storing pairs.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        areas: Precomputed areas of shape (N,).
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum metric value counted as a match.
    """

    def __init__(self, boxes: np.ndarray, areas: np.ndarray, match_metric: str, match_threshold: float) -> None:
        self.boxes = boxes
        self.areas = areas
        self.match_metric = match_metric
        self.match_threshold = match_threshold
        self.geoms = shapely_box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
        self.tree = STRtree(self.geoms)

    def row(self, i: int) -> np.ndarray:
        """Return the matches of box ``i``, ascending, excluding ``i`` itself.

        Equivalent to one row of the CSR ``build_sparse_matches`` would build.
        """
        candidates = self.tree.query(self.geoms[i], predicate="intersects")
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            return candidates.astype(np.intp)

        boxes, areas = self.boxes, self.areas
        inter_x1 = np.maximum(boxes[i, 0], boxes[candidates, 0])
        inter_y1 = np.maximum(boxes[i, 1], boxes[candidates, 1])
        inter_x2 = np.minimum(boxes[i, 2], boxes[candidates, 2])
        inter_y2 = np.minimum(boxes[i, 3], boxes[candidates, 3])
        inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        if self.match_metric == "IOU":
            denom = areas[i] + areas[candidates] - inter
        else:  # IOS
            denom = np.minimum(areas[i], areas[candidates])
        metric = _safe_ratio(inter, denom)

        return np.sort(candidates[metric >= self.match_threshold]).astype(np.intp)


def nms_streaming(
    boxes: np.ndarray,
    areas: np.ndarray,
    match_metric: str,
    match_threshold: float,
    sorted_idxs: np.ndarray,
) -> list[int]:
    """NMS reading match rows on demand. Same result as ``nms_sparse``.

    Only boxes that survive are ever queried, so on crowded inputs, where almost
    everything is suppressed, this touches a small fraction of the pairs the CSR
    would have stored.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        areas: Precomputed areas of shape (N,).
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum metric value counted as a match.
        sorted_idxs: Indices sorted by score descending.

    Returns:
        List of kept indices sorted by score descending.
    """
    matches = MatchQuery(boxes, areas, match_metric, match_threshold)
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep: list[int] = []

    for idx in sorted_idxs:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        suppressed[matches.row(int(idx))] = True

    return keep


def greedy_nmm_streaming(
    boxes: np.ndarray,
    areas: np.ndarray,
    match_metric: str,
    match_threshold: float,
    sorted_idxs: np.ndarray,
) -> dict[int, list[int]]:
    """Greedy NMM reading match rows on demand. Same result as ``greedy_nmm_sparse``.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        areas: Precomputed areas of shape (N,).
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum metric value counted as a match.
        sorted_idxs: Indices sorted by score descending.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    n = len(boxes)
    matches = MatchQuery(boxes, areas, match_metric, match_threshold)
    suppressed = np.zeros(n, dtype=bool)

    # The dense loop only considers candidates that come later in score order,
    # and emits them in that order.
    rank = np.empty(n, dtype=np.intp)
    rank[sorted_idxs] = np.arange(n)

    keep_to_merge_list: dict[int, list[int]] = {}
    for position, idx in enumerate(sorted_idxs):
        if suppressed[idx]:
            continue

        neighbours = matches.row(int(idx))
        merge_indices = neighbours[(rank[neighbours] > position) & ~suppressed[neighbours]]
        merge_indices = merge_indices[np.argsort(rank[merge_indices])]

        suppressed[merge_indices] = True
        keep_to_merge_list[int(idx)] = merge_indices.tolist()

    return keep_to_merge_list


def estimate_mean_degree(boxes: np.ndarray, sample: int = 512) -> float:
    """Estimate how many other boxes an average box intersects.

    Queries the tree with an evenly spaced subset instead of every box, so the
    cost is a small fraction of the full query the sparse path would run.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        sample: Number of boxes to probe with.

    Returns:
        Mean number of intersecting neighbours, self-pair excluded.
    """
    n = len(boxes)
    geoms = shapely_box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    tree = STRtree(geoms)

    k = min(n, sample)
    probe = geoms[np.linspace(0, n - 1, k).astype(np.intp)]
    rows, _ = tree.query(probe, predicate="intersects")
    return max(0.0, len(rows) / k - 1.0)


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
    metric = _safe_ratio(inter, denom)

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


def _dominates_all(indptr: np.ndarray, indices: np.ndarray, scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return, for every CSR entry, whether its row may claim its column.

    A box claims another when the other scores lower, or scores equal and does
    not sort before it lexicographically by coordinates. Same rule as the dense
    ``dominates`` matrix, evaluated for all stored pairs at once so the merge
    loop below does no per-row numpy work.

    Args:
        indptr: CSR row pointers of length N + 1.
        indices: CSR column indices.
        scores: Prediction scores of shape (N,).
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].

    Returns:
        Boolean array aligned with ``indices``, True where the row may claim it.
    """
    n = len(indptr) - 1
    rows = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))

    lower_score = scores[rows] > scores[indices]
    score_equal = scores[rows] == scores[indices]

    row_lt = np.zeros(len(indices), dtype=bool)
    still_equal = np.ones(len(indices), dtype=bool)
    for col in range(4):
        col_lt = boxes[rows, col] < boxes[indices, col]
        col_eq = boxes[rows, col] == boxes[indices, col]
        row_lt |= still_equal & col_lt
        still_equal &= col_eq

    return lower_score | (score_equal & ~row_lt)


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

    dominates = _dominates_all(indptr, indices, scores, boxes)

    keep_to_merge_list: dict[int, list[int]] = {}
    merge_to_keep = np.full(n, -1, dtype=np.intp)

    for idx_pos in range(n):
        current_idx = int(sorted_idxs[idx_pos])
        start, end = indptr[current_idx], indptr[current_idx + 1]
        matched = indices[start:end][dominates[start:end]]

        if merge_to_keep[current_idx] < 0:
            # current_idx is a keeper. Point it at itself so that a later box
            # cannot claim it: keepers are never merged into anything.
            merge_to_keep[current_idx] = current_idx
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


def nmm_streaming(
    boxes: np.ndarray,
    areas: np.ndarray,
    match_metric: str,
    match_threshold: float,
    sorted_idxs: np.ndarray,
    scores: np.ndarray,
) -> dict[int, list[int]]:
    """NMM reading match rows on demand. Same result as ``nmm_sparse``.

    Unlike NMS and greedy NMM this visits every box, not just the survivors, so
    it issues one tree query per box and is markedly slower than the stored-pair
    version. It exists for inputs whose pair list does not fit in memory.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].
        areas: Precomputed areas of shape (N,).
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum metric value counted as a match.
        sorted_idxs: Indices sorted by score descending.
        scores: Prediction scores of shape (N,).

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    n = len(boxes)
    matches = MatchQuery(boxes, areas, match_metric, match_threshold)

    keep_to_merge_list: dict[int, list[int]] = {}
    merge_to_keep = np.full(n, -1, dtype=np.intp)

    for idx_pos in range(n):
        current_idx = int(sorted_idxs[idx_pos])
        matched = _dominated_row(matches.row(current_idx), current_idx, scores, boxes)

        if merge_to_keep[current_idx] < 0:
            # current_idx is a keeper. Point it at itself so that a later box
            # cannot claim it: keepers are never merged into anything.
            merge_to_keep[current_idx] = current_idx
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


def _dominated_row(candidates: np.ndarray, i: int, scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Restrict one match row to the columns its row may claim.

    Single-row form of ``_dominates_all``.
    """
    if len(candidates) == 0:
        return candidates

    lower_score = scores[i] > scores[candidates]
    score_equal = scores[i] == scores[candidates]

    row_lt = np.zeros(len(candidates), dtype=bool)
    still_equal = np.ones(len(candidates), dtype=bool)
    for col in range(4):
        col_lt = boxes[i, col] < boxes[candidates, col]
        col_eq = boxes[i, col] == boxes[candidates, col]
        row_lt |= still_equal & col_lt
        still_equal &= col_eq

    return candidates[lower_score | (score_equal & ~row_lt)]
