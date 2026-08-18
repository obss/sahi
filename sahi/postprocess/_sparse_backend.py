"""Sparse postprocessing path for large prediction counts.

The greedy loops only need ``metric >= threshold``, never the metric itself, so
this module skips the dense ``N x N`` matrix and builds the thresholded
adjacency directly from the box pairs a shapely STRtree reports as
intersecting. The adjacency is CSR: row ``i`` is
``indices[indptr[i]:indptr[i + 1]]``, sorted ascending.

Storing every pair is only a saving while boxes are spread out. Boxes piled on
top of each other intersect nearly everything, so the pair count approaches
``N ^ 2`` and the CSR gets as expensive as the matrix it replaced. The loops
read one row at a time, so ``MatchQuery`` answers rows from the tree on demand
instead, which holds peak memory at ``O(N + max_degree)`` whatever the layout.
Each loop also hands boxes back to the query as it finishes with them, so a
crowded input shrinks the tree it is querying as it goes.
"""

from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely import box as shapely_box

# Below this many boxes the dense path wins: the STRtree build costs more than
# the N x N matrix it saves.
SPARSE_MIN_BOXES = 2000

# NMM reads a row for every box, not just for the survivors, so answering its
# rows from a stored pair list is worth the memory while boxes have few
# neighbours. Past this many the list costs more to build than the queries it
# saves, and keeps growing with the square of the crowding. The measured
# crossover barely moves with N, so it is a degree and not a pair budget.
NMM_MAX_DEGREE = 40.0

# Probing the tree in one call would return sample x degree pairs at once, which
# on crowded boxes dwarfs everything the path being chosen goes on to allocate.
# Only the count is wanted, so the probe is spent a chunk at a time.
DEGREE_PROBE_CHUNK = 32


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


def _overlap_metric(
    boxes: np.ndarray, areas: np.ndarray, left: int | np.ndarray, right: np.ndarray, match_metric: str
) -> np.ndarray:
    """Overlap of each ``left`` box against the ``right`` box paired with it.

    ``left`` may be a single index, which broadcasts against ``right``. Holding the
    IOU and IOS rule in one place keeps a row read on demand and a stored pair from
    drifting apart.
    """
    inter_x1 = np.maximum(boxes[left, 0], boxes[right, 0])
    inter_y1 = np.maximum(boxes[left, 1], boxes[right, 1])
    inter_x2 = np.minimum(boxes[left, 2], boxes[right, 2])
    inter_y2 = np.minimum(boxes[left, 3], boxes[right, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    if match_metric == "IOU":
        denom = areas[left] + areas[right] - inter
    else:  # IOS
        denom = np.minimum(areas[left], areas[right])
    return _safe_ratio(inter, denom)


def _dominates(left: int | np.ndarray, right: np.ndarray, scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Whether each ``left`` box may claim the ``right`` box paired with it.

    A box claims another when the other scores lower, or scores equal and does not
    sort before it lexicographically by coordinates. ``left`` may be a single index,
    which broadcasts against ``right``. Same rule as the dense ``dominates`` matrix.
    """
    lower_score = scores[left] > scores[right]
    score_equal = scores[left] == scores[right]

    left_lt = np.zeros(len(right), dtype=bool)
    still_equal = np.ones(len(right), dtype=bool)
    for col in range(4):
        col_lt = boxes[left, col] < boxes[right, col]
        col_eq = boxes[left, col] == boxes[right, col]
        left_lt |= still_equal & col_lt
        still_equal &= col_eq

    return lower_score | (score_equal & ~left_lt)


def should_stream_nmm(boxes: np.ndarray) -> bool:
    """Return whether NMM should answer rows from the tree instead of storing pairs.

    Args:
        boxes: Array of shape (N, 4) with columns [x1, y1, x2, y2].

    Returns:
        True when the boxes intersect each other often enough that reading rows
        from the tree beats storing them.
    """
    return estimate_mean_degree(boxes) > NMM_MAX_DEGREE


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

    Every loop below settles each box exactly once, as a keeper or into a group,
    and never revisits that decision, so a settled box is dropped from later
    rows. It stays available as a query geometry for when its own turn arrives;
    it just cannot be returned as a candidate again. STRtree is immutable, so
    settled boxes are masked out and the tree is rebuilt once the live
    population has halved, which is O(N log N) of rebuilding over a whole run.

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
        self.active = np.ones(len(boxes), dtype=bool)
        self.active_count = len(boxes)
        self.tree_indices = np.arange(len(boxes), dtype=np.intp)
        # Until the first rebuild the tree holds every box, so its own indices
        # are the box indices and the remap in ``_candidate_indices`` is a no-op.
        self.tree_is_whole = True

    def deactivate(self, index: int) -> None:
        """Drop one settled box from every later row."""
        if self.active[index]:
            self.active[index] = False
            self.active_count -= 1

    def deactivate_row(self, indices: np.ndarray) -> None:
        """Drop a whole settled row. Its entries are active by construction."""
        if len(indices):
            self.active[indices] = False
            self.active_count -= len(indices)

    def row(self, i: int) -> np.ndarray:
        """Return the unsettled matches of box ``i``, ascending, excluding ``i``.

        With nothing deactivated yet this is one row of the CSR
        ``build_sparse_matches`` would build.
        """
        candidates = self._candidate_indices(i)
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            return candidates.astype(np.intp)

        metric = _overlap_metric(self.boxes, self.areas, i, candidates, self.match_metric)
        return np.sort(candidates[metric >= self.match_threshold]).astype(np.intp)

    def _candidate_indices(self, i: int) -> np.ndarray:
        """Return the active box indices whose envelopes intersect box ``i``."""
        if self.active_count == 0:
            return np.empty(0, dtype=np.intp)

        if self.active_count * 2 <= len(self.tree_indices):
            self.tree_indices = np.flatnonzero(self.active)
            self.tree = STRtree(self.geoms[self.tree_indices])
            self.tree_is_whole = False

        # Every indexed geometry is an axis-aligned rectangle, so it is its own
        # envelope and the tree's envelope query is already exact. Asking GEOS
        # for "intersects" as well would repeat that test for every pair, and
        # the metric above is the real filter regardless.
        candidates = self.tree.query(self.geoms[i])
        if not self.tree_is_whole:
            candidates = self.tree_indices[candidates]
        return candidates[self.active[candidates]]


def nms_streaming(
    boxes: np.ndarray,
    areas: np.ndarray,
    match_metric: str,
    match_threshold: float,
    sorted_idxs: np.ndarray,
) -> list[int]:
    """NMS reading match rows on demand. Same result as ``nms_from_matrix``.

    Only boxes that survive are ever queried, and suppressed boxes leave the
    query, so on crowded inputs this touches a small fraction of the pairs the
    CSR would have stored.

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
    keep: list[int] = []

    for idx in sorted_idxs:
        if matches.active_count == 0:
            break

        current_idx = int(idx)
        if not matches.active[current_idx]:
            continue
        keep.append(current_idx)

        # A survivor can never be suppressed afterwards. The metrics are
        # symmetric, so any box able to suppress it was suppressed by it first.
        matches.deactivate(current_idx)
        matches.deactivate_row(matches.row(current_idx))

    return keep


def greedy_nmm_streaming(
    boxes: np.ndarray,
    areas: np.ndarray,
    match_metric: str,
    match_threshold: float,
    sorted_idxs: np.ndarray,
) -> dict[int, list[int]]:
    """Greedy NMM reading match rows on demand. Same result as ``greedy_nmm_from_matrix``.

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

    # The dense loop only considers candidates that come later in score order,
    # and emits them in that order. A row cannot return an earlier one, since
    # every box before this position has been settled and left the query, so
    # only that order is left to reproduce.
    rank = np.empty(n, dtype=np.intp)
    rank[sorted_idxs] = np.arange(n)

    keep_to_merge_list: dict[int, list[int]] = {}
    for idx in sorted_idxs:
        if matches.active_count == 0:
            break

        current_idx = int(idx)
        if not matches.active[current_idx]:
            continue

        matches.deactivate(current_idx)
        merge_indices = matches.row(current_idx)
        merge_indices = merge_indices[np.argsort(rank[merge_indices])]
        matches.deactivate_row(merge_indices)
        keep_to_merge_list[current_idx] = merge_indices.tolist()

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
    # The indexed geometries are axis-aligned rectangles, so their envelopes
    # intersect exactly when the rectangles do. Avoid a duplicate GEOS predicate
    # evaluation for every candidate pair.
    matches = sum(
        len(tree.query(probe[start : start + DEGREE_PROBE_CHUNK])[0]) for start in range(0, k, DEGREE_PROBE_CHUNK)
    )
    return max(0.0, matches / k - 1.0)


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
    rows, cols = tree.query(geoms)

    self_pair = rows == cols
    if self_pair.any():
        rows, cols = rows[~self_pair], cols[~self_pair]

    matched = _overlap_metric(boxes, areas, rows, cols, match_metric) >= match_threshold
    rows, cols = rows[matched], cols[matched]

    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]

    indptr = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols.astype(np.intp)


def _dominates_all(indptr: np.ndarray, indices: np.ndarray, scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return, for every CSR entry, whether its row may claim its column.

    Evaluated for all stored pairs at once so the merge loop does no per-row
    numpy work.

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
    return _dominates(rows, indices, scores, boxes)


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
            keep_idx = current_idx
            merge_list: list[int] = []
            keep_to_merge_list[keep_idx] = merge_list
        else:
            keep_idx = int(merge_to_keep[current_idx])
            merge_list = keep_to_merge_list[keep_idx]

        # A claimed box always has a non-negative merge_to_keep entry, so that
        # test alone decides membership; scanning merge_list would repeat it.
        for m in matched:
            m_int = int(m)
            if merge_to_keep[m_int] < 0:
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

    Unlike NMS and greedy NMM this reads a row for every box, not just for the
    survivors, since a claimed box still propagates its keeper's label. Only
    unclaimed boxes can change groups, so claimed ones leave the query and the
    run stops early once every box has been assigned.

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
        if matches.active_count == 0:
            break

        current_idx = int(sorted_idxs[idx_pos])

        if merge_to_keep[current_idx] < 0:
            # current_idx is a keeper. Point it at itself so that a later box
            # cannot claim it: keepers are never merged into anything.
            merge_to_keep[current_idx] = current_idx
            keep_idx = current_idx
            merge_list: list[int] = []
            keep_to_merge_list[keep_idx] = merge_list
        else:
            keep_idx = int(merge_to_keep[current_idx])
            merge_list = keep_to_merge_list[keep_idx]

        # The current box and every returned candidate are now assigned. They
        # stay valid query geometries for when their turns arrive, but cannot be
        # claimed by a second group, so they leave the candidate rows.
        matches.deactivate(current_idx)
        matched = _dominated_row(matches.row(current_idx), current_idx, scores, boxes)
        merge_list.extend(matched.tolist())
        merge_to_keep[matched] = keep_idx
        matches.deactivate_row(matched)

    return keep_to_merge_list


def _dominated_row(candidates: np.ndarray, i: int, scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Restrict one match row to the columns its row may claim.

    Single-row form of ``_dominates_all``.
    """
    if len(candidates) == 0:
        return candidates
    return candidates[_dominates(i, candidates, scores, boxes)]
