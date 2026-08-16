"""Parity tests for the sparse postprocess path against the dense matrix path."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

from sahi.postprocess._numpy_backend import (
    _prepare_matrix,
    _score_tiebreak_order,
    greedy_nmm_from_matrix,
    nmm_from_matrix,
    nmm_numpy,
    nms_from_matrix,
)
from sahi.postprocess._sparse_backend import (
    SPARSE_MIN_BOXES,
    MatchQuery,
    build_sparse_matches,
    greedy_nmm_sparse,
    greedy_nmm_streaming,
    nmm_sparse,
    nmm_streaming,
    nms_sparse,
    nms_streaming,
    should_stream_nmm,
    should_use_sparse,
)

# Every parity test runs the same metric x threshold x spread grid.
PARITY_GRID = pytest.mark.parametrize(
    ("match_metric", "match_threshold", "spread"),
    [(m, t, s) for m in ("IOU", "IOS") for t in (0.1, 0.5, 0.9) for s in (200.0, 20000.0)],
)


def _make_predictions(n: int, spread: float, seed: int) -> np.ndarray:
    """Build (N, 6) predictions; larger spread means fewer overlapping boxes."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, spread, n)
    y1 = rng.uniform(0, spread, n)
    w = rng.uniform(1, 40, n)
    h = rng.uniform(1, 40, n)
    # Rounded scores produce ties, exercising the coordinate tie-break.
    scores = np.round(rng.uniform(0, 1, n), 2)
    categories = rng.integers(0, 3, n)
    return np.stack([x1, y1, x1 + w, y1 + h, scores, categories], axis=1)


class _Case(NamedTuple):
    """Both representations of one fixture: dense matrix and CSR adjacency."""

    boxes: np.ndarray
    areas: np.ndarray
    scores: np.ndarray
    matrix: np.ndarray
    sorted_idxs: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray


def _case(predictions: np.ndarray, match_metric: str, match_threshold: float) -> _Case:
    """Set up what the parity tests compare. Builds the NxN matrix, so keep N small."""
    boxes, scores = predictions[:, :4], predictions[:, 4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    indptr, indices = build_sparse_matches(boxes, areas, match_metric, match_threshold)
    return _Case(boxes, areas, scores, matrix, sorted_idxs, indptr, indices)


def _assert_partition(result: dict[int, list[int]], n: int, label: str) -> None:
    """Every index is either a keeper or merged into exactly one keeper, never both."""
    merged_by: dict[int, int] = {}
    for keeper, merged in result.items():
        assert keeper not in merged, f"{label}: keeper {keeper} merged into itself"
        for m in merged:
            assert m not in merged_by, f"{label}: {m} merged into {merged_by[m]} and {keeper}"
            merged_by[m] = keeper
    assert not set(result) & set(merged_by), f"{label}: an index is both keeper and merged"
    assert set(result) | set(merged_by) == set(range(n)), f"{label}: some index is in no group"


def test_should_use_sparse_thresholds() -> None:
    """Small inputs and non-positive thresholds stay on the dense path."""
    assert should_use_sparse(SPARSE_MIN_BOXES, 0.5) is True
    assert should_use_sparse(SPARSE_MIN_BOXES - 1, 0.5) is False
    # A zero threshold matches disjoint pairs too, which the prefilter skips.
    assert should_use_sparse(SPARSE_MIN_BOXES, 0.0) is False


@PARITY_GRID
def test_sparse_adjacency_matches_dense(match_metric: str, match_threshold: float, spread: float) -> None:
    """The CSR adjacency equals the thresholded dense matrix, minus the diagonal."""
    case = _case(_make_predictions(400, spread, seed=1), match_metric, match_threshold)

    expected = case.matrix >= match_threshold
    np.fill_diagonal(expected, False)

    for i in range(len(case.boxes)):
        got = case.indices[case.indptr[i] : case.indptr[i + 1]]
        assert sorted(got.tolist()) == got.tolist(), "CSR columns must be ascending"
        assert set(got.tolist()) == set(np.where(expected[i])[0].tolist())


@PARITY_GRID
def test_dense_csr_and_streaming_agree(match_metric: str, match_threshold: float, spread: float) -> None:
    """NMS, greedy NMM and NMM return identical results on all three paths."""
    predictions = _make_predictions(400, spread, seed=2)
    case = _case(predictions, match_metric, match_threshold)
    streaming_args = (case.boxes, case.areas, match_metric, match_threshold, case.sorted_idxs)

    dense_nms = nms_from_matrix(case.matrix, case.sorted_idxs, match_threshold)
    assert nms_sparse(case.indptr, case.indices, case.sorted_idxs) == dense_nms
    assert nms_streaming(*streaming_args) == dense_nms

    dense_greedy = greedy_nmm_from_matrix(case.matrix, case.sorted_idxs, match_threshold)
    assert greedy_nmm_sparse(case.indptr, case.indices, case.sorted_idxs) == dense_greedy
    assert greedy_nmm_streaming(*streaming_args) == dense_greedy

    dense_nmm = nmm_from_matrix(case.matrix, case.sorted_idxs, case.scores, case.boxes, match_threshold)
    assert nmm_sparse(case.indptr, case.indices, case.sorted_idxs, case.scores, case.boxes) == dense_nmm
    assert nmm_streaming(*streaming_args, case.scores) == dense_nmm
    _assert_partition(dense_nmm, len(predictions), "nmm")


@PARITY_GRID
def test_match_query_row_equals_csr_row(match_metric: str, match_threshold: float, spread: float) -> None:
    """A row read on demand is the row the CSR builder would have stored."""
    case = _case(_make_predictions(300, spread, seed=4), match_metric, match_threshold)
    query = MatchQuery(case.boxes, case.areas, match_metric, match_threshold)

    for i in range(len(case.boxes)):
        assert query.row(i).tolist() == case.indices[case.indptr[i] : case.indptr[i + 1]].tolist()


def test_nmm_streams_only_when_pair_list_would_be_large() -> None:
    """Scattered boxes keep the stored-pair path; crowded ones switch to streaming."""
    scattered = _make_predictions(4000, spread=30000.0, seed=7)
    crowded = _make_predictions(4000, spread=80.0, seed=7)

    assert should_stream_nmm(scattered[:, :4]) is False
    assert should_stream_nmm(crowded[:, :4]) is True

    # Both routes must agree with the stored-pair result and still group every box.
    for name, predictions in (("scattered", scattered), ("crowded", crowded)):
        boxes, scores = predictions[:, :4], predictions[:, 4]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # The NxN matrix is skipped here; only the score order it returns is needed.
        sorted_idxs = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)
        indptr, indices = build_sparse_matches(boxes, areas, "IOS", 0.3)

        result = nmm_numpy(predictions, "IOS", 0.3)
        assert result == nmm_sparse(indptr, indices, sorted_idxs, scores, boxes)
        _assert_partition(result, len(predictions), name)


def test_crowded_input_stays_within_memory() -> None:
    """Regression guard for #1374 on layouts where storing every pair is not sparse.

    These boxes all sit on top of each other, so the pair count is close to
    N squared and the stored-pair path needs hundreds of MB for it.
    """
    import tracemalloc

    from sahi.postprocess._numpy_backend import greedy_nmm_numpy, nms_numpy

    predictions = _make_predictions(8000, spread=80.0, seed=6)
    boxes = predictions[:, :4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    indptr, _ = build_sparse_matches(boxes, areas, "IOS", 0.3)
    assert indptr[-1] > 5_000_000, "fixture must be crowded enough to matter"

    tracemalloc.start()
    nms_numpy(predictions, "IOS", 0.3)
    greedy_nmm_numpy(predictions, "IOS", 0.3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 32 * 2**20, f"peak {peak / 2**20:.0f} MB suggests the pair list was materialized"


def test_large_input_takes_sparse_path_and_matches_dense() -> None:
    """Above the cutoff the public numpy functions still agree with the dense path."""
    from sahi.postprocess._numpy_backend import greedy_nmm_numpy, nms_numpy

    n = SPARSE_MIN_BOXES + 500
    predictions = _make_predictions(n, spread=30000.0, seed=3)
    assert should_use_sparse(n, 0.3) is True
    case = _case(predictions, "IOS", 0.3)

    assert nms_numpy(predictions, "IOS", 0.3) == nms_from_matrix(case.matrix, case.sorted_idxs, 0.3)
    assert greedy_nmm_numpy(predictions, "IOS", 0.3) == greedy_nmm_from_matrix(case.matrix, case.sorted_idxs, 0.3)
    assert nmm_numpy(predictions, "IOS", 0.3) == nmm_from_matrix(
        case.matrix, case.sorted_idxs, case.scores, case.boxes, 0.3
    )


@pytest.mark.parametrize("match_threshold", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("match_metric", ["IOU", "IOS"])
def test_nmm_never_merges_a_keeper_into_itself(match_metric: str, match_threshold: float) -> None:
    """Tied scores on duplicate boxes used to let a later box claim an earlier keeper.

    A keeper is never merged into anything, no index lands in two merge lists,
    and every index is either a keeper or merged exactly once.
    """
    rng = np.random.default_rng(11)
    n = 80
    x1 = np.round(rng.uniform(0, 200, n) / 25) * 25
    y1 = np.round(rng.uniform(0, 200, n) / 25) * 25
    w = np.round(rng.uniform(1, 40, n) / 25) * 25
    h = np.round(rng.uniform(1, 40, n) / 25) * 25
    scores = np.round(rng.uniform(0, 1, n), 1)
    predictions = np.stack([x1, y1, x1 + w, y1 + h, scores, rng.integers(0, 3, n)], axis=1)
    case = _case(predictions, match_metric, match_threshold)

    results = {
        "dense": nmm_from_matrix(case.matrix, case.sorted_idxs, case.scores, case.boxes, match_threshold),
        "csr": nmm_sparse(case.indptr, case.indices, case.sorted_idxs, case.scores, case.boxes),
        "streaming": nmm_streaming(
            case.boxes, case.areas, match_metric, match_threshold, case.sorted_idxs, case.scores
        ),
    }
    assert results["dense"] == results["csr"] == results["streaming"]

    for name, result in results.items():
        _assert_partition(result, n, name)


def test_metric_does_not_warn_on_zero_area_boxes() -> None:
    """Degenerate boxes make the denominator zero; that must not divide by zero."""
    import warnings

    from sahi.postprocess._numpy_backend import compute_metric_matrix

    predictions = _make_predictions(300, spread=200.0, seed=12)
    predictions[::7, 2] = predictions[::7, 0]  # zero width
    predictions[::11, 3] = predictions[::11, 1]  # zero height
    boxes = predictions[:, :4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for metric in ("IOU", "IOS"):
            compute_metric_matrix(boxes, areas, metric)
            build_sparse_matches(boxes, areas, metric, 0.3)
            MatchQuery(boxes, areas, metric, 0.3).row(0)
