"""Parity tests for the sparse postprocess path against the dense matrix path."""

from __future__ import annotations

import numpy as np
import pytest

from sahi.postprocess._numpy_backend import (
    _prepare_matrix,
    greedy_nmm_from_matrix,
    nmm_from_matrix,
    nms_from_matrix,
)
from sahi.postprocess._sparse_backend import (
    SPARSE_MIN_BOXES,
    build_sparse_matches,
    greedy_nmm_sparse,
    nmm_sparse,
    nms_sparse,
    should_use_sparse,
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


def test_should_use_sparse_thresholds() -> None:
    """Small inputs and non-positive thresholds stay on the dense path."""
    assert should_use_sparse(SPARSE_MIN_BOXES, 0.5) is True
    assert should_use_sparse(SPARSE_MIN_BOXES - 1, 0.5) is False
    # A zero threshold matches disjoint pairs too, which the prefilter skips.
    assert should_use_sparse(SPARSE_MIN_BOXES, 0.0) is False


@pytest.mark.parametrize("match_metric", ["IOU", "IOS"])
@pytest.mark.parametrize("match_threshold", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("spread", [200.0, 20000.0])
def test_sparse_adjacency_matches_dense(match_metric: str, match_threshold: float, spread: float) -> None:
    """The CSR adjacency equals the thresholded dense matrix, minus the diagonal."""
    predictions = _make_predictions(400, spread, seed=1)
    boxes = predictions[:, :4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    matrix, _ = _prepare_matrix(predictions, match_metric)
    indptr, indices = build_sparse_matches(boxes, areas, match_metric, match_threshold)

    expected = matrix >= match_threshold
    np.fill_diagonal(expected, False)

    for i in range(len(boxes)):
        got = indices[indptr[i] : indptr[i + 1]]
        assert sorted(got.tolist()) == got.tolist(), "CSR columns must be ascending"
        assert set(got.tolist()) == set(np.where(expected[i])[0].tolist())


@pytest.mark.parametrize("match_metric", ["IOU", "IOS"])
@pytest.mark.parametrize("match_threshold", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("spread", [200.0, 20000.0])
def test_sparse_algorithms_match_dense(match_metric: str, match_threshold: float, spread: float) -> None:
    """NMS, greedy NMM and NMM return identical results on both paths."""
    predictions = _make_predictions(400, spread, seed=2)
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    indptr, indices = build_sparse_matches(boxes, areas, match_metric, match_threshold)

    assert nms_sparse(indptr, indices, sorted_idxs) == nms_from_matrix(matrix, sorted_idxs, match_threshold)
    assert greedy_nmm_sparse(indptr, indices, sorted_idxs) == greedy_nmm_from_matrix(
        matrix, sorted_idxs, match_threshold
    )
    assert nmm_sparse(indptr, indices, sorted_idxs, scores, boxes) == nmm_from_matrix(
        matrix, sorted_idxs, scores, boxes, match_threshold
    )


def test_large_input_takes_sparse_path_and_matches_dense() -> None:
    """Above the cutoff the public numpy functions still agree with the dense path."""
    from sahi.postprocess._numpy_backend import greedy_nmm_numpy, nmm_numpy, nms_numpy

    n = SPARSE_MIN_BOXES + 500
    predictions = _make_predictions(n, spread=30000.0, seed=3)
    assert should_use_sparse(n, 0.3) is True

    matrix, sorted_idxs = _prepare_matrix(predictions, "IOS")
    boxes, scores = predictions[:, :4], predictions[:, 4]

    assert nms_numpy(predictions, "IOS", 0.3) == nms_from_matrix(matrix, sorted_idxs, 0.3)
    assert greedy_nmm_numpy(predictions, "IOS", 0.3) == greedy_nmm_from_matrix(matrix, sorted_idxs, 0.3)
    assert nmm_numpy(predictions, "IOS", 0.3) == nmm_from_matrix(matrix, sorted_idxs, scores, boxes, 0.3)
