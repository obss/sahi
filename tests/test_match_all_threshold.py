"""Tests for the degenerate ``match_threshold <= 0`` path.

Both metrics are non-negative, so such a threshold matches every pair including
disjoint ones. The dense path answered that by building an N x N matrix, which
made the one configuration that needs no metric at all the most expensive one,
and reintroduced the out-of-memory failure from issue #1374 at large N.
"""

from __future__ import annotations

import numpy as np
import pytest

from sahi.postprocess._numpy_backend import (
    _prepare_matrix,
    _score_tiebreak_order,
    greedy_nmm_from_matrix,
    greedy_nmm_numpy,
    matches_all_pairs,
    nmm_from_matrix,
    nmm_numpy,
    nms_from_matrix,
    nms_numpy,
)

BACKENDS: list[tuple[str, tuple]] = [("numpy", (nms_numpy, greedy_nmm_numpy, nmm_numpy))]

try:
    from sahi.postprocess._numba_backend import greedy_nmm_numba, nmm_numba, nms_numba

    BACKENDS.append(("numba", (nms_numba, greedy_nmm_numba, nmm_numba)))
except ImportError:  # pragma: no cover - numba is optional
    pass

try:
    from sahi.postprocess._torchvision_backend import (
        greedy_nmm_torchvision,
        nmm_torchvision,
        nms_torchvision,
    )

    BACKENDS.append(("torchvision", (nms_torchvision, greedy_nmm_torchvision, nmm_torchvision)))
except ImportError:  # pragma: no cover - torchvision is optional
    pass

BACKEND_IDS = [name for name, _ in BACKENDS]
BACKEND_FUNCS = [funcs for _, funcs in BACKENDS]


def _predictions(n: int, spread: float, seed: int, score_decimals: int | None = None) -> np.ndarray:
    """Build (N, 6) predictions; round the scores to force ties when asked."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, spread, n)
    y1 = rng.uniform(0, spread, n)
    w = rng.uniform(1, 40, n)
    h = rng.uniform(1, 40, n)
    scores = rng.uniform(0, 1, n)
    if score_decimals is not None:
        scores = np.round(scores, score_decimals)
    categories = rng.integers(0, 3, n)
    return np.stack([x1, y1, x1 + w, y1 + h, scores, categories], axis=1).astype(np.float32)


def _top_index(predictions: np.ndarray) -> int:
    boxes = predictions[:, :4]
    order = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], predictions[:, 4])
    return int(order[0])


def test_matches_all_pairs_boundary() -> None:
    """Only non-positive thresholds admit every pair."""
    assert matches_all_pairs(0.0) is True
    assert matches_all_pairs(-0.5) is True
    assert matches_all_pairs(1e-9) is False
    assert matches_all_pairs(0.5) is False


@pytest.mark.parametrize("funcs", BACKEND_FUNCS, ids=BACKEND_IDS)
@pytest.mark.parametrize("match_metric", ["IOU", "IOS"])
@pytest.mark.parametrize("match_threshold", [0.0, -0.5])
@pytest.mark.parametrize("spread", [200.0, 20000.0])
def test_matches_dense_path(funcs: tuple, match_metric: str, match_threshold: float, spread: float) -> None:
    """With a unique top score every backend reproduces the dense matrix result."""
    f_nms, f_greedy, f_nmm = funcs
    predictions = _predictions(120, spread, seed=1)
    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    boxes, scores = predictions[:, :4], predictions[:, 4]

    assert f_nms(predictions, match_metric, match_threshold) == nms_from_matrix(matrix, sorted_idxs, match_threshold)
    assert f_greedy(predictions, match_metric, match_threshold) == greedy_nmm_from_matrix(
        matrix, sorted_idxs, match_threshold
    )
    assert f_nmm(predictions, match_metric, match_threshold) == nmm_from_matrix(
        matrix, sorted_idxs, scores, boxes, match_threshold
    )


@pytest.mark.parametrize("funcs", BACKEND_FUNCS, ids=BACKEND_IDS)
@pytest.mark.parametrize("match_metric", ["IOU", "IOS"])
def test_collapses_to_single_keeper(funcs: tuple, match_metric: str) -> None:
    """Everything merges into the top-scoring box."""
    f_nms, f_greedy, f_nmm = funcs
    predictions = _predictions(120, 200.0, seed=2)
    boxes = predictions[:, :4]
    order = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], predictions[:, 4])
    top = int(order[0])
    rest = [i for i in range(len(predictions)) if i != top]

    assert f_nms(predictions, match_metric, 0.0) == [top]
    # greedy emits merged indices in score order, nmm in ascending index order
    assert f_greedy(predictions, match_metric, 0.0) == {top: [int(i) for i in order[1:]]}
    assert f_nmm(predictions, match_metric, 0.0) == {top: rest}


@pytest.mark.parametrize("funcs", BACKEND_FUNCS, ids=BACKEND_IDS)
@pytest.mark.parametrize("match_metric", ["IOU", "IOS"])
def test_tied_top_score_matches_dense_path(funcs: tuple, match_metric: str) -> None:
    """A tie for the top score is handled exactly as the dense loop handles it.

    Boxes sharing the top score are keepers with nothing merged into them, and
    no keeper ever appears inside a merge list.
    """
    f_nms, f_greedy, f_nmm = funcs
    predictions = _predictions(120, 200.0, seed=3, score_decimals=1)
    top_score = predictions[:, 4].max()
    assert (predictions[:, 4] == top_score).sum() > 1, "fixture must tie at the top"

    matrix, sorted_idxs = _prepare_matrix(predictions, match_metric)
    boxes, scores = predictions[:, :4], predictions[:, 4]

    assert f_nms(predictions, match_metric, 0.0) == nms_from_matrix(matrix, sorted_idxs, 0.0)
    assert f_greedy(predictions, match_metric, 0.0) == greedy_nmm_from_matrix(matrix, sorted_idxs, 0.0)

    result = f_nmm(predictions, match_metric, 0.0)
    assert result == nmm_from_matrix(matrix, sorted_idxs, scores, boxes, 0.0)
    for keeper, merged in result.items():
        assert keeper not in merged


@pytest.mark.parametrize("funcs", BACKEND_FUNCS, ids=BACKEND_IDS)
def test_empty_input(funcs: tuple) -> None:
    """Empty predictions stay empty."""
    f_nms, f_greedy, f_nmm = funcs
    empty = np.zeros((0, 6), dtype=np.float32)
    assert f_nms(empty, "IOU", 0.0) == []
    assert f_greedy(empty, "IOU", 0.0) == {}
    assert f_nmm(empty, "IOU", 0.0) == {}


@pytest.mark.parametrize("funcs", BACKEND_FUNCS, ids=BACKEND_IDS)
def test_large_input_allocates_no_matrix(funcs: tuple) -> None:
    """Regression guard for #1374.

    40000 boxes would need a 6.4 GiB float32 matrix, so reaching the dense path
    here fails with MemoryError or a CUDA OOM rather than returning.
    """
    f_nms, f_greedy, f_nmm = funcs
    predictions = _predictions(40000, 30000.0, seed=4)
    top = _top_index(predictions)

    assert f_nms(predictions, "IOS", 0.0) == [top]
    assert list(f_greedy(predictions, "IOS", 0.0)) == [top]
    assert list(f_nmm(predictions, "IOS", 0.0)) == [top]
