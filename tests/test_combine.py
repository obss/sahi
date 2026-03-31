import numpy as np
import pytest

from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    batched_greedy_nmm,
    batched_nmm,
    batched_nms,
    greedy_nmm,
    nmm,
    nms,
)


def make_pred(x1, y1, x2, y2, score, cid):
    return [x1, y1, x2, y2, score, cid]


PREDS_NMS = [
    make_pred(0, 0, 10, 10, 0.9, 1),
    make_pred(1, 1, 9, 9, 0.8, 1),
    make_pred(100, 100, 110, 110, 0.7, 1),
]

PREDS_BATCHED_NMS = [
    make_pred(0, 0, 10, 10, 0.9, 1),
    make_pred(1, 1, 9, 9, 0.8, 2),
    make_pred(0, 0, 10, 10, 0.85, 1),
]

PREDS_NMM = [
    make_pred(0, 0, 20, 20, 0.95, 1),
    make_pred(2, 2, 10, 10, 0.5, 1),
    make_pred(5, 5, 15, 15, 0.4, 1),
]

PREDS_BATCHED_NMM = [
    make_pred(0, 0, 20, 20, 0.95, 1),
    make_pred(2, 2, 10, 10, 0.5, 2),
    make_pred(3, 3, 11, 11, 0.4, 2),
]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_nms_empty(self):
        preds = np.array([], dtype=np.float32).reshape(0, 6)
        assert nms(preds) == []

    def test_nms_single(self):
        preds = np.array([make_pred(0, 0, 10, 10, 0.9, 1)], dtype=np.float32)
        assert nms(preds) == [0]

    def test_greedy_nmm_empty(self):
        preds = np.array([], dtype=np.float32).reshape(0, 6)
        assert greedy_nmm(preds) == {}

    def test_greedy_nmm_single(self):
        preds = np.array([make_pred(0, 0, 10, 10, 0.9, 1)], dtype=np.float32)
        result = greedy_nmm(preds)
        assert result == {0: []}

    def test_nmm_single(self):
        preds = np.array([make_pred(0, 0, 10, 10, 0.9, 1)], dtype=np.float32)
        result = nmm(preds)
        assert result == {0: []}

    def test_nms_identical_boxes(self):
        preds = np.array([
            make_pred(0, 0, 10, 10, 0.9, 1),
            make_pred(0, 0, 10, 10, 0.8, 1),
        ], dtype=np.float32)
        keep = nms(preds, match_threshold=0.5)
        assert len(keep) == 1
        assert 0 in keep

    def test_nms_non_overlapping(self):
        preds = np.array([
            make_pred(0, 0, 10, 10, 0.9, 1),
            make_pred(50, 50, 60, 60, 0.8, 1),
            make_pred(100, 100, 110, 110, 0.7, 1),
        ], dtype=np.float32)
        keep = nms(preds, match_threshold=0.5)
        assert len(keep) == 3  # all kept — no overlap

    def test_nms_equal_scores_deterministic(self):
        preds = np.array([
            make_pred(0, 0, 10, 10, 0.9, 1),
            make_pred(1, 1, 11, 11, 0.9, 1),
        ], dtype=np.float32)
        keep1 = nms(preds, match_threshold=0.5)
        keep2 = nms(preds, match_threshold=0.5)
        assert keep1 == keep2  # deterministic tie-breaking


# ===========================================================================
# IOS metric tests
# ===========================================================================


class TestIOSMetric:
    def test_nms_ios_small_inside_large(self):
        """Small box fully inside large box: IOS=1.0 (suppressed), IOU < 1.0."""
        preds = np.array([
            make_pred(0, 0, 100, 100, 0.9, 1),  # large box
            make_pred(10, 10, 20, 20, 0.8, 1),   # small box fully inside
        ], dtype=np.float32)
        keep_ios = nms(preds, match_metric="IOS", match_threshold=0.5)
        assert len(keep_ios) == 1  # small box suppressed (IOS = 1.0)

    def test_nms_iou_vs_ios_differ(self):
        """IOU and IOS give different results for asymmetric overlap."""
        preds = np.array([
            make_pred(0, 0, 100, 100, 0.9, 1),
            make_pred(10, 10, 20, 20, 0.8, 1),
        ], dtype=np.float32)
        keep_iou = nms(preds, match_metric="IOU", match_threshold=0.5)
        keep_ios = nms(preds, match_metric="IOS", match_threshold=0.5)
        # IOU of small inside large is ~1% (100/10000+100-100), below threshold
        # IOS is 100% (small fully inside large)
        assert len(keep_iou) == 2  # both kept with IOU
        assert len(keep_ios) == 1  # small suppressed with IOS

    def test_greedy_nmm_ios(self):
        preds = np.array([
            make_pred(0, 0, 100, 100, 0.9, 1),
            make_pred(10, 10, 20, 20, 0.8, 1),
        ], dtype=np.float32)
        result = greedy_nmm(preds, match_metric="IOS", match_threshold=0.5)
        assert 0 in result
        assert 1 in result[0]

    def test_nmm_ios(self):
        preds = np.array([
            make_pred(0, 0, 100, 100, 0.9, 1),
            make_pred(10, 10, 20, 20, 0.8, 1),
        ], dtype=np.float32)
        result = nmm(preds, match_metric="IOS", match_threshold=0.5)
        assert 0 in result
        assert 1 in result[0]

    def test_batched_nms_ios(self):
        preds = np.array([
            make_pred(0, 0, 100, 100, 0.9, 1),
            make_pred(10, 10, 20, 20, 0.8, 1),
            make_pred(0, 0, 100, 100, 0.7, 2),
            make_pred(10, 10, 20, 20, 0.6, 2),
        ], dtype=np.float32)
        keep = batched_nms(preds, match_metric="IOS", match_threshold=0.5)
        assert len(keep) == 2  # one per category


# ===========================================================================
# Core NMS/NMM tests
# ===========================================================================


class TestNMS:
    def test_basic(self):
        preds = np.array(PREDS_NMS, dtype=np.float32)
        keep = nms(preds, match_metric="IOU", match_threshold=0.5)
        assert 0 in keep and 2 in keep
        assert 1 not in keep

    def test_batched_class_aware(self):
        preds = np.array(PREDS_BATCHED_NMS, dtype=np.float32)
        keep = batched_nms(preds, match_metric="IOU", match_threshold=0.5)
        assert 0 in keep and 1 in keep
        assert 2 not in keep

    def test_high_threshold_keeps_all(self):
        preds = np.array(PREDS_NMS, dtype=np.float32)
        keep = nms(preds, match_metric="IOU", match_threshold=0.99)
        assert len(keep) == 3  # threshold too high, nothing suppressed


class TestNMM:
    def test_merge_mapping(self):
        preds = np.array(PREDS_NMM, dtype=np.float32)
        result = nmm(preds, match_metric="IOU", match_threshold=0.1)
        assert 0 in result
        assert 1 in result[0] and 2 in result[0]

    def test_batched_class_aware(self):
        preds = np.array(PREDS_BATCHED_NMM, dtype=np.float32)
        result = batched_nmm(preds, match_metric="IOU", match_threshold=0.1)
        merged_any = any(2 in v for v in result.values())
        assert merged_any


class TestGreedyNMM:
    def test_basic(self):
        preds = np.array(PREDS_NMM, dtype=np.float32)
        result = greedy_nmm(preds, match_metric="IOU", match_threshold=0.1)
        assert 0 in result
        assert set(result[0]) == {1, 2}

    def test_batched(self):
        preds = np.array(PREDS_NMM, dtype=np.float32)
        result = batched_greedy_nmm(preds, match_metric="IOU", match_threshold=0.1)
        assert any(len(v) > 0 for v in result.values())


# ===========================================================================
# PostprocessPredictions classes
# ===========================================================================


def _make_object_predictions(preds_data):
    """Create ObjectPrediction objects for testing postprocess classes."""
    from sahi.prediction import ObjectPrediction

    predictions = []
    for p in preds_data:
        predictions.append(
            ObjectPrediction(
                bbox=p[:4],
                score=p[4],
                category_id=int(p[5]),
                category_name=str(int(p[5])),
            )
        )
    return predictions


class TestPostprocessClasses:
    def test_nms_postprocess(self):
        obj_preds = _make_object_predictions(PREDS_NMS)
        pp = NMSPostprocess(match_threshold=0.5, match_metric="IOU")
        result = pp(obj_preds)
        assert len(result) == 2  # 3 input → 2 after suppression

    def test_nms_postprocess_class_agnostic_false(self):
        obj_preds = _make_object_predictions(PREDS_BATCHED_NMS)
        pp = NMSPostprocess(match_threshold=0.5, match_metric="IOU", class_agnostic=False)
        result = pp(obj_preds)
        assert len(result) == 2  # class 1: keep 1, class 2: keep 1

    def test_nmm_postprocess(self):
        obj_preds = _make_object_predictions(PREDS_NMM)
        pp = NMMPostprocess(match_threshold=0.1, match_metric="IOU")
        result = pp(obj_preds)
        assert len(result) >= 1  # at least the keeper survives

    def test_greedy_nmm_postprocess(self):
        obj_preds = _make_object_predictions(PREDS_NMM)
        pp = GreedyNMMPostprocess(match_threshold=0.1, match_metric="IOU")
        result = pp(obj_preds)
        assert len(result) >= 1

    def test_nms_postprocess_single(self):
        obj_preds = _make_object_predictions([make_pred(0, 0, 10, 10, 0.9, 1)])
        pp = NMSPostprocess(match_threshold=0.5)
        result = pp(obj_preds)
        assert len(result) == 1


# ===========================================================================
# Backend registry tests
# ===========================================================================


class TestBackendRegistry:
    def test_set_and_get(self):
        from sahi.postprocess.backends import get_postprocess_backend, set_postprocess_backend

        original = get_postprocess_backend()
        try:
            for name in ("numpy", "numba", "torchvision", "auto"):
                set_postprocess_backend(name)
                assert get_postprocess_backend() == name
        finally:
            set_postprocess_backend(original)

    def test_invalid_backend_raises(self):
        from sahi.postprocess.backends import set_postprocess_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            set_postprocess_backend("invalid_backend")

    def test_resolve_cache_invalidation(self):
        from sahi.postprocess.backends import _resolved_cache, resolve_backend, set_postprocess_backend

        original = get_postprocess_backend()
        try:
            set_postprocess_backend("numpy")
            assert resolve_backend() == "numpy"
            set_postprocess_backend("auto")
            # After reset, cache should be cleared and re-resolved
            result = resolve_backend()
            assert result in ("numpy", "numba", "torchvision")
        finally:
            set_postprocess_backend(original)

    def test_forced_numpy_backend_dispatch(self):
        """Force numpy backend and verify NMS dispatches correctly."""
        from sahi.postprocess.backends import set_postprocess_backend

        original = get_postprocess_backend()
        try:
            set_postprocess_backend("numpy")
            preds = np.array(PREDS_NMS, dtype=np.float32)
            keep = nms(preds, match_metric="IOU", match_threshold=0.5)
            assert 0 in keep and 2 in keep
        finally:
            set_postprocess_backend(original)


def get_postprocess_backend():
    from sahi.postprocess.backends import get_postprocess_backend

    return get_postprocess_backend()


# ===========================================================================
# Numpy backend direct tests
# ===========================================================================


class TestNumpyBackend:
    def test_nms(self):
        from sahi.postprocess._numpy_backend import nms_numpy

        preds = np.array(PREDS_NMS, dtype=np.float32)
        keep = nms_numpy(preds, match_metric="IOU", match_threshold=0.5)
        assert 0 in keep and 2 in keep
        assert 1 not in keep

    def test_greedy_nmm(self):
        from sahi.postprocess._numpy_backend import greedy_nmm_numpy

        preds = np.array(PREDS_NMM, dtype=np.float32)
        result = greedy_nmm_numpy(preds, match_metric="IOU", match_threshold=0.1)
        assert 0 in result
        assert set(result[0]) == {1, 2}

    def test_nmm(self):
        from sahi.postprocess._numpy_backend import nmm_numpy

        preds = np.array(PREDS_NMM, dtype=np.float32)
        result = nmm_numpy(preds, match_metric="IOU", match_threshold=0.1)
        assert 0 in result
        assert 1 in result[0] and 2 in result[0]

    def test_metric_matrix_iou(self):
        from sahi.postprocess._numpy_backend import compute_metric_matrix

        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        areas = np.array([100, 100], dtype=np.float32)
        matrix = compute_metric_matrix(boxes, areas, "IOU")
        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == pytest.approx(1.0, abs=1e-5)  # identical boxes
        assert matrix[0, 0] == pytest.approx(1.0, abs=1e-5)  # self

    def test_metric_matrix_ios(self):
        from sahi.postprocess._numpy_backend import compute_metric_matrix

        boxes = np.array([[0, 0, 100, 100], [10, 10, 20, 20]], dtype=np.float32)
        areas = np.array([10000, 100], dtype=np.float32)
        matrix = compute_metric_matrix(boxes, areas, "IOS")
        # Small box fully inside large: IOS = 100/100 = 1.0
        assert matrix[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_metric_matrix_no_overlap(self):
        from sahi.postprocess._numpy_backend import compute_metric_matrix

        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
        areas = np.array([100, 100], dtype=np.float32)
        matrix = compute_metric_matrix(boxes, areas, "IOU")
        assert matrix[0, 1] == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# Numba backend parity tests
# ===========================================================================


class TestNumbaBackend:
    @pytest.fixture(autouse=True)
    def _skip_no_numba(self):
        pytest.importorskip("numba")

    def test_nms_parity(self):
        from sahi.postprocess._numba_backend import nms_numba
        from sahi.postprocess._numpy_backend import nms_numpy

        preds = np.array(PREDS_NMS, dtype=np.float32)
        assert nms_numpy(preds, "IOU", 0.5) == nms_numba(preds, "IOU", 0.5)

    def test_greedy_nmm_parity(self):
        from sahi.postprocess._numba_backend import greedy_nmm_numba
        from sahi.postprocess._numpy_backend import greedy_nmm_numpy

        preds = np.array(PREDS_NMM, dtype=np.float32)
        assert greedy_nmm_numpy(preds, "IOU", 0.1) == greedy_nmm_numba(preds, "IOU", 0.1)

    def test_nmm_parity(self):
        from sahi.postprocess._numba_backend import nmm_numba
        from sahi.postprocess._numpy_backend import nmm_numpy

        preds = np.array(PREDS_NMM, dtype=np.float32)
        assert nmm_numpy(preds, "IOU", 0.1) == nmm_numba(preds, "IOU", 0.1)

    def test_nms_ios_parity(self):
        from sahi.postprocess._numba_backend import nms_numba
        from sahi.postprocess._numpy_backend import nms_numpy

        preds = np.array([
            make_pred(0, 0, 100, 100, 0.9, 1),
            make_pred(10, 10, 20, 20, 0.8, 1),
        ], dtype=np.float32)
        assert nms_numpy(preds, "IOS", 0.5) == nms_numba(preds, "IOS", 0.5)

    def test_random_parity(self):
        """Verify numpy and numba agree on random data."""
        from sahi.postprocess._numba_backend import greedy_nmm_numba, nms_numba
        from sahi.postprocess._numpy_backend import greedy_nmm_numpy, nms_numpy

        rng = np.random.RandomState(42)
        n = 50
        x1 = rng.uniform(0, 100, n).astype(np.float32)
        y1 = rng.uniform(0, 100, n).astype(np.float32)
        x2 = x1 + rng.uniform(5, 30, n).astype(np.float32)
        y2 = y1 + rng.uniform(5, 30, n).astype(np.float32)
        scores = rng.uniform(0.1, 1.0, n).astype(np.float32)
        cats = rng.choice([0, 1], n).astype(np.float32)
        preds = np.stack([x1, y1, x2, y2, scores, cats], axis=1)

        assert nms_numpy(preds, "IOU", 0.5) == nms_numba(preds, "IOU", 0.5)
        assert greedy_nmm_numpy(preds, "IOU", 0.3) == greedy_nmm_numba(preds, "IOU", 0.3)


# ===========================================================================
# Torch parity tests (skipped without torch)
# ===========================================================================


class TestTorchParity:
    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def _to_tensor(self, data):
        import torch

        return torch.tensor(data, dtype=torch.float32)

    def test_nms_torch_input(self):
        preds = self._to_tensor(PREDS_NMS).numpy()
        keep = nms(preds, match_metric="IOU", match_threshold=0.5)
        assert 0 in keep and 2 in keep
        assert 1 not in keep

    def test_numpy_torch_parity(self):
        for preds_data, func, kwargs in [
            (PREDS_NMS, nms, {"match_metric": "IOU", "match_threshold": 0.5}),
            (PREDS_BATCHED_NMS, batched_nms, {"match_metric": "IOU", "match_threshold": 0.5}),
            (PREDS_NMM, nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
            (PREDS_NMM, greedy_nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
            (PREDS_NMM, batched_greedy_nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
            (PREDS_BATCHED_NMM, batched_nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
        ]:
            np_result = func(np.array(preds_data, dtype=np.float32), **kwargs)
            torch_result = func(self._to_tensor(preds_data).numpy(), **kwargs)
            assert np_result == torch_result, f"{func.__name__}: mismatch"
