import numpy as np
import pytest

from sahi.postprocess.combine import (
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


# --- numpy tests (always run) ---


def test_nms_basic():
    preds = np.array(PREDS_NMS, dtype=np.float32)
    keep = nms(preds, match_metric="IOU", match_threshold=0.5)
    assert 0 in keep and 2 in keep
    assert 1 not in keep


def test_batched_nms_class_aware():
    preds = np.array(PREDS_BATCHED_NMS, dtype=np.float32)
    keep = batched_nms(preds, match_metric="IOU", match_threshold=0.5)
    assert 0 in keep
    assert 1 in keep
    assert 2 not in keep


def test_nmm_merge_mapping():
    preds = np.array(PREDS_NMM, dtype=np.float32)
    keep_to_merge = nmm(preds, match_metric="IOU", match_threshold=0.1)
    assert isinstance(keep_to_merge, dict)
    assert 0 in keep_to_merge
    merged = set(keep_to_merge[0])
    assert 1 in merged and 2 in merged


def test_greedy_nmm_and_batched():
    preds = np.array(PREDS_NMM, dtype=np.float32)
    greedy_map = greedy_nmm(preds, match_metric="IOU", match_threshold=0.1)
    assert 0 in greedy_map
    assert set(greedy_map[0]) == {1, 2}

    batched = batched_greedy_nmm(preds, match_metric="IOU", match_threshold=0.1)
    assert any(len(v) > 0 for v in batched.values())


def test_batched_nmm_class_aware():
    preds = np.array(PREDS_BATCHED_NMM, dtype=np.float32)
    keep_to_merge = batched_nmm(preds, match_metric="IOU", match_threshold=0.1)
    assert 0 in keep_to_merge or 1 in keep_to_merge or 2 in keep_to_merge
    merged_any = any(2 in v for v in keep_to_merge.values())
    assert merged_any


# --- torch tensor tests (skipped when torch is not installed) ---

torch = pytest.importorskip("torch")


def _to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)


def test_nms_basic_torch():
    """Torch tensors are auto-converted via .numpy() by callers; verify numpy() output matches."""
    preds_torch = _to_tensor(PREDS_NMS)
    preds_np = preds_torch.numpy()
    keep = nms(preds_np, match_metric="IOU", match_threshold=0.5)
    assert 0 in keep and 2 in keep
    assert 1 not in keep


def test_batched_nms_class_aware_torch():
    preds_torch = _to_tensor(PREDS_BATCHED_NMS)
    preds_np = preds_torch.numpy()
    keep = batched_nms(preds_np, match_metric="IOU", match_threshold=0.5)
    assert 0 in keep
    assert 1 in keep
    assert 2 not in keep


def test_nmm_merge_mapping_torch():
    preds_torch = _to_tensor(PREDS_NMM)
    preds_np = preds_torch.numpy()
    keep_to_merge = nmm(preds_np, match_metric="IOU", match_threshold=0.1)
    assert isinstance(keep_to_merge, dict)
    assert 0 in keep_to_merge
    merged = set(keep_to_merge[0])
    assert 1 in merged and 2 in merged


def test_greedy_nmm_and_batched_torch():
    preds_torch = _to_tensor(PREDS_NMM)
    preds_np = preds_torch.numpy()

    greedy_map = greedy_nmm(preds_np, match_metric="IOU", match_threshold=0.1)
    assert 0 in greedy_map
    assert set(greedy_map[0]) == {1, 2}

    batched = batched_greedy_nmm(preds_np, match_metric="IOU", match_threshold=0.1)
    assert any(len(v) > 0 for v in batched.values())


def test_batched_nmm_class_aware_torch():
    preds_torch = _to_tensor(PREDS_BATCHED_NMM)
    preds_np = preds_torch.numpy()
    keep_to_merge = batched_nmm(preds_np, match_metric="IOU", match_threshold=0.1)
    assert 0 in keep_to_merge or 1 in keep_to_merge or 2 in keep_to_merge
    merged_any = any(2 in v for v in keep_to_merge.values())
    assert merged_any


def test_numpy_torch_parity():
    """Verify numpy and torch-derived inputs produce identical results across all functions."""
    for preds_data, func, kwargs in [
        (PREDS_NMS, nms, {"match_metric": "IOU", "match_threshold": 0.5}),
        (PREDS_BATCHED_NMS, batched_nms, {"match_metric": "IOU", "match_threshold": 0.5}),
        (PREDS_NMM, nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
        (PREDS_NMM, greedy_nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
        (PREDS_NMM, batched_greedy_nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
        (PREDS_BATCHED_NMM, batched_nmm, {"match_metric": "IOU", "match_threshold": 0.1}),
    ]:
        np_result = func(np.array(preds_data, dtype=np.float32), **kwargs)
        torch_result = func(_to_tensor(preds_data).numpy(), **kwargs)
        assert np_result == torch_result, f"{func.__name__}: numpy vs torch mismatch"
