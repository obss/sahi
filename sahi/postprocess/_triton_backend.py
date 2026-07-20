"""Triton postprocessing backend.

Only GreedyNMM + IOS has a Triton path for now. Other operations fall back to
the numpy backend to keep the backend interface complete while the accelerated
path remains narrow and explicit.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from sahi.postprocess._numpy_backend import (
    _score_tiebreak_order,
    greedy_nmm_from_packed_mask,
    greedy_nmm_numpy,
    nmm_numpy,
    nms_numpy,
)
from sahi.utils.torch_utils import select_device

_WORD_BITS = 32
_BLOCK_ROWS = 16


def _get_cuda_device() -> str | None:
    """Return the selected CUDA device, or None when CUDA is unavailable."""
    device = select_device()
    if getattr(device, "type", None) != "cuda":
        return None
    return str(device)


@triton.jit
def _ios_match_bitset_kernel(
    boxes_ptr,
    out_ptr,
    n: tl.constexpr,
    num_words: tl.constexpr,
    threshold,
    BLOCK_ROWS: tl.constexpr,
    WORD_BITS: tl.constexpr,
):
    row_block = tl.program_id(0)
    word_id = tl.program_id(1)

    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    bits = tl.arange(0, WORD_BITS)
    cols = word_id * WORD_BITS + bits

    row_mask = rows < n
    col_mask = cols < n

    row_x1 = tl.load(boxes_ptr + rows * 4 + 0, mask=row_mask, other=0.0)[:, None]
    row_y1 = tl.load(boxes_ptr + rows * 4 + 1, mask=row_mask, other=0.0)[:, None]
    row_x2 = tl.load(boxes_ptr + rows * 4 + 2, mask=row_mask, other=0.0)[:, None]
    row_y2 = tl.load(boxes_ptr + rows * 4 + 3, mask=row_mask, other=0.0)[:, None]

    col_x1 = tl.load(boxes_ptr + cols * 4 + 0, mask=col_mask, other=0.0)[None, :]
    col_y1 = tl.load(boxes_ptr + cols * 4 + 1, mask=col_mask, other=0.0)[None, :]
    col_x2 = tl.load(boxes_ptr + cols * 4 + 2, mask=col_mask, other=0.0)[None, :]
    col_y2 = tl.load(boxes_ptr + cols * 4 + 3, mask=col_mask, other=0.0)[None, :]

    inter_x1 = tl.maximum(row_x1, col_x1)
    inter_y1 = tl.maximum(row_y1, col_y1)
    inter_x2 = tl.minimum(row_x2, col_x2)
    inter_y2 = tl.minimum(row_y2, col_y2)
    inter_w = tl.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tl.maximum(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h

    row_area = tl.maximum(row_x2 - row_x1, 0.0) * tl.maximum(row_y2 - row_y1, 0.0)
    col_area = tl.maximum(col_x2 - col_x1, 0.0) * tl.maximum(col_y2 - col_y1, 0.0)
    smaller = tl.minimum(row_area, col_area)

    valid = (rows[:, None] < n) & (cols[None, :] < n) & (smaller > 0.0)
    matches = valid & (inter >= threshold * smaller)

    bit_values = tl.full((WORD_BITS,), 1, tl.uint32) << bits
    packed_values = tl.where(matches, bit_values[None, :], tl.zeros((BLOCK_ROWS, WORD_BITS), dtype=tl.uint32))
    packed = tl.sum(packed_values, axis=1)

    tl.store(out_ptr + rows * num_words + word_id, packed, mask=row_mask)


def _compute_ios_match_bitset_triton(predictions: np.ndarray, match_threshold: float) -> np.ndarray:
    """Compute packed IOS threshold matches with Triton."""
    device = _get_cuda_device()
    if device is None:
        raise RuntimeError("Triton GreedyNMM requires a CUDA device")

    n = len(predictions)
    num_words = triton.cdiv(n, _WORD_BITS)
    boxes_np = np.ascontiguousarray(predictions[:, :4], dtype=np.float32)
    boxes = torch.tensor(boxes_np, dtype=torch.float32, device=device)
    packed = torch.empty((n, num_words), dtype=torch.int32, device=device)

    grid = (triton.cdiv(n, _BLOCK_ROWS), num_words)
    _ios_match_bitset_kernel[grid](
        boxes,
        packed,
        n,
        num_words,
        float(match_threshold),
        BLOCK_ROWS=_BLOCK_ROWS,
        WORD_BITS=_WORD_BITS,
    )

    return np.ascontiguousarray(packed.cpu().numpy().view(np.uint32))


def nms_triton(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
    return nms_numpy(predictions, match_metric, match_threshold)


def greedy_nmm_triton(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    if len(predictions) == 0:
        return {}

    if match_metric != "IOS":
        return greedy_nmm_numpy(predictions, match_metric, match_threshold)

    if _get_cuda_device() is None:
        return greedy_nmm_numpy(predictions, match_metric, match_threshold)

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    sorted_idxs = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)
    match_bitset = _compute_ios_match_bitset_triton(predictions, match_threshold)
    return greedy_nmm_from_packed_mask(match_bitset, sorted_idxs)


def nmm_triton(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    return nmm_numpy(predictions, match_metric, match_threshold)
