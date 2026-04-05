"""Torchvision GPU-accelerated postprocessing backend.

Uses torchvision.ops.nms for GPU NMS (IOU only).
For IOS and NMM, computes the IoU/IoS matrix on GPU via torch
and runs the merge logic on CPU via shared functions.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision

from sahi.postprocess._numpy_backend import (
    _score_tiebreak_order,
    greedy_nmm_from_matrix,
    nmm_from_matrix,
    nms_from_matrix,
)
from sahi.utils.torch_utils import select_device


def _get_device() -> str:
    """Get the best available torch device."""
    return str(select_device())


def _compute_metric_matrix_torch(predictions: np.ndarray, match_metric: str) -> np.ndarray:
    """Compute pairwise IoU/IoS matrix using torch on GPU, return as numpy."""
    device = _get_device()
    boxes = torch.tensor(predictions[:, :4], dtype=torch.float32, device=device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    inter_x1 = torch.maximum(x1[:, None], x1[None, :])
    inter_y1 = torch.maximum(y1[:, None], y1[None, :])
    inter_x2 = torch.minimum(x2[:, None], x2[None, :])
    inter_y2 = torch.minimum(y2[:, None], y2[None, :])
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    if match_metric == "IOU":
        union = areas[:, None] + areas[None, :] - inter
        matrix = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    else:  # IOS
        smaller = torch.minimum(areas[:, None], areas[None, :])
        matrix = torch.where(smaller > 0, inter / smaller, torch.zeros_like(inter))

    return matrix.cpu().numpy().astype(np.float32)


def _prepare_matrix_torch(predictions: np.ndarray, match_metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute metric matrix on GPU and sort indices. Shared prep for all functions."""
    matrix = _compute_metric_matrix_torch(predictions, match_metric)
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    sorted_idxs = _score_tiebreak_order(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores)
    return matrix, sorted_idxs


# ---------------------------------------------------------------------------
# Public torchvision backend functions
# ---------------------------------------------------------------------------


def nms_torchvision(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
    """NMS using torchvision.ops.nms (GPU-accelerated for IOU)."""
    if len(predictions) == 0:
        return []

    if match_metric == "IOU":
        # Use torchvision's native CUDA NMS kernel
        device = _get_device()
        boxes = torch.tensor(predictions[:, :4], dtype=torch.float32, device=device)
        scores = torch.tensor(predictions[:, 4], dtype=torch.float32, device=device)
        keep = torchvision.ops.nms(boxes, scores, match_threshold)
        return keep.cpu().tolist()

    # IOS: compute matrix on GPU, greedy loop on CPU
    matrix, sorted_idxs = _prepare_matrix_torch(predictions, match_metric)
    return nms_from_matrix(matrix, sorted_idxs, match_threshold)


def greedy_nmm_torchvision(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Greedy NMM: compute metric matrix on GPU, merge logic on CPU."""
    matrix, sorted_idxs = _prepare_matrix_torch(predictions, match_metric)
    return greedy_nmm_from_matrix(matrix, sorted_idxs, match_threshold)


def nmm_torchvision(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """NMM: compute metric matrix on GPU, transitive merge logic on CPU."""
    matrix, sorted_idxs = _prepare_matrix_torch(predictions, match_metric)
    return nmm_from_matrix(matrix, sorted_idxs, predictions[:, 4], predictions[:, :4], match_threshold)
