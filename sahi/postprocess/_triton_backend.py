from __future__ import annotations

import torch
import numpy as np

import triton
import triton.language as tl

from sahi.postprocess._numpy_backend import (
    greedy_nmm_numpy,
    nmm_numpy,
    nms_numpy,
)
from sahi.utils.torch_utils import select_device


def _get_device() -> str:
    """Get the best available torch device."""
    return str(select_device())


def _compute_metric_matrix_triton(predictions: np.ndarray, match_metric: str) -> np.ndarray:
    device = _get_device()
    boxes = torch.tensor(predictions[:, :4], dtype=torch.float32, device=device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    inter_x1 = torch.maximum(x1[:, None], x1[None, :])
    inter_y1 = torch.maximum(y1[:, None], y1[None, :])
    inter_x2 = torch.minimum(x2[:, None], x2[None, :])
    inter_y2 = torch.minimum(y2[:, None], y2[None, :])
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    


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
    if match_metric != "IOS":
        return greedy_nmm_numpy(predictions, match_metric, match_threshold)

    return greedy_nmm_triton(predictions, match_metric, match_threshold)


def nmm_triton(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    return nmm_numpy(predictions, match_metric, match_threshold)