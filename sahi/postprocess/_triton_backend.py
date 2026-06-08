import torch
import numpy as np

import triton
import triton.language as tl

def nms_triton(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
  pass

def greedy_nmm_torchvision(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    pass


def nmm_torchvision(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    pass