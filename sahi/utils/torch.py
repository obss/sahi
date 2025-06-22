# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.


import os
from typing import Any, Optional, Union

import numpy as np
from PIL.Image import Image

import re

try:
    import torch
    from torch import Tensor, device

    has_torch_cuda = torch.cuda.is_available()
    try:
        has_torch_mps: bool = torch.backends.mps.is_available()  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        has_torch_mps = False
    has_torch = True
except ImportError:
    has_torch_cuda = False
    has_torch_mps = False
    has_torch = False


def empty_cuda_cache():
    if has_torch_cuda:
        return torch.cuda.empty_cache()


def to_float_tensor(img: Union[np.ndarray, Image]) -> Tensor:
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W).
    Args:
        img: PIL.Image or numpy array
    Returns:
        torch.tensor
    """
    nparray: np.ndarray
    if isinstance(img, np.ndarray):
        nparray = img
    else:
        nparray = np.array(img)
    nparray = nparray.transpose((2, 0, 1))
    tens = torch.from_numpy(np.array(nparray)).float()
    if tens.max() > 1:
        tens /= 255
    return tens


def torch_to_numpy(img: Any) -> np.ndarray:
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    return img.transpose((1, 2, 0))


def select_device(device: Optional[str] = None) -> device:
    """
    Selects torch device

    Args:
        device: "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.
                When no device string is given, the order of preference
                to try is: cuda:0 > mps > cpu

    Returns:
        torch.device

    Inspired by https://github.com/ultralytics/yolov5/blob/6371de8879e7ad7ec5283e8b95cc6dd85d6a5e72/utils/torch_utils.py#L107
    """
    if device == "cuda" or device is None:
        device = "cuda:0"
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()

    cuda_id_pattern = r"^(0|[1-9]\d*)$"
    valid_cuda_id = bool(re.fullmatch(cuda_id_pattern, device))

    if not cpu and not mps and has_torch_cuda and valid_cuda_id:  # prefer GPU if available
        arg = "cuda:" + (device if int(device) < torch.cuda.device_count() else "0")
    elif mps and getattr(torch, "has_mps", False) and has_torch_mps:  # prefer MPS if available
        arg = "mps"
    else:  # revert to CPU
        arg = "cpu"

    return torch.device(arg)
