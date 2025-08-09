# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.


import re
from os import environ
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL.Image import Image


def empty_cuda_cache() -> None:
    torch.cuda.empty_cache()


def to_float_tensor(img: Union[np.ndarray, Image]) -> "torch.Tensor":
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
    tensor = torch.from_numpy(np.array(nparray)).float()
    if tensor.max() > 1:
        tensor /= 255
    return tensor


def torch_to_numpy(img: Any) -> np.ndarray:
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    return img.transpose((1, 2, 0))


def select_device(device: Optional[str] = None) -> "torch.device":
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
        environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()

    cuda_id_pattern = r"^(0|[1-9]\d*)$"
    valid_cuda_id = bool(re.fullmatch(cuda_id_pattern, device))

    if not cpu and not mps and torch.cuda.is_available() and valid_cuda_id:  # prefer GPU if available
        arg = "cuda:" + (device if int(device) < torch.cuda.device_count() else "0")
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        arg = "mps"
    else:  # revert to CPU
        arg = "cpu"

    return torch.device(arg)
