# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import torch
from torch import stack as torch_stack
from torch.cuda import empty_cache as empty_cuda_cache
from torch.cuda import is_available as cuda_is_available


def to_float_tensor(img) -> torch.tensor:
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W).

    Args:
        img: np.ndarray
    Returns:
        torch.tensor
    """
    #
    return torch.from_numpy(img.transpose(2, 0, 1)).float()
