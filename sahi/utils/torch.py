# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.


from sahi.utils.import_utils import is_torch_available


def torch_load(path):
    if is_torch_available():
        import torch

        return torch.load(path)
    else:
        raise ImportError("You need to install PyTorch to use this method.")


def empty_cuda_cache():
    if is_torch_cuda_available():
        import torch

        return empty_cuda_cache()
    else:
        raise ImportError("You need to install PyTorch to use this method.")


def torch_stack():
    if is_torch_available():
        import torch

        return torch_stack
    else:
        raise ImportError("You need to install PyTorch to use this method.")


def to_float_tensor(img):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W).
    Args:
        img: np.ndarray
    Returns:
        torch.tensor
    """
    if is_torch_available():
        import torch

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        if img.max() > 1:
            img /= 255

        return img

    else:
        raise ImportError("You need to install PyTorch to use this method.")


def torch_to_numpy(img):
    if is_torch_available():
        import torch

        img = img.numpy()
        if img.max() > 1:
            img /= 255
        return img.transpose((1, 2, 0))
    else:
        raise ImportError("You need to install PyTorch to use this method.")


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False
