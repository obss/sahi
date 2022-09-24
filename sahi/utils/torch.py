# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.


from sahi.utils.import_utils import is_available


def empty_cuda_cache():
    if is_torch_cuda_available():
        import torch

        return torch.cuda.empty_cache()


def to_float_tensor(img):
    """
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W).
    Args:
        img: np.ndarray
    Returns:
        torch.tensor
    """

    import torch

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255

    return img


def torch_to_numpy(img):
    import torch

    img = img.numpy()
    if img.max() > 1:
        img /= 255
    return img.transpose((1, 2, 0))


def is_torch_cuda_available():
    if is_available("torch"):
        import torch

        return torch.cuda.is_available()
    else:
        return False
