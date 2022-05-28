import importlib.util
import logging

from sahi.utils.versions import importlib_metadata

# adapted from https://github.com/huggingface/transformers/src/transformers/utils/import_utils.py

logger = logging.getLogger(__name__)


_torch_version = "N/A"


_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
        logger.info(f"PyTorch version {_torch_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False


def is_torch_available():
    return _torch_available
