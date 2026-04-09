from __future__ import annotations

import importlib.metadata as importlib_metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahi.annotation import BoundingBox as BoundingBox
    from sahi.annotation import Category as Category
    from sahi.annotation import Mask as Mask
    from sahi.auto_model import AutoDetectionModel as AutoDetectionModel
    from sahi.models.base import DetectionModel as DetectionModel
    from sahi.prediction import ObjectPrediction as ObjectPrediction

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"

# Lazy imports — heavy modules (cv2, shapely, requests, tqdm) are only
# loaded when the user actually accesses one of these names.
_LAZY_IMPORTS = {
    "BoundingBox": "sahi.annotation",
    "Category": "sahi.annotation",
    "Mask": "sahi.annotation",
    "AutoDetectionModel": "sahi.auto_model",
    "DetectionModel": "sahi.models.base",
    "ObjectPrediction": "sahi.prediction",
}


def __getattr__(name: str) -> object:
    """Lazily import public symbols on first access.

    When a name listed in ``_LAZY_IMPORTS`` is accessed on the ``sahi``
    module, this function dynamically imports the corresponding submodule,
    retrieves the attribute, and caches it in the module globals so that
    subsequent accesses bypass this hook.

    Raises:
        AttributeError: If ``name`` is not a known lazy import.
    """
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sahi' has no attribute {name!r}")


__all__ = [
    "AutoDetectionModel",
    "BoundingBox",
    "Category",
    "DetectionModel",
    "Mask",
    "ObjectPrediction",
    "__version__",
]
