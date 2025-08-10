import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"


from sahi.annotation import BoundingBox, Category, Mask
from sahi.auto_model import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction

__all__ = [
    "__version__",
    "BoundingBox",
    "Category",
    "Mask",
    "AutoDetectionModel",
    "DetectionModel",
    "ObjectPrediction",
]
