"""RT-DETR model utilities and constants."""

from __future__ import annotations

from sahi.utils.file import download_from_url


class RTDETRTestConstants:
    """RT-DETR test model configurations."""

    RTDETRL_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-l.pt"
    RTDETRL_MODEL_PATH = "tests/data/models/rtdetr/rtdetr-l.pt"

    RTDETRX_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-x.pt"
    RTDETRX_MODEL_PATH = "tests/data/models/rtdetr/rtdetr-x.pt"


def download_rtdetrl_model(destination_path: str | None = None) -> None:
    """Download the RT-DETR-L model for testing."""
    download_from_url(RTDETRTestConstants.RTDETRL_MODEL_URL, destination_path or RTDETRTestConstants.RTDETRL_MODEL_PATH)


def download_rtdetrx_model(destination_path: str | None = None) -> None:
    """Download the RT-DETR-X model for testing."""
    download_from_url(RTDETRTestConstants.RTDETRX_MODEL_URL, destination_path or RTDETRTestConstants.RTDETRX_MODEL_PATH)
