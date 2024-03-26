import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class RTDETRTestConstants:
    RTDETRL_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-l.pt"
    RTDETRL_MODEL_PATH = "tests/data/models/rtdetr/rtdetr-l.pt"

    RTDETRX_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-x.pt"
    RTDETRX_MODEL_PATH = "tests/data/models/rtdetr/rtdetr-x.pt"


def download_rtdetrl_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = RTDETRTestConstants.RTDETRL_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            RTDETRTestConstants.RTDETRX_MODEL_URL,
            destination_path,
        )


def download_rtdetrx_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = RTDETRTestConstants.RTDETRX_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            RTDETRTestConstants.RTDETRX_MODEL_URL,
            destination_path,
        )
