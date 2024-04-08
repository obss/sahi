import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov9TestConstants:
    YOLOV9C_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9c.pt"
    YOLOV9C_MODEL_PATH = "tests/data/models/yolov9/yolov9c.pt"

    YOLOV9E_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov9e.pt"
    YOLOV9E_MODEL_PATH = "tests/data/models/yolov9/yolov9e.pt"


def download_yolov9c_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9C_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9C_MODEL_URL,
            destination_path,
        )


def download_yolov9e_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9E_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9E_MODEL_URL,
            destination_path,
        )
