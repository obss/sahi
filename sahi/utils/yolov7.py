import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov7TestConstants:
    YOLOV7_MODEL_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    YOLOV7_MODEL_PATH = "tests/data/models/yolov7/yolov7.pt"

    YOLOV7_TINY_MODEL_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    YOLOV7_TINY_MODEL_PATH = "tests/data/models/yolov7/yolov7-tiny.pt"

    YOLOV7_E6_MODEL_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt"
    YOLOV7_E6_MODEL_PATH = "tests/data/models/yolov7/yolov7-e6.pt"


def download_yolov7_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov7TestConstants.YOLOV7_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov7TestConstants.YOLOV7_MODEL_URL,
            destination_path,
        )


def download_yolov7e6_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov7TestConstants.YOLOV7_E6_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov7TestConstants.YOLOV7_E6_MODEL_URL,
            destination_path,
        )
