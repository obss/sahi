import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov5TestConstants:
    YOLOV_MODEL_URL = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96"
