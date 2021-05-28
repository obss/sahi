import urllib.request
from os import path

from sahi.utils.file import create_dir


def mmdet_version_as_integer():
    import mmdet

    return int(mmdet.__version__.replace(".", ""))


class MmdetTestConstants:
    MMDET_CASCADEMASKRCNN_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
    MMDET_CASCADEMASKRCNN_MODEL_PATH = "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
    MMDET_RETINANET_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
    MMDET_RETINANET_MODEL_PATH = "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"

    @property
    def MMDET_CASCADEMASKRCNN_CONFIG_PATH(self):
        if mmdet_version_as_integer() < 290:
            return "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_v280.py"
        else:
            return "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"

    @property
    def MMDET_RETINANET_CONFIG_PATH(self):
        if mmdet_version_as_integer() < 290:
            return "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_1x_coco_v280.py"
        else:
            return "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_1x_coco.py"


class Yolov5TestConstants:
    YOLOV5S6_MODEL_URL = (
        "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s6.pt"
    )
    YOLOV5S6_MODEL_PATH = "tests/data/models/yolov5/yolov5s6.pt"


def download_mmdet_cascade_mask_rcnn_model():

    create_dir("tests/data/models/mmdet_cascade_mask_rcnn/")

    if not path.exists(MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH):
        urllib.request.urlretrieve(
            MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_URL,
            MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
        )


def download_mmdet_retinanet_model():

    create_dir("tests/data/models/mmdet_retinanet/")

    if not path.exists(MmdetTestConstants.MMDET_RETINANET_MODEL_PATH):
        urllib.request.urlretrieve(
            MmdetTestConstants.MMDET_RETINANET_MODEL_URL,
            MmdetTestConstants.MMDET_RETINANET_MODEL_PATH,
        )


def download_yolov5s6_model():

    create_dir("tests/data/models/yolov5/")

    if not path.exists(Yolov5TestConstants.YOLOV5S6_MODEL_PATH):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5S6_MODEL_URL,
            Yolov5TestConstants.YOLOV5S6_MODEL_PATH,
        )
