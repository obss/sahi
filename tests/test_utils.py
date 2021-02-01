import urllib.request
from os import path

from sahi.utils.file import create_dir

MMDET_CASCADEMASKRCNN_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
MMDET_CASCADEMASKRCNN_MODEL_PATH = "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
MMDET_CASCADEMASKRCNN_CONFIG_PATH = (
    "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"
)

MMDET_RETINANET_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
MMDET_RETINANET_MODEL_PATH = (
    "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
)
MMDET_RETINANET_CONFIG_PATH = (
    "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_1x_coco.py"
)


def download_mmdet_cascade_mask_rcnn_model():

    create_dir("tests/data/models/mmdet_cascade_mask_rcnn/")

    if not path.exists(MMDET_CASCADEMASKRCNN_MODEL_PATH):
        urllib.request.urlretrieve(
            MMDET_CASCADEMASKRCNN_MODEL_URL,
            MMDET_CASCADEMASKRCNN_MODEL_PATH,
        )


def download_mmdet_retinanet_model():

    create_dir("tests/data/models/mmdet_retinanet/")

    if not path.exists(MMDET_RETINANET_MODEL_PATH):
        urllib.request.urlretrieve(
            MMDET_RETINANET_MODEL_URL,
            MMDET_RETINANET_MODEL_PATH,
        )
