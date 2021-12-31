import urllib.request
from os import path
from pathlib import Path
from typing import Optional

import detectron2


def detectron2_version_as_integer():
    return int(detectron2.__version__.split(".")[0])


class Detectron2TestConstants:
    try:
        mask_rcnn_R_50_C4_1x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl"
        mask_rcnn_R_50_C4_1x_path = "model/mask_rcnn_R_50_C4_1x.pkl"

        mask_rcnn_R_50_C4_3x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl"
        mask_rcnn_R_50_C4_3x_path = "model/mask_rcnn_R_50_C4_3x.pkl"

        mask_rcnn_R_50_DC5_1x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl"
        mask_rcnn_R_50_DC5_1x_path = "model/mask_rcnn_R_50_DC5_1x.pkl"

        mask_rcnn_R_50_DC5_3x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl"
        mask_rcnn_R_50_DC5_3x_path = "model/mask_rcnn_R_50_DC5_3x.pkl"

        mask_rcnn_R_50_FPN_3x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
        mask_rcnn_R_50_FPN_3x_path = "model/mask_rcnn_R_50_FPN_3x.pkl"

        mask_rcnn_R_101_C4_3x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
        mask_rcnn_R_101_C4_3x_path = "model/mask_rcnn_R_101_C4_3x.pkl"

        mask_rcnn_R_101_DC5_3x_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
        mask_rcnn_R_101_DC5_3x_path = "model/mask_rcnn_R_101_DC5_3x.pkl"

    except ImportError:
        print("Import Error")


def download_detectron2_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Detectron2TestConstants.mask_rcnn_R_101_C4_3x_path

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Detectron2TestConstants.mask_rcnn_R_101_C4_3x_url,
            destination_path,
        )
