mmdet_cascade_mask_rcnn_model_url = "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
mmdet_cascade_mask_rcnn_model_path = "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
mmdet_cascade_mask_rcnn_config_url = "https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"
mmdet_cascade_mask_rcnn_config_path = (
    "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"
)


def download_mmdet_cascade_mask_rcnn_model():
    import urllib.request
    from os import path

    from sahi.utils.file import create_dir

    create_dir("tests/data/models/mmdet_cascade_mask_rcnn/")

    if not path.exists(mmdet_cascade_mask_rcnn_model_path):
        urllib.request.urlretrieve(
            mmdet_cascade_mask_rcnn_model_url,
            mmdet_cascade_mask_rcnn_model_path,
        )
