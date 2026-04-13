"""MMDetection configuration file for cascade-mask-rcnn_r50_fpn_1x_coco.py."""

_base_ = [
    "../_base_/models/cascade-mask-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
