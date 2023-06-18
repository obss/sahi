tta_model = dict(type="DetTTAModel", tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100))

img_scales = [(1333, 800), (666, 400), (2000, 1200)]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale=s, keep_ratio=True) for s in img_scales],
            [dict(type="RandomFlip", prob=1.0), dict(type="RandomFlip", prob=0.0)],
            [dict(type="LoadAnnotations", with_bbox=True)],
            [
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]
