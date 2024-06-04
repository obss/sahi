optimizer = dict(
    type="SGD",
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="YOLOX",
    warmup="exp",
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05,
)
runner = dict(type="EpochBasedRunner", max_epochs=300)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=15, priority=48),
    dict(type="SyncNormHook", num_last_epochs=15, interval=10, priority=48),
    dict(type="ExpMomentumEMAHook", resume_from=None, momentum=0.0001, priority=49),
]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
img_scale = (640, 640)
model = dict(
    type="YOLOX",
    input_size=(640, 640),
    random_size_range=(10, 20),
    random_size_interval=10,
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.375),
    neck=dict(type="YOLOXPAFPN", in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(type="YOLOXHead", num_classes=80, in_channels=96, feat_channels=96),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)
data_root = "data/coco/"
dataset_type = "CocoDataset"
train_pipeline = [
    dict(type="Mosaic", img_scale=(640, 640), pad_val=114.0),
    dict(type="RandomAffine", scaling_ratio_range=(0.5, 1.5), border=(-320, -320)),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017/",
        pipeline=[dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True)],
        filter_empty_gt=False,
    ),
    pipeline=[
        dict(type="Mosaic", img_scale=(640, 640), pad_val=114.0),
        dict(type="RandomAffine", scaling_ratio_range=(0.5, 1.5), border=(-320, -320)),
        dict(type="YOLOXHSVRandomAug"),
        dict(type="RandomFlip", flip_ratio=0.5),
        dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
        dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
    ],
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017/",
            pipeline=[dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True)],
            filter_empty_gt=False,
        ),
        pipeline=[
            dict(type="Mosaic", img_scale=(640, 640), pad_val=114.0),
            dict(type="RandomAffine", scaling_ratio_range=(0.5, 1.5), border=(-320, -320)),
            dict(type="YOLOXHSVRandomAug"),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
            dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
    val=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type="DefaultFormatBundle"),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(416, 416),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type="DefaultFormatBundle"),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)
max_epochs = 300
num_last_epochs = 15
interval = 10
evaluation = dict(save_best="auto", interval=10, dynamic_intervals=[(285, 1)], metric="bbox")
