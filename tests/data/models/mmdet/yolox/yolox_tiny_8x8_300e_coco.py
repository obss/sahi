auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
base_lr = 0.01
custom_hooks = [
    dict(num_last_epochs=15, priority=48, type="YOLOXModeSwitchHook"),
    dict(priority=48, type="SyncNormHook"),
    dict(ema_type="ExpMomentumEMA", momentum=0.0001, priority=49, type="EMAHook", update_buffers=True),
]
data_root = "data/coco/"
dataset_type = "CocoDataset"
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False, dist_cfg=dict(backend="nccl"), mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)
)
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        act_cfg=dict(type="Swish"),
        deepen_factor=0.33,
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type="CSPDarknet",
        use_depthwise=False,
        widen_factor=0.375,
    ),
    bbox_head=dict(
        act_cfg=dict(type="Swish"),
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(eps=1e-16, loss_weight=5.0, mode="square", reduction="sum", type="IoULoss"),
        loss_cls=dict(loss_weight=1.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction="sum", type="L1Loss"),
        loss_obj=dict(loss_weight=1.0, reduction="sum", type="CrossEntropyLoss", use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        num_classes=80,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        type="YOLOXHead",
        use_depthwise=False,
    ),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=10,
                random_size_range=(
                    320,
                    640,
                ),
                size_divisor=32,
                type="BatchSyncRandomResize",
            ),
        ],
        pad_size_divisor=32,
        type="DetDataPreprocessor",
    ),
    neck=dict(
        act_cfg=dict(type="Swish"),
        in_channels=[
            96,
            192,
            384,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type="BN"),
        num_csp_blocks=1,
        out_channels=96,
        type="YOLOXPAFPN",
        upsample_cfg=dict(mode="nearest", scale_factor=2),
        use_depthwise=False,
    ),
    test_cfg=dict(nms=dict(iou_threshold=0.65, type="nms"), score_thr=0.01),
    train_cfg=dict(assigner=dict(center_radius=2.5, type="SimOTAAssigner")),
    type="YOLOX",
)
num_last_epochs = 15
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, nesterov=True, type="SGD", weight_decay=0.0005),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=True, convert_to_iter_based=True, end=5, type="mmdet.QuadraticWarmupLR"),
    dict(
        T_max=285, begin=5, by_epoch=True, convert_to_iter_based=True, end=285, eta_min=0.0005, type="CosineAnnealingLR"
    ),
    dict(begin=285, by_epoch=True, end=300, factor=1, type="ConstantLR"),
]
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file="annotations/instances_val2017.json",
        backend_args=None,
        data_prefix=dict(img="val2017/"),
        data_root="data/coco/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    416,
                    416,
                ),
                type="Resize",
            ),
            dict(
                pad_to_square=True,
                pad_val=dict(
                    img=(
                        114.0,
                        114.0,
                        114.0,
                    )
                ),
                type="Pad",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="data/coco/annotations/instances_val2017.json", backend_args=None, metric="bbox", type="CocoMetric"
)
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            416,
            416,
        ),
        type="Resize",
    ),
    dict(
        pad_to_square=True,
        pad_val=dict(
            img=(
                114.0,
                114.0,
                114.0,
            )
        ),
        type="Pad",
    ),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(max_epochs=300, type="EpochBasedTrainLoop", val_interval=10)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file="annotations/instances_train2017.json",
            backend_args=None,
            data_prefix=dict(img="train2017/"),
            data_root="data/coco/",
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(backend_args=None, type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
            type="CocoDataset",
        ),
        pipeline=[
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                type="Mosaic",
            ),
            dict(
                border=(
                    -320,
                    -320,
                ),
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type="RandomAffine",
            ),
            dict(type="YOLOXHSVRandomAug"),
            dict(prob=0.5, type="RandomFlip"),
            dict(
                keep_ratio=True,
                scale=(
                    640,
                    640,
                ),
                type="Resize",
            ),
            dict(
                pad_to_square=True,
                pad_val=dict(
                    img=(
                        114.0,
                        114.0,
                        114.0,
                    )
                ),
                type="Pad",
            ),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type="FilterAnnotations",
            ),
            dict(type="PackDetInputs"),
        ],
        type="MultiImageMixDataset",
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_dataset = dict(
    dataset=dict(
        ann_file="annotations/instances_train2017.json",
        backend_args=None,
        data_prefix=dict(img="train2017/"),
        data_root="data/coco/",
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        type="CocoDataset",
    ),
    pipeline=[
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            type="Mosaic",
        ),
        dict(
            border=(
                -320,
                -320,
            ),
            scaling_ratio_range=(
                0.1,
                2,
            ),
            type="RandomAffine",
        ),
        dict(
            img_scale=(
                640,
                640,
            ),
            pad_val=114.0,
            ratio_range=(
                0.8,
                1.6,
            ),
            type="MixUp",
        ),
        dict(type="YOLOXHSVRandomAug"),
        dict(prob=0.5, type="RandomFlip"),
        dict(
            keep_ratio=True,
            scale=(
                640,
                640,
            ),
            type="Resize",
        ),
        dict(
            pad_to_square=True,
            pad_val=dict(
                img=(
                    114.0,
                    114.0,
                    114.0,
                )
            ),
            type="Pad",
        ),
        dict(
            keep_empty=False,
            min_gt_bbox_wh=(
                1,
                1,
            ),
            type="FilterAnnotations",
        ),
        dict(type="PackDetInputs"),
    ],
    type="MultiImageMixDataset",
)
train_pipeline = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        type="Mosaic",
    ),
    dict(
        border=(
            -320,
            -320,
        ),
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type="RandomAffine",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        keep_ratio=True,
        scale=(
            640,
            640,
        ),
        type="Resize",
    ),
    dict(
        pad_to_square=True,
        pad_val=dict(
            img=(
                114.0,
                114.0,
                114.0,
            )
        ),
        type="Pad",
    ),
    dict(
        keep_empty=False,
        min_gt_bbox_wh=(
            1,
            1,
        ),
        type="FilterAnnotations",
    ),
    dict(type="PackDetInputs"),
]
tta_model = dict(tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.65, type="nms")), type="DetTTAModel")
tta_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scale=(
                        640,
                        640,
                    ),
                    type="Resize",
                ),
                dict(
                    keep_ratio=True,
                    scale=(
                        320,
                        320,
                    ),
                    type="Resize",
                ),
                dict(
                    keep_ratio=True,
                    scale=(
                        960,
                        960,
                    ),
                    type="Resize",
                ),
            ],
            [
                dict(prob=1.0, type="RandomFlip"),
                dict(prob=0.0, type="RandomFlip"),
            ],
            [
                dict(
                    pad_to_square=True,
                    pad_val=dict(
                        img=(
                            114.0,
                            114.0,
                            114.0,
                        )
                    ),
                    type="Pad",
                ),
            ],
            [
                dict(type="LoadAnnotations", with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                    type="PackDetInputs",
                ),
            ],
        ],
        type="TestTimeAug",
    ),
]
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file="annotations/instances_val2017.json",
        backend_args=None,
        data_prefix=dict(img="val2017/"),
        data_root="data/coco/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    416,
                    416,
                ),
                type="Resize",
            ),
            dict(
                pad_to_square=True,
                pad_val=dict(
                    img=(
                        114.0,
                        114.0,
                        114.0,
                    )
                ),
                type="Pad",
            ),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CocoDataset",
    ),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file="data/coco/annotations/instances_val2017.json", backend_args=None, metric="bbox", type="CocoMetric"
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
