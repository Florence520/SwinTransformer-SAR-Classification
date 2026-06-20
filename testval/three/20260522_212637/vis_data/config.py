class_weight = None
crop_size = (
    224,
    224,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(
        224,
        224,
    ),
    type='MulSegDataPreProcessor')
data_root = 'D:/bishe/dataset/MTS12dataset'
dataset_type = 'MulSloveniaDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_norm_cfg = dict(
    mean1=[
        127.09,
        52.14,
    ],
    mean2=[
        0.12444697,
        0.10736255,
        0.0919176,
        0.24317962,
        0.15813877,
        0.09011517,
    ],
    std1=[
        43.3,
        40.76,
    ],
    std2=[
        0.09699202,
        0.09535561,
        0.11081194,
        0.13308774,
        0.09035148,
        0.06907019,
    ],
    to_rgb=False)
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    500,
    500,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        img_size=(
            16,
            224,
            224,
        ),
        in_channels=2,
        out_channels=8,
        type='SwinUNETR'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(
            224,
            224,
        ),
        type='MulSegDataPreProcessor'),
    decode_head=dict(
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        num_classes=8,
        type='PseudoHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        224,
        224,
    ), mode='slide', stride=(
        200,
        200,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
optical_background = (
    0,
    0,
    0,
)
optical_num_bands = 3
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=1e-05),
    type='OptimWrapper')
optimizer = dict(lr=0.0001, type='AdamW', weight_decay=1e-05)
pad_size = (
    224,
    224,
)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
resume = True
sar_background = (
    0,
    0,
    0,
)
sar_num_bands = 16
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_p1_path='s1/test',
            img_p2_path='s2/test',
            seg_map_path='label/test'),
        data_root='D:\\bishe\\dataset\\MTS12 dataset',
        pipeline=[
            dict(type='LoadDataFromFile'),
            dict(img_scale=(
                500,
                500,
            ), keep_ratio=True, type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                mean1=[
                    127.09,
                    52.14,
                ],
                mean2=[
                    0.12444697,
                    0.10736255,
                    0.0919176,
                    0.24317962,
                    0.15813877,
                    0.09011517,
                ],
                std1=[
                    43.3,
                    40.76,
                ],
                std2=[
                    0.09699202,
                    0.09535561,
                    0.11081194,
                    0.13308774,
                    0.09035148,
                    0.06907019,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='MulSloveniaDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], nan_to_num=0, type='IoUMetric')
test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(img_scale=(
        500,
        500,
    ), keep_ratio=True, type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(
        mean1=[
            127.09,
            52.14,
        ],
        mean2=[
            0.12444697,
            0.10736255,
            0.0919176,
            0.24317962,
            0.15813877,
            0.09011517,
        ],
        std1=[
            43.3,
            40.76,
        ],
        std2=[
            0.09699202,
            0.09535561,
            0.11081194,
            0.13308774,
            0.09035148,
            0.06907019,
        ],
        to_rgb=False,
        type='Normalize'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_p1_path='s1/train',
            img_p2_path='s2/train',
            seg_map_path='label/train'),
        data_root='D:\\bishe\\dataset\\MTS12 dataset',
        pipeline=[
            dict(type='LoadDataFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                img_scale=(
                    500,
                    500,
                ),
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                type='Resize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    224,
                    224,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                mean1=[
                    127.09,
                    52.14,
                ],
                mean2=[
                    0.12444697,
                    0.10736255,
                    0.0919176,
                    0.24317962,
                    0.15813877,
                    0.09011517,
                ],
                std1=[
                    43.3,
                    40.76,
                ],
                std2=[
                    0.09699202,
                    0.09535561,
                    0.11081194,
                    0.13308774,
                    0.09035148,
                    0.06907019,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='MulSloveniaDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(
        img_scale=(
            500,
            500,
        ),
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        type='Resize'),
    dict(cat_max_ratio=0.75, crop_size=(
        224,
        224,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        mean1=[
            127.09,
            52.14,
        ],
        mean2=[
            0.12444697,
            0.10736255,
            0.0919176,
            0.24317962,
            0.15813877,
            0.09011517,
        ],
        std1=[
            43.3,
            40.76,
        ],
        std2=[
            0.09699202,
            0.09535561,
            0.11081194,
            0.13308774,
            0.09035148,
            0.06907019,
        ],
        to_rgb=False,
        type='Normalize'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(reduce_zero_label=True, type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_p1_path='s1/val',
            img_p2_path='s2/val',
            seg_map_path='label/val'),
        data_root='D:\\bishe\\dataset\\MTS12 dataset',
        pipeline=[
            dict(type='LoadDataFromFile'),
            dict(img_scale=(
                500,
                500,
            ), keep_ratio=True, type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                mean1=[
                    127.09,
                    52.14,
                ],
                mean2=[
                    0.12444697,
                    0.10736255,
                    0.0919176,
                    0.24317962,
                    0.15813877,
                    0.09011517,
                ],
                std1=[
                    43.3,
                    40.76,
                ],
                std2=[
                    0.09699202,
                    0.09535561,
                    0.11081194,
                    0.13308774,
                    0.09035148,
                    0.06907019,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='MulSloveniaDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], nan_to_num=0, type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'D:/bishe/SL-MULNet/testval/three'
