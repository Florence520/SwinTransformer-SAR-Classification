# dataset settings
dataset_type = "MulSloveniaDataset"  # 假设有一个支持多模态的数据集类
data_root = r"D:\bishe\dataset\MTS12 dataset"

img_norm_cfg = dict(mean1=[127.09, 52.14], std1=[43.30, 40.76],mean2=[0.12444697,0.10736255,0.0919176,0.24317962,0.15813877,0.09011517], std2=[0.09699202,0.09535561,0.11081194,0.13308774,0.09035148,0.06907019], to_rgb=False)

# SAR数据配置
sar_background = (0, 0, 0)  # BGR，但实际上SAR可能不是三通道，这里只是示例
sar_num_bands = 16

# 光学数据配置
optical_background = (0, 0, 0)  # 光学数据的背景，通常为黑色
optical_num_bands = 3  # 光学数据通常是三通道

img_scale = (500, 500)
crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='Resize',
        img_scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth1
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations', reduce_zero_label=True)], [dict(type='PackSegInputs')]
        ])
]


train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_p1_path='s1/train',img_p2_path='s2/train', seg_map_path='label/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_p1_path='s1/val', img_p2_path='s2/val', seg_map_path='label/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_p1_path='s1/test', img_p2_path='s2/test', seg_map_path='label/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'], nan_to_num=0)
test_evaluator = val_evaluator
