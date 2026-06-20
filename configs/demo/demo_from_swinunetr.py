# 基于原始模型的演示数据集配置
_base_ = [
    "../swinUnetr/SwinUNETR_224x224_40k_slovenia_s1.py",
]

# 覆盖数据路径为小数据集
data_root = r"D:\bishe\testdataset"

# 针对小数据集调整训练参数
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100, val_interval=50)

# 调整学习率调度
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, eta_min=1e-6, by_epoch=False)
]

# 调整日志间隔
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'))