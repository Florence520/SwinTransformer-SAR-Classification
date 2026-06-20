#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建最小的演示数据集
从原始数据集选取最少的样本用于前端演示
"""
import os
import shutil
from pathlib import Path

def create_minimal_dataset():
    """创建最小数据集"""
    
    # 原始数据集路径
    src_root = Path(r"D:\bishe\dataset\MTS12 dataset")
    dst_root = Path(r"D:\bishe\testdataset")
    
    # 创建目标目录结构
    dirs_to_create = [
        dst_root / "s1" / "train",
        dst_root / "s1" / "val",
        dst_root / "s1" / "test",
        dst_root / "s2" / "train",
        dst_root / "s2" / "val",
        dst_root / "s2" / "test",
        dst_root / "label" / "train",
        dst_root / "label" / "val",
        dst_root / "label" / "test",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("目录结构已创建")
    
    # 选取样本
    # 训练集：2个样本
    train_samples = ["eopatch_id_0_col_0_row_19", "eopatch_id_107_col_7_row_14"]
    
    # 验证集：1个样本（选择实际存在的）
    val_samples = ["eopatch_id_105_col_7_row_12"]
    
    # 测试集：1个样本（用于推理演示）
    test_samples = ["eopatch_id_129_col_8_row_11"]
    
    print("\n开始复制文件...")
    print("="*60)
    
    def copy_file(src, dst):
        """复制文件并打印结果"""
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {src.name}")
            return True
        else:
            print(f"  ✗ 缺失: {src.name}")
            return False
    
    # 复制训练集
    print("\n【训练集】")
    for sample in train_samples:
        copy_file(src_root / "s1" / "train" / f"{sample}.mat", 
                  dst_root / "s1" / "train" / f"{sample}.mat")
        copy_file(src_root / "s2" / "train" / f"{sample}.mat", 
                  dst_root / "s2" / "train" / f"{sample}.mat")
        copy_file(src_root / "label" / "train" / f"{sample}_label.mat", 
                  dst_root / "label" / "train" / f"{sample}_label.mat")
    
    # 复制验证集
    print("\n【验证集】")
    for sample in val_samples:
        copy_file(src_root / "s1" / "val" / f"{sample}.mat", 
                  dst_root / "s1" / "val" / f"{sample}.mat")
        copy_file(src_root / "s2" / "val" / f"{sample}.mat", 
                  dst_root / "s2" / "val" / f"{sample}.mat")
        copy_file(src_root / "label" / "val" / f"{sample}_label.mat", 
                  dst_root / "label" / "val" / f"{sample}_label.mat")
    
    # 复制测试集
    print("\n【测试集】")
    for sample in test_samples:
        copy_file(src_root / "s1" / "test" / f"{sample}.mat", 
                  dst_root / "s1" / "test" / f"{sample}.mat")
        copy_file(src_root / "s2" / "test" / f"{sample}.mat", 
                  dst_root / "s2" / "test" / f"{sample}.mat")
        copy_file(src_root / "label" / "test" / f"{sample}_label.mat", 
                  dst_root / "label" / "test" / f"{sample}_label.mat")
    
    print("\n" + "="*60)
    print("数据集创建完成！")
    print(f"目标路径: {dst_root}")
    
    # 统计数据集大小
    total_size = sum(f.stat().st_size for f in dst_root.rglob('*.mat') if f.exists())
    print(f"总大小: {total_size / 1024 / 1024:.2f} MB")
    
    # 生成配置文件
    print("\n生成配置文件...")
    config_content = f'''# 演示数据集配置文件
dataset_type = "MulSloveniaDataset"
data_root = r"{dst_root}"

img_norm_cfg = dict(
    mean1=[127.09, 52.14], 
    std1=[43.30, 40.76],
    mean2=[0.12444697,0.10736255,0.0919176,0.24317962,0.15813877,0.09011517], 
    std2=[0.09699202,0.09535561,0.11081194,0.13308774,0.09035148,0.06907019], 
    to_rgb=False
)

img_scale = (500, 500)
crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadDataFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_p1_path='s1/train',
            img_p2_path='s2/train', 
            seg_map_path='label/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_p1_path='s1/val', 
            img_p2_path='s2/val', 
            seg_map_path='label/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=True,
        data_prefix=dict(
            img_p1_path='s1/test', 
            img_p2_path='s2/test', 
            seg_map_path='label/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'], nan_to_num=0)
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'], nan_to_num=0)
'''
    
    config_path = dst_root / "demo_config.py"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"配置文件已保存: {config_path}")
    
    # 生成使用说明
    usage_content = '''# 演示数据集使用说明

## 数据集信息
- 训练集：2个样本
- 验证集：1个样本
- 测试集：1个样本
- 总大小：约224 MB

## 样本列表
训练集：
  - eopatch_id_0_col_0_row_19
  - eopatch_id_107_col_7_row_14

验证集：
  - eopatch_id_105_col_7_row_12

测试集：
  - eopatch_id_129_col_8_row_11

## 使用方法
1. 前端训练：使用 demo_config.py 配置文件
2. 推理演示：使用测试集中的样本进行推理

## 注意事项
- 由于数据集极小，仅用于功能演示
- 训练结果不具统计意义
- batch_size 已设为1以适应小数据集
'''
    
    readme_path = dst_root / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(usage_content)
    print(f"使用说明已保存: {readme_path}")

if __name__ == "__main__":
    create_minimal_dataset()