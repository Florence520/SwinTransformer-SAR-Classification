#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单独处理一个SAR MAT文件并生成图像"""
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_single_sar(mat_file):
    """可视化单个SAR文件"""
    print(f"正在处理: {mat_file}")
    
    # 加载数据
    data = sio.loadmat(mat_file)
    band = data['band']  # (12, 500, 500, 2)
    
    print(f"数据形状: {band.shape}")
    print(f"数据类型: {band.dtype}")
    print(f"数据范围: [{band.min()}, {band.max()}]")
    
    # 获取文件名
    filename = Path(mat_file).stem
    
    # 创建输出目录
    output_dir = Path("single_sar_output")
    output_dir.mkdir(exist_ok=True)
    
    # 显示第1、第6、第12个时相
    time_indices = [0, 5, 11]
    
    for t_idx in time_indices:
        sar_data = band[t_idx, ...]
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # VV极化
        ax = axes[0]
        im = ax.imshow(sar_data[:, :, 0], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Time Phase {t_idx+1}\nVV Polarization', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # VH极化
        ax = axes[1]
        im = ax.imshow(sar_data[:, :, 1], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Time Phase {t_idx+1}\nVH Polarization', fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 伪彩色合成
        ax = axes[2]
        r = sar_data[:, :, 0] / 255.0
        g = sar_data[:, :, 1] / 255.0
        b = np.zeros_like(r)
        rgb_image = np.stack([r, g, b], axis=-1)
        im = ax.imshow(rgb_image)
        ax.set_title(f'Time Phase {t_idx+1}\nComposite (VV+VH)', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        # 保存
        output_file = output_dir / f"{filename}_phase{t_idx+1}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close(fig)

if __name__ == "__main__":
    mat_file = r"D:\bishe\dataset\MTS12 dataset\s1\test\eopatch_id_129_col_8_row_11.mat"
    
    if os.path.exists(mat_file):
        visualize_single_sar(mat_file)
    else:
        print(f"文件不存在: {mat_file}")