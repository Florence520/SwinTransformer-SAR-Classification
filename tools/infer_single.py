#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单图推理脚本"""
import sys
import os
import argparse
import tempfile
import cv2
import numpy as np
import scipy.io as sio
import torch

# 添加项目根目录到路径
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, ".."))

# 显式导入 mmseg.models 和 MulSegDataPreProcessor，确保所有自定义模块被注册
import mmseg.models
from mmseg.models.data_preprocessor import MulSegDataPreProcessor, SegDataPreProcessor
from mmseg.registry import MODELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', required=True, help='.mat 文件路径')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径')
    parser.add_argument('--output', default=None, help='输出图片路径（可选，默认保存到 testval/testimages 目录）')
    args = parser.parse_args()
    
    # 固定输出目录
    output_dir = r"D:\bishe\SL-MULNet\testval\testimages"
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定输出路径，自动生成文件名
    if args.output is None:
        # 从输入文件名生成输出文件名
        input_filename = os.path.basename(args.mat)
        output_filename = os.path.splitext(input_filename)[0] + '_pred.png'
        args.output = os.path.join(output_dir, output_filename)
    
    print(f"配置文件路径：{args.config}")
    print(f"检查点路径：{args.checkpoint}")
    print(f"已注册的模型数量：{len(MODELS.module_dict)}")
    print(f"MulSegDataPreProcessor 已注册：{'MulSegDataPreProcessor' in MODELS.module_dict}")
    
    try:
        # 根据输入的 mat 文件路径找到对应的 S1 和 S2 文件
        input_path = args.mat
        
        # 查找对应的 S2 文件
        s2_file = None
        if 's1' in input_path:
            s2_file = input_path.replace('s1', 's2')
        elif 'S1' in input_path:
            s2_file = input_path.replace('S1', 'S2')
        else:
            # 尝试从文件名推断
            dir_path = os.path.dirname(input_path)
            filename = os.path.basename(input_path)
            # 尝试查找同文件名的 S2 文件
            s2_file = os.path.join(dir_path.replace('s1', 's2').replace('S1', 'S2'), filename)
        
        if s2_file is None or not os.path.exists(s2_file):
            print(f"警告：找不到对应的 S2 文件！", file=sys.stderr)
            print(f"输入路径：{input_path}", file=sys.stderr)
            print(f"尝试的 S2 路径：{s2_file}", file=sys.stderr)
            sys.exit(1)
        
        print(f"S1 文件：{input_path}")
        print(f"S2 文件：{s2_file}")
        
        # 直接使用推理 API，传入数据字典
        from mmseg.apis import init_model, inference_model
        
        model = init_model(args.config, args.checkpoint, device='cpu')
        
        # 构建数据字典
        data_dict = {
            'img_p1_path': input_path,
            'img_p2_path': s2_file
        }
        
        # 调用 inference_model，传入数据字典
        result = inference_model(model, data_dict)
        
        pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
        
        # 颜色映射 -
        palette = [
            [255, 255, 255],    # 0: No Data (白色)
            [255, 255, 0],      # 1: Cultivated Land (黄色)
            [0, 102, 0],        # 2: Forest (深绿色)
            [255, 165, 0],      # 3: Grassland (橙色)
            [139, 69, 19],      # 4: Shrubland (棕色)
            [0, 0, 255],        # 5: Water (蓝色)
            [135, 206, 235],    # 6: Wetlands (浅蓝色)
            [255, 0, 255],      # 7: Artificial Surface (洋红色)
            [128, 128, 128],    # 8: Bare Land (灰色)
        ]
        
        color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for class_idx, color in enumerate(palette):
            color_mask[pred_mask == class_idx] = color
        
        cv2.imwrite(args.output, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        print(f"推理完成，结果已保存到：{args.output}")
        
    except Exception as e:
        print(f"推理失败：{str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
