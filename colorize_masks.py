import os
import cv2
import numpy as np

# 定义颜色映射表（根据你的数据集类别定义）
# 这里使用常见的地物分类颜色
PALETTE = [
    [255, 255, 255],    # 0: No Data（白色）
    [255, 255, 0],      # 1: Cultivated Land（耕地，黄色）
    [0, 102, 0],        # 2: Forest（森林，深绿色）
    [255, 165, 0],      # 3: Grassland（草地，橙色）
    [139, 69, 19],      # 4: Shrubland（灌丛，棕色）
    [0, 0, 255],        # 5: Water（水体，蓝色）
    [135, 206, 235],    # 6: Wetlands（湿地，浅蓝色）
    [255, 0, 255],      # 7: Artificial Surface（人工地表，洋红色）
    [128, 128, 128]     # 8: Bare Land（裸地，灰色）
]

def colorize_mask(input_path, output_path, palette):
    """将灰度分割标签转换为彩色图"""
    # 读取灰度图
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"无法读取文件: {input_path}")
        return
    
    # 创建彩色图
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 应用颜色映射
    for class_idx, color in enumerate(palette):
        color_mask[mask == class_idx] = color
    
    # 保存彩色图
    cv2.imwrite(output_path, color_mask)
    print(f"已保存: {output_path}")

def batch_colorize(input_dir, output_dir, palette):
    """批量处理目录中的所有分割图"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"color_{filename}")
            colorize_mask(input_path, output_path, palette)

if __name__ == "__main__":
    # 输入目录（你的分割结果目录）
    input_dir = r"D:\bishe\SL-MULNet\testval"
    # 输出目录（彩色结果保存位置）
    output_dir = r"D:\bishe\SL-MULNet\testval_color"
    
    batch_colorize(input_dir, output_dir, PALETTE)
    print("批量处理完成！")