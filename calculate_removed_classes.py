#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据数据集类别分布计算去除 Wetland 和 Bare Land 后的指标
"""

def calculate_metrics_after_removal():
    """计算去除两类后的指标"""
    # 原始类别数据（来自图片）
    original_data = {
        'No Data': {'pixels': 4055553, 'weight': 0.124786},
        'Cultivated Land': {'pixels': 3596368, 'weight': 0.110657},
        'Forest': {'pixels': 16146522, 'weight': 0.496816},
        'Grassland': {'pixels': 5205043, 'weight': 0.160155},
        'Shrubland': {'pixels': 1115888, 'weight': 0.034335},
        'Water': {'pixels': 288595, 'weight': 0.00888},
        'Wetland': {'pixels': 13203, 'weight': 0.000406},
        'Artificial Surface': {'pixels': 1624166, 'weight': 0.049974},
        'Bare Land': {'pixels': 454662, 'weight': 0.01399}
    }
    
    # 需要去除的类别
    remove_classes = ['Wetland', 'Bare Land']
    
    # 计算剩余类别的总像素数
    remaining_pixels = sum([v['pixels'] for k, v in original_data.items() if k not in remove_classes])
    
    # 计算新权重（归一化后）
    new_weights = {}
    for cls, data in original_data.items():
        if cls not in remove_classes:
            new_weights[cls] = data['pixels'] / remaining_pixels
    
    print("="*70)
    print("数据集类别分布（去除 Wetland 和 Bare Land 后）")
    print("="*70)
    print(f"{'类别':<20} {'原始像素数':>15} {'新占比':>10} {'新权重':>12}")
    print("-"*70)
    
    total_new_weight = 0
    for cls in ['No Data', 'Cultivated Land', 'Forest', 'Grassland', 'Shrubland', 'Water', 'Artificial Surface']:
        pixels = original_data[cls]['pixels']
        weight = new_weights[cls]
        total_new_weight += weight
        print(f"{cls:<20} {pixels:>15,} {weight*100:>9.2f}% {weight:>12.6f}")
    
    print("-"*70)
    print(f"{'总计':<20} {remaining_pixels:>15,} {total_new_weight*100:>9.2f}% {total_new_weight:>12.6f}")
    print("="*70)
    
    return new_weights, original_data

def calculate_new_wf1_for_models(new_weights):
    """计算三个模型去除两类后的 WF1"""
    # 假设各模型各类别的F1分数（根据原始WF1反推）
    # 这是基于原始WF1=0.690的估算，实际需要真实的各类别F1
    model_data = {
        'RF [21]': {'OA': 0.638, 'WF1': 0.690, 'mF1': 0.286},
        'XGBoost [22]': {'OA': 0.602, 'WF1': 0.667, 'mF1': 0.275},
        '2D U-Net [49]': {'OA': 0.761, 'WF1': 0.738, 'mF1': 0.316}
    }
    
    print("\n" + "="*70)
    print("去除 Wetland 和 Bare Land 后的指标计算")
    print("="*70)
    print(f"{'模型':<20} {'原始OA':>10} {'原始WF1':>10} {'原始mF1':>10} {'新WF1':>10} {'新mF1':>10}")
    print("-"*70)
    
    for model, metrics in model_data.items():
        # 由于没有各类别F1，我们基于假设计算
        # 假设去除低F1的两类后，WF1会提升
        # 新mF1 = (原始mF1 * 8 - 低F1类别) / 6
        # 这里假设 Wetland 和 Bare Land 的F1很低（接近0）
        
        # 估算新的mF1（假设去除的两类F1较低）
        new_mF1 = metrics['mF1'] * 8 / 6
        
        # 估算新的WF1（权重重新分配）
        # 剩余类别权重和 = 1 - 0.000406 - 0.01399 = 0.985604
        # 假设原始WF1是各类别F1加权平均
        # 新WF1 = (原始WF1 - 低权重*低F1) / 剩余权重和
        # 假设 Wetland 和 Bare Land 的F1为0.1和0.05
        wetland_f1 = 0.1
        bare_f1 = 0.05
        remaining_wf1_numerator = metrics['WF1'] - original_data['Wetland']['weight'] * wetland_f1 - original_data['Bare Land']['weight'] * bare_f1
        remaining_weight_sum = 1 - original_data['Wetland']['weight'] - original_data['Bare Land']['weight']
        new_WF1 = remaining_wf1_numerator / remaining_weight_sum
        
        print(f"{model:<20} {metrics['OA']:>9.3f} {metrics['WF1']:>10.3f} {metrics['mF1']:>10.3f} {new_WF1:>10.3f} {new_mF1:>10.3f}")
    
    print("="*70)
    print("注：新WF1和新mF1是基于假设的估算值")
    print("如需精确计算，请提供各模型各类别的F1分数")

if __name__ == "__main__":
    new_weights, original_data = calculate_metrics_after_removal()
    calculate_new_wf1_for_models(new_weights)