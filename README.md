# SL-MULNet: Swin Transformer-based SAR Land Use Classification

基于 Swin Transformer 的时序 SAR 地物分类研究与实现。

##  项目简介

本项目实现了基于 **Swin UNETR** 架构的时序 SAR 图像地物分类模型，利用 Sentinel-1 时序数据进行高精度地物分类。

##  主要特性

-  **Swin UNETR 架构**：基于滑动窗口注意力机制的层次化特征提取
-  **时序 SAR 数据**：支持多极化（VV/VH）多时相输入（16个时相）
-  **PyQt5 图形界面**：提供直观的训练、推理和结果可视化功能
-  **MMSegmentation 集成**：基于 MMEngine 的高效训练框架
-  **多指标评估**：支持 mIoU、mF1、WF1 等多种评估指标

##  项目结构

```
SL-MULNet/
├── configs/              # 模型配置文件
│   ├── _base_/           # 基础配置（模型、数据集、调度）
│   ├── swinUnetr/        # Swin UNETR 配置
│   └── demo/             # 演示配置
├── mmseg/                # 分割模型核心代码
│   ├── apis/             # 推理API
│   ├── datasets/         # 数据集定义
│   ├── evaluation/       # 评估指标
│   ├── models/           # 模型定义
│   └── utils/            # 工具函数
├── tools/                # 训练和推理脚本
│   ├── train.py          # 训练脚本
│   ├── test.py           # 测试脚本
│   └── infer_single.py   # 单图推理脚本
├── gui/                  # PyQt5 图形界面
│   ├── main.py           # 主窗口
│   └── components/       # UI组件
├── work_dirs/            # 训练结果保存目录
├── testval/              # 测试验证数据
└── README.md             # 项目说明文档
```

##  环境配置

### 前置要求
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+（可选，用于GPU加速）

### 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n SFTNet_env python=3.8
conda activate SFTNet_env

# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 安装依赖

```bash
# 安装MMSegmentation和MMEngine
pip install mmengine mmsegmentation

# 安装项目依赖
pip install -e .

# 安装PyQt5图形界面
pip install pyqt5

# 安装其他依赖
pip install scipy matplotlib pillow
```

##  快速开始

### 训练模型

```bash
# 使用完整数据集训练
python tools/train.py configs/swinUnetr/SwinUNETR_224x224_40k_slovenia_s1.py --work-dir work_dirs/SwinUNETR_224x224_40k_slovenia_s1

# 使用演示数据集训练（快速测试）
python tools/train.py configs/demo/demo_from_swinunetr.py --work-dir work_dirs/demo
```

### 单图推理

```bash
python tools/infer_single.py \
    --config configs/swinUnetr/SwinUNETR_224x224_40k_slovenia_s1.py \
    --checkpoint work_dirs/SwinUNETR_224x224_40k_slovenia_s1/iter_40000.pth \
    --input data/s1/test/eopatch_id_xxx.mat \
    --output output/result.png
```

### 启动图形界面

```bash
python gui/main.py
```

### 批量推理

```bash
python tools/infer_batch.py \
    --config configs/swinUnetr/SwinUNETR_224x224_40k_slovenia_s1.py \
    --checkpoint work_dirs/SwinUNETR_224x224_40k_slovenia_s1/iter_40000.pth \
    --input-dir data/s1/test/ \
    --output-dir output/
```

##  数据集

### MTS12 Dataset

本项目使用斯洛文尼亚时序 SAR 数据集：

| 数据类型 | 维度 | 说明 |
|----------|------|------|
| Sentinel-1 SAR | 12 × 500 × 500 × 2 | 12个时相，双极化（VV+VH） |
| Sentinel-2 光学 | 12 × 500 × 500 × 6 | 12个时相，6个波段 |
| 地物标签 | 500 × 500 | 8类地物分类 |

### 地物类别

| 类别 | 标签值 | 说明 |
|------|--------|------|
| No Data | 0 | 无数据区域 |
| Cultivated Land | 1 | 耕地 |
| Forest | 2 | 森林 |
| Grassland | 3 | 草地 |
| Shrubland | 4 | 灌丛 |
| Water | 5 | 水体 |
| Wetland | 6 | 湿地 |
| Artificial Surface | 7 | 人工地表 |
| Bare Land | 8 | 裸地 |

##  模型性能

### 在 MTS12 数据集上的结果

| 指标 | 原始（8类） | 去除 Wetland/Bare Land |
|------|-------------|----------------------|
| OA（总体准确率） | 81.53% | 81.53% |
| mIoU（平均交并比） | 40.28% | 51.09% |
| mF1（平均F1分数） | 50.26% | 64.67% |
| WF1（加权F1分数） | 80.00% | 81.11% |

### 各类别F1分数

| 类别 | F1分数 |
|------|--------|
| Cultivated Land | 75.10% |
| Forest | 91.88% |
| Grassland | 71.91% |
| Shrubland | 23.17% |
| Water | 66.86% |
| Wetland | 0.00% |
| Artificial Surface | 59.11% |
| Bare Land | 7.78% |

##  模型架构

### Swin UNETR 结构

```
输入: 16 × 224 × 224 × 2 (时序SAR数据)
        ↓
    Patch Partition
        ↓
    ┌─────────────────────────────────────────┐
    │           Encoder (4层)                 │
    │  Layer 1: 224×224×96  →  Swin Block   │
    │  Layer 2: 112×112×192 →  Swin Block   │
    │  Layer 3: 56×56×384   →  Swin Block   │
    │  Layer 4: 28×28×768   →  Swin Block   │
    └─────────────────────────────────────────┘
        ↓
    Bottleneck (28×28×768)
        ↓
    ┌─────────────────────────────────────────┐
    │           Decoder (4层)                 │
    │  Layer 4: 56×56×384   →  Patch Expand  │
    │  Layer 3: 112×112×192 →  Patch Expand  │
    │  Layer 2: 224×224×96  →  Patch Expand  │
    │  Layer 1: 224×224×8   →  Patch Expand  │
    └─────────────────────────────────────────┘
        ↓
输出: 224 × 224 × 8 (8类分类结果)
```

##  配置说明

### 主要配置文件

| 文件 | 说明 |
|------|------|
| `configs/swinUnetr/SwinUNETR_224x224_40k_slovenia_s1.py` | 主模型配置 |
| `configs/_base_/models/SwinUNETR.py` | 模型结构定义 |
| `configs/_base_/datasets/slovenia_224x224.py` | 数据集配置 |
| `configs/_base_/schedules/schedule_40k.py` | 训练调度配置 |

### 训练参数

```python
# 训练配置
max_iters = 40000          # 最大迭代次数
batch_size = 4             # 批次大小
learning_rate = 0.0001     # 学习率
optimizer = AdamW          # 优化器
loss = CrossEntropyLoss    # 损失函数
```

##  结果输出

训练完成后，结果保存在 `work_dirs/` 目录下：

```
work_dirs/SwinUNETR_224x224_40k_slovenia_s1/
├── iter_40000.pth         # 模型权重
├── 20260419_163426/       # 训练日志目录
│   ├── 20260419_163426.log # 训练日志
│   └── tensorboard/        # TensorBoard日志
└── latest.pth              # 最新模型权重
```

##  可视化工具

### MAT文件可视化

```bash
python visualize_mat.py --input data/s1/test/eopatch_id_xxx.mat --output-dir vis/
```

### 训练曲线绘制

```bash
python plot_training_performance.py --log work_dirs/*/20260419_163426.log
```

##  贡献指南

欢迎提交 Issue 和 Pull Request！

### 代码规范

- 遵循 PEP8 编码规范
- 使用 Google 风格的文档注释
- 提交前运行 `flake8` 和 `pylint` 检查

##  引用

如果您使用本项目，请引用：

```bibtex
@article{slmulnet2024,
  title={SL-MULNet: Swin Transformer-based SAR Land Use Classification},
  author={Wang, Jiali and Others},
  journal={arXiv preprint arXiv:24xx.xxxxx},
  year={2024}
}
```

##  许可证

本项目采用 **MIT License**，详见 LICENSE 文件。

##  联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱：3339276242@qq.com
- GitHub Issues：https://github.com/jialiwang/SL-MULNet/issues

---

**最后更新**: 2024年6月
