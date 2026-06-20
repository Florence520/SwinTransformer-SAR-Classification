
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmseg.registry import MODELS


@MODELS.register_module()
class MultiModalBackbone(nn.Module):
    def __init__(self, num_channels_sar, num_channels_optical, num_classes, num_timesteps,upsample_size):
        super(MultiModalBackbone, self).__init__()

        # CNN部分，分别处理SAR和光学遥感数据
        self.cnn_sar = nn.Sequential(
            nn.Conv2d(num_channels_sar, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.cnn_optical = nn.Sequential(
            nn.Conv2d(num_channels_optical, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 假设经过CNN后的特征维度相同，我们可以将它们拼接在一起
        self.features_dim = 64 * (224 // 4) * (224 // 4)  # 假设输入图像为224x224，且经过一次MaxPool2d

        # RNN部分，用于提取时间特征
        self.rnn = nn.LSTM(input_size=self.features_dim * 2, hidden_size=512, num_layers=2, batch_first=True)

        # 分类器
        self.classifier = nn.Linear(512, num_classes)

        # 新增：记录上采样目标尺寸
        self.upsample_size = upsample_size

        self.num_timesteps = num_timesteps

    def forward(self, x_sar, x_optical):
        # x_sar和x_optical的形状应为 (batch_size, num_timesteps, num_channels, H, W)
        batch_size, num_timesteps, _, H, W = x_sar.size()

        # 分别对SAR和光学遥感数据应用CNN
        x_sar = x_sar.view(-1, x_sar.size(2), x_sar.size(3), x_sar.size(4))
        x_sar = self.cnn_sar(x_sar)
        x_sar = x_sar.view(batch_size, num_timesteps, -1)

        x_optical = x_optical.view(-1, x_optical.size(2), x_optical.size(3), x_optical.size(4))
        x_optical = self.cnn_optical(x_optical)
        x_optical = x_optical.view(batch_size, num_timesteps, -1)

        # 将SAR和光学遥感数据的特征拼接在一起
        x = torch.cat((x_sar, x_optical), dim=2)

        # 将拼接后的特征输入到RNN中
        x, _ = self.rnn(x)

        # 取RNN的最后一个时间步的输出进行分类
        x = x[:, -1, :]

        # 新增：上采样层
        x = F.interpolate(x.unsqueeze(-1).unsqueeze(-1), size=self.upsample_size, mode='bilinear', align_corners=False)
        x = x.squeeze(-1).squeeze(-1)  # 去掉因为上采样而添加的两个维度

        return x
