import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

class MultiModalProcess(nn.Module):
    def __init__(self, shallow_channels, deep_channels, out_channels):
        super(MultiModalProcess, self).__init__()

        self.shallow_fusion_conv = nn.Sequential(
            nn.Conv3d(shallow_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )

        self.deep_fusion_conv = nn.Sequential(
            nn.Conv3d(deep_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )

        self.concat_relu = nn.ReLU()

        self.attention_conv_bn_relu = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.final_conv_bn = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels)
        )

        self.sigmoid = nn.Sigmoid()

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, shallow_fusion, deep_fusion):
        shallow_fusion = self.shallow_fusion_conv(shallow_fusion)
        deep_fusion = self.deep_fusion_conv(deep_fusion)

        # 初始化一个空的张量来存储插值结果
        # 由于 deep_fusion 的深度维度为 1，我们不需要额外的深度维度来存储结果
        # 初始化一个空的张量来存储插值结果
        # 我们需要创建一个与 shallow_fusion 形状相同的张量来存储结果
        deep_fusion_resized = torch.empty_like(shallow_fusion)

        # 对 deep_fusion 的每个深度切片进行插值
        # 由于 deep_fusion 的深度维度只有 1 个元素，我们只需要处理这一个切片
        deep_fusion_slice = deep_fusion[:, :, 0, :, :]  # 提取深度为 0 的切片，形状为 (2, 512, 14, 14)
        deep_fusion_resized_slice = nn.functional.interpolate(deep_fusion_slice, size=shallow_fusion.size()[3:], mode='bilinear',
                                                  align_corners=False)  # 对切片进行插值，形状变为 (2, 512, 28, 28)

        # 由于 deep_fusion 只有一个深度切片，我们需要决定如何处理这个结果以匹配 shallow_fusion 的深度维度
        # 一种简单的方法是复制这个插值结果以填充整个深度维度
        # 注意：这种方法可能不是您想要的，因为它没有利用 shallow_fusion 的额外深度信息
        deep_fusion_resized[:, :, 0, :, :] = deep_fusion_resized_slice  # 将插值结果复制到深度为 0 的位置
        deep_fusion_resized[:, :, 1, :, :] = deep_fusion_resized_slice.clone()  # 复制插值结果到深度为 1 的位置（这里只是简单复制，可能不是最佳做法）

        concatenated = torch.cat((shallow_fusion, deep_fusion_resized), dim=1)
        concatenated = self.concat_relu(concatenated)

        attention_output = self.attention_conv_bn_relu(concatenated)
        attention_weights = self.sigmoid(attention_output)

        weighted_shallow_fusion = shallow_fusion * attention_weights

        weighted_shallow_fusion = self.fusion_conv(weighted_shallow_fusion)

        deep_fusion = deep_fusion_resized + weighted_shallow_fusion

        return deep_fusion

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm3d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose3d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm3d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm3d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

@MODELS.register_module()
class MultiDecoderHead(BaseDecodeHead):
    def __init__(self,num_classes, in_index=-1,**kwargs):
        meaningless_value = 8
        super().__init__(in_channels=meaningless_value,
                         channels=meaningless_value,
                         num_classes=num_classes,
                         in_index=in_index,
                         **kwargs)

        filters = [64, 256, 512, 1024]

        self.multi_modal_process1 = MultiModalProcess(filters[2], filters[3], filters[2])
        self.multi_modal_process2 = MultiModalProcess(filters[1], filters[2], filters[1])
        self.multi_modal_process3 = MultiModalProcess(filters[0], filters[1], filters[0])

        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])

        self.finaldeconv1 = nn.ConvTranspose3d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv3d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv3d(32, self.num_classes, 2, padding=1)
        pass
    def forward(self,x):
        e02, e12, e22, e33 = x
        x3 = self.multi_modal_process1(e22, e33)
        d3 = self.decoder3(e33) + x3

        x2 = self.multi_modal_process2(e12, d3)
        l0 = self.decoder2(d3)
        l0 = l0[:, :, :3, :, :]
        d2 = l0 + x2

        x1 = self.multi_modal_process3(e02, d2)
        d1 = self.decoder1(d2) + x1
        fuse = self.finaldeconv1(d1)
        fuse = self.finalrelu1(fuse)
        fuse = self.finalconv2(fuse)
        fuse = self.finalrelu2(fuse)
        fuse = self.finalconv3(fuse)
        fuse = torch.mean(fuse, dim=2, keepdim=False)
        return fuse