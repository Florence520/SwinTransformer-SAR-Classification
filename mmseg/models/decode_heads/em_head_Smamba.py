import torch.nn as nn
import torch
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Sequence, Tuple, Union, Optional
from monai.networks.blocks.dynunet_block import get_conv_layer, UnetResBlock,UnetBasicBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
# from mamba_ssm import Mamba
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..utils.SMamba import StructureAwareSSM
from fvcore.nn.weight_init import c2_msra_fill
from einops import rearrange


# @MODELS.register_module()
# class EMHead_Smamba(BaseDecodeHead):
#     def __init__(self,
#                  num_classes=8,
#                  hidden_size: int = 288,
#                  out_chans=8,
#                 #  feat_size=[48, 96, 192, 384],
#                  feat_size=[36, 72, 144, 288],
#                  ds: bool = False,
#                  spatial_dims=3,
#                  norm_name="instance",
#                  res_block: bool = True,
#                  conv_decoder: bool = False,
#                  in_shpae=[224, 224, 16],
#                  **kwargs):
#         meaningless_value = 8
#         super().__init__(in_channels=meaningless_value,
#                          num_classes=meaningless_value,
#                          channels=meaningless_value,
#                          **kwargs)
#         self.hidden_size = hidden_size
#         self.out_chans = out_chans
#         self.feat_size = feat_size
#         self.ds = ds  # Deep supervision
#         self.spatial_dims = spatial_dims

#         self.use_conv = conv_decoder
#         if self.use_conv:
#             self.decoder4 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=hidden_size,
#                 out_channels=self.feat_size[2],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.decoder3 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[2],
#                 out_channels=self.feat_size[1],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.decoder2 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[1],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.decoder1 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[0],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 upsample_kernel_size=4,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#         else:
#             self.decoder4 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.hidden_size,
#                 out_channels=self.feat_size[2],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 in_shape=in_shpae,
#                 stage=3,
#             )
#             self.decoder3 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[2],
#                 out_channels=self.feat_size[1],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 stage=2,
#                 in_shape=in_shpae,
#             )
#             self.decoder2 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[1],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 stage=1,
#                 in_shape=in_shpae,
#             )
#             self.decoder1 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[0],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 # upsample_kernel_size=2,
#                 upsample_kernel_size=(2,4,4),
#                 norm_name=norm_name,
#                 res_block=False,
#                 stage=0,
#                 in_shape=in_shpae,
#             )

#         self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0],
#                                 out_channels=self.out_chans)  # Final outputs, stage 1
#         if self.ds:
#             self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
#             self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=self.out_chans)

#     def forward(self, x):
#         outs,enc1 = x
#         ### Normal skip connections
#         dec3 = self.decoder4(outs[3], outs[2])
#         dec2 = self.decoder3(dec3, outs[1])
#         dec1 = self.decoder2(dec2, outs[0])

#         # b,48,96,96,96 -> b,48,96,96,96
#         out = self.decoder1(dec1, enc1)
#         # b,48,96,96,96 -> b,out_c,96,96,96
#         if self.ds:
#             logits = [torch.mean(self.out(out), dim=2, keepdim=False), torch.mean(self.out2(dec1), dim=2, keepdim=False), torch.mean(self.out3(dec2), dim=2, keepdim=False)]
#         else:
#             logits = torch.mean(self.out(out), dim=2, keepdim=False)
#         # logits = torch.mean(logits, dim=2, keepdim=False)
#         return logits


# class EMNetUpBlock(nn.Module):
#     """
#     An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
#     UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
#     """

#     def __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             upsample_kernel_size: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             res_block: bool = False,
#             stage: int = 1,
#             in_shape: list = [224, 224, 16],
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: convolution kernel size.
#             upsample_kernel_size: convolution kernel size for transposed convolution layers.
#             norm_name: feature normalization type and arguments.
#             res_block: bool argument to determine if residual block is used.

#         """

#         super().__init__()
#         # print("<<<<< this is new up block >>>>>>>>>")
#         upsample_stride = upsample_kernel_size
#         self.transp_conv = get_conv_layer(
#             spatial_dims,
#             in_channels,
#             out_channels,
#             kernel_size=upsample_kernel_size,
#             stride=upsample_stride,
#             conv_only=True,
#             is_transposed=True,
#         )
#         self.decoder_block = nn.ModuleList()

#         if res_block:
#             stage_blocks = []
#             for i in range(3):
#                 stage_blocks.append(MambaLayer(dim=out_channels, stage=stage,
#                                                d_state=16, d_conv=4, expand=2,
#                                                pos_embed=True, input_shpae=in_shape))
#             self.decoder_block.append(nn.Sequential(*stage_blocks))
#         else:
#             conv_block = UnetBasicBlock(  # type: ignore
#                 spatial_dims,
#                 out_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 norm_name=norm_name,
#             )
#             self.decoder_block.append(conv_block)

#     def forward(self, inp, skip):
#         # number of channels for skip should equals to out_channels
#         out = self.transp_conv(inp)
#         out = out + skip
#         out = self.decoder_block[0](out)
#         return out


# class MambaLayer(nn.Module):
#     def __init__(self, dim, stage=1, d_state=16, d_conv=4, expand=2, pos_embed=True, input_shpae=[224, 224, 16]):
#         super().__init__()
#         self.dim = dim
#         self.norm = nn.LayerNorm(dim)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mamba = StructureAwareSSM(d_model=dim, dropout=0.1, d_state=8,expand=1)
#         x = input_shpae[0] // 2 ** (stage) // 2
#         y = input_shpae[1] // 2 ** (stage) // 2
#         z = input_shpae[2] // 2 ** (stage)
#         if pos_embed:
#             self.pos_embed = nn.Parameter(torch.zeros(1, x * y * z, self.dim))  # Temp n*n*n/2

#         self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
#         # self.res_gate = EnhancedGatedResidual(dim)  # 替换原有门控
#         # self.TCA_module = TCA(dim)
#         self.TCP_att_module = TCP_att_module(dim=dim,frame=z)

#         self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim // 4)
#         self.conv51 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
#         # self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

#     def forward(self, x):
#         B, C, D, H, W = x.shape
#         img_dims = (D, H, W)
#         assert C == self.dim
#         n_tokens = x.shape[2:].numel()
#         img_dims = x.shape[2:]
#         x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

#         if self.pos_embed is not None:
#             x_flat = x_flat + self.pos_embed
#         x_norm = self.norm(x_flat)
#         x_restored = x_norm.reshape(B, *img_dims, C)
#         x_norm = x_restored.permute(0, 2, 3, 1, 4)
#         x_mamba = self.mamba(x_norm)
#         x_mamba = x_mamba.reshape(B, H * W * D, C)
#         x_mamba = self.norm1(x_mamba)
#         x_spatial = self.norm2(
#             self.mlp(x_norm.transpose(-1, -2).reshape(B, C, *img_dims)).reshape(B, C, n_tokens).transpose(-1, -2))

#         out = x_flat + self.gamma * (x_mamba + x_spatial)
#         out = out.transpose(-1, -2).reshape(B, C, *img_dims)

#         # attn = self.conv51(out)
#         # out = out + self.conv8(attn)
#         # out = self.TCA_module(out)  # 增强门控直接处理out
#         attn = self.conv51(out)
#         attn = attn.permute(0,2,1,3,4)
#         attn = rearrange(attn, "b d c h w -> (b d) c h w")
#         attn = self.TCP_att_module(attn)
#         attn = rearrange(attn, "(b d) c h w -> b d c h w",b=B,d=D)
#         attn = attn.permute(0,2,1,3,4)
#         out = out + attn
#         return out

# class MlpChannel(nn.Module):
#     def __init__(self,hidden_size, mlp_dim, ):
#         super().__init__()
#         self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
#         self.act = nn.GELU()
#         self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x


# class EnhancedGatedResidual(nn.Module):
#     def __init__(self, dim, reduction_ratio=4):
#         super().__init__()
#         self.dim = dim
        
#         # 多尺度特征提取分支
#         self.conv3x3 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim//4)
#         self.conv5x5 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim//4)
        
#         # 通道注意力（类似SE模块）
#         self.channel_gate = nn.Sequential(
#             nn.AdaptiveAvgPool3d(1),
#             nn.Conv3d(dim, dim//reduction_ratio, 1),
#             nn.ReLU(),
#             nn.Conv3d(dim//reduction_ratio, dim, 1),
#             nn.Sigmoid()
#         )
        
#         # 空间注意力
#         self.spatial_gate = nn.Sequential(
#             nn.Conv3d(2, 1, kernel_size=7, padding=3),  # 输入通道2（mean+max）
#             nn.Sigmoid()
#         )
        
#         # 门控融合层
#         self.fusion = nn.Sequential(
#             nn.Conv3d(dim*3, dim, kernel_size=1),  # 融合3个分支
#             nn.GroupNorm(num_groups=1, num_channels=dim),
#             nn.GELU()
#         )

#         # 最终门控输出
#         self.output_gate = nn.Sequential(
#             nn.Conv3d(dim, dim, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 多尺度特征提取
#         feat3 = self.conv3x3(x)
#         feat5 = self.conv5x5(x)
        
#         # 通道注意力
#         channel_weight = self.channel_gate(x)
        
#         # 空间注意力
#         mean_max = torch.cat([
#             x.mean(dim=1, keepdim=True),
#             x.max(dim=1, keepdim=True)[0]
#         ], dim=1)
#         spatial_weight = self.spatial_gate(mean_max)
        
#         # 特征融合
#         fused = self.fusion(
#             torch.cat([
#                 feat3 * channel_weight,
#                 feat5 * spatial_weight,
#                 x
#             ], dim=1)
#         )
        
#         # 动态门控输出
#         gate = self.output_gate(fused)
#         return x + gate * fused

# # class TCA(nn.Module):
# #     def __init__(self, dim):
# #         super(TCA, self).__init__()
# #         self.dim = dim

# #         self.g1 = nn.Conv2d(
# #             self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
# #         )
# #         self.g2 = nn.Conv2d(
# #             self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
# #         )
# #         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

# #         self.sigmoid = nn.Sigmoid()
# #         self.relu = nn.ReLU(inplace=True)
# #         self.conv_1d = nn.Conv3d(
# #                 self.dim, self.dim, kernel_size=(3,1,1),
# #                 stride=1, padding=(1,0,0),
# #                 padding_mode='circular'
# #             )
# #         self.relu_out = nn.ReLU(inplace=True)

# #     def forward(self, x):
# #         B,C,D,H_in,W_in = x.shape
# #         x_in = x.clone()
# #         x = x.permute(0, 2, 1, 3, 4).reshape(B*D, C, H_in, W_in)
# #         x_mid = x.clone()
# #         x = self.pool(x)
# #         [N, C, H, W] = x.size()
# #         V = N // D
# #         x_t = x.view(V, D, C, H, W)
# #         x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
# #         x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

# #         x_diff1 = x_t - x_t1
# #         x_diff1 = x_diff1.view(N, C, H, W)
# #         x_diff2 = (x_t - x_t2)*0.5
# #         x_diff2 = x_diff2.view(N, C, H, W)

# #         x_diff1 = self.TCA_diff(x_diff1)
# #         x_diff2 = self.TCA_diff(x_diff2)
# #         out = x_mid * (x_diff1  + x_diff2)*0.5
# #         out = out.reshape(B, D, C, H_in, W_in).permute(0, 2, 1, 3, 4)
# #         out = self.conv_1d(out)
# #         return  self.relu_out(x_in + out)


# #     def TCA_diff(self, x):
# #         x = self.g1(x)
# #         x = self.relu(x)
# #         x = self.g2(x)
# #         x = self.sigmoid(x)
# #         return x

# class TSA(nn.Module):

#     def __init__(
#         self,
#         dim,
#         frame=8,
#         instantiation="softmax",
#         norm_eps=1e-5,
#         norm_momentum=0.1,
#         norm_module=nn.BatchNorm2d,
#     ):
#         """
#         Args:
#             dim (int): number of dimension for the input.
#             frame (int): frame number.
#             instantiation (string): supports two different instantiation method:
#                 "dot_product": normalizing correlation matrix with L2.
#                 "softmax": normalizing correlation matrix with Softmax.
#             norm_module (nn.Module): nn.Module for the normalization layer. The
#                 default is nn.BatchNorm2d.
#         """
#         super(TSA, self).__init__()
#         self.dim = dim
#         self.dim = dim
#         self.frame = frame
#         self.instantiation = instantiation
#         self.norm_eps = norm_eps
#         self.norm_momentum = norm_momentum
#         self._construct_TSA(
#             norm_module
#         )

#     def _construct_TSA(
#         self, norm_module
#     ):
#         # Three convolution : conv_phi0 (x), conv_phi_1 (x-1), and conv_phi_2 (x-2).
#         self.conv_phi_1 = nn.Conv2d(
#             self.dim, self.dim, kernel_size=1, stride=1, padding=0
#         )
#         self.conv_phi_2 = nn.Conv2d(
#             self.dim, self.dim, kernel_size=1, stride=1, padding=0
#         )
#         self.conv_phi0 = nn.Conv2d(
#             self.dim, self.dim, kernel_size=1, stride=1, padding=0
#         )

#         # TODO: change the name to `norm`
#         self.norm = norm_module(
#             self.dim,
#             eps=self.norm_eps,
#         )

#         self.init_weights()


#     def init_weights(self):
#             """
#             Performs ResNet style weight initialization.
#             """
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     """
#                     Follow the initialization method proposed in:
#                     {He, Kaiming, et al.
#                     "Delving deep into rectifiers: Surpassing human-level
#                     performance on imagenet classification."
#                     arXiv preprint arXiv:1502.01852 (2015)}
#                     """
#                     c2_msra_fill(m)
#                 elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
#                     batchnorm_weight = 0.
#                     if m.weight is not None:
#                         m.weight.data.fill_(batchnorm_weight)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 if isinstance(m, nn.Linear):
#                     m.weight.data.normal_(mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         m.bias.data.zero_()

#     def forward(self, x):
#         x_identity = x
#         N, C, H, W = x.size()

#         T = self.frame
#         V = N // T
#         x = x.view(V, T, C, H, W)
#         x_1 = torch.cat((x[:,0,:,:,:].unsqueeze(1),x[:,:-1,:,:,:]), 1)
#         x_2 = torch.cat((x_1[:, 0, :, :, :].unsqueeze(1), x_1[:, :-1, :, :, :]), 1)
#         x_1 = x_1.view(N, C, H, W)
#         x_2 = x_2.view(N, C, H, W)
#         x = x.view(-1, C, H, W)

#         phi_x_1 = self.conv_phi_1(x_1)
#         phi_x_2 = self.conv_phi_2(x_2)
#         phi_x0 = self.conv_phi0(x)

#         phi_x_1 = phi_x_1.view(N, self.dim, -1)
#         phi_x_2 = phi_x_2.view(N, self.dim, -1)
#         phi_x0 = phi_x0.view(N, self.dim, -1)

#         # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW).
#         theta_x_1_2 = torch.einsum("nca,ncb->nab", (phi_x_1, phi_x_2))
#         # For original Non-local paper, there are two main ways to normalize
#         # the affinity tensor:
#         #   1) Softmax normalization (norm on exp).
#         #   2) dot_product normalization.
#         if self.instantiation == "softmax":
#             # Normalizing the affinity tensor theta_x_1_2 before softmax.
#             theta_x_1_2 = theta_x_1_2 * (self.dim ** -0.5)
#             theta_x_1_2 = nn.functional.softmax(theta_x_1_2, dim=2)
#         elif self.instantiation == "dot_product":
#             spatial_temporal_dim = theta_x_1_2.shape[2]
#             theta_x_1_2 = theta_x_1_2 / spatial_temporal_dim
#         else:
#             raise NotImplementedError(
#                 "Unknown norm type {}".format(self.instantiation)
#             )

#         # (N, HxW, HxW) * (N, C, HxW) => (N, C, HxW).
#         x_out = torch.einsum("ntg,ncg->nct", (theta_x_1_2, phi_x0))

#         # (N, C, HxW) => (N, C, H, W).
#         x_out = x_out.contiguous().view(N, self.dim, H, W)

#         x_out = self.norm(x_out)
#         return x_identity + x_out

# class TCA(nn.Module):
#     def __init__(self, dim, frame):
#         super(TCA, self).__init__()
#         self.dim = dim
#         self.frame = frame

#         print("-" * 20 + "TCA called" + '-' * 20)
#         self.g1 = nn.Conv2d(
#             self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
#         )
#         self.g2 = nn.Conv2d(
#             self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
#         )

#         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)


#     def forward(self, x, x_att):
#         x = self.pool(x)
#         [N, C, H, W] = x.size()
#         V = N // self.frame
#         x_t = x.view(V, self.frame, C, H, W)
#         x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
#         x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

#         x_diff1 = x_t - x_t1
#         x_diff1 = x_diff1.view(N, C, H, W)
#         x_diff2 = (x_t - x_t2)*0.5
#         x_diff2 = x_diff2.view(N, C, H, W)

#         x_diff1 = self.TCA_diff(x_diff1)
#         x_diff2 = self.TCA_diff(x_diff2)

#         return  x_att * (x_diff1  + x_diff2)*0.5


#     def TCA_diff(self, x):
#         x = self.g1(x)
#         x = self.relu(x)
#         x = self.g2(x)
#         x = self.sigmoid(x)
#         return x

# class TCP_att_module(nn.Module):

#     def __init__(
#         self,
#         dim,
#         frame=8,
#         ch_flag=True,
#         sp_flag=True,
#         conv_1d_flag=True,
#     ):
#         """
#         Args:
#             dim (int): TCP_dim.
#             frame (int): frame number.
#             ch_flag (bool): whether use temporal-based channel attention, default: True.
#             sp_flag (bool): whether use temporal-based spatial attention, default: True.
#             conv_1d_flag (bool): whether use temporal convolution, default: True.

#         """
#         super(TCP_att_module, self).__init__()
#         self.dim = dim
#         self.frame = frame
#         self.ch_flag = ch_flag
#         self.sp_flag = sp_flag
#         self.conv_1d_flag = conv_1d_flag
#         self.k = frame // 2 + 1 #temporal conv kernel
#         if self.k % 2 == 0:  # 若k为偶数，加1变为奇数
#             self.k += 1
#         self._construct_TCP_att(
#         )

#     def _construct_TCP_att(self
#     ):
#         self.relu = nn.ReLU(inplace=True)

#         if self.ch_flag:
#             self.TCA = TCA(self.dim, self.frame)

#         if self.conv_1d_flag :
#             k_padding = (self.k-1)//2
#             self.conv_1d = nn.Conv3d(
#                 self.dim, self.dim, kernel_size=(self.k,1,1),
#                 stride=1, padding=(k_padding,0,0),
#                 padding_mode='circular'
#             )

#         if self.sp_flag:
#             print("-" * 20 + "TSA Module called" + '-' * 20)
#             self.TSA = TSA(self.dim, self.frame)

#         self.init_weights()



#     def init_weights(self):
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#                     #c2_msra_fill(m)
#                     m.weight.data.normal_(mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d) \
#                         or isinstance(m, nn.BatchNorm3d):
#                     batchnorm_weight = 0.
#                     if m.weight is not None:
#                         m.weight.data.fill_(batchnorm_weight)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 if isinstance(m, nn.Linear):
#                     m.weight.data.normal_(mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#             if self.conv_1d_flag:
#                 self.conv_1d.weight.data.fill_(0.)
#             if self.sp_flag:
#                 self.TSA.init_weights()

#     def forward(self, x):
#         if self.sp_flag:
#             x_att = self.TSA(x)
#         else:
#             x_att = x

#         if self.ch_flag:
#             x_att = self.TCA(x, x_att)

#         if self.conv_1d_flag:
#             [N, C, H, W] = x_att.size()
#             V = N // self.frame
#             x_att = x_att.reshape(V, self.frame, C, H, W).permute(0,2,1,3,4)
#             x_att = self.conv_1d(x_att)
#             x_att = x_att.permute(0,2,1,3,4,).reshape(N, C, H, W)
#             return self.relu(x_att + x)
#         else :
#             return x_att  # following SE, w/o shortcut





@MODELS.register_module()
class EMHead_Smamba(BaseDecodeHead):
    def __init__(self,
                 num_classes=8,
                 hidden_size: int = 288,
                 out_chans=8,
                #  feat_size=[48, 96, 192, 384],
                 feat_size=[36, 72, 144, 288],
                 ds: bool = False,
                 spatial_dims=3,
                 norm_name="instance",
                 res_block: bool = True,
                 conv_decoder: bool = False,
                 in_shpae=[224, 224, 16],
                 **kwargs):
        meaningless_value = 18
        super().__init__(in_channels=meaningless_value,
                         num_classes=meaningless_value,
                         channels=meaningless_value,
                         **kwargs)
        self.hidden_size = hidden_size
        self.out_chans = out_chans
        self.feat_size = feat_size
        self.ds = ds  # Deep supervision
        self.spatial_dims = spatial_dims

        self.use_conv = conv_decoder
        if self.use_conv:
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=4,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.decoder4 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=self.feat_size[2],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                in_shape=in_shpae,
                stage=3,
            )
            self.decoder3 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.feat_size[1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=2,
                in_shape=in_shpae,
            )
            self.decoder2 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.feat_size[0],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                stage=1,
                in_shape=in_shpae,
            )
            self.decoder1 = EMNetUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.feat_size[0],
                kernel_size=3,
                # upsample_kernel_size=2,
                upsample_kernel_size=(8,2,2),
                norm_name=norm_name,
                res_block=False,
                stage=0,
                in_shape=in_shpae,
            )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0],
                                out_channels=self.out_chans)  # Final outputs, stage 1
        if self.ds:
            self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
            self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=self.out_chans)

    def forward(self, x):
        outs,enc1 = x
        ### Normal skip connections
        dec3 = self.decoder4(outs[3], outs[2])
        dec2 = self.decoder3(dec3, outs[1])
        dec1 = self.decoder2(dec2, outs[0])

        # b,48,96,96,96 -> b,48,96,96,96
        out = self.decoder1(dec1, enc1)
        # b,48,96,96,96 -> b,out_c,96,96,96
        if self.ds:
            logits = [torch.mean(self.out(out), dim=2, keepdim=False), torch.mean(self.out2(dec1), dim=2, keepdim=False), torch.mean(self.out3(dec2), dim=2, keepdim=False)]
        else:
            logits = torch.mean(self.out(out), dim=2, keepdim=False)
        # logits = torch.mean(logits, dim=2, keepdim=False)
        return logits


class EMNetUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            res_block: bool = False,
            stage: int = 1,
            in_shape: list = [224, 224, 16],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        # print("<<<<< this is new up block >>>>>>>>>")
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        self.decoder_block = nn.ModuleList()

        if res_block:
            stage_blocks = []
            for i in range(3):
                stage_blocks.append(MambaLayer(dim=out_channels, stage=stage,
                                               d_state=16, d_conv=4, expand=2,
                                               pos_embed=True, input_shpae=in_shape))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        else:
            conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
            self.decoder_block.append(conv_block)

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)
        return out


class MambaLayer(nn.Module):
    def __init__(self, dim, stage=1, d_state=16, d_conv=4, expand=2, pos_embed=True, input_shpae=[224, 224, 16]):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = StructureAwareSSM(d_model=dim, dropout=0.1, d_state=8,expand=1)
        x = input_shpae[0] // 2 ** (stage)
        y = input_shpae[1] // 2 ** (stage)
        z = input_shpae[2] // 2 ** (stage+2)
#         x = input_shpae[0] // (2 ** (stage - 1)) if stage > 0 else input_shpae[0]
#         y = input_shpae[1] // (2 ** (stage - 1)) if stage > 0 else input_shpae[1]
#         z = input_shpae[2] // 2 ** (stage+1)  # 保持z维度的计算方式
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, x * y * z, self.dim))  # Temp n*n*n/2

        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
        # self.res_gate = EnhancedGatedResidual(dim)  # 替换原有门控
        # self.TCA_module = TCA(dim)
        self.TCP_att_module = TCP_att_module(dim=dim,frame=z)

        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim // 4)
        self.conv51 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
        # self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        img_dims = (D, H, W)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        if self.pos_embed is not None:
            x_flat = x_flat + self.pos_embed
        x_norm = self.norm(x_flat)
        x_restored = x_norm.reshape(B, *img_dims, C)
        x_norm = x_restored.permute(0, 2, 3, 1, 4)
        x_mamba = self.mamba(x_norm)
        x_mamba = x_mamba.reshape(B, H * W * D, C)
        x_mamba = self.norm1(x_mamba)
        x_spatial = self.norm2(
            self.mlp(x_norm.transpose(-1, -2).reshape(B, C, *img_dims)).reshape(B, C, n_tokens).transpose(-1, -2))

        out = x_flat + self.gamma * (x_mamba + x_spatial)
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)

        # attn = self.conv51(out)
        # out = out + self.conv8(attn)
        # out = self.TCA_module(out)  # 增强门控直接处理out
        attn = self.conv51(out)
        attn = attn.permute(0,2,1,3,4)
        attn = rearrange(attn, "b d c h w -> (b d) c h w")
        attn = self.TCP_att_module(attn)
        attn = rearrange(attn, "(b d) c h w -> b d c h w",b=B,d=D)
        attn = attn.permute(0,2,1,3,4)
        out = out + attn
        return out

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class EnhancedGatedResidual(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        
        # 多尺度特征提取分支
        self.conv3x3 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim//4)
        self.conv5x5 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim//4)
        
        # 通道注意力（类似SE模块）
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim//reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv3d(dim//reduction_ratio, dim, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),  # 输入通道2（mean+max）
            nn.Sigmoid()
        )
        
        # 门控融合层
        self.fusion = nn.Sequential(
            nn.Conv3d(dim*3, dim, kernel_size=1),  # 融合3个分支
            nn.GroupNorm(num_groups=1, num_channels=dim),
            nn.GELU()
        )

        # 最终门控输出
        self.output_gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        
        # 通道注意力
        channel_weight = self.channel_gate(x)
        
        # 空间注意力
        mean_max = torch.cat([
            x.mean(dim=1, keepdim=True),
            x.max(dim=1, keepdim=True)[0]
        ], dim=1)
        spatial_weight = self.spatial_gate(mean_max)
        
        # 特征融合
        fused = self.fusion(
            torch.cat([
                feat3 * channel_weight,
                feat5 * spatial_weight,
                x
            ], dim=1)
        )
        
        # 动态门控输出
        gate = self.output_gate(fused)
        return x + gate * fused

# class TCA(nn.Module):
#     def __init__(self, dim):
#         super(TCA, self).__init__()
#         self.dim = dim

#         self.g1 = nn.Conv2d(
#             self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
#         )
#         self.g2 = nn.Conv2d(
#             self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
#         )
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_1d = nn.Conv3d(
#                 self.dim, self.dim, kernel_size=(3,1,1),
#                 stride=1, padding=(1,0,0),
#                 padding_mode='circular'
#             )
#         self.relu_out = nn.ReLU(inplace=True)

#     def forward(self, x):
#         B,C,D,H_in,W_in = x.shape
#         x_in = x.clone()
#         x = x.permute(0, 2, 1, 3, 4).reshape(B*D, C, H_in, W_in)
#         x_mid = x.clone()
#         x = self.pool(x)
#         [N, C, H, W] = x.size()
#         V = N // D
#         x_t = x.view(V, D, C, H, W)
#         x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
#         x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

#         x_diff1 = x_t - x_t1
#         x_diff1 = x_diff1.view(N, C, H, W)
#         x_diff2 = (x_t - x_t2)*0.5
#         x_diff2 = x_diff2.view(N, C, H, W)

#         x_diff1 = self.TCA_diff(x_diff1)
#         x_diff2 = self.TCA_diff(x_diff2)
#         out = x_mid * (x_diff1  + x_diff2)*0.5
#         out = out.reshape(B, D, C, H_in, W_in).permute(0, 2, 1, 3, 4)
#         out = self.conv_1d(out)
#         return  self.relu_out(x_in + out)


#     def TCA_diff(self, x):
#         x = self.g1(x)
#         x = self.relu(x)
#         x = self.g2(x)
#         x = self.sigmoid(x)
#         return x

class TSA(nn.Module):

    def __init__(
        self,
        dim,
        frame=8,
        instantiation="softmax",
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim (int): number of dimension for the input.
            frame (int): frame number.
            instantiation (string): supports two different instantiation method:
                "dot_product": normalizing correlation matrix with L2.
                "softmax": normalizing correlation matrix with Softmax.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm2d.
        """
        super(TSA, self).__init__()
        self.dim = dim
        self.dim = dim
        self.frame = frame
        self.instantiation = instantiation
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_TSA(
            norm_module
        )

    def _construct_TSA(
        self, norm_module
    ):
        # Three convolution : conv_phi0 (x), conv_phi_1 (x-1), and conv_phi_2 (x-2).
        self.conv_phi_1 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi_2 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi0 = nn.Conv2d(
            self.dim, self.dim, kernel_size=1, stride=1, padding=0
        )

        # TODO: change the name to `norm`
        self.norm = norm_module(
            self.dim,
            eps=self.norm_eps,
        )

        self.init_weights()


    def init_weights(self):
            """
            Performs ResNet style weight initialization.
            """
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    """
                    Follow the initialization method proposed in:
                    {He, Kaiming, et al.
                    "Delving deep into rectifiers: Surpassing human-level
                    performance on imagenet classification."
                    arXiv preprint arXiv:1502.01852 (2015)}
                    """
                    c2_msra_fill(m)
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    batchnorm_weight = 0.
                    if m.weight is not None:
                        m.weight.data.fill_(batchnorm_weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        x_identity = x
        N, C, H, W = x.size()

        T = self.frame
        V = N // T
        x = x.view(V, T, C, H, W)
        x_1 = torch.cat((x[:,0,:,:,:].unsqueeze(1),x[:,:-1,:,:,:]), 1)
        x_2 = torch.cat((x_1[:, 0, :, :, :].unsqueeze(1), x_1[:, :-1, :, :, :]), 1)
        x_1 = x_1.view(N, C, H, W)
        x_2 = x_2.view(N, C, H, W)
        x = x.view(-1, C, H, W)

        phi_x_1 = self.conv_phi_1(x_1)
        phi_x_2 = self.conv_phi_2(x_2)
        phi_x0 = self.conv_phi0(x)

        phi_x_1 = phi_x_1.view(N, self.dim, -1)
        phi_x_2 = phi_x_2.view(N, self.dim, -1)
        phi_x0 = phi_x0.view(N, self.dim, -1)

        # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW).
        theta_x_1_2 = torch.einsum("nca,ncb->nab", (phi_x_1, phi_x_2))
        # For original Non-local paper, there are two main ways to normalize
        # the affinity tensor:
        #   1) Softmax normalization (norm on exp).
        #   2) dot_product normalization.
        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_x_1_2 before softmax.
            theta_x_1_2 = theta_x_1_2 * (self.dim ** -0.5)
            theta_x_1_2 = nn.functional.softmax(theta_x_1_2, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_x_1_2.shape[2]
            theta_x_1_2 = theta_x_1_2 / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, HxW, HxW) * (N, C, HxW) => (N, C, HxW).
        x_out = torch.einsum("ntg,ncg->nct", (theta_x_1_2, phi_x0))

        # (N, C, HxW) => (N, C, H, W).
        x_out = x_out.contiguous().view(N, self.dim, H, W)

        x_out = self.norm(x_out)
        return x_identity + x_out

class TCA(nn.Module):
    def __init__(self, dim, frame):
        super(TCA, self).__init__()
        self.dim = dim
        self.frame = frame

        print("-" * 20 + "TCA called" + '-' * 20)
        self.g1 = nn.Conv2d(
            self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
        )
        self.g2 = nn.Conv2d(
            self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, x_att):
        x = self.pool(x)
        [N, C, H, W] = x.size()
        V = N // self.frame
        x_t = x.view(V, self.frame, C, H, W)
        x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
        x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

        x_diff1 = x_t - x_t1
        x_diff1 = x_diff1.view(N, C, H, W)
        x_diff2 = (x_t - x_t2)*0.5
        x_diff2 = x_diff2.view(N, C, H, W)

        x_diff1 = self.TCA_diff(x_diff1)
        x_diff2 = self.TCA_diff(x_diff2)

        return  x_att * (x_diff1  + x_diff2)*0.5


    def TCA_diff(self, x):
        x = self.g1(x)
        x = self.relu(x)
        x = self.g2(x)
        x = self.sigmoid(x)
        return x

class TCP_att_module(nn.Module):

    def __init__(
        self,
        dim,
        frame=8,
        ch_flag=True,
        sp_flag=True,
        conv_1d_flag=True,
    ):
        """
        Args:
            dim (int): TCP_dim.
            frame (int): frame number.
            ch_flag (bool): whether use temporal-based channel attention, default: True.
            sp_flag (bool): whether use temporal-based spatial attention, default: True.
            conv_1d_flag (bool): whether use temporal convolution, default: True.

        """
        super(TCP_att_module, self).__init__()
        self.dim = dim
        self.frame = frame
        self.ch_flag = ch_flag
        self.sp_flag = sp_flag
        self.conv_1d_flag = conv_1d_flag
        self.k = frame // 2 + 1 #temporal conv kernel
        if self.k % 2 == 0:  # 若k为偶数，加1变为奇数
            self.k += 1
        self._construct_TCP_att(
        )

    def _construct_TCP_att(self
    ):
        self.relu = nn.ReLU(inplace=True)

        if self.ch_flag:
            self.TCA = TCA(self.dim, self.frame)

        if self.conv_1d_flag :
            k_padding = (self.k-1)//2
            self.conv_1d = nn.Conv3d(
                self.dim, self.dim, kernel_size=(self.k,1,1),
                stride=1, padding=(k_padding,0,0),
                padding_mode='circular'
            )

        if self.sp_flag:
            print("-" * 20 + "TSA Module called" + '-' * 20)
            self.TSA = TSA(self.dim, self.frame)

        self.init_weights()



    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    #c2_msra_fill(m)
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d) \
                        or isinstance(m, nn.BatchNorm3d):
                    batchnorm_weight = 0.
                    if m.weight is not None:
                        m.weight.data.fill_(batchnorm_weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
            if self.conv_1d_flag:
                self.conv_1d.weight.data.fill_(0.)
            if self.sp_flag:
                self.TSA.init_weights()

    def forward(self, x):
        if self.sp_flag:
            x_att = self.TSA(x)
        else:
            x_att = x

        if self.ch_flag:
            x_att = self.TCA(x, x_att)

        if self.conv_1d_flag:
            [N, C, H, W] = x_att.size()
            V = N // self.frame
            x_att = x_att.reshape(V, self.frame, C, H, W).permute(0,2,1,3,4)
            x_att = self.conv_1d(x_att)
            x_att = x_att.permute(0,2,1,3,4,).reshape(N, C, H, W)
            return self.relu(x_att + x)
        else :
            return x_att  # following SE, w/o shortcut







# @MODELS.register_module()
# class EMHead_Smamba(BaseDecodeHead):
#     def __init__(self,
#                  num_classes=8,
#                  hidden_size: int = 288,
#                  out_chans=8,
#                 #  feat_size=[48, 96, 192, 384],
#                  feat_size=[36, 72, 144, 288],
#                  ds: bool = False,
#                  spatial_dims=3,
#                  norm_name="instance",
#                  res_block: bool = True,
#                  conv_decoder: bool = False,
#                  in_shpae=[224, 224, 16],
#                  **kwargs):
#         meaningless_value = 4
#         super().__init__(in_channels=meaningless_value,
#                          num_classes=meaningless_value,
#                          channels=meaningless_value,
#                          **kwargs)
#         self.hidden_size = hidden_size
#         self.out_chans = out_chans
#         self.feat_size = feat_size
#         self.ds = ds  # Deep supervision
#         self.spatial_dims = spatial_dims

#         self.use_conv = conv_decoder
#         if self.use_conv:
#             self.decoder4 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=hidden_size,
#                 out_channels=self.feat_size[2],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.decoder3 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[2],
#                 out_channels=self.feat_size[1],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.decoder2 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[1],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#             self.decoder1 = UnetrUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[0],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 upsample_kernel_size=4,
#                 norm_name=norm_name,
#                 res_block=res_block,
#             )
#         else:
#             self.decoder4 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.hidden_size,
#                 out_channels=self.feat_size[2],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 in_shape=in_shpae,
#                 stage=3,
#             )
#             self.decoder3 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[2],
#                 out_channels=self.feat_size[1],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 stage=2,
#                 in_shape=in_shpae,
#             )
#             self.decoder2 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[1],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 upsample_kernel_size=2,
#                 norm_name=norm_name,
#                 res_block=res_block,
#                 stage=1,
#                 in_shape=in_shpae,
#             )
#             self.decoder1 = EMNetUpBlock(
#                 spatial_dims=spatial_dims,
#                 in_channels=self.feat_size[0],
#                 out_channels=self.feat_size[0],
#                 kernel_size=3,
#                 # upsample_kernel_size=2,
#                 upsample_kernel_size=(4,1,1),
#                 norm_name=norm_name,
#                 res_block=False,
#                 stage=0,
#                 in_shape=in_shpae,
#             )

#         self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0],
#                                 out_channels=self.out_chans)  # Final outputs, stage 1
#         if self.ds:
#             self.out2 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=self.out_chans)
#             self.out3 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[1], out_channels=self.out_chans)

#     def forward(self, x):
#         outs,enc1 = x
#         ### Normal skip connections
#         dec3 = self.decoder4(outs[3], outs[2])
#         dec2 = self.decoder3(dec3, outs[1])
#         dec1 = self.decoder2(dec2, outs[0])

#         # b,48,96,96,96 -> b,48,96,96,96
#         out = self.decoder1(dec1, enc1)
#         # b,48,96,96,96 -> b,out_c,96,96,96
#         if self.ds:
#             logits = [torch.mean(self.out(out), dim=2, keepdim=False), torch.mean(self.out2(dec1), dim=2, keepdim=False), torch.mean(self.out3(dec2), dim=2, keepdim=False)]
#         else:
#             logits = torch.mean(self.out(out), dim=2, keepdim=False)
#         # logits = torch.mean(logits, dim=2, keepdim=False)
#         return logits


# class EMNetUpBlock(nn.Module):
#     """
#     An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
#     UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
#     """

#     def __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             upsample_kernel_size: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             res_block: bool = False,
#             stage: int = 1,
#             in_shape: list = [224, 224, 16],
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: convolution kernel size.
#             upsample_kernel_size: convolution kernel size for transposed convolution layers.
#             norm_name: feature normalization type and arguments.
#             res_block: bool argument to determine if residual block is used.

#         """

#         super().__init__()
#         # print("<<<<< this is new up block >>>>>>>>>")
#         upsample_stride = upsample_kernel_size
#         self.transp_conv = get_conv_layer(
#             spatial_dims,
#             in_channels,
#             out_channels,
#             kernel_size=upsample_kernel_size,
#             stride=upsample_stride,
#             conv_only=True,
#             is_transposed=True,
#         )
#         self.decoder_block = nn.ModuleList()

#         if res_block:
#             stage_blocks = []
#             for i in range(3):
#                 stage_blocks.append(MambaLayer(dim=out_channels, stage=stage,
#                                                d_state=16, d_conv=4, expand=2,
#                                                pos_embed=True, input_shpae=in_shape))
#             self.decoder_block.append(nn.Sequential(*stage_blocks))
#         else:
#             conv_block = UnetBasicBlock(  # type: ignore
#                 spatial_dims,
#                 out_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 norm_name=norm_name,
#             )
#             self.decoder_block.append(conv_block)

#     def forward(self, inp, skip):
#         # number of channels for skip should equals to out_channels
#         out = self.transp_conv(inp)
#         out = out + skip
#         out = self.decoder_block[0](out)
#         return out


# class MambaLayer(nn.Module):
#     def __init__(self, dim, stage=1, d_state=16, d_conv=4, expand=2, pos_embed=True, input_shpae=[224, 224, 16]):
#         super().__init__()
#         self.dim = dim
#         self.norm = nn.LayerNorm(dim)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mamba = StructureAwareSSM(d_model=dim, dropout=0.1, d_state=8,expand=1)
#         x = input_shpae[0] // (2 ** (stage - 1)) if stage > 0 else input_shpae[0]
#         y = input_shpae[1] // (2 ** (stage - 1)) if stage > 0 else input_shpae[1]
#         z = input_shpae[2] // 2 ** (stage+1)  # 保持z维度的计算方式
#         if pos_embed:
#             self.pos_embed = nn.Parameter(torch.zeros(1, x * y * z, self.dim))  # Temp n*n*n/2

#         self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)
#         # self.res_gate = EnhancedGatedResidual(dim)  # 替换原有门控
#         # self.TCA_module = TCA(dim)
#         self.TCP_att_module = TCP_att_module(dim=dim,frame=z)

#         self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim // 4)
#         self.conv51 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
#         # self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

#     def forward(self, x):
#         B, C, D, H, W = x.shape
#         img_dims = (D, H, W)
#         assert C == self.dim
#         n_tokens = x.shape[2:].numel()
#         img_dims = x.shape[2:]
#         x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

#         if self.pos_embed is not None:
#             x_flat = x_flat + self.pos_embed
#         x_norm = self.norm(x_flat)
#         x_restored = x_norm.reshape(B, *img_dims, C)
#         x_norm = x_restored.permute(0, 2, 3, 1, 4)
#         x_mamba = self.mamba(x_norm)
#         x_mamba = x_mamba.reshape(B, H * W * D, C)
#         x_mamba = self.norm1(x_mamba)
#         x_spatial = self.norm2(
#             self.mlp(x_norm.transpose(-1, -2).reshape(B, C, *img_dims)).reshape(B, C, n_tokens).transpose(-1, -2))

#         out = x_flat + self.gamma * (x_mamba + x_spatial)
#         out = out.transpose(-1, -2).reshape(B, C, *img_dims)

#         # attn = self.conv51(out)
#         # out = out + self.conv8(attn)
#         # out = self.TCA_module(out)  # 增强门控直接处理out
#         attn = self.conv51(out)
#         attn = attn.permute(0,2,1,3,4)
#         attn = rearrange(attn, "b d c h w -> (b d) c h w")
#         attn = self.TCP_att_module(attn)
#         attn = rearrange(attn, "(b d) c h w -> b d c h w",b=B,d=D)
#         attn = attn.permute(0,2,1,3,4)
#         out = out + attn
#         return out

# class MlpChannel(nn.Module):
#     def __init__(self,hidden_size, mlp_dim, ):
#         super().__init__()
#         self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
#         self.act = nn.GELU()
#         self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return x


# class EnhancedGatedResidual(nn.Module):
#     def __init__(self, dim, reduction_ratio=4):
#         super().__init__()
#         self.dim = dim
        
#         # 多尺度特征提取分支
#         self.conv3x3 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim//4)
#         self.conv5x5 = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim//4)
        
#         # 通道注意力（类似SE模块）
#         self.channel_gate = nn.Sequential(
#             nn.AdaptiveAvgPool3d(1),
#             nn.Conv3d(dim, dim//reduction_ratio, 1),
#             nn.ReLU(),
#             nn.Conv3d(dim//reduction_ratio, dim, 1),
#             nn.Sigmoid()
#         )
        
#         # 空间注意力
#         self.spatial_gate = nn.Sequential(
#             nn.Conv3d(2, 1, kernel_size=7, padding=3),  # 输入通道2（mean+max）
#             nn.Sigmoid()
#         )
        
#         # 门控融合层
#         self.fusion = nn.Sequential(
#             nn.Conv3d(dim*3, dim, kernel_size=1),  # 融合3个分支
#             nn.GroupNorm(num_groups=1, num_channels=dim),
#             nn.GELU()
#         )

#         # 最终门控输出
#         self.output_gate = nn.Sequential(
#             nn.Conv3d(dim, dim, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 多尺度特征提取
#         feat3 = self.conv3x3(x)
#         feat5 = self.conv5x5(x)
        
#         # 通道注意力
#         channel_weight = self.channel_gate(x)
        
#         # 空间注意力
#         mean_max = torch.cat([
#             x.mean(dim=1, keepdim=True),
#             x.max(dim=1, keepdim=True)[0]
#         ], dim=1)
#         spatial_weight = self.spatial_gate(mean_max)
        
#         # 特征融合
#         fused = self.fusion(
#             torch.cat([
#                 feat3 * channel_weight,
#                 feat5 * spatial_weight,
#                 x
#             ], dim=1)
#         )
        
#         # 动态门控输出
#         gate = self.output_gate(fused)
#         return x + gate * fused

# # class TCA(nn.Module):
# #     def __init__(self, dim):
# #         super(TCA, self).__init__()
# #         self.dim = dim

# #         self.g1 = nn.Conv2d(
# #             self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
# #         )
# #         self.g2 = nn.Conv2d(
# #             self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
# #         )
# #         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

# #         self.sigmoid = nn.Sigmoid()
# #         self.relu = nn.ReLU(inplace=True)
# #         self.conv_1d = nn.Conv3d(
# #                 self.dim, self.dim, kernel_size=(3,1,1),
# #                 stride=1, padding=(1,0,0),
# #                 padding_mode='circular'
# #             )
# #         self.relu_out = nn.ReLU(inplace=True)

# #     def forward(self, x):
# #         B,C,D,H_in,W_in = x.shape
# #         x_in = x.clone()
# #         x = x.permute(0, 2, 1, 3, 4).reshape(B*D, C, H_in, W_in)
# #         x_mid = x.clone()
# #         x = self.pool(x)
# #         [N, C, H, W] = x.size()
# #         V = N // D
# #         x_t = x.view(V, D, C, H, W)
# #         x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
# #         x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

# #         x_diff1 = x_t - x_t1
# #         x_diff1 = x_diff1.view(N, C, H, W)
# #         x_diff2 = (x_t - x_t2)*0.5
# #         x_diff2 = x_diff2.view(N, C, H, W)

# #         x_diff1 = self.TCA_diff(x_diff1)
# #         x_diff2 = self.TCA_diff(x_diff2)
# #         out = x_mid * (x_diff1  + x_diff2)*0.5
# #         out = out.reshape(B, D, C, H_in, W_in).permute(0, 2, 1, 3, 4)
# #         out = self.conv_1d(out)
# #         return  self.relu_out(x_in + out)


# #     def TCA_diff(self, x):
# #         x = self.g1(x)
# #         x = self.relu(x)
# #         x = self.g2(x)
# #         x = self.sigmoid(x)
# #         return x

# class TSA(nn.Module):

#     def __init__(
#         self,
#         dim,
#         frame=8,
#         instantiation="softmax",
#         norm_eps=1e-5,
#         norm_momentum=0.1,
#         norm_module=nn.BatchNorm2d,
#     ):
#         """
#         Args:
#             dim (int): number of dimension for the input.
#             frame (int): frame number.
#             instantiation (string): supports two different instantiation method:
#                 "dot_product": normalizing correlation matrix with L2.
#                 "softmax": normalizing correlation matrix with Softmax.
#             norm_module (nn.Module): nn.Module for the normalization layer. The
#                 default is nn.BatchNorm2d.
#         """
#         super(TSA, self).__init__()
#         self.dim = dim
#         self.dim = dim
#         self.frame = frame
#         self.instantiation = instantiation
#         self.norm_eps = norm_eps
#         self.norm_momentum = norm_momentum
#         self._construct_TSA(
#             norm_module
#         )

#     def _construct_TSA(
#         self, norm_module
#     ):
#         # Three convolution : conv_phi0 (x), conv_phi_1 (x-1), and conv_phi_2 (x-2).
#         self.conv_phi_1 = nn.Conv2d(
#             self.dim, self.dim, kernel_size=1, stride=1, padding=0
#         )
#         self.conv_phi_2 = nn.Conv2d(
#             self.dim, self.dim, kernel_size=1, stride=1, padding=0
#         )
#         self.conv_phi0 = nn.Conv2d(
#             self.dim, self.dim, kernel_size=1, stride=1, padding=0
#         )

#         # TODO: change the name to `norm`
#         self.norm = norm_module(
#             self.dim,
#             eps=self.norm_eps,
#         )

#         self.init_weights()


#     def init_weights(self):
#             """
#             Performs ResNet style weight initialization.
#             """
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     """
#                     Follow the initialization method proposed in:
#                     {He, Kaiming, et al.
#                     "Delving deep into rectifiers: Surpassing human-level
#                     performance on imagenet classification."
#                     arXiv preprint arXiv:1502.01852 (2015)}
#                     """
#                     c2_msra_fill(m)
#                 elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
#                     batchnorm_weight = 0.
#                     if m.weight is not None:
#                         m.weight.data.fill_(batchnorm_weight)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 if isinstance(m, nn.Linear):
#                     m.weight.data.normal_(mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         m.bias.data.zero_()

#     def forward(self, x):
#         x_identity = x
#         N, C, H, W = x.size()

#         T = self.frame
#         V = N // T
#         x = x.view(V, T, C, H, W)
#         x_1 = torch.cat((x[:,0,:,:,:].unsqueeze(1),x[:,:-1,:,:,:]), 1)
#         x_2 = torch.cat((x_1[:, 0, :, :, :].unsqueeze(1), x_1[:, :-1, :, :, :]), 1)
#         x_1 = x_1.view(N, C, H, W)
#         x_2 = x_2.view(N, C, H, W)
#         x = x.view(-1, C, H, W)

#         phi_x_1 = self.conv_phi_1(x_1)
#         phi_x_2 = self.conv_phi_2(x_2)
#         phi_x0 = self.conv_phi0(x)

#         phi_x_1 = phi_x_1.view(N, self.dim, -1)
#         phi_x_2 = phi_x_2.view(N, self.dim, -1)
#         phi_x0 = phi_x0.view(N, self.dim, -1)

#         # (N, C, HxW) * (N, C, HxW) => (N, HxW, HxW).
#         theta_x_1_2 = torch.einsum("nca,ncb->nab", (phi_x_1, phi_x_2))
#         # For original Non-local paper, there are two main ways to normalize
#         # the affinity tensor:
#         #   1) Softmax normalization (norm on exp).
#         #   2) dot_product normalization.
#         if self.instantiation == "softmax":
#             # Normalizing the affinity tensor theta_x_1_2 before softmax.
#             theta_x_1_2 = theta_x_1_2 * (self.dim ** -0.5)
#             theta_x_1_2 = nn.functional.softmax(theta_x_1_2, dim=2)
#         elif self.instantiation == "dot_product":
#             spatial_temporal_dim = theta_x_1_2.shape[2]
#             theta_x_1_2 = theta_x_1_2 / spatial_temporal_dim
#         else:
#             raise NotImplementedError(
#                 "Unknown norm type {}".format(self.instantiation)
#             )

#         # (N, HxW, HxW) * (N, C, HxW) => (N, C, HxW).
#         x_out = torch.einsum("ntg,ncg->nct", (theta_x_1_2, phi_x0))

#         # (N, C, HxW) => (N, C, H, W).
#         x_out = x_out.contiguous().view(N, self.dim, H, W)

#         x_out = self.norm(x_out)
#         return x_identity + x_out

# class TCA(nn.Module):
#     def __init__(self, dim, frame):
#         super(TCA, self).__init__()
#         self.dim = dim
#         self.frame = frame

#         print("-" * 20 + "TCA called" + '-' * 20)
#         self.g1 = nn.Conv2d(
#             self.dim, self.dim * 4, kernel_size=1, stride=1, padding=0
#         )
#         self.g2 = nn.Conv2d(
#             self.dim * 4, self.dim, kernel_size=1, stride=1, padding=0
#         )

#         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)


#     def forward(self, x, x_att):
#         x = self.pool(x)
#         [N, C, H, W] = x.size()
#         V = N // self.frame
#         x_t = x.view(V, self.frame, C, H, W)
#         x_t1 = torch.cat((x_t[:,-1,:,:,:].unsqueeze(1), x_t[:,:-1,:,:,:,]), 1)
#         x_t2 = torch.cat((x_t[:, -2:, :, :, :], x_t[:, :-2, :, :, :, ]), 1)

#         x_diff1 = x_t - x_t1
#         x_diff1 = x_diff1.view(N, C, H, W)
#         x_diff2 = (x_t - x_t2)*0.5
#         x_diff2 = x_diff2.view(N, C, H, W)

#         x_diff1 = self.TCA_diff(x_diff1)
#         x_diff2 = self.TCA_diff(x_diff2)

#         return  x_att * (x_diff1  + x_diff2)*0.5


#     def TCA_diff(self, x):
#         x = self.g1(x)
#         x = self.relu(x)
#         x = self.g2(x)
#         x = self.sigmoid(x)
#         return x

# class TCP_att_module(nn.Module):

#     def __init__(
#         self,
#         dim,
#         frame=8,
#         ch_flag=True,
#         sp_flag=True,
#         conv_1d_flag=True,
#     ):
#         """
#         Args:
#             dim (int): TCP_dim.
#             frame (int): frame number.
#             ch_flag (bool): whether use temporal-based channel attention, default: True.
#             sp_flag (bool): whether use temporal-based spatial attention, default: True.
#             conv_1d_flag (bool): whether use temporal convolution, default: True.

#         """
#         super(TCP_att_module, self).__init__()
#         self.dim = dim
#         self.frame = frame
#         self.ch_flag = ch_flag
#         self.sp_flag = sp_flag
#         self.conv_1d_flag = conv_1d_flag
#         self.k = frame // 2 + 1 #temporal conv kernel
#         if self.k % 2 == 0:  # 若k为偶数，加1变为奇数
#             self.k += 1
#         self._construct_TCP_att(
#         )

#     def _construct_TCP_att(self
#     ):
#         self.relu = nn.ReLU(inplace=True)

#         if self.ch_flag:
#             self.TCA = TCA(self.dim, self.frame)

#         if self.conv_1d_flag :
#             k_padding = (self.k-1)//2
#             self.conv_1d = nn.Conv3d(
#                 self.dim, self.dim, kernel_size=(self.k,1,1),
#                 stride=1, padding=(k_padding,0,0),
#                 padding_mode='circular'
#             )

#         if self.sp_flag:
#             print("-" * 20 + "TSA Module called" + '-' * 20)
#             self.TSA = TSA(self.dim, self.frame)

#         self.init_weights()



#     def init_weights(self):
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#                     #c2_msra_fill(m)
#                     m.weight.data.normal_(mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 elif isinstance(m, nn.BatchNorm2d) \
#                         or isinstance(m, nn.BatchNorm3d):
#                     batchnorm_weight = 0.
#                     if m.weight is not None:
#                         m.weight.data.fill_(batchnorm_weight)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#                 if isinstance(m, nn.Linear):
#                     m.weight.data.normal_(mean=0.0, std=0.01)
#                     if m.bias is not None:
#                         m.bias.data.zero_()
#             if self.conv_1d_flag:
#                 self.conv_1d.weight.data.fill_(0.)
#             if self.sp_flag:
#                 self.TSA.init_weights()

#     def forward(self, x):
#         if self.sp_flag:
#             x_att = self.TSA(x)
#         else:
#             x_att = x

#         if self.ch_flag:
#             x_att = self.TCA(x, x_att)

#         if self.conv_1d_flag:
#             [N, C, H, W] = x_att.size()
#             V = N // self.frame
#             x_att = x_att.reshape(V, self.frame, C, H, W).permute(0,2,1,3,4)
#             x_att = self.conv_1d(x_att)
#             x_att = x_att.permute(0,2,1,3,4,).reshape(N, C, H, W)
#             return self.relu(x_att + x)
#         else :
#             return x_att  # following SE, w/o shortcut