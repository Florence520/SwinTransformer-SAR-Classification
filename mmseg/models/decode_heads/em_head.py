import torch.nn as nn
import torch
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Sequence, Tuple, Union, Optional
from monai.networks.blocks.dynunet_block import get_conv_layer, UnetResBlock,UnetBasicBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
# from mamba_ssm import Mamba
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


import torch.nn as nn
import torch
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Sequence, Tuple, Union, Optional
from monai.networks.blocks.dynunet_block import get_conv_layer, UnetResBlock,UnetBasicBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
# from mamba_ssm import Mamba
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
# from ..utils.SMamba import StructureAwareSSM
from fvcore.nn.weight_init import c2_msra_fill
from einops import rearrange



@MODELS.register_module()
class EMHead(BaseDecodeHead):
    def __init__(self,
                 num_classes=8,
                 hidden_size: int = 288,
                 out_chans=8,
                 feat_size=[36, 72, 144, 288],
                 ds: bool = False,
                 spatial_dims=3,
                 norm_name="instance",
                 res_block: bool = True,
                 conv_decoder: bool = False,
                 in_shpae=[224, 224, 16],
                 **kwargs):
        meaningless_value = 8
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
                upsample_kernel_size=(2,4,4),
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
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=8,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=1,  # Block expansion factor
            bimamba_type="v2",
        )
        x = input_shpae[0] // 2 ** (stage) // 2
        y = input_shpae[1] // 2 ** (stage) // 2
        z = input_shpae[2] // 2 ** (stage)
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, x * y * z, self.dim))  # Temp n*n*n/2

        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

        self.mlp = MlpChannel(hidden_size=dim, mlp_dim=dim // 4)
        self.conv51 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

    def forward(self, x):
        B, C = x.shape[:2]

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        if self.pos_embed is not None:
            x_flat = x_flat + self.pos_embed
        x_norm = self.norm(x_flat)
        x_mamba = self.norm1(self.mamba(x_norm))
        x_spatial = self.norm2(
            self.mlp(x_norm.transpose(-1, -2).reshape(B, C, *img_dims)).reshape(B, C, n_tokens).transpose(-1, -2))

        out = x_flat + self.gamma * (x_mamba + x_spatial)
        out = out.transpose(-1, -2).reshape(B, C, *img_dims)

        attn = self.conv51(out)
        out = out + self.conv8(attn)

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