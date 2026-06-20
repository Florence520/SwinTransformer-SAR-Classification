import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
# from fvcore.nn import flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .Smamba_utils import selective_scan_fn
except:
    from .Smamba_utils import selective_scan_fn

class DepthwiseFunction3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, padding, is_bias):
        ctx.save_for_backward(x, w)
        ctx.padding = padding
        ctx.is_bias = is_bias

        # 3D卷积
        output = F.conv3d(
            x, w, bias=b,
            padding=padding,
            groups=x.size(1)  # 深度可分离卷积
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播逻辑修正
        x, w = ctx.saved_tensors
        padding = ctx.padding             # 从 ctx 属性获取 padding
        is_bias = ctx.is_bias             # 从 ctx 属性获取 is_bias
        N, C, D, H, W = x.shape
        kD, kH, kW = w.shape[-3:]

        # 输入梯度计算
        w_inv = torch.flip(w, dims=[-3, -2, -1])  # 翻转卷积核
        grad_input = torch.nn.functional.conv3d(
            grad_output, w_inv, padding=padding, groups=C
        )

        # 权重梯度修正（关键步骤）
        x_pad = F.pad(x, (padding,) * 6)  # 3D填充每个维度两侧
        x_unfold = x_pad.unfold(2, kD, 1).unfold(3, kH, 1).unfold(4, kW, 1)  # [N, C, D_unf, H_unf, W_unf, kD, kH, kW]
        x_unfold = x_unfold.permute(0, 1, 5, 6, 7, 2, 3, 4)  # [N, C, kD, kH, kW, D_unf, H_unf, W_unf]
        x_unfold = x_unfold.contiguous().view(N, C, kD*kH*kW, -1)  # [N, C, kernel_elements, num_patches]

        grad_reshaped = grad_output.view(N, C, -1)  # [N, C, num_patches]
        grad_weight = torch.einsum('ncp,nckp->ck', grad_reshaped, x_unfold)  # [C, kernel_elements]
        grad_weight = grad_weight.view(C, 1, kD, kH, kW)  # [3,1,3,3,3]

        # 偏置梯度
        grad_bias = grad_output.sum(dim=(0,2,3,4)) if is_bias else None

        return grad_input, grad_weight, grad_bias, None, None


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # B, H, W, D, C = x.shape
        # x = x.reshape(B, -1, C)  # 形状变为 (B, H*W*D, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = x.reshape(B, H, W, D, -1)  # 恢复原始形状
        return x


class StateFusion(nn.Module):
    def __init__(self, dim,d_state):
        super(StateFusion, self).__init__()

        self.dim = dim
        self.d_state = d_state
        self.num = self.dim*self.d_state
        # 定义三维卷积核 (depth, height, width)
        self.kernel_3 = nn.Parameter(torch.ones(self.num, 1, 3, 3, 3))  # 形状 (C, 1, 3, 3, 3)
        self.kernel_3_1 = nn.Parameter(torch.ones(self.num, 1, 3, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(self.num, 1, 3, 3, 3))

        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):

        if self.training:
            h1 = F.conv3d(self.padding(h, (1, 1, 1, 1, 1, 1)), self.kernel_3, padding=0, dilation=1, groups=self.num)
            h2 = F.conv3d(self.padding(h, (3, 3, 3, 3, 3, 3)), self.kernel_3_1, padding=0, dilation=3, groups=self.num)
            h3 = F.conv3d(self.padding(h, (5, 5, 5, 5, 5, 5)), self.kernel_3_2, padding=0, dilation=5, groups=self.num)

            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out

        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.num, 1, 11, 11, 11), device=h.device)

                # 填充 dilation=1 的权重到中心
                self._merge_weight[:, :, 4:7, 4:7, 4:7] += self.alpha[0] * self.kernel_3

                # dilation=3的修正
                for dz in [2, 5, 8]:
                    for dy in [2, 5, 8]:
                        for dx in [2, 5, 8]:
                            z_idx = (dz - 2) // 3
                            y_idx = (dy - 2) // 3
                            x_idx = (dx - 2) // 3
                            self._merge_weight[:, :, dz, dy, dx] += self.alpha[1] * self.kernel_3_1[:, :, z_idx, y_idx,
                                                                                    x_idx]

                for dz in [0,5,10]:
                    for dy in [0,5,10]:
                        for dx in [0,5,10]:
                            # ❌ 原错误：dz//5 导致索引超出范围（当dz=5时5//5=1，但kernel_3_2的尺寸为3x3x3）
                            z_idx = 1 if dz ==5 else (0 if dz ==0 else 2)  # ✅ 修正索引计算
                            y_idx = 1 if dy ==5 else (0 if dy ==0 else 2)
                            x_idx = 1 if dx ==5 else (0 if dx ==0 else 2)
                            self._merge_weight[:, :, dz, dy, dx] += self.alpha[2] * self.kernel_3_2[:, :, z_idx, y_idx, x_idx]
            h_padded = self.padding(h, (5,5,5,5,5,5))  # 每个维度前后各pad5
            # out = DepthwiseFunction3D.apply(h_padded, self._merge_weight, None, 0, False)
            out = F.conv3d(h_padded, self._merge_weight, groups=self.num)
            return out

class StructureAwareSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,  # 输入通道数
            out_channels=self.d_inner,  # 输出通道数
            groups=self.d_inner,  # 分组卷积，与输入通道数相同表示深度可分离卷积
            bias=conv_bias,  # 是否使用偏置
            kernel_size=(d_conv, d_conv, d_conv),  # 卷积核大小，三维卷积需要三个值
            padding=((d_conv - 1) // 2, (d_conv - 1) // 2, (d_conv - 1) // 2),  # 三维填充
            **factory_kwargs,  # 其他可选参数
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion(self.d_inner,self.d_state)

        self.out_norm = nn.LayerNorm(self.d_inner*self.d_state)
        self.out_proj = nn.Linear(self.d_inner*self.d_state, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):

        B, C, D, H, W = x.shape  
        Z = C * D
        L = H * W  

        xs = x.view(B, Z, L)  

        #+++++++++++++++++++++++++++++++++++++++++++++修改+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        x_proj_weight = self.x_proj_weight.unsqueeze(0)  # [1, K, C]
        x_proj_weight = x_proj_weight.expand(D, -1, -1)  # [D, K, C] (虚拟扩展)
        x_proj_weight = x_proj_weight.reshape(-1, C)     # [D*K, C] (物理存储连续化)
        x_dbl = torch.matmul(x_proj_weight.view(-1, Z), xs)
        # x_proj_weight = self.x_proj_weight.repeat(D, 1)
        # x_dbl = torch.matmul(x_proj_weight.view(1, -1, Z), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        #+++++++++++++++++++++++++++++++++++++++++++++修改+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        dt_projs_weight = self.dt_projs_weight.unsqueeze(0)  # [1, K, C]
        dt_projs_weight = dt_projs_weight.expand(D, -1, -1)  # [D, K, C] (虚拟扩展)
        dt_projs_weight = dt_projs_weight.reshape(-1, C)     # [D*K, C] (物理存储连续化)
        dts = torch.matmul(dt_projs_weight.view(Z, -1), dts)
        # dt_projs_weight = self.dt_projs_weight.repeat(D, 1)
        # dts = torch.matmul(dt_projs_weight.view(1, Z, -1), dts)
        
        # 动态生成A_logs和Ds
        As = -torch.exp(self.A_logs).repeat(D, 1)  # [CD, d_state]
        Ds = self.Ds.repeat(D)  # [CD]
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias.repeat(D)

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        #从上面的2,64,64变成2,64,2,64
        h = rearrange(h, "b (c d) o (h w) -> b (c o) d h w", d=D, h=H, w=W)

        h = self.state_fusion(h)

        # 最终投影（合并 D 维度）
        h = rearrange(h, "b c d h w -> b (c d) (h w)")
        Cs = Cs.unsqueeze(1)              # (B, 1, 16, 1024)
        Cs = Cs.repeat(1, C*D, 1, 1)      # (B, 64, 16, 1024)
        Cs = Cs.reshape(B, C*D*self.d_state, -1)  # (B, 2048, 1024)
        y = h * Cs

        # 步骤2: 调整 Ds 维度（无需手动计算 C*D）
        Ds = Ds.view(-1, 1)          # 形状 (C*D, 1)
        Ds = Ds.unsqueeze(0)               # 形状 (1, C*D, 1)
        Ds = Ds.unsqueeze(-1)              # 形状 (1, C*D, 1, 1)
        Ds = Ds.repeat(1, 1, self.d_state, 1)  # 关键扩展 → 形状 (1, C*D, d_state, 1)

        # 步骤3: 调整 xs 维度
        xs = x.view(B, C*D, L).unsqueeze(2)  # 形状 (B, C*D, 1, L)

        # 步骤4: 广播计算
        p = (xs * Ds).reshape(B, -1, L)     # 形状 (B, C*D*d_state, L)

        # Ds = Ds.view(-1, 1).repeat(1, self.d_state)
        # p = xs.repeat_interleave(self.d_state, dim=1) * Ds.view(1, -1, 1)
        y = y + p

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        #添加了一个维度
        B, H, W, D, C = x.shape

        xz = self.in_proj(x)  # x的最后一个维度c必须等于线形层的输入维度
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, 'b h w d c -> b c d h w').contiguous()

        x = self.act(self.conv3d(x)) #类似于6这个符号

        y = self.ssm(x) #SASF

        #+++++++++++++++++++++++++++++++++++++++++++++修改+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        y = rearrange(y, "b (c d) (h w) -> b h w d c", c=C*self.d_state*self.expand, d=D, h=H, w=W)
        # y = rearrange(y, "b (c d) (h w) -> b h w d c", c=C*self.d_state, d=D, h=H, w=W)

        y = self.out_norm(y)
        #+++++++++++++++++++++++++++++++++++++++++++++修改+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        z = F.silu(z)  # [2,14,14,2,384]
        z = rearrange(z, 'b h w d c -> b h w d c 1')  # 添加状态维度 [2,14,14,2,384,1]
        z = z.expand(-1, -1, -1, -1, -1, self.d_state)  # 扩展s维度到16 
        z = rearrange(z, 'b h w d c s -> b h w d (c s)')  # 合并后得到6144 
        y = y * z
        # z = z.repeat_interleave(self.d_state, dim=-1)
        # y = y * F.silu(z)
        
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class SpatialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        num_heads: int = 8,
        mlp_ratio = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.cpe1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
       
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = StructureAwareSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(in_features=hidden_dim, hidden_features=int(hidden_dim*mlp_ratio), act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)
    
    def forward(self, x: torch.Tensor):
        # B H W D C  need B C D H W
        x = x + self.cpe1(x.permute(0, 4, 3, 1, 2).contiguous()).permute(0, 3, 4, 2, 1)
        # self.self_attention之后的形状为b h w d c
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 4, 3, 1, 2).contiguous()).permute(0, 3, 4, 2, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
    


def test_spatial_mamba_block():
    # 配置参数
    batch_size = 2
    hidden_dim = 192
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试输入 (batch, H, W, D, channels)
    x = torch.randn(batch_size, 14, 14, 2, hidden_dim).to(device)
    
    # 初始化模块，显式设置expand=1
    model = SpatialMambaBlock(
        hidden_dim=hidden_dim,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        attn_drop_rate=0.1,
        d_state=16,
        dt_init="random",
        num_heads=4,
        expand=1  # 添加expand参数设置为1
    ).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        output = model(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    # 测试反向传播
    x.requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Input gradients not computed"
    
    # 进入eval模式并检查合并权重形状
    model.eval()
    with torch.no_grad():
        output = model(x)
        
    # 检查StateFusion的合并权重形状是否正确
    for module in model.modules():
        if isinstance(module, StateFusion):
            assert hasattr(module, "_merge_weight"), "Merge weights not created"
            # 此时expand=1，dim应为hidden_dim=8
            expected_shape = (hidden_dim*16, 1, 11, 11, 11)
            assert module._merge_weight.shape == expected_shape, \
                f"Wrong merge shape: {module._merge_weight.shape}, expected {expected_shape}"
    
    print("All tests passed!")

# 运行测试
if __name__ == "__main__":
    test_spatial_mamba_block()