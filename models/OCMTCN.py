from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from OtherFile.Tools import ConfigTools

is_transfer = ConfigTools.load_config_data("is_transfer")  # 是否加入逆序矩阵约束
is_muti = ConfigTools.load_config_data("is_muti")     # 是否使用多分辨率融合

class LayerNorm(nn.Module):
    """Layer normalization for 4D tensor [B, M, D, N]"""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)  # [B, M, N, D]
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)  # [B, M, D, N]
        return x
class MonotonicPositionEncoding(nn.Module):
    def __init__(self, seq_len, init_min=0.0, init_max=1.0):
        super().__init__()
        # 可学习的原始位置参数，初始线性
        raw = torch.linspace(init_min, init_max, seq_len)
        self.raw_pos = nn.Parameter(raw)  # (N,)

    def forward(self):
        # 强制单调不减：累积最大值
        mono_pos = torch.cummax(self.raw_pos, dim=0)[0]
        return mono_pos  # (N,)

class MonotonicSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, dropout=0.1, ddmodel=512):
        """
        d_model: M * D (合并后的特征维度)
        n_heads: 注意力头数
        seq_len: 序列长度
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len

        # 单调位置编码
        if seq_len is not None:
            # 创建单调位置编码
            self.pos_enc = MonotonicPositionEncoding(seq_len)
        else:
            self.pos_enc = None
        # self.pos_enc = MonotonicPositionEncoding(seq_len)

        # 多头注意力
        self.attn = nn.MultiheadAttention(ddmodel, n_heads, dropout=dropout, batch_first=True)

        # 偏置缩放因子（可学习，初始为正值）
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

        # 层归一化（可选）
        self.norm = nn.LayerNorm(ddmodel)
        self.lin1 = nn.Linear(d_model, ddmodel)
        self.lin2 = nn.Linear(ddmodel, d_model)

    def forward(self, x):
        """
        x: (B, M, D, N) -> 输出相同形状
        """
        B, M, D, N = x.shape
        # 合并变量维度和通道维度: (B, N, M*D)
        x_flat = x.permute(0, 3, 1, 2).reshape(B, N, M*D)
        x_flat = self.lin1(x_flat)

        # 获取单调位置编码 (N,)
        p = self.pos_enc()  # (N,)

        # 计算顺序偏置矩阵 (N, N)
        # delta[i,j] = p_i - p_j，若 >0 表示 i 比 j 深
        delta = p.unsqueeze(0) - p.unsqueeze(1)  # (N, N)
        # 惩罚逆序：如果 i 深于 j，则 i 关注 j 时加负偏置
        bias = -self.bias_scale * torch.relu(delta)  # (N, N)
        # 扩展为多头注意力需要的形状: (1, 1, N, N) 或 (B, H, N, N)
        # MultiheadAttention 的 attn_mask 要求形状 (N, N) 或 (B*H, N, N)
        attn_mask = bias  # (1, 1, N, N)

        # 自注意力
        # attn_out, _ = self.attn(x_flat, x_flat, x_flat, attn_mask=attn_mask)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)

        # 残差连接 + 层归一化
        out = self.norm(x_flat + attn_out)
        out = self.lin2(out)

        # 恢复形状
        out = out.reshape(B, N, M, D).permute(0, 2, 3, 1)  # (B, M, D, N)
        return out


class StratigraphyTransferConstraintLayer(nn.Module):
    """
    地层转移约束层：将三对角转移矩阵嵌入模型内部的特征计算中
    输入输出形状完全一致，可插入到ModernTCN的任意Block/Stage之间
    """
    def __init__(self,
                 hidden_dim: int,        # 输入隐藏特征维度（M*D，变量数×单变量特征维度）
                 num_classes: int,       # 地层类别数K（转移矩阵的维度）
                 transition_matrix: torch.Tensor,  # 你的三对角转移矩阵 [K, K]
                 alpha_init: float = 0.1, # 约束强度初始值（可学习）
                 freeze_forbidden: bool = True  # 是否冻结非法转移位置（推荐True）
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.freeze_forbidden = freeze_forbidden

        # 1. 降维投影：隐藏维度 → 类别空间K
        self.down_proj = nn.Linear(hidden_dim, num_classes)

        # 2. 可学习的转移矩阵：初始化为你的地质先验矩阵
        self.transition_matrix = nn.Parameter(transition_matrix.clone())

        # 3. 构建非法转移掩码：forbidden_mask[i][j] = 1 表示i→j是非法转移
        self.register_buffer('forbidden_mask', (1 - transition_matrix).bool())

        # 4. 升维投影：类别空间K → 原始隐藏维度
        self.up_proj = nn.Linear(num_classes, hidden_dim)

        # 5. 可学习的约束强度系数（避免过度修正原始特征）
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # 初始化：如果冻结非法转移，将其梯度屏蔽
        if self.freeze_forbidden:
            self.transition_matrix.register_hook(self._mask_forbidden_grad)

    def _mask_forbidden_grad(self, grad):
        """梯度钩子：非法转移位置的梯度强制为0，不参与训练"""
        grad.masked_fill_(self.forbidden_mask, 0.0)
        return grad


    def forward(self, x):
        """
        前向传播：输入输出形状完全一致
        Args:
            x: 输入隐藏特征，形状 [B, M, D, N]
               (B=批次, M=变量数, D=单变量特征维度, N=序列长度)
        Returns:
            out: 约束后的特征，形状与输入完全相同
        """
        B, M, D, N = x.shape
        identity = x  # 残差连接的原始特征

        # --------------------------
        # 步骤1：维度对齐与降维投影
        # --------------------------
        # 合并变量维度和特征维度：[B, M, D, N] → [B, N, M*D]
        x_flat = x.permute(0, 3, 1, 2).reshape(B, N, M * D)

        # 投影到类别空间：[B, N, M*D] → [B, N, K]
        h = self.down_proj(x_flat)

        # --------------------------
        # 步骤2：转移矩阵约束（核心）
        # --------------------------
        # 计算前一个时间步的特征：h_prev[t] = h[t-1]
        # 第一个时间步没有前序，用自身填充
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0), mode='replicate')  # [B, N, K]

        # 合法转移特征计算：h_prev @ 转移矩阵
        # 物理意义：每个时间步的特征 = 前一个时间步所有合法转移的特征加权和
        h_transfer = torch.matmul(h_prev, self.transition_matrix.T)  # [B, N, K]

        # 融合原始类别特征与转移约束特征
        h_constrained = h + h_transfer
        # h_constrained = h + 0.1 * h_transfer

        # --------------------------
        # 步骤3：升维投影与残差连接
        # --------------------------
        # 投影回原始隐藏维度：[B, N, K] → [B, N, M*D]
        x_constrained = self.up_proj(h_constrained)

        # 恢复原始形状：[B, N, M*D] → [B, M, D, N]
        x_constrained = x_constrained.reshape(B, N, M, D).permute(0, 2, 3, 1)

        # 残差连接：仅对原始特征做增量修正
        out = identity + self.alpha * x_constrained
        # out = identity + 0.1 * x_constrained


        return out
class Flatten_Head(nn.Module):
    """
    平展头模块，用于将特征映射到目标预测窗口

    支持两种模式：
    1. 独立模式（individual）：每个变量使用独立的线性层
    2. 共享模式：所有变量共享同一个线性层
    """
    def __init__(self, individual, seq_len, nf, num_class, head_dropout=0):
        """
        初始化平展头模块

        Args:
            individual: 是否对每个变量使用独立的线性层
            n_vars: 变量数量（通道数）
            nf: 输入特征维度（d_model * patch_num）
            target_window: 目标预测窗口长度
            head_dropout: Dropout比率，防止过拟合
        """
        super(Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = seq_len

        if self.individual:
            # 独立模式：为每个变量创建独立的层
            self.linears = nn.ModuleList()      # 线性变换层列表
            self.dropouts = nn.ModuleList()      # Dropout层列表
            self.flattens = nn.ModuleList()      # 平展层列表
            for i in range(self.n_vars):
                # 将最后两个维度平展为一维
                self.flattens.append(nn.Flatten(start_dim=-2))
                # 从特征维度映射到目标窗口长度
                self.linears.append(nn.Linear(nf, num_class))
                # 应用Dropout正则化
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # 共享模式：所有变量共享同一组层
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, num_class)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        """
        前向传播，将特征映射到预测结果

        Args:
            x: 输入特征张量，形状为 [batch_size, n_vars, d_model, patch_num]

        Returns:
            预测结果张量，形状为 [batch_size, n_vars, target_window]
        """
        x = x.permute(0, 3, 1, 2)


        if self.individual:
            # 独立模式：逐个变量处理
            x_out = []
            for i in range(self.n_vars):
                # 平展第i个变量的特征维度
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                # 线性变换到目标窗口长度
                z = self.linears[i](z)  # z: [bs x target_window]
                # 应用Dropout
                z = self.dropouts[i](z)
                x_out.append(z)
            # 将所有变量的输出堆叠在一起
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            # 共享模式：一次性处理所有变量
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x



class LayerNorm(nn.Module):
    """Layer normalization for 4D tensor [B, M, D, N]"""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)  # [B, M, N, D]
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)  # [B, M, D, N]
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    """重参数化大卷积核 - ModernTCN核心组件"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2

        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1,
                                         groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1,
                                      groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2,
                                          groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):
        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dim=-1)
        x = torch.cat([x, pad_right], dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size,
                                     stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding,
                                     dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    """ModernTCN核心Block - 双维度ConvFFN"""

    def __init__(self, large_size, small_size, dmodel, dff, nvars,
                 small_kernel_merged=False, drop=0.1, attn_heads=4, num_classes=14):
        super(Block, self).__init__()

        # 深度可分离大卷积
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
            nvars=nvars
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # ConvFFN1 - 变量维度分组
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff,
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel,
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # ConvFFN2 - 通道维度分组
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff,
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel,
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

        # 新增：顺序约束注意力
        # 注意：dmodel 是每个变量的通道数，合并后的特征维度 = nvars * dmodel
        self.monotonic_attn = MonotonicSelfAttention(
            d_model=nvars*dmodel,
            n_heads=attn_heads,
            seq_len=None,  # 动态获取，在 forward 中确定
            dropout=drop
        )

        transition_matrix = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            transition_matrix[i, i] = 1.0  # 允许同一地层连续
            if i < num_classes - 1:
                transition_matrix[i, i+1] = 1.0  # 允许向下沉积到下一层

        # 新增：地层转移约束层
        if transition_matrix is not None:
            hidden_dim = nvars * dmodel  # 合并后的隐藏维度
            self.transfer_constraint = StratigraphyTransferConstraintLayer(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                transition_matrix=transition_matrix,
                alpha_init=0.1
            )
        else:
            self.transfer_constraint = nn.Identity()

    def forward(self, x):
        input = x
        B, M, D, N = x.shape

        # 大卷积处理
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)

        # 归一化
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        # ConvFFN1
        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        # ConvFFN2
        x = x.permute(0, 2, 1, 3)  # [B, D, M, N]
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)  # [B, M, D, N]

        # 残差连接
        x = input + x
        if is_transfer:
            x = self.transfer_constraint(x)

        return x


class Stage(nn.Module):
    """Stage模块 - 包含多个Block"""

    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel,
                 dw_model, nvars, small_kernel_merged=False, drop=0.1,
                 use_order_constraint=True, num_classes=14, constraint_reg_weight=0.001):
        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel,
                        dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

        transition_matrix = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            transition_matrix[i, i] = 1.0  # 允许同一地层连续
            if i < num_classes - 1:
                transition_matrix[i, i+1] = 1.0  # 允许向下沉积到下一层

        # 新增：地层转移约束层
        if transition_matrix is not None:
            hidden_dim = nvars * dmodel  # 合并后的隐藏维度
            self.transfer_constraint = StratigraphyTransferConstraintLayer(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                transition_matrix=transition_matrix,
                alpha_init=0.1
            )
        else:
            self.transfer_constraint = nn.Identity()

        self.monotonic_attn = MonotonicSelfAttention(
            d_model=nvars*dmodel,
            n_heads=4,
            seq_len=None,  # 动态获取，在 forward 中确定
            dropout=drop
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))


    def forward(self, x):
        B, M, D, N = x.shape
        for blk in self.blocks:
            x = blk(x)
        return x

    def get_constraint_reg_loss(self):
        if hasattr(self.constraint, 'regularization_loss'):
            return self.constraint.regularization_loss()
        return 0.0


class UpsampleLayer(nn.Module):
    """上采样层 - 用于解码器恢复序列长度"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, target_shape=None):
        # x: [B, C, L]

        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        x = self.bn(x)

        # 如果提供了目标形状，调整到精确尺寸
        if target_shape is not None:
            target_len = target_shape[-1]
            if x.shape[-1] != target_len:
                x = F.interpolate(x, size=target_len, mode='nearest')

        return x



class ModernTCN_Classification(nn.Module):
    """
    基于ModernTCN的地层识别模型（逐点分类）

    核心改进：
    1. 移除Flatten_Head，使用PointwiseClassificationHead
    2. 编码器-解码器架构，支持多尺度特征融合
    3. Patch嵌入强制stride=1，保持序列长度
    4. 保留RevIN、重参数化大核、双维度ConvFFN等核心优势
    """

    def __init__(self, task_name, patch_size, patch_stride, stem_ratio, downsample_ratio,
                 ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1,
                 use_multi_scale=True, revin=True, affine=True, subtract_last=False,
                 seq_len=512, c_in=7, target_window=96, class_drop=0., class_num=10,
                 avg_slice_thickness=25.0,sampling_interval=0.125):

        super(ModernTCN_Classification, self).__init__()

        self.task_name = task_name
        self.class_drop = class_drop
        self.class_num = class_num
        self.seq_len = seq_len
        self.c_in = c_in
        self.n_vars = nvars
        self.num_stage = len(num_blocks)
        self.use_multi_scale = use_multi_scale
        self.downsample_ratio = downsample_ratio
        self.dims = dims
        self.avg_slice_thickness =avg_slice_thickness
        self.sampling_interval = sampling_interval


        # Stem layer - Patch嵌入（分类任务强制stride=1）
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # 下采样层（仅当use_multi_scale=True且num_stage>1时启用）
        if self.num_stage > 1 and use_multi_scale:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio,
                              stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

         # Backbone - 多阶段特征提取
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                ffn_ratio, num_blocks[stage_idx],
                large_size[stage_idx], small_size[stage_idx],
                dmodel=dims[stage_idx], dw_model=dw_dims[stage_idx],
                nvars=self.n_vars, small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout,
                num_classes=self.class_num
            )
            self.stages.append(layer)


        self.upsample = None
        if self.num_stage > 1 and use_multi_scale:
            self.upsample = UpsampleLayer(
                    in_channels=dims[-1],
                    out_channels=dims[0],
                    scale_factor=downsample_ratio ** (self.num_stage - 1)
                )
        # region 多尺度上采样
        if self.use_multi_scale and self.num_stage > 1:
            self.use_multi_scale = use_multi_scale
            self.up_sample_ratio = downsample_ratio

            # 为每个stage创建通道调整层（包括第0个）
            self.fusion_conv_channel_adjust = nn.ModuleList()

            for i in range(self.num_stage):
                # 通道调整层
                fusion = nn.Sequential(
                    nn.Conv1d(dims[i], dims[0], kernel_size=1, stride=1),
                    nn.BatchNorm1d(dims[0])
                )
                self.fusion_conv_channel_adjust.append(fusion)
            # 融合层：将所有stage的通道数相加后映射回dims[-1]
            total_channels = dims[0] * self.num_stage
            self.fusion_conv = nn.Sequential(
                nn.Conv1d(total_channels, dims[0], kernel_size=1, stride=1),
                nn.BatchNorm1d(dims[0]),
                nn.GELU()
            )

            # 2. 独立分类头：每个stage有独立的Flatten_Head（使用原始维度）
            self.aux_heads = nn.ModuleList([
                Flatten_Head(
                    individual=False,
                    seq_len=seq_len,
                    nf=dims[i] * self.n_vars,  # 使用原始通道数 * 变量数
                    num_class=class_num,
                    head_dropout=0
                )
                for i in range(self.num_stage)
            ])

            # 3. 上采样倍率记录
            self.upsample_factors = torch.tensor([
                2**i for i in range(self.num_stage)
            ])  # [1, 2, 4]

            # 4. 可学习的温度参数（控制权重衰减速度）
            self.scale_temperature = nn.Parameter(torch.tensor(1.0))

            # 5. 不确定性参数（用于自适应损失加权）
            # num_losses = 1(main) + num_stage(aux)
            self.log_vars = nn.Parameter(torch.zeros(1 + self.num_stage))

        # endregion

        # 分类头
        d_model = dims[0]
        self.head_nf = d_model * self.n_vars
        self.classification_head = Flatten_Head(False, seq_len, self.head_nf, class_num, head_dropout=0)

        # ========== 新增 CRF 层 ==========
        self.crf = CRF(class_num, batch_first=True)

        # 初始化转移矩阵：强制地层顺序约束
        with torch.no_grad():
            # 所有转移初始化为较小的负值
            self.crf.transitions.fill_(-10.0)

            # 允许合法转移：i->i (自环) 和 i->i+1 (向下沉积)
            for i in range(class_num):
                self.crf.transitions[i, i] = 0.0          # 允许同一地层连续
                if i + 1 < class_num:
                    self.crf.transitions[i, i + 1] = 0.0  # 允许向下沉积一层

            # 禁止向上逆序转移（保持为很大的负值）

        # 冻结转移矩阵（可选，如果想让模型学习转移概率则设为True）
        self.crf.transitions.requires_grad = False
        # =================================

        self.scale_weights = nn.Parameter(torch.ones(self.num_stage))

        # self.flatten = nn.Flatten(start_dim=-2)


    def crf_loss(self, logits, labels):
        """
        计算CRF负对数似然损失

        Args:
            logits: [B, seq_len, num_classes] 模型输出的原始分数
            labels: [B, seq_len] 真实标签索引
        Returns:
            CRF 负对数似然损失（标量）
        """
        # CRF的forward返回的是log likelihood，取负号得到loss
        return -self.crf(logits, labels)

    def crf_decode(self, logits):
        """
        使用CRF解码得到最优预测序列

        Args:
            logits: [B, seq_len, num_classes]
        Returns:
            predictions: [B, seq_len] 最优路径预测
        """
        return self.crf.decode(logits)

    def forward_feature(self, x):
        """
        编码器前向传播

        Args:
            x: [B, 1, C, L] - 输入特征

        Returns:
            features: 编码后的特征
            stage_shapes: 每个stage的输出形状（用于解码器）
        """
        stage_shapes = []
        stage_features = []  # 新增：保存所有stage特征
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            # 动态padding处理（如果不能整除）
            if i > 0 and N % self.downsample_ratio != 0:
                pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            # 下采样
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            # Block处理
            x = self.stages[i](x)

            stage_shapes.append(x.shape)
            stage_features.append(x)  # 新增：保存当前stage特征

        return stage_features, stage_shapes


    def decode_features(self, stage_features, stage_shapes):
        """
        多尺度特征融合解码器

        Args:
            stage_features: 所有stage的特征列表
            stage_shapes: 各stage的形状记录

        Returns:
            融合后的多尺度特征
        """
        if not self.use_multi_scale or self.num_stage <= 1:
            return stage_features[-1]

        B, M, D, N = stage_features[0].shape
        upsampled_features = []

        # 将每个stage的特征上采样到原始尺寸
        for i, features in enumerate(stage_features):
            current_N = features.shape[3]  # 当前stage的序列长度
            feat = features.reshape(B * M, features.shape[2], current_N)
            if current_N == N:
                # 第一个stage已经是原始尺寸，直接使用
                # upsampled_features.append(features)
                a = 1
            else:
                # 计算需要上采样的倍数
                scale_factor = N // current_N  # 关键修改：动态计算倍数

                # 上采样到原始尺寸
                # feat = features.reshape(B * M, features.shape[2], current_N)
                feat = F.interpolate(feat, scale_factor=scale_factor, mode='nearest')

            if i > 0:
                # if i < len(stage_features) - 1:
                feat = self.fusion_conv_channel_adjust[i](feat)
            feat = feat.reshape(B, M, self.dims[0], N)
            upsampled_features.append(feat)



        fused = torch.cat(upsampled_features, dim=2)  # [B, M, sum_channels, N]
        fused = self.fusion_conv(fused.reshape(B*M, -1, N)).reshape(B, M, self.dims[0], N)
        return fused

    def decode_features_single(self, x, stage_shapes):
        """
        解码器前向传播 - 恢复序列长度

        Args:
            x: 编码器输出
            stage_shapes: 各stage的形状记录

        Returns:
            恢复到原始尺度的特征
        """
        x = x[-1]
        B, M, D, N = x.shape

        if not self.use_multi_scale or self.num_stage <= 1:
            return x

        x = x.reshape(B * M, D, N)
        x = self.upsample(x)
        # 恢复4D格式
        N = x.shape[-1]
        D = x.shape[-2]
        x = x.reshape(B, M, D, N)
        return x

    def decode_features_with_aux(self, stage_features, stage_shapes):
        """
        多尺度特征融合 + 辅助分类

        Args:
            stage_features: 所有stage的特征列表
            stage_shapes: 各stage的形状记录

        Returns:
            fused: [B, M, C, N] 融合后的特征（用于主分类）
            aux_logits_list: [(B, N, num_classes), ...] 辅助logits列表
            scale_weights: [num_stage] 当前批次的尺度权重
        """
        B, M, D, N = stage_features[0].shape
        upsampled_features = []
        projected_features = []

        # === Step 1: 对每个stage的特征进行上采样和双分支处理 ===
        for i, features in enumerate(stage_features):
            current_N = features.shape[3]  # 当前stage的序列长度
            feat = features.reshape(B * M, features.shape[2], current_N)

            if current_N == N:
                # 第一个stage已经是原始尺寸，不需要上采样
                pass
            else:
                # 计算需要上采样的倍数
                scale_factor = N // current_N

                # 上采样到原始尺寸（使用最近邻插值）
                feat = F.interpolate(feat, scale_factor=scale_factor, mode='nearest')

            # === 分支1：融合分支 - 投影到 dims[0] ===
            if i > 0:
                feat_for_fusion = self.fusion_conv_channel_adjust[i](feat)  # [B*M, dims[0], N]
            else:
                feat_for_fusion = feat  # Stage 0 已经是 dims[0]，无需投影

            feat_reshaped = feat_for_fusion.reshape(B, M, self.dims[0], N)
            upsampled_features.append(feat_reshaped)

            # === 分支2：辅助分类分支 - 保持原始维度 ===
            # 直接使用上采样后的原始特征（未投影）
            feat_for_aux = feat.reshape(B, M, -1, N)  # [B, M, dims[i], N]
            projected_features.append(feat_for_aux)

        # === Step 2: 计算尺度权重（基于上采样倍率）===
        scale_weights = self.compute_scale_weights()

        # === Step 3: 生成辅助分类logits（每个stage独立分类头，使用原始维度）===
        aux_logits_list = []
        for i, aux_feat in enumerate(projected_features):
        # for i, aux_feat in enumerate(upsampled_features):

            # 通过独立分类头（该头的输入维度是 dims[i] * n_vars）
            aux_logits = self.aux_heads[i](aux_feat)
            # shape: [B, N, num_classes]

            aux_logits_list.append(aux_logits)


        fused = torch.cat(upsampled_features, dim=2)  # [B, M, sum_channels, N]
        fused = self.fusion_conv(fused.reshape(B*M, -1, N)).reshape(B, M, self.dims[0], N)

        return fused, aux_logits_list, scale_weights

    def compute_adaptive_loss(self, logits_main, aux_logits_list, labels, main_loss, smooth_loss):
        """
        使用不确定性自适应加权的损失计算

        Args:
            logits_main: [B, N, num_classes] 主分类logits
            aux_logits_list: [(B, N, num_classes), ...] 辅助logits列表
            labels: [B, N] 真实标签

        Returns:
            total_loss: 标量，总损失
            loss_dict: dict，包含各个损失的详细信息（用于监控）
        """
        B, N = labels.shape
        num_classes = logits_main.shape[-1]

        # 展平标签用于交叉熵计算
        labels_flat = labels.reshape(-1)  # [B*N]

        # === 1. 计算各个任务的原始损失 ===

        # 主损失（使用CRF负对数似然）
        main_loss_nll = main_loss  # 标量

        # 辅助损失列表（使用交叉熵，不用CRF）
        aux_losses_ce = []
        aux_losses_smooth = []  # 新增：每个辅助分支的平滑损失

        # 配置：每个Stage的平滑窗口大小（匹配感受野）
        # 索引0对应最浅层（高分辨率），索引越大越深（低分辨率）
        # smooth_windows = [3, 3, 5, 7]  # 3个Stage分别用3、5、7的窗口
        smooth_windows = [3, 3, 3, 3]  # 3个Stage分别用3、5、7的窗口
        # 配置：每个Stage平滑损失的基础权重
        smooth_base_weights = [0.1, 0.05, 0.02, 0.07]  # 高分辨率权重高，低分辨率权重低

        for i, aux_logits in enumerate(aux_logits_list):
            # aux_logits shape: [B, N, num_classes]
            aux_loss_ce = F.cross_entropy(
                aux_logits.reshape(-1, num_classes),  # [B*N, num_classes]
                labels_flat,                           # [B*N]
                reduction='mean'
            )
            aux_losses_ce.append(aux_loss_ce)

            # 新增：辅助分支的多尺度平滑损失
            # 不同Stage使用不同窗口大小的平滑损失
            aux_smooth_loss = smooth_loss(
                aux_logits,
                scales=[smooth_windows[i]],
                weights=[1.0],
                use_gaussian_weight=True
            )
            aux_losses_smooth.append(aux_smooth_loss * smooth_base_weights[i])


        # === 2. 获取不确定性参数 ===
        # log_vars[0] 对应主任务，log_vars[1:] 对应辅助任务
        main_log_var = self.log_vars[0]
        aux_log_vars = self.log_vars[1:]

        # === 关键修复：限制 log_vars 的范围 ===
        # 防止 log_var 过小导致 precision 爆炸，或过大导致权重为0
        with torch.no_grad():
            self.log_vars.clamp_(min=-2.0, max=2.0)  # σ ∈ [0.135, 7.39]

        # 重新读取限制后的值
        main_log_var = self.log_vars[0]
        aux_log_vars = self.log_vars[1:]

        # === 3. 计算自适应加权损失 ===
        # 公式: L_total = sum(0.5 * exp(-log_var_i) * L_i + log_var_i)

        # 主损失的加权
        main_precision = torch.exp(-main_log_var)
        weighted_main_loss = main_precision * main_loss_nll + main_log_var ** 2

        # 辅助损失的加权
        weighted_aux_loss = 0
        for i, aux_loss_ce in enumerate(aux_losses_ce):
            aux_precision = torch.exp(-aux_log_vars[i])
            # 每个辅助分支的总损失 = CE损失 + 平滑损失
            aux_total_loss = aux_losses_ce[i] + aux_losses_smooth[i]
            weighted_aux_loss += aux_precision * aux_total_loss + aux_log_vars[i] ** 2

        # 总损失
        total_loss = weighted_main_loss + weighted_aux_loss
        # total_loss = main_loss + weighted_aux_loss

        # === 4. 记录详细信息（用于调试和监控）===
        loss_dict = {
            'total_loss': total_loss.item(),
            'main_loss_nll': main_loss_nll.item(),
            'main_loss_weighted': weighted_main_loss.item(),
            'main_uncertainty': torch.exp(main_log_var).item(),
            'main_precision': main_precision.item(),
        }

        for i, aux_loss_ce in enumerate(aux_losses_ce):
            aux_precision_i = torch.exp(-aux_log_vars[i]).item()
            loss_dict[f'aux_loss_{i}_ce'] = aux_loss_ce.item()
            loss_dict[f'aux_loss_{i}_weighted'] = (
                    0.5 * torch.exp(-aux_log_vars[i]) * aux_loss_ce + aux_log_vars[i]
            ).item()
            loss_dict[f'aux_uncertainty_{i}'] = torch.exp(aux_log_vars[i]).item()
            loss_dict[f'aux_precision_{i}'] = aux_precision_i

        # 尺度权重（用于监控）
        scale_weights = self.compute_scale_weights()
        for i, w in enumerate(scale_weights):
            loss_dict[f'scale_weight_{i}'] = w.item()

        return total_loss, loss_dict

    def compute_scale_weights(self):
        """
        基于上采样倍率和温度参数计算尺度权重

        公式: weight_i = exp(-factor_i / temperature) / sum(exp(-factor_j / temperature))

        Returns:
            scale_weights: [num_stage] 归一化的尺度权重，和为1
        """
        # 计算原始权重：指数衰减形式
        # factor越大（上采样倍数越高），权重越小
        raw_weights = torch.exp(-self.upsample_factors.to('cuda') /
                                (self.scale_temperature + 1e-8))  # 添加eps防止除零

        # 归一化到和为1
        scale_weights = raw_weights / (raw_weights.sum() + 1e-8)

        return scale_weights


    def classification(self, x):
        """分类任务前向传播"""
        x = x.permute(0, 2, 1)
        B, M, L = x.shape
        # 转换为[B, 1, C, L]格式
        x = x.unsqueeze(-2)
        # 编码器
        stage_features, stage_shapes = self.forward_feature(x)
        projected_features = 0
        # 解码器（如果启用多尺度）
        if self.use_multi_scale and self.num_stage > 1:
            
            if is_muti:
                # features = self.decode_features(stage_features, stage_shapes)
                features, projected_features, _ = self.decode_features_with_aux(stage_features, stage_shapes)
            else:
                features = self.decode_features_single(stage_features, stage_shapes)
        else:
            features = stage_features[-1]

        # features = features.permute(0, 3, 1, 2)
        # features = self.flatten(features)

        # 分类头

        logits = self.classification_head(features)

        return logits, projected_features, features
        # return logits

    def forward(self, x, lengths=None):
        """
        完整前向传播

        Args:
            x: [B, L, C] - 输入序列
            te: 时间编码（可选，当前未使用）

        Returns:
            [B, L, num_classes] - 逐点分类logits
        """
        if self.task_name == 'classification':
            return self.classification(x)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented")

def get_ocmtcn_net(input_size, num_classes, seq_len):

    configs = ModernTCNConfig()

    # 超参数配置
    task_name = configs.task_name
    stem_ratio = configs.stem_ratio
    downsample_ratio = configs.downsample_ratio
    ffn_ratio = configs.ffn_ratio
    num_blocks = configs.num_blocks
    large_size = configs.large_size
    small_size = configs.small_size
    dims = configs.dims
    dw_dims = configs.dw_dims

    nvars = input_size
    small_kernel_merged = configs.small_kernel_merged
    drop_backbone = configs.dropout
    use_multi_scale = configs.use_multi_scale
    revin = configs.revin
    affine = configs.affine
    subtract_last = configs.subtract_last

    seq_len = seq_len
    c_in = input_size
    target_window = configs.pred_len

    patch_size = configs.patch_size
    patch_stride = configs.patch_stride

    # 分类任务参数
    class_dropout = configs.class_dropout
    class_num = configs.num_class

    # 构建模型
    return ModernTCN_Classification(
        task_name=task_name,
        patch_size=patch_size,
        patch_stride=patch_stride,
        stem_ratio=stem_ratio,
        downsample_ratio=downsample_ratio,
        ffn_ratio=ffn_ratio,
        num_blocks=num_blocks,
        large_size=large_size,
        small_size=small_size,
        dims=dims,
        dw_dims=dw_dims,
        nvars=nvars,
        small_kernel_merged=small_kernel_merged,
        backbone_dropout=drop_backbone,
        use_multi_scale=use_multi_scale,
        revin=revin,
        affine=affine,
        subtract_last=subtract_last,
        seq_len=seq_len,
        c_in=c_in,
        target_window=target_window,
        class_drop=class_dropout,
        class_num=num_classes
    )




class ModernTCNConfig:
    """ModernTCN地层识别配置"""

    # 任务类型
    task_name = 'classification'

    # 数据相关
    seq_len = 512          # 输入序列长度（WINDOW_SIZE）
    enc_in = 7             # 输入特征维度
    num_class = 10         # 地层类别数
    pred_len = 96          # 兼容参数（分类任务不使用）

    # Patch嵌入（分类任务推荐stride=1）
    patch_size = 1        # 小核局部平滑
    patch_stride = 1       # 强制不降采样

    # 网络结构
    stem_ratio = 1         # stem层扩展比率
    downsample_ratio = 2   # 下采样倍数（仅在多尺度时使用）
    ffn_ratio = 4          # FFN扩展比率

    # Stage配置（两个stage实现多尺度）
    # num_blocks = [3, 3, 6, 12]    # 每个stage的block数量
    # large_size = [31,29,27,25]  # 大卷积核尺寸
    # small_size = [5,5,5,5]    # 小卷积核尺寸
    # dims = [64, 128, 256, 512]       # 各stage通道数
    # dw_dims = [64, 128, 256, 512]    # 深度卷积通道数
    
    num_blocks = [3, 3, 6]    # 每个stage的block数量
    large_size = [31,29,27]  # 大卷积核尺寸
    small_size = [5,5,5]    # 小卷积核尺寸
    dims = [64, 128, 256]       # 各stage通道数
    dw_dims = [64, 128, 256]    # 深度卷积通道数

    # Dropout
    dropout = 0.1          # backbone dropout
    class_dropout = 0.1    # 分类头dropout
    head_dropout = 0.1     # 兼容参数

    # 特性开关
    small_kernel_merged = False  # 训练时为False，推理时可设为True
    use_multi_scale = True       # 启用多尺度特征
    revin = True                 # 启用RevIN归一化
    affine = True                # RevIN可学习参数
    subtract_last = False        # 使用均值归一化

    # 个体化处理（分类任务通常设为False）
    individual = False



