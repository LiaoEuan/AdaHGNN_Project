# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ==============================================================================
# 1. 域对抗适应 (DANN) 组件
# ==============================================================================

class GradientReversalFunction(Function):
    """梯度反转层 (GRL) 的自动求导函数实现"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时将梯度取反并乘以 alpha
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (Gradient Reversal Layer).
    在前向传播中表现为恒等变换，在反向传播中将梯度取反。
    用于让特征提取器学习到“域无关”的特征。
    """
    def __init__(self, alpha: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        current_alpha = alpha if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, current_alpha)


# ==============================================================================
# 2. 时序对齐组件
# ==============================================================================

class LearnableTemporalShift(nn.Module):
    """
    可学习的时序平移模块 (Learnable Temporal Shift).
    
    作用: 自动学习每个 EEG 通道的最佳时间延迟，以对齐脑活动与音频刺激之间的延迟差异。
    原理: 使用可微的网格采样 (Grid Sampling) 实现亚像素级的时间偏移。
    """
    def __init__(self, num_channels, sampling_rate, max_shift_sec=0.2):
        super().__init__()
        self.num_channels = num_channels
        self.max_shift_steps = int(max_shift_sec * sampling_rate)
        
        # 定义可学习参数，初始化为 0
        self.shift_param = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: [Batch, Channel, Time]
        Returns:
            x_shifted: 对齐后的数据
        """
        B, C, T = x.shape
        
        # 计算偏移步数 (限制在 -max 到 +max 之间)
        shift_steps = self.tanh(self.shift_param) * self.max_shift_steps
        
        # 构建基础时间网格
        base_grid = torch.arange(T, device=x.device, dtype=torch.float32).view(1, 1, T)
        
        # 生成偏移后的采样网格: [1, C, T]
        new_grid = base_grid + shift_steps
        
        # --- 线性插值采样 ---
        idx_floor = torch.floor(new_grid).long()
        idx_ceil = idx_floor + 1
        alpha = new_grid - torch.floor(new_grid) # 插值权重
        
        # 边界处理 (Replicate Padding)
        idx_floor = torch.clamp(idx_floor, 0, T-1).expand(B, -1, -1)
        idx_ceil = torch.clamp(idx_ceil, 0, T-1).expand(B, -1, -1)
        alpha = alpha.expand(B, -1, -1)
        
        val_floor = torch.gather(x, 2, idx_floor)
        val_ceil = torch.gather(x, 2, idx_ceil)
        
        x_shifted = (1 - alpha) * val_floor + alpha * val_ceil
        return x_shifted


# ==============================================================================
# 3. 基础卷积组件
# ==============================================================================

class TemporalStem(nn.Module):
    """
    多尺度时间特征提取主干。
    使用多分支卷积捕捉不同时间尺度的特征，并进行降采样分块 (Patching)。
    """
    def __init__(self, out_planes: int, kernel_size: int, patch_size: int, radix: int = 2):
        super(TemporalStem, self).__init__()
        self.radix = radix
        self.in_planes = 1 # 单通道输入
        self.out_planes = out_planes
        self.mid_planes = self.out_planes * self.radix

        # 初始升维卷积
        self.sconv = nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.mid_planes)

        # 多分支并行卷积
        self.tconv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.out_planes, self.out_planes, ks, 1, padding=ks // 2, bias=False, groups=self.out_planes),
                nn.BatchNorm1d(self.out_planes)
            ) for ks in [kernel_size // (2**i) for i in range(self.radix)]
        ])

        # 特征融合与降采样
        self.interFre = lambda x: F.gelu(sum(x))
        self.downSampling = nn.AvgPool1d(patch_size, stride=patch_size)
        self.dp = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [Batch, 1, Time]
        out = self.bn1(self.sconv(x))
        branches = torch.split(out, self.out_planes, dim=1) if self.radix > 1 else [out]
        out_branches = [conv(branch) for conv, branch in zip(self.tconv, branches)]
        out_fused = self.interFre(out_branches)
        out_patched = self.downSampling(out_fused)
        return self.dp(out_patched)