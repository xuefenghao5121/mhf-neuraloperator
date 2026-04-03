"""
Cross-Head Attention (CoDA) for MHF-NeuralOperator
===================================================

CoDA = Cross-head Attention，用于解决 MHF 的头独立性假设问题。

参考：
    - mhf-fno 项目: /home/huawei/.openclaw/workspace/tianyuan-mhf-fno/mhf_fno/mhf_attention.py
    - TransFourier: Multi-Head Attention in Spectral Domain
    - Squeeze-and-Excitation Networks (SENet)

版本: 1.0.0
作者: Tianyuan Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossHeadAttention(nn.Module):
    """
    轻量级跨头注意力模块 (SENet 风格)
    
    用于 MHF 的多头之间建立信息交互，解决头独立性假设问题。
    
    设计原理:
        1. 全局平均池化: [B, n_heads, C, H, W] -> [B, n_heads, C]
        2. 头间注意力: 使用 Q/K/V 投影计算注意力权重
        3. 门控融合: 混合原始特征和注意力增强特征
    
    参数效率:
        - 参数量: ~3 * C^2 (Q/K/V 投影) + FFN
        - 相比展开空间维度的方案，参数减少 99%+
    
    Args:
        n_heads: MHF 头数量
        channels_per_head: 每个头的通道数
        reduction: 中间层缩减比例，默认 4
        dropout: Dropout 概率，默认 0.0
    """
    
    def __init__(
        self,
        n_heads: int,
        channels_per_head: int,
        reduction: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.channels_per_head = channels_per_head
        
        # 嵌入维度 = 每个头的通道数
        self.embed_dim = channels_per_head
        hidden_dim = max(channels_per_head // reduction, 4)
        
        # 头间注意力层 (Q/K/V 投影)
        self.query = nn.Linear(channels_per_head, channels_per_head, bias=False)
        self.key = nn.Linear(channels_per_head, channels_per_head, bias=False)
        self.value = nn.Linear(channels_per_head, channels_per_head, bias=False)
        
        # 缩放因子
        self.scale = channels_per_head ** -0.5
        
        # 输出投影 + FFN
        self.out_proj = nn.Linear(channels_per_head, channels_per_head)
        self.ffn = nn.Sequential(
            nn.Linear(channels_per_head, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels_per_head),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(channels_per_head)
        self.norm2 = nn.LayerNorm(channels_per_head)
        
        # 门控参数
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (B, n_heads, C_per_head, H, W) 或 (B, n_heads, C_per_head, L)
            
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        B = x.shape[0]
        original_shape = x.shape
        
        # Step 1: 全局平均池化
        if x.dim() == 4:  # 1D: [B, n_heads, C, L]
            x_pooled = x.mean(dim=-1)  # [B, n_heads, C]
        else:  # 2D: [B, n_heads, C, H, W]
            x_pooled = x.mean(dim=(-2, -1))  # [B, n_heads, C]
        
        # Step 2: 头间注意力
        residual = x_pooled
        x_norm = self.norm1(x_pooled)
        
        # Q, K, V 投影
        Q = self.query(x_norm)  # [B, n_heads, C]
        K = self.key(x_norm)
        V = self.value(x_norm)
        
        # 计算注意力权重 [B, n_heads, n_heads]
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 注意力输出
        attn_out = torch.matmul(attn_weights, V)  # [B, n_heads, C]
        attn_out = self.out_proj(attn_out)
        
        # 残差连接
        x_attn = residual + attn_out
        
        # Step 3: FFN
        residual = x_attn
        x_norm = self.norm2(x_attn)
        ffn_out = self.ffn(x_norm)
        x_out = residual + ffn_out
        
        # Step 4: 门控融合
        gate = torch.sigmoid(self.gate)
        attention_weights = gate * (x_out - x_pooled) + x_pooled
        
        # 广播回空间维度并应用
        if x.dim() == 4:  # 1D
            attention_weights = attention_weights.unsqueeze(-1)  # [B, n_heads, C, 1]
        else:  # 2D
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, C, 1, 1]
        
        # 逐元素调制
        out = x * (1 + attention_weights)
        
        return out


class SpectralConvMHFWithCoDA(nn.Module):
    """
    带 CoDA 的 MHF 频谱卷积层
    
    在 MHF 的基础上添加跨头注意力机制。
    
    工作流程:
        1. MHF 频谱卷积 (多头分解)
        2. IFFT 转换到空域
        3. CoDA 跨头注意力
        4. 合并多头输出
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        n_modes: 频率模式数元组
        n_heads: 多头数量，默认 4
        use_coda: 是否启用 CoDA，默认 True
        coda_reduction: CoDA 中间层缩减比例，默认 4
        coda_dropout: CoDA dropout 概率，默认 0.0
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        n_heads: int = 4,
        use_coda: bool = True,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        from mhf.spectral_mhf import SpectralConvMHF
        
        # MHF 频谱卷积
        self.mhf_conv = SpectralConvMHF(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            **kwargs
        )
        
        # CoDA 配置
        self.use_coda = use_coda
        self.n_heads = n_heads
        
        if use_coda:
            # 检查通道数是否能被 n_heads 整除
            if out_channels % n_heads != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) 必须能被 n_heads ({n_heads}) 整除"
                )
            
            channels_per_head = out_channels // n_heads
            
            # CoDA 模块
            self.coda = CrossHeadAttention(
                n_heads=n_heads,
                channels_per_head=channels_per_head,
                reduction=coda_reduction,
                dropout=coda_dropout
            )
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(out_channels, *(1,) * len(n_modes)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量 [B, C, H, W] 或 [B, C, L]
            
        Returns:
            输出张量，形状与输入相同
        """
        # MHF 频谱卷积
        x_out = self.mhf_conv(x)
        
        # CoDA 跨头注意力
        if self.use_coda:
            if x.dim() == 3:  # 1D: [B, C, L]
                B, C, L = x_out.shape
                x_heads = x_out.view(B, self.n_heads, C // self.n_heads, L)
                x_heads = self.coda(x_heads)
                x_out = x_heads.view(B, C, L)
            else:  # 2D: [B, C, H, W]
                B, C, H, W = x_out.shape
                x_heads = x_out.view(B, self.n_heads, C // self.n_heads, H, W)
                x_heads = self.coda(x_heads)
                x_out = x_heads.view(B, C, H, W)
        
        # 添加偏置
        x_out = x_out + self.bias
        
        return x_out


# 导出公共 API
__all__ = [
    'CrossHeadAttention',
    'SpectralConvMHFWithCoDA',
]
