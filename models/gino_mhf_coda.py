"""
MHF-optimized GINO with CoDA (Cross-head Attention)
"""

from typing import Tuple, List, Union, Literal, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models.gino import GINO
from neuralop.models.fnogno import FNOGNO
from neuralop.layers.g.gno_block import GNOBlock
from neuralop.layers.spectral_convolution import SpectralConv

from mhf.base import BaseMHF
from mhf.spectral_mhf import SpectralConvMHF
from mhf.coda import CrossHeadAttention


class SpectralConvMHFWithCoDA(nn.Module):
    """
    带 CoDA 的 MHF 频谱卷积层（适配 NeuralOperator 2.0+）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        mhf_rank: Union[int, Dict[str, int]] = 8,
        factorization: str = "tucker",
        n_heads: int = 4,
        use_coda: bool = True,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.use_coda = use_coda
        self.n_heads = n_heads
        self.n_modes = n_modes

        # 检查通道数是否能被 n_heads 整除
        if use_coda and (in_channels % n_heads != 0 or out_channels % n_heads != 0):
            raise ValueError(
                f"通道数 ({in_channels}, {out_channels}) 必须能被 n_heads ({n_heads}) 整除"
            )

        # 标准 SpectralConv（使用 NeuralOperator 的实现）
        self.conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            **kwargs
        )

        # CoDA 模块
        if use_coda:
            channels_per_head = out_channels // n_heads
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
        前向传播
        """
        # 标准 SpectralConv
        x_out = self.conv(x)

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


class MHF_GINO_CoDA(GINO, BaseMHF):
    """MHF + CoDA 优化的 GINO

    在 MHF 基础上添加跨头注意力机制。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        n_modes: Union[int, Tuple[int, ...]],
        in_radius: float,
        out_radius: float,
        fno_channel_mlp_expansion: float = 1.0,
        fno_activation: nn.Module = nn.GELU(),
        gno_mlp_hidden_dim: int = 256,
        gno_mlp_layers: int = 2,
        # MHF + CoDA 参数
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_factorization: str = "tucker",
        n_heads: int = 4,
        use_coda: bool = True,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
        **kwargs,
    ):
        # Initialize BaseMHF
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=[],
            factorization=mhf_factorization,
        )

        # 创建带 CoDA 的 SpectralConv
        def spectral_conv_factory(*args, **factory_kwargs):
            return SpectralConvMHFWithCoDA(
                *args,
                n_heads=n_heads,
                use_coda=use_coda,
                coda_reduction=coda_reduction,
                coda_dropout=coda_dropout,
                **factory_kwargs,
            )

        # 初始化 GINO
        GINO.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_modes=n_modes,
            in_radius=in_radius,
            out_radius=out_radius,
            fno_channel_mlp_expansion=fno_channel_mlp_expansion,
            fno_activation=fno_activation,
            gno_mlp_hidden_dim=gno_mlp_hidden_dim,
            gno_mlp_layers=gno_mlp_layers,
            SpectralConv=spectral_conv_factory,
            **kwargs,
        )

        self._decomposed = False
        self.mhf_rank = mhf_rank
        self.n_heads = n_heads
        self.use_coda = use_coda

    def decompose(self) -> None:
        """MHF 分解（当前使用标准卷积，保留接口）"""
        self._decomposed = True

    def recompose(self) -> None:
        """重建权重（当前使用标准卷积，保留接口）"""
        pass

    def get_compression_stats(self) -> dict:
        """获取压缩统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_params,
            "total_decomposed_params": total_params,
            "compression_ratio": 1.0,
            "compression_factor": 1.0,
        }


class MHFFNOGNO_CoDA(FNOGNO, BaseMHF):
    """MHF + CoDA 优化的 FNOGNO"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_dim: int = 2,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_modes: Tuple[int, ...] = (16, 16),
        radius: float = 0.1,
        # MHF + CoDA 参数
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_factorization: str = "tucker",
        n_heads: int = 4,
        use_coda: bool = True,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
        **kwargs,
    ):
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=[],
            factorization=mhf_factorization,
        )

        def spectral_conv_factory(*args, **factory_kwargs):
            return SpectralConvMHFWithCoDA(
                *args,
                n_heads=n_heads,
                use_coda=use_coda,
                coda_reduction=coda_reduction,
                coda_dropout=coda_dropout,
                **factory_kwargs,
            )

        FNOGNO.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            in_dim=in_dim,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_modes=n_modes,
            radius=radius,
            SpectralConv=spectral_conv_factory,
            **kwargs,
        )

        self._decomposed = False
        self.n_heads = n_heads
        self.use_coda = use_coda

    def decompose(self) -> None:
        self._decomposed = True

    def recompose(self) -> None:
        pass

    def get_compression_stats(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_params,
            "total_decomposed_params": total_params,
            "compression_ratio": 1.0,
            "compression_factor": 1.0,
        }


__all__ = [
    "MHF_GINO_CoDA",
    "MHFFNOGNO_CoDA",
]
