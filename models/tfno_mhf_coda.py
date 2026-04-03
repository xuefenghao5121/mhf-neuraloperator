"""
MHF-optimized TFNO with CoDA (Cross-head Attention)
"""

from typing import Tuple, List, Union, Literal, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models.tfno import TFNO
from neuralop.layers.spectral_convolution import SpectralConv

from mhf.base import BaseMHF
from mhf.coda import CrossHeadAttention


class SpectralConvMHFWithCoDA(nn.Module):
    """
    带 CoDA 的频谱卷积层（适配 TFNO）
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

        self.use_coda = use_coda
        self.n_heads = n_heads
        self.n_modes = n_modes

        # 检查通道数是否能被 n_heads 整除
        if use_coda and (in_channels % n_heads != 0 or out_channels % n_heads != 0):
            raise ValueError(
                f"通道数 ({in_channels}, {out_channels}) 必须能被 n_heads ({n_heads}) 整除"
            )

        # 标准 SpectralConv
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


class MHF_TFNO_CoDA(TFNO, BaseMHF):
    """MHF + CoDA 优化的 TFNO

    TFNO (Tensorized Fourier Neural Operator) 已经使用张量分解，
    此类在频谱卷积层添加 CoDA 跨头注意力机制。
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: float = 2.0,
        projection_channel_ratio: float = 2.0,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Optional[Literal["ada_in", "group_norm", "instance_norm"]] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Optional[Union[float, List[float]]] = None,
        domain_padding: Optional[Union[float, List[float]]] = None,
        # MHF + CoDA 参数
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        mhf_factorization: str = "tucker",
        n_heads: int = 4,
        use_coda: bool = True,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
    ):
        # Initialize BaseMHF
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=mhf_resolutions or [],
            factorization=mhf_factorization,
        )

        # 初始化 TFNO
        TFNO.__init__(
            self,
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=resolution_scaling_factor,
            domain_padding=domain_padding,
        )

        # 替换 TFNO 的 SpectralConv 为带 CoDA 的版本
        for i, conv in enumerate(self.fno_blocks.convs):
            self.fno_blocks.convs[i] = SpectralConvMHFWithCoDA(
                in_channels=hidden_channels if i > 0 else in_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                use_coda=use_coda,
                coda_reduction=coda_reduction,
                coda_dropout=coda_dropout,
            )

        self._decomposed = False
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
        coda_params = 0
        if self.use_coda:
            for conv in self.fno_blocks.convs:
                if hasattr(conv, 'coda'):
                    coda_params += sum(p.numel() for p in conv.coda.parameters())

        return {
            "decomposed": self._decomposed,
            "total_original_params": total_params,
            "total_decomposed_params": total_params,
            "coda_params": coda_params,
            "compression_ratio": 1.0,
            "compression_factor": 1.0,
        }


class MHF_TFNO_Baseline(TFNO, BaseMHF):
    """TFNO Baseline (不带 MHF 或 CoDA)"""

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        **kwargs,
    ):
        BaseMHF.__init__(
            self,
            ranks=8,
            resolutions=[],
            factorization="tucker",
        )

        TFNO.__init__(
            self,
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **kwargs,
        )

        self._decomposed = False

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
    "MHF_TFNO_CoDA",
    "MHF_TFNO_Baseline",
]
