"""
MHF-optimized Geometry-Informed Neural Operator (GINO)
"""

from typing import Tuple, List, Union, Literal, Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn as F

from neuralop.models.gino import GINO
from neuralop.models.fnogno import FNOGNO
from neuralop.layers.gno_block import GNOBlock
from neuralop.layers.spectral_convolution import SpectralConv

from mhf.base import BaseMHF
from mhf.factorization import get_factorization


class MHF_GINO(GINO):
    """MHF-optimized Geometry-Informed Neural Operator
    
    GINO combines FNO on regular grids with GNO on point clouds.
    MHF optimizes the FNO spectral convolutions.
    
    Parameters
    ----------
    All parameters from GINO are preserved, plus MHF parameters:
    
    mhf_rank : Union[int, Dict[str, int]], default=8
        Rank for MHF factorization on FNO layers
    mhf_factorization : str, default="tucker"
        Tensor factorization type
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
        gno_coord_dim: int = 3,
        # MHF parameters
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_factorization: str = "tucker",
        **kwargs,
    ):
        from mhf.spectral_mhf import SpectralConvMHF
        
        # Set MHF attributes before GINO init because spectral_conv_factory needs them
        self._decomposed = False
        self.mhf_rank = mhf_rank
        self.mhf_factorization = mhf_factorization
        
        # Calculate resolutions based on n_modes - stored for MHF interface
        if isinstance(n_modes, tuple):
            self._resolutions = [2 * m for m in n_modes]
        else:
            self._resolutions = [2 * n_modes]
        
        # Factory for MHF spectral convolutions
        def spectral_conv_factory(*args, **factory_kwargs):
            # args: (in_channels, out_channels, n_modes)
            n_modes_arg = args[2] if len(args) >= 3 else 16
            # Remove any existing factorization from factory_kwargs
            if 'factorization' in factory_kwargs:
                del factory_kwargs['factorization']
            return SpectralConvMHF(
                *args,
                mhf_rank=self._get_rank_for_conv(n_modes_arg),
                factorization=self.mhf_factorization,
                **factory_kwargs,
            )
        
        # Convert parameter names to match base GINO
        fno_hidden_channels = hidden_channels
        fno_n_layers = n_layers
        fno_n_modes = n_modes
        in_gno_radius = in_radius
        out_gno_radius = out_radius
        
        # Initialize GINO
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            fno_hidden_channels=fno_hidden_channels,
            fno_n_layers=fno_n_layers,
            fno_n_modes=fno_n_modes,
            gno_coord_dim=gno_coord_dim,
            in_gno_radius=in_gno_radius,
            out_gno_radius=out_gno_radius,
            fno_channel_mlp_expansion=fno_channel_mlp_expansion,
            fno_non_linearity=fno_activation,
            in_gno_channel_mlp_hidden_layers=[gno_mlp_hidden_dim, gno_mlp_hidden_dim] if gno_mlp_layers >= 2 else [gno_mlp_hidden_dim],
            out_gno_channel_mlp_hidden_layers=[gno_mlp_hidden_dim * 2, gno_mlp_hidden_dim] if gno_mlp_layers >= 2 else [gno_mlp_hidden_dim],
            gno_channel_mlp_non_linearity=fno_activation,
            fno_conv_module=spectral_conv_factory,
            gno_use_open3d=False,
            **kwargs,
        )
    
    def _get_rank_for_conv(self, n_modes: Union[int, Tuple[int, ...]]) -> int:
        """Get rank for a specific convolution based on modes count"""
        if isinstance(self.mhf_rank, dict):
            key = str(min(n_modes) if isinstance(n_modes, tuple) else n_modes)
            if key in self.mhf_rank:
                return self.mhf_rank[key]
            elif "default" in self.mhf_rank:
                return self.mhf_rank["default"]
            else:
                return 8
        else:
            return self.mhf_rank
    
    def decompose(self) -> None:
        """Perform MHF decomposition on all applicable layers"""
        from mhf.spectral_mhf import SpectralConvMHF
        
        def decompose_recursive(module: nn.Module):
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            for child in module.children():
                decompose_recursive(child)
        
        decompose_recursive(self)
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct full weights from decomposition"""
        from mhf.spectral_mhf import SpectralConvMHF
        
        def recompose_recursive(module: nn.Module):
            if isinstance(module, SpectralConvMHF):
                # module.recompose() handles the case when weight doesn't exist
                module.recompose()
            for child in module.children():
                recompose_recursive(child)
        
        recompose_recursive(self)
    
    def forward_mhf(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass guaranteed to use MHF decomposition"""
        assert self._decomposed, "Must call decompose() before using forward_mhf"
        return self.forward(x, *args, **kwargs)
    
    def get_compression_stats(self) -> dict:
        """Get overall compression statistics"""
        from mhf.spectral_mhf import SpectralConvMHF
        total_original = 0
        total_decomposed = 0
        
        def collect_stats(module: nn.Module):
            nonlocal total_original, total_decomposed
            if isinstance(module, SpectralConvMHF):
                orig, decomp = module.count_parameters()
                total_original += orig
                total_decomposed += decomp
            for child in module.children():
                collect_stats(child)
        
        collect_stats(self)
        
        if total_original == 0:
            return {
                "decomposed": self._decomposed,
                "total_original_params": 0,
                "total_decomposed_params": 0,
                "overall_compression_ratio": 1.0,
                "overall_compression_factor": 1.0,
            }
        
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_original,
            "total_decomposed_params": total_decomposed,
            "overall_compression_ratio": total_decomposed / total_original,
            "overall_compression_factor": total_original / total_decomposed,
        }


class MHFFNOGNO(FNOGNO, BaseMHF):
    """MHF-optimized FNOGNO (hybrid FNO-GNN operator)"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_dim: int = 2,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_modes: Tuple[int, ...] = (16, 16),
        radius: float = 0.1,
        # MHF parameters
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_factorization: str = "tucker",
        **kwargs,
    ):
        from mhf.spectral_mhf import SpectralConvMHF
        
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=[],
            factorization=mhf_factorization,
        )
        
        def spectral_conv_factory(*args, **factory_kwargs):
            return SpectralConvMHF(*args, mhf_rank=mhf_rank, 
                                  factorization=mhf_factorization, **factory_kwargs)
        
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
        
        self.mhf_rank = mhf_rank
        self._decomposed = False
    
    def decompose(self) -> None:
        from mhf.spectral_mhf import SpectralConvMHF
        def decompose_r(module):
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            for child in module.children():
                decompose_r(child)
        decompose_r(self)
        self._decomposed = True
    
    def recompose(self) -> None:
        from mhf.spectral_mhf import SpectralConvMHF
        def recompose_r(module):
            if isinstance(module, SpectralConvMHF):
                # module.recompose() handles the case when weight doesn't exist
                module.recompose()
            for child in module.children():
                recompose_r(child)
        recompose_r(self)
    
    def get_compression_stats(self) -> dict:
        from mhf.spectral_mhf import SpectralConvMHF
        total_o = 0
        total_d = 0
        def stats_r(module):
            nonlocal total_o, total_d
            if isinstance(module, SpectralConvMHF):
                o, d = module.count_parameters()
                total_o += o
                total_d += d
            for child in module.children():
                stats_r(child)
        stats_r(self)
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_o,
            "total_decomposed_params": total_d,
            "compression_ratio": total_d / max(1, total_o),
            "compression_factor": total_o / max(1, total_d),
        }
