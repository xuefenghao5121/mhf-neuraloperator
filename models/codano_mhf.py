"""
MHF-optimized Conditional Neural Operator (CODANO)
"""

from typing import Tuple, List, Union, Literal, Optional, Any

import torch
import torch.nn as nn

from neuralop.models.codano import CODANO
from mhf.spectral_mhf import SpectralConvMHF
from mhf.base import BaseMHF


class MHFCODANO(CODANO, BaseMHF):
    """MHF-optimized Conditional Neural Operator (CODANO)
    
    CODANO uses Fourier-based transformer architecture.
    MHF optimizes the spectral convolution layers in the FNO backbone.
    
    Parameters
    ----------
    All CODANO parameters preserved, plus MHF parameters:
    
    mhf_rank : Union[int, List[int]], default=8
        Rank for MHF factorization
    mhf_factorization : str, default="tucker"
        Tensor factorization type
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        n_modes: Tuple[int, ...],
        n_cond_layers: int = 0,
        cond_projection_channels: int = 128,
        use_nonlinear_cond: bool = True,
        # MHF parameters
        mhf_rank: Union[int, List[int]] = 8,
        mhf_factorization: str = "tucker",
        **kwargs,
    ):
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=[],
            factorization=mhf_factorization,
        )
        
        # Create MHF spectral conv factory
        def spectral_conv_factory(in_ch, out_ch, modes, **factory_kwargs):
            return SpectralConvMHF(
                in_ch, out_ch, modes,
                mhf_rank=mhf_rank,
                factorization=mhf_factorization,
                **factory_kwargs
            )
        
        CODANO.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_modes=n_modes,
            n_cond_layers=n_cond_layers,
            cond_projection_channels=cond_projection_channels,
            use_nonlinear_cond=use_nonlinear_cond,
            SpectralConv=spectral_conv_factory,
            **kwargs,
        )
        
        self._decomposed = False
        self.mhf_rank = mhf_rank
        self.mhf_factorization = mhf_factorization
    
    def decompose(self) -> None:
        """Perform MHF decomposition on all spectral convs"""
        def decompose_r(module: nn.Module):
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            for child in module.children():
                decompose_r(child)
        
        decompose_r(self)
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct full weights from MHF decomposition"""
        def recompose_r(module: nn.Module):
            if isinstance(module, SpectralConvMHF):
                full = module.recompose()
                with torch.no_grad():
                    module.weight.copy_(full)
            for child in module.children():
                recompose_r(child)
        
        recompose_r(self)
    
    def forward_mhf(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with MHF decomposition"""
        assert self._decomposed, "Call decompose first"
        return self.forward(x, **kwargs)
    
    def get_compression_stats(self) -> dict:
        """Get overall compression statistics"""
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
        
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_original,
            "total_decomposed_params": total_decomposed,
            "compression_ratio": total_decomposed / max(1, total_original),
            "compression_factor": total_original / max(1, total_decomposed),
        }
