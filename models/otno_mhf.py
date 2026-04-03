"""
MHF-optimized Optimal Transport Neural Operator (OTNO)
"""

from typing import Tuple, List, Union, Literal, Optional

import torch
import torch.nn as nn

from neuralop.models.otno import OTNO
from mhf.base import BaseMHF


class MHFOTNO(OTNO, BaseMHF):
    """MHF-optimized Optimal Transport Neural Operator
    
    OTNO uses optimal transport for grid alignment. The core spectral
    convolution can be optimized with MHF.
    
    Parameters
    ----------
    All OTNO parameters plus MHF parameters
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: Tuple[int, ...],
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
        
        # For OTNO, we use the base class and will decompose
        # any spectral convolutions it contains
        from mhf.spectral_mhf import SpectralConvMHF
        
        OTNO.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            **kwargs,
        )
        
        self._decomposed = False
    
    def decompose(self) -> None:
        """Perform MHF decomposition on any spectral convs"""
        from mhf.spectral_mhf import SpectralConvMHF
        
        def decompose_r(module):
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            for child in module.children():
                decompose_r(child)
        
        decompose_r(self)
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct full weights"""
        from mhf.spectral_mhf import SpectralConvMHF
        
        def recompose_r(module):
            if isinstance(module, SpectralConvMHF):
                full = module.recompose()
                with torch.no_grad():
                    module.weight.copy_(full)
            for child in module.children():
                recompose_r(child)
        
        recompose_r(self)
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        from mhf.spectral_mhf import SpectralConvMHF
        total_o = 0
        total_d = 0
        
        def collect(module):
            nonlocal total_o, total_d
            if isinstance(module, SpectralConvMHF):
                o, d = module.count_parameters()
                total_o += o
                total_d += d
            for child in module.children():
                collect(child)
        
        collect(self)
        
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_o,
            "total_decomposed_params": total_d,
            "compression_ratio": total_d / max(1, total_o),
            "compression_factor": total_o / max(1, total_d),
        }
