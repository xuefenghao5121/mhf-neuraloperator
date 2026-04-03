"""
MHF-optimized Recurrent Neural Operator (RNO)
"""

from typing import Tuple, List, Union, Literal, Optional

import torch
import torch.nn as nn

from neuralop.models.rno import RNO
from neuralop.layers.spectral_convolution import SpectralConv
from mhf.spectral_mhf import SpectralConvMHF
from mhf.base import BaseMHF


class MHFRNO(RNO, BaseMHF):
    """MHF-optimized Recurrent Neural Operator (RNO)
    
    RNO uses recurrent architecture with spectral convolutions.
    MHF optimizes the spectral convolution layers.
    
    Parameters
    ----------
    All RNO parameters plus:
    mhf_rank : Union[int, List[int]], default=8
        Rank for MHF factorization
    mhf_factorization : str, default="tucker"
        Tensor factorization type
    """
    
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        recurrent_features: int = 32,
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
        
        def spectral_conv_factory(in_ch, out_ch, modes, **factory_kwargs):
            return SpectralConvMHF(
                in_ch, out_ch, modes,
                mhf_rank=mhf_rank,
                factorization=mhf_factorization,
                **factory_kwargs
            )
        
        RNO.__init__(
            self,
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            recurrent_features=recurrent_features,
            SpectralConv=spectral_conv_factory,
            **kwargs,
        )
        
        self._decomposed = False
    
    def decompose(self) -> None:
        """Perform MHF decomposition"""
        def decompose_r(module):
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            for child in module.children():
                decompose_r(child)
        decompose_r(self)
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct full weights"""
        def decompose_r(module):
            if isinstance(module, SpectralConvMHF):
                full = module.recompose()
                with torch.no_grad():
                    module.weight.copy_(full)
            for child in module.children():
                decompose_r(child)
        decompose_r(self)
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
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
