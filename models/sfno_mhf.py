"""
MHF-optimized Spherical Fourier Neural Operator (SFNO)
"""

from typing import Tuple, List, Union, Literal, Optional

import torch
import torch.nn as nn

from neuralop.models.sfno import SFNO
from mhf.base import BaseMHF
from mhf.spectral_mhf import SpectralConvMHF


class MHSFNO(SFNO, BaseMHF):
    """MHF-optimized Spherical Fourier Neural Operator (SFNO)
    
    SFNO works on spherical domains using spherical harmonics.
    MHF can optimize the spherical spectral convolutions.
    
    Parameters
    ----------
    All SFNO parameters plus MHF parameters
    """
    
    def __init__(
       self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        max_sh_degree: int = 20,
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
        
        # SFNO requires torch_harmonics
        # We use the same constructor but replace spectral conv with MHF if needed
        SFNO.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            max_sh_degree=max_sh_degree,
            **kwargs,
        )
        
        self._decomposed = False
        self.mhf_rank = mhf_rank
        self.mhf_factorization = mhf_factorization
    
    def decompose(self) -> None:
        """Perform MHF decomposition on spherical spectral convolutions"""
        # SFNO uses SphericalConv which inherits from base spectral conv
        # We can apply MHF if it's compatible
        from neuralop.layers.spherical_convolution import SphericalConv
        
        def decompose_r(module: nn.Module):
            # For SFNO, we can still use MHF on any spectral conv components
            from mhf.spectral_mhf import SpectralConvMHF
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            elif isinstance(module, SphericalConv):
                # TODO: Add MHF support for SphericalConv
                pass
            for child in module.children():
                decompose_r(child)
        
        decompose_r(self)
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct full weights from decomposition"""
        from mhf.spectral_mhf import SpectralConvMHF
        
        def recompose_r(module: nn.Module):
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
            "note": "SFNO spherical convolution MHF support is work in progress",
        }
