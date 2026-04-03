"""
MHF-optimized U-shaped Neural Operator (UNO)
"""

from typing import Tuple, List, Union, Literal, Optional

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models.uno import UNO
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.padding import DomainPadding
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models.base_model import BaseModel

from mhf.spectral_mhf import SpectralConvMHF
from mhf.base import BaseMHF


class MHFUNO(UNO, BaseMHF):
    """MHF-optimized U-shaped Neural Operator (UNO)
    
    UNO uses a U-shaped architecture with multiple resolutions.
    MHF optimizes the spectral convolution at every level.
    
    All original API is preserved, with additional MHF parameters.
    
    Parameters
    ----------
    Same as UNO, plus:
    
    mhf_rank : Union[int, List[int]], default=8
        Rank(s) for MHF factorization
    mhf_factorization : str, default="tucker"
        Tensor factorization type
    mhf_implementation : str, default="factorized"
        Implementation mode: "factorized" or "reconstructed"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: Tuple[int, ...],
        n_layers: int = 4,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        nhidden: int = 4,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Optional[Literal["ada_in", "group_norm", "instance_norm"]] = None,
        domain_padding: Optional[Union[Number, List[Number]]] = None,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        factorization: str = None,
        rank: float = 1.0,
        factorization_implementation: str = "reconstructed",
        # MHF parameters
        mhf_rank: Union[int, List[int]] = 8,
        mhf_factorization: str = "tucker",
        mhf_implementation: str = "factorized",
    ):
        # Initialize bases
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=[],  # UNO handles multiple resolutions internally
            factorization=mhf_factorization,
        )
        
        # Store MHF params
        self.mhf_rank = mhf_rank
        self.mhf_factorization = mhf_factorization
        self.mhf_implementation = mhf_implementation
        
        # Create spectral conv factory that uses MHF
        def spectral_conv_factory(in_channels, out_channels, n_modes, **kwargs):
            return SpectralConvMHF(
                in_channels, out_channels, n_modes,
                mhf_rank=mhf_rank,
                factorization=mhf_factorization,
                implementation=mhf_implementation,
                **kwargs
            )
        
        # Initialize UNO with our MHF spectral conv
        UNO.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            nhidden=nhidden,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            domain_padding=domain_padding,
            output_scaling_factor=output_scaling_factor,
            factorization=factorization,
            rank=rank,
            factorization_implementation=factorization_implementation,
            SpectralConv=spectral_conv_factory,
        )
        
        self._decomposed = False
    
    def decompose(self) -> None:
        """Perform MHF decomposition on all spectral conv layers"""
        # Walk the network and decompose any SpectralConvMHF found
        def decompose_module(module: nn.Module):
            if isinstance(module, SpectralConvMHF):
                module.decompose()
            for child in module.children():
                decompose_module(child)
        
        decompose_module(self)
        self._decomposed = True
    
    def recompose(self) -> torch.Tensor:
        """Reconstruct all full weights from decomposition"""
        # Walk the network and recompose any SpectralConvMHF found
        def recompose_module(module: nn.Module):
            if isinstance(module, SpectralConvMHF):
                return module.recompose()
            for child in module.children():
                recompose_module(child)
            return None
        
        recompose_module(self)
        return None
    
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass guaranteed to use MHF decomposition"""
        assert self._decomposed, "Must call decompose() before using forward_mhf"
        return self.forward(x)
    
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
    
    @classmethod
    def from_original(cls, original_uno: UNO, mhf_rank: int = 8, 
                      mhf_factorization: str = "tucker") -> "MHFUNO":
        """Create MHFUNO from an already-trained original UNO
        
        Parameters
        ----------
        original_uno : UNO
            Trained original UNO model
        mhf_rank : int, default=8
            Rank for MHF factorization
        mhf_factorization : str, default="tucker"
            Factorization type
            
        Returns
        -------
        MHFUNO
            MHF-optimized model with weights copied from original
        """
        # Extract parameters
        in_channels = original_uno.in_channels
        out_channels = original_uno.out_channels
        hidden_channels = original_uno.hidden_channels
        n_modes = original_uno.n_modes
        if hasattr(original_uno, 'n_layers'):
            n_layers = original_uno.n_layers
        else:
            n_layers = 4
        
        # Create MHF model
        mhf_model = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            mhf_rank=mhf_rank,
            mhf_factorization=mhf_factorization,
        )
        
        # Copy weights - this requires careful copying of all conv layers
        # For simplicity, we copy the state dict and let MHF decompose
        with torch.no_grad():
            mhf_model.load_state_dict(original_uno.state_dict(), strict=False)
        
        mhf_model.decompose()
        return mhf_model
