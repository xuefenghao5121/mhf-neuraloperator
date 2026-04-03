"""
MHF-optimized spectral convolution for Fourier-based neural operators
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import tensorly as tl

from neuralop.layers.base_spectral_conv import BaseSpectralConv
from .base import MultiResolutionHierarchicalFactorization


class SpectralConvMHF(BaseSpectralConv):
    """MHF-optimized Spectral Convolution
    
    This class extends the original spectral convolution with
    Multi-Resolution Hierarchical Factorization for parameter compression.
    
    It maintains the same interface as the original SpectralConv in neuraloperator,
    so it can be used as a drop-in replacement.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : Tuple[int, ...]
        Number of Fourier modes to keep along each dimension
    mhf_rank : Union[int, List[int]], default=8
        Rank(s) for MHF factorization. If list, specifies rank per level.
    mhf_resolutions : List[int], optional
        Resolutions for MHF hierarchy. If None, automatically generated.
    factorization : str, default="tucker"
        Tensor factorization type: "cp", "tucker", or "tt"
    implementation : str, default="factorized"
        Implementation mode:
        - "reconstructed": Reconstruct full weight before forward
        - "factorized": Use factors directly for contraction (more memory efficient)
    separable : bool, default=False
        Whether to use separable factorization
    complex_data : bool, default=False
        Whether the data is complex-valued
    **kwargs
        Additional parameters passed to base SpectralConv
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        factorization: str = "tucker",
        implementation: str = "factorized",
        separable: bool = False,
        complex_data: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, n_modes, complex_data=complex_data)
        
        # Store original dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.dim = len(n_modes)
        
        # MHF parameters
        self.mhf_rank = mhf_rank
        self.factorization = factorization
        self.implementation = implementation
        self.separable = separable
        
        # Auto-generate resolutions if not provided
        if mhf_resolutions is None:
            self.mhf_resolutions = self._auto_resolutions(n_modes)
        else:
            self.mhf_resolutions = sorted(mhf_resolutions)
        
        # Create MHF decomposition
        self.mhf = MultiResolutionHierarchicalFactorization(
            resolutions=self.mhf_resolutions,
            ranks=mhf_rank,
            factorization_type=factorization,
        )
        
        # Flag indicating whether MHF decomposition has been performed
        self._mhf_decomposed = False
        
        # Initialize weight as usual (same as original SpectralConv)
        # This means the model can be trained normally first, then compressed
        self._init_weight()
    
    def _auto_resolutions(self, n_modes: Tuple[int, ...]) -> List[int]:
        """Automatically generate hierarchical resolutions
        
        Creates a logarithmic sequence from min(n_modes) to max(n_modes)
        """
        max_res = max(n_modes)
        resolutions = []
        current = 4
        while current <= max_res:
            resolutions.append(current)
            current = current * 2
        if resolutions[-1] != max_res:
            resolutions.append(max_res)
        return resolutions
    
    def _init_weight(self):
        """Initialize weight same as original spectral convolution"""
        # This follows the original initialization in neuraloperator
        scale = (1 / (self.in_channels * self.out_channels))
        if self.dim == 1:
            self.weight = nn.Parameter(scale * torch.rand(self.in_channels, self.out_channels, self.n_modes[0], dtype=self.cdtype))
        elif self.dim == 2:
            self.weight = nn.Parameter(scale * torch.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1], dtype=self.cdtype))
        elif self.dim == 3:
            self.weight = nn.Parameter(scale * torch.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1], self.n_modes[2], dtype=self.cdtype))
        elif self.dim == 4:
            self.weight = nn.Parameter(scale * torch.rand(self.in_channels, self.out_channels, self.n_modes[0], self.n_modes[1], self.n_modes[2], self.n_modes[3], dtype=self.cdtype))
    
    def decompose(self) -> None:
        """Perform MHF decomposition on the current weight
        
        This should be called after training the full model to compress it.
        """
        # Get spatial dimensions (after in_channels, out_channels)
        spatial_dims = list(range(2, 2 + self.dim))
        
        # Perform MHF decomposition
        self.mhf.decompose(self.weight.data, spatial_dims=spatial_dims)
        
        # Optionally free the original weight to save memory
        # We still keep it for compatibility/recompose
        self._mhf_decomposed = True
    
    def recompose(self) -> torch.Tensor:
        """Reconstruct full weight from MHF decomposition"""
        assert self._mhf_decomposed, "Must decompose before recomposing"
        return self.mhf.reconstruct()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        If MHF decomposition has been performed, uses the decomposed factors
        directly for more efficient computation. Otherwise falls back to
        the original dense weight computation.
        """
        if self._mhf_decomposed and self.implementation == "factorized":
            return self._forward_mhf(x)
        else:
            return self._forward_original(x)
    
    def _forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using MHF decomposition"""
        # Contract input with MHF factors directly
        # This follows the same contraction pattern as original SpectralConv
        # but uses the decomposed factors to avoid storing full weight
        
        # The einsum pattern is:
        # batch, in_channels, x_1, x_2, ... -> batch, out_channels, x_1, x_2, ...
        # This contraction is handled by the MHF class
        
        return self.mhf.forward(x)
    
    def _forward_original(self, x: torch.Tensor) -> torch.Tensor:
        """Original forward pass with dense weight
        
        Fallback when MHF hasn't been applied or reconstructed implementation is used.
        """
        # Original implementation from neuraloperator
        if self.dim == 1:
            return self._forward_1d(x)
        elif self.dim == 2:
            return self._forward_2d(x)
        elif self.dim == 3:
            return self._forward_3d(x)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """1D spectral convolution"""
        return torch.einsum("bix,iox->box", x, self.weight)
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D spectral convolution"""
        return torch.einsum("bixy,ioxy->boxy", x, self.weight)
    
    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """3D spectral convolution"""
        return torch.einsum("bixyz,ioxyz->boxyz", x, self.weight)
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        if not self._mhf_decomposed:
            return {
                "decomposed": False,
                "original_params": self.weight.numel(),
                "decomposed_params": None,
                "compression_ratio": None,
                "compression_factor": None,
            }
        
        original_params = self.weight.numel()
        decomposed_params = self.mhf.count_params()
        
        return {
            "decomposed": True,
            "original_params": original_params,
            "decomposed_params": decomposed_params,
            "compression_ratio": decomposed_params / original_params,
            "compression_factor": original_params / decomposed_params,
        }
    
    @property
    def is_mhf_decomposed(self) -> bool:
        """Check if MHF decomposition has been performed"""
        return self._mhf_decomposed
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count original vs decomposed parameters"""
        original = self.weight.numel()
        if not self._mhf_decomposed:
            return original, original
        decomposed = self.mhf.count_params()
        return original, decomposed


def from_original(
    original_conv: BaseSpectralConv,
    mhf_rank: Union[int, List[int]] = 8,
    mhf_resolutions: Optional[List[int]] = None,
    factorization: str = "tucker",
    implementation: str = "factorized",
) -> SpectralConvMHF:
    """Create MHF-optimized convolution from an existing original convolution
    
    This is a convenience function to convert an already-trained
    original spectral convolution to MHF-optimized version.
    
    Parameters
    ----------
    original_conv : BaseSpectralConv
        Original trained spectral convolution
    mhf_rank : Union[int, List[int]], default=8
        Rank for MHF factorization
    mhf_resolutions : List[int], optional
        Resolutions for MHF hierarchy
    factorization : str, default="tucker"
        Factorization type
    implementation : str, default="factorized"
        Implementation mode
        
    Returns
    -------
    SpectralConvMHF
        MHF-optimized convolution with weights copied from original
    """
    # Extract dimensions from original
    in_channels = original_conv.in_channels
    out_channels = original_conv.out_channels
    n_modes = original_conv.n_modes
    complex_data = hasattr(original_conv, "complex_data") and original_conv.complex_data
    
    # Create new MHF convolution
    mhf_conv = SpectralConvMHF(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        mhf_rank=mhf_rank,
        mhf_resolutions=mhf_resolutions,
        factorization=factorization,
        implementation=implementation,
        complex_data=complex_data,
    )
    
    # Copy weight
    if hasattr(original_conv, "weight"):
        with torch.no_grad():
            mhf_conv.weight.copy_(original_conv.weight)
    
    # Perform decomposition
    mhf_conv.decompose()
    
    return mhf_conv
