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
        # Only pass device and dtype to parent
        parent_kwargs = {}
        if 'device' in kwargs:
            parent_kwargs['device'] = kwargs['device']
        if 'dtype' in kwargs:
            parent_kwargs['dtype'] = kwargs['dtype']
        super().__init__(**parent_kwargs)
        
        # Store dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.dim = len(n_modes)
        self.complex_data = complex_data
        
        # Set complex dtype
        if complex_data:
            self.cdtype = torch.complex64
        else:
            self.cdtype = torch.float32
        
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
        
        For direct factorization on the original resolution, we only need one level.
        Multi-level hierarchical is optional but not required for basic factorization.
        """
        max_res = max(n_modes)
        return [max_res]
    
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
        
        # Remove original weight to save parameters after decomposition
        # After decomposition, we only need the factors for forward pass
        # Original weight is not needed for factorized inference
        del self.weight
        self._mhf_decomposed = True
    
    def recompose(self) -> torch.Tensor:
        """Reconstruct full weight from MHF decomposition"""
        assert self._mhf_decomposed, "Must decompose before recomposing"
        full_weight = self.mhf.reconstruct()
        if not hasattr(self, 'weight'):
            # If original weight was removed during decomposition,
            # reconstruct it as a buffer (not a parameter)
            self.register_buffer('weight', full_weight)
            return full_weight
        else:
            with torch.no_grad():
                self.weight.copy_(full_weight)
        return full_weight
    
    def forward(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        """Forward pass
        
        If MHF decomposition has been performed, uses the decomposed factors
        directly for more efficient computation. Otherwise falls back to
        the original dense weight computation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        output_shape : tuple, optional
            Output shape for interpolation (required by FNOBlocks interface)
        """
        if output_shape is not None:
            # Transform input if output_shape is provided
            x = self.transform(x, output_shape)
        
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
        # If weight was deleted (after decompose), we need to reconstruct it
        if not hasattr(self, 'weight'):
            weight = self.mhf.reconstruct()
        else:
            weight = self.weight
        
        # Check if input size matches weight size
        # If input is larger (due to resolution scaling), we need to INTERPOLATE our weight
        # to match the input spatial dimensions. Because weight only keeps n_modes modes,
        # but input can be larger due to upsampling in FNOBlocks.
        input_spatial_sizes = x.shape[2:]
        weight_spatial_sizes = weight.shape[2:]
        
        if input_spatial_sizes != weight_spatial_sizes:
            # Reshape weight to (1, in_channels * out_channels, *weight_spatial_sizes)
            # interpolate expects: (N, C, d1, d2, ...) where N is batch, C is channels
            reshaped_weight = weight.reshape(1, self.in_channels * self.out_channels, *weight_spatial_sizes)
            reshaped_weight = nn.functional.interpolate(
                reshaped_weight, 
                size=input_spatial_sizes, 
                mode='bilinear' if self.dim == 2 else 'trilinear' if self.dim == 3 else 'linear',
                align_corners=False
            )
            # Reshape back
            weight = reshaped_weight.reshape(self.in_channels, self.out_channels, *input_spatial_sizes)
        
        # Einsum based on dimension
        if self.dim == 1:
            return torch.einsum("bix,iox->box", x, weight)
        elif self.dim == 2:
            return torch.einsum("bixy,ioxy->boxy", x, weight)
        elif self.dim == 3:
            return torch.einsum("bixyz,ioxyz->boxyz", x, weight)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """1D spectral convolution - kept for compatibility"""
        return self._forward_original(x)
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D spectral convolution - kept for compatibility"""
        return self._forward_original(x)
    
    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """3D spectral convolution - kept for compatibility"""
        return self._forward_original(x)
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        if not self._mhf_decomposed:
            original_params = self.weight.numel()
            return {
                "decomposed": False,
                "original_params": original_params,
                "decomposed_params": None,
                "compression_ratio": None,
                "compression_factor": None,
            }
        
        # If weight was deleted, we need to get original shape from MHF
        if hasattr(self, 'weight'):
            original_params = self.weight.numel()
        else:
            # Get original shape from MHF and calculate parameters
            original_shape = self.mhf._original_shape
            original_params = 1
            for dim in original_shape:
                original_params *= dim
        
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
        if hasattr(self, 'weight'):
            original = self.weight.numel()
        else:
            original_shape = self.mhf._original_shape
            original = 1
            for dim in original_shape:
                original *= dim
        
        if not self._mhf_decomposed:
            return original, original
        decomposed = self.mhf.count_params()
        return original, decomposed
    
    def transform(self, x, output_shape=None):
        """Implement transform for skip connection (required by BaseSpectralConv interface)
        
        Since we are not changing the resolution in MHF, this is identity.
        If your spectral conv changes resolution, override this in subclass.
        """
        if output_shape is not None:
            # Interpolate to output shape if required
            # This follows what original SpectralConv does
            spatial_dims = list(range(2, 2 + self.dim))
            scale_factors = [output_shape[i] / x.shape[i + 2] for i in range(self.dim)]
            return nn.functional.interpolate(
                x, scale_factor=scale_factors, mode='bilinear' if self.dim == 2 else 'trilinear'
            )
        return x


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
