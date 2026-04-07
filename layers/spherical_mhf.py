"""
MHF-optimized Spherical Convolution for SFNO

This module implements MHF optimization for spherical spectral convolution
used in Spherical Fourier Neural Operator (SFNO).

Spherical convolution uses spherical harmonics (via torch_harmonics library)
to perform spectral operations on spherical domains.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from torch_harmonics import RealSHT, InverseRealSHT
    TORCH_HARMONICS_AVAILABLE = True
except ImportError:
    TORCH_HARMONICS_AVAILABLE = False
    # Will be defined as placeholders
    RealSHT = None
    InverseRealSHT = None

from mhf.base import MultiResolutionHierarchicalFactorization


class SphericalConvMHF(nn.Module):
    """MHF-optimized Spherical Spectral Convolution
    
    This class extends spherical spectral convolution with
    Multi-Resolution Hierarchical Factorization for parameter compression.
    
    The spherical convolution operates on spectral coefficients using
    spherical harmonics. MHF decomposes the weight tensor across
    different spherical harmonic orders.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    grid_shape : Tuple[int, int]
        Grid shape for spherical grid (nlat, nlon)
        nlat: number of latitudes, nlon: number of longitudes
    max_degree : int, optional
        Maximum spherical harmonic degree. If None, uses grid_shape[0] // 2
    mhf_rank : Union[int, List[int]], default=8
        Rank(s) for MHF factorization. If list, specifies rank per level.
    mhf_resolutions : List[int], optional
        Resolutions for MHF hierarchy (spherical harmonic orders). 
        If None, automatically generated.
    factorization : str, default="tucker"
        Tensor factorization type: "cp", "tucker", or "tt"
    implementation : str, default="factorized"
        Implementation mode:
        - "reconstructed": Reconstruct full weight before forward
        - "factorized": Use factors directly for contraction (more memory efficient)
    grid_type : str, default="equiangular"
        Type of spherical grid: "equiangular" or "legendre-gauss"
    norm : str, default="ortho"
        Normalization type for spherical harmonics
    hard_cutoff : bool, default=False
        Whether to use hard cutoff for maximum degree
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_shape: Tuple[int, int],
        max_degree: Optional[int] = None,
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        factorization: str = "tucker",
        implementation: str = "factorized",
        grid_type: str = "equiangular",
        norm: str = "ortho",
        hard_cutoff: bool = False,
    ):
        if not TORCH_HARMONICS_AVAILABLE:
            raise ImportError(
                "torch_harmonics is required for SphericalConvMHF. "
                "Please install it: pip install torch-harmonics"
            )
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_shape = grid_shape
        self.nlat, self.nlon = grid_shape
        
        # Determine maximum spherical harmonic degree
        if max_degree is None:
            self.max_degree = self.nlat // 2
        else:
            self.max_degree = max_degree
        
        # Initialize spherical harmonic transforms
        # Note: hard_cutoff parameter not supported in current torch-harmonics
        self.sht = RealSHT(
            nlat=self.nlat,
            nlon=self.nlon,
            lmax=self.max_degree,
            grid=grid_type,
            norm=norm,
        )

        self.isht = InverseRealSHT(
            nlat=self.nlat,
            nlon=self.nlon,
            lmax=self.max_degree,
            grid=grid_type,
            norm=norm,
        )
        
        # Number of spectral coefficients per channel
        # For RealSHT, the output shape depends on the implementation
        # We'll calculate it based on actual SHT output
        test_input = torch.randn(self.nlat, self.nlon)
        test_output = self.sht(test_input)
        # Store the number of complex coefficients (before stacking real/imag)
        self.num_coeffs_complex = test_output.numel()
        # But num_coeffs should match the flattened real representation
        self.num_coeffs = test_output.numel() * 2  # Real + Imaginary parts
        
        # Initialize weight for spectral domain
        # Shape: [in_channels, out_channels, num_coeffs]
        scale = (1 / (in_channels * out_channels * self.num_coeffs)) ** 0.5
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.num_coeffs))
        
        # MHF parameters
        self.mhf_rank = mhf_rank
        self.factorization = factorization
        self.implementation = implementation
        
        # Auto-generate resolutions if not provided
        if mhf_resolutions is None:
            self.mhf_resolutions = self._auto_resolutions()
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
    
    def _auto_resolutions(self) -> List[int]:
        """Automatically generate hierarchical resolutions
        
        Creates a logarithmic sequence from low to max_degree
        """
        resolutions = []
        current = 2  # Start from degree 2 (lowest meaningful)
        while current <= self.max_degree:
            resolutions.append((current + 1) ** 2)  # Store as number of coefficients
            current = current * 2
        if resolutions[-1] != self.num_coeffs:
            resolutions.append(self.num_coeffs)
        return resolutions
    
    def transform_to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Transform spatial input to spectral domain using SHT

        Parameters
        ----------
        x : torch.Tensor
            Spatial domain input, shape [B, in_channels, nlat, nlon]

        Returns
        -------
        torch.Tensor
            Spectral domain coefficients, shape [B, in_channels, num_coeffs]
        """
        B, C, nlat, nlon = x.shape
        assert nlat == self.nlat and nlon == self.nlon, \
            f"Input grid shape ({nlat}, {nlon}) doesn't match expected ({self.nlat}, {self.nlon})"

        # Reshape for SHT: [B*C, nlat, nlon]
        x_flat = x.reshape(B * C, nlat, nlon)

        # Apply spherical harmonic transform (returns complex tensor)
        x_spec_complex = self.sht(x_flat)

        # Convert complex to real (stack real and imaginary parts)
        x_spec_real = torch.stack([x_spec_complex.real, x_spec_complex.imag], dim=-1)
        # x_spec_real shape: [B*C, ..., 2]

        # Flatten the spatial dimensions
        x_spec_flat = x_spec_real.reshape(B * C, -1)

        # Reshape back: [B, C, num_coeffs]
        # num_coeffs should match the flattened real representation
        x_spec = x_spec_flat.reshape(B, C, -1)

        return x_spec
    
    def transform_to_spatial(self, x_spec: torch.Tensor) -> torch.Tensor:
        """Transform spectral coefficients back to spatial domain using inverse SHT

        Parameters
        ----------
        x_spec : torch.Tensor
            Spectral domain coefficients, shape [B, out_channels, num_coeffs]

        Returns
        -------
        torch.Tensor
            Spatial domain output, shape [B, out_channels, nlat, nlon]
        """
        B, C, num_coeffs = x_spec.shape

        # Reshape for ISHT: [B*C, num_coeffs]
        x_spec_flat = x_spec.reshape(B * C, -1)

        # Reshape to original SHT output shape (complex)
        # The SHT output was complex with shape [lmax, lmax+1]
        # We stored real and imaginary parts
        x_spec_complex = torch.view_as_complex(
            x_spec_flat.reshape(B * C, -1, 2).contiguous()
        )
        # Now reshape to [B*C, lmax, lmax+1]
        x_spec_complex = x_spec_complex.reshape(B * C, self.max_degree, self.max_degree + 1)

        # Apply inverse spherical harmonic transform
        x = self.isht(x_spec_complex)  # [B*C, nlat, nlon]

        # Reshape back: [B, C, nlat, nlon]
        x = x.reshape(B, C, self.nlat, self.nlon)

        return x
    
    def spectral_convolution(self, x_spec: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Perform spectral domain convolution
        
        Parameters
        ----------
        x_spec : torch.Tensor
            Input spectral coefficients, shape [B, in_channels, num_coeffs]
        weight : torch.Tensor
            Spectral weight, shape [in_channels, out_channels, num_coeffs]
        
        Returns
        -------
        torch.Tensor
            Output spectral coefficients, shape [B, out_channels, num_coeffs]
        """
        # Spectral convolution is point-wise multiplication in spectral domain
        # x_spec: [B, in_channels, num_coeffs]
        # weight: [in_channels, out_channels, num_coeffs]
        
        # Contract over in_channels and multiply element-wise over coefficients
        # Using einsum: bix,iox -> box
        out_spec = torch.einsum('bix,iox->box', x_spec, weight)
        
        return out_spec
    
    def decompose(self) -> None:
        """Perform MHF decomposition on the current weight
        
        This should be called after training the full model to compress it.
        """
        # The weight tensor shape is [in_channels, out_channels, num_coeffs]
        # We decompose along the spectral coefficient dimension (last dimension)
        spatial_dims = [2]  # Decompose along coefficient dimension
        
        # Perform MHF decomposition
        self.mhf.decompose(self.weight.data, spatial_dims=spatial_dims)
        
        self._mhf_decomposed = True
    
    def recompose(self) -> torch.Tensor:
        """Reconstruct full weight from MHF decomposition"""
        assert self._mhf_decomposed, "Must decompose before recomposing"
        return self.mhf.reconstruct()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Spatial domain input, shape [B, in_channels, nlat, nlon]
        
        Returns
        -------
        torch.Tensor
            Spatial domain output, shape [B, out_channels, nlat, nlon]
        """
        # Transform to spectral domain
        x_spec = self.transform_to_spectral(x)
        
        # Perform spectral convolution
        if self._mhf_decomposed and self.implementation == "factorized":
            # Use MHF decomposition for forward pass
            out_spec = self._forward_mhf(x_spec)
        else:
            # Use dense weight
            out_spec = self.spectral_convolution(x_spec, self.weight)
        
        # Transform back to spatial domain
        out = self.transform_to_spatial(out_spec)
        
        return out
    
    def _forward_mhf(self, x_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass using MHF decomposition
        
        This uses the decomposed factors directly to avoid reconstructing
        the full weight tensor, saving memory.
        
        Parameters
        ----------
        x_spec : torch.Tensor
            Input spectral coefficients, shape [B, in_channels, num_coeffs]
        
        Returns
        -------
        torch.Tensor
            Output spectral coefficients, shape [B, out_channels, num_coeffs]
        """
        # Contract input with MHF factors
        # x_spec: [B, in_channels, num_coeffs]
        # We need to produce: [B, out_channels, num_coeffs]
        
        # MHF handles the decomposition along the coefficient dimension
        # The forward pass needs to apply the MHF contraction
        return self.mhf.forward(x_spec)
    
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
    original_conv,
    mhf_rank: Union[int, List[int]] = 8,
    mhf_resolutions: Optional[List[int]] = None,
    factorization: str = "tucker",
    implementation: str = "factorized",
) -> SphericalConvMHF:
    """Create MHF-optimized spherical convolution from an existing spherical convolution
    
    This is a convenience function to convert an already-trained
    spherical convolution to MHF-optimized version.
    
    Parameters
    ----------
    original_conv
        Original trained spherical convolution
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
    SphericalConvMHF
        MHF-optimized convolution with weights copied from original
    """
    # Extract parameters from original
    in_channels = original_conv.in_channels
    out_channels = original_conv.out_channels
    grid_shape = original_conv.grid_shape if hasattr(original_conv, 'grid_shape') else (original_conv.nlat, original_conv.nlon)
    max_degree = getattr(original_conv, 'max_degree', None)
    
    # Create new MHF convolution
    mhf_conv = SphericalConvMHF(
        in_channels=in_channels,
        out_channels=out_channels,
        grid_shape=grid_shape,
        max_degree=max_degree,
        mhf_rank=mhf_rank,
        mhf_resolutions=mhf_resolutions,
        factorization=factorization,
        implementation=implementation,
    )
    
    # Copy weight
    if hasattr(original_conv, 'weight'):
        with torch.no_grad():
            mhf_conv.weight.copy_(original_conv.weight)
    
    # Perform decomposition
    mhf_conv.decompose()
    
    return mhf_conv


__all__ = ["SphericalConvMHF", "from_original", "TORCH_HARMONICS_AVAILABLE", "SphericalConvMHF as SphericalSpectralConvMHF"]
