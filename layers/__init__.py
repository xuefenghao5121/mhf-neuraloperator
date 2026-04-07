"""
MHF-optimized neural operator layers
"""

from .spectral_conv_mhf import SpectralConvMHF
from .fno_block_mhf import FNOBlocksMHF
from .gno_block_mhf import GNOBlockMHF

try:
    from .spherical_mhf import SphericalConvMHF as SphericalSpectralConvMHF, TORCH_HARMONICS_AVAILABLE
    _SPHERICAL_CONV_MHF_AVAILABLE = True
except (ImportError, SyntaxError) as e:
    SphericalSpectralConvMHF = None
    TORCH_HARMONICS_AVAILABLE = False
    _SPHERICAL_CONV_MHF_AVAILABLE = False

__all__ = [
    "SpectralConvMHF",
    "FNOBlocksMHF",
    "GNOBlockMHF",
    "SphericalSpectralConvMHF",
    "TORCH_HARMONICS_AVAILABLE",
    "_SPHERICAL_CONV_MHF_AVAILABLE",
]
