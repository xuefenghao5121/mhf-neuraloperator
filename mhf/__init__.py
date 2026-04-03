"""
MHF-NeuralOperator
==================

Multi-Resolution Hierarchical Factorization for universal neural operator optimization.

This package extends the original neuraloperator library with MHF optimization
for all supported neural operator types.

Core Components:
- MultiResolutionHierarchicalFactorization: Core decomposition algorithm
- BaseMHF: Base interface for all MHF-optimized modules
- MHF-enabled models for all neuraloperator operator types

Available Models:
- MHFNO: MHF-optimized Fourier Neural Operator
- MHFUNO: MHF-optimized U-shaped Neural Operator
- MHFGINO: MHF-optimized Geometry-Informed Neural Operator
- MHFCODANO: MHF-optimized Conditional Neural Operator
- MHFRNO: MHF-optimized Recurrent Neural Operator
- MHFLocalNO: MHF-optimized Local Neural Operator
- MHFOTNO: MHF-optimized Optimal Transport Neural Operator
"""

__version__ = "1.0.0"
__author__ = "天渊团队 (team_tianyuan_fft)"

from .base import (
    BaseMHF,
    MHFMetadata,
    MultiResolutionHierarchicalFactorization
)
from .factorization import (
    CPFactorization,
    TuckerFactorization,
    TTFactorization
)
from .factory import get_mhf_model

__all__ = [
    # Base classes
    "BaseMHF",
    "MHFMetadata",
    "MultiResolutionHierarchicalFactorization",
    
    # Factorization methods
    "CPFactorization",
    "TuckerFactorization",
    "TTFactorization",
    
    # Factory
    "get_mhf_model",
]
