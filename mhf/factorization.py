"""
Specific factorization method implementations for MHF
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import tensorly as tl
from tensorly import TensorlyTensor
from tensorly.decomposition import CP as CPDecomposition
from tensorly.decomposition import Tucker as TuckerDecomposition
from tensorly.decomposition import TT as TTDecomposition


class BaseFactorization:
    """Base class for all factorization methods"""
    
    def __init__(self, rank: Union[int, Tuple[int, ...]]):
        self.rank = rank
        self.decomposed_ = None
        
    @abstractmethod
    def decompose(self, tensor: torch.Tensor) -> Any:
        """Decompose the input tensor"""
        pass
    
    @abstractmethod
    def reconstruct(self, factors: Any) -> torch.Tensor:
        """Reconstruct tensor from factors"""
        pass
    
    @abstractmethod
    def count_params(self, factors: Any) -> int:
        """Count number of parameters in the factorization"""
        pass


class CPFactorization(BaseFactorization):
    """CP (Canonical Polyadic) decomposition
    
    Good for tensors with low-rank structure.
    Higher compression ratio than Tucker for many cases.
    """
    
    def __init__(
        self,
        rank: int,
        tol: float = 1e-6,
        n_iter_max: int = 100,
        init: str = "svd",
    ):
        super().__init__(rank)
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.init = init
        
    def decompose(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Perform CP decomposition"""
        weights, factors = CPDecomposition(
            rank=self.rank,
            tol=self.tol,
            n_iter_max=self.n_iter_max,
            init=self.init,
        ).fit_transform(tensor)
        self.decomposed_ = (weights, factors)
        return weights, factors
    
    def reconstruct(self, factors: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Reconstruct tensor from CP factors"""
        weights, factors = factors
        return tl.cp_to_tensor((weights, factors))
    
    def count_params(self, factors: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]) -> int:
        """Count parameters in CP factorization"""
        weights, factors_list = factors
        total = weights.numel()
        for f in factors_list:
            total += f.numel()
        return total


class TuckerFactorization(BaseFactorization):
    """Tucker decomposition
    
    More flexible than CP, usually gives good approximation with moderate ranks.
    Most commonly used in spectral convolutions.
    """
    
    def __init__(
        self,
        rank: Union[int, Tuple[int, ...]],
        tol: float = 1e-6,
        n_iter_max: int = 100,
        init: str = "svd",
    ):
        super().__init__(rank)
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.init = init
        
    def decompose(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Perform Tucker decomposition"""
        core, factors = TuckerDecomposition(
            rank=self.rank,
            tol=self.tol,
            n_iter_max=self.n_iter_max,
            init=self.init,
        ).fit_transform(tensor)
        self.decomposed_ = (core, factors)
        return core, factors
    
    def reconstruct(self, factors: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Reconstruct tensor from Tucker factors"""
        core, factors = factors
        return tl.tucker_to_tensor((core, factors))
    
    def count_params(self, factors: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]) -> int:
        """Count parameters in Tucker factorization"""
        core, factors_list = factors
        total = core.numel()
        for f in factors_list:
            total += f.numel()
        return total


class TTFactorization(BaseFactorization):
    """Tensor Train (TT) decomposition
    
    Excellent for high-dimensional tensors, very high compression ratio.
    Slightly slower computation but great for large problems.
    """
    
    def __init__(
        self,
        rank: Union[int, Tuple[int, ...]],
        tol: float = 1e-6,
        n_iter_max: int = 100,
    ):
        super().__init__(rank)
        self.tol = tol
        self.n_iter_max = n_iter_max
        
    def decompose(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Perform Tensor Train decomposition"""
        factors = TTDecomposition(
            rank=self.rank,
            tol=self.tol,
            n_iter_max=self.n_iter_max,
        ).fit_transform(tensor)
        self.decomposed_ = factors
        return factors
    
    def reconstruct(self, factors: Tuple[torch.Tensor, ...]) -> torch.T:
        """Reconstruct tensor from TT factors"""
        return tl.tt_to_tensor(factors)
    
    def count_params(self, factors: Tuple[torch.Tensor, ...]) -> int:
        """Count parameters in TT factorization"""
        total = 0
        for core in factors:
            total += core.numel()
        return total


def get_factorization(
    factorization_type: str,
    rank: Union[int, Tuple[int, ...]],
    **kwargs,
) -> BaseFactorization:
    """Factory function to get factorization instance
    
    Parameters
    ----------
    factorization_type : str
        Type of factorization: "cp", "tucker", or "tt"
    rank : Union[int, Tuple[int, ...]]
        Rank for factorization
    **kwargs
        Additional kwargs passed to factorization constructor
        
    Returns
    -------
    BaseFactorization
        Factorization instance
    """
    if factorization_type == "cp":
        return CPFactorization(rank, **kwargs)
    elif factorization_type == "tucker":
        return TuckerFactorization(rank, **kwargs)
    elif factorization_type == "tt":
        return TTFactorization(rank, **kwargs)
    else:
        raise ValueError(
            f"Unknown factorization type: {factorization_type}. "
            "Supported types: 'cp', 'tucker', 'tt'"
        )
