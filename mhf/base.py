"""
Base definitions for MHF (Multi-Resolution Hierarchical Factorization)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import tensorly as tl


@dataclass
class MHFMetadata:
    """Metadata for MHF decomposition
    
    Stores information about the decomposition for debugging and analysis.
    """
    original_shape: Tuple[int, ...]
    decomposed_shape: Tuple[int, ...]
    resolutions: List[int]
    ranks: Union[int, List[int], Dict[str, int]]
    factorization_type: str
    original_num_params: int
    decomposed_num_params: int
    decomposed: bool = False
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio (decomposed / original)"""
        return self.decomposed_num_params / self.original_num_params
    
    @property
    def compression_factor(self) -> float":
        """Compression factor (original / decomposed)"""
        return self.original_num_params / max(1, self.decomposed_num_params)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_shape": self.original_shape,
            "decomposed_shape": self.decomposed_shape,
            "resolutions": self.resolutions,
            "ranks": self.ranks,
            "factorization_type": self.factorization_type,
            "original_num_params": self.original_num_params,
            "decomposed_num_params": self.decomposed_num_params,
            "compression_ratio": self.compression_ratio,
            "compression_factor": self.compression_factor,
            "decomposed": self.decomposed,
        }


class BaseMHF(ABC, nn.Module):
    """Abstract base class for all MHF-optimized modules
    
    All MHF-optimized layers and models should inherit from this base class
    to ensure a consistent interface for decomposition and reconstruction.
    """
    
    def __init__(
        self,
        ranks: Union[int, List[int], Dict[str, int]],
        resolutions: List[int],
        factorization: str = "tucker",
    ):
        super().__init__()
        self.ranks = ranks
        self.resolutions = sorted(resolutions) if isinstance(resolutions, list) else [resolutions]
        self.factorization = factorization
        self._decomposed = False
        self._metadata: Optional[MHFMetadata] = None
        
    @abstractmethod
    def decompose(self) -> None:
        """Execute MHF decomposition on the current weights
        
        This method takes the current dense weights and performs
        multi-resolution hierarchical factorization.
        """
        pass
    
    @abstractmethod
    def recompose(self) -> torch.Tensor:
        """Reconstruct the full weight tensor from decomposition factors
        
        Returns
        -------
        torch.Tensor
            Reconstructed full weight tensor
        """
        pass
    
    @abstractmethod
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using MHF decomposition directly
        
        Performs forward pass with the decomposed factors without
        reconstructing the full weight tensor, which saves memory.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        pass
    
    def is_decomposed(self) -> bool:
        """Check if decomposition has been performed"""
        return self._decomposed
    
    def get_metadata(self) -> Optional[MHFMetadata]:
        """Get decomposition metadata"""
        return self._metadata
    
    def get_compression_ratio(self) -> Optional[float]:
        """Get current compression ratio"""
        if self._metadata is not None:
            return self._metadata.compression_ratio
        return None
    
    def get_compression_factor(self) -> Optional[float]:
        """Get current compression factor"""
        if self._metadata is not None:
            return self._metadata.compression_factor
        return None


class MultiResolutionHierarchicalFactorization:
    """Core MHF decomposition algorithm
    
    This class implements the Multi-Resolution Hierarchical Factorization:
    1. Decompose the weight tensor progressively from low to high resolution
    2. Each level captures the residual error from previous levels
    3. Hierarchical tensor factorization is applied at each level
    
    Parameters
    ----------
    resolutions : List[int]
        List of resolutions to use, ordered from low to high
        The last resolution should match the original resolution
    ranks : Union[int, Dict[str, int]]
        Rank(s) to use for factorization at each level
    factorization_type : str, default="tucker"
        Type of tensor factorization to use: "cp", "tucker", or "tt"
    """
    
    def __init__(
        self,
        resolutions: List[int],
        ranks: Union[int, Dict[str, int]],
        factorization_type: str = "tucker",
    ):
        self.resolutions = sorted(resolutions)
        self.ranks = ranks
        self.factorization_type = factorization_type
        
        # Store decomposition factors for each level
        self.factors: Dict[int, Any] = {}
        self._original_shape: Optional[Tuple[int, ...]] = None
        self._decomposed = False
        
    def decompose(
        self,
        weight: torch.Tensor,
        spatial_dims: Optional[List[int]] = None,
    ) -> Dict[int, Any]:
        """Perform multi-resolution hierarchical decomposition
        
        Parameters
        ----------
        weight : torch.Tensor
            Original weight tensor to decompose
        spatial_dims : List[int], optional
            Dimensions corresponding to spatial/frequency modes
            If None, the last N dimensions are treated as spatial
            where N is the number of resolutions
            
        Returns
        -------
        Dict[int, Any]
            Decomposition factors for each resolution level
        """
        self._original_shape = weight.shape
        
        # Auto-detect spatial dimensions if not provided
        if spatial_dims is None:
            # Assume spatial dimensions are the last N dimensions
            # where N is the length of resolutions
            n_spatial = len(self.resolutions)
            spatial_dims = list(range(len(weight.shape) - n_spatial, len(weight.shape)))
        
        self._spatial_dims = spatial_dims
        current_residual = weight.clone()
        prev_recon = None
        
        for level, res in enumerate(self.resolutions):
            # Downsample current residual to this level's resolution
            resampled = self._downsample_to_resolution(current_residual, res, spatial_dims)
            
            # Perform tensor factorization on resampled weights
            factor = self._factorize(resampled, self._get_rank_for_level(level))
            self.factors[res] = factor
            
            # Reconstruct from this level's factorization
            recon_level = self._reconstruct_level(factor)
            
            # Upsample back to original resolution (or next level)
            recon_full = self._upsample_to_original(recon_level, self._original_shape, spatial_dims)
            
            if prev_recon is not None:
                recon_full = prev_recon + recon_full
            
            # Update residual for next level
            if level < len(self.resolutions) - 1:
                current_residual = current_residual - recon_full
                prev_recon = recon_full
            else:
                # At final level, residual is added directly to the reconstruction
                pass
        
        self._decomposed = True
        return self.factors
    
    def reconstruct(self) -> torch.Tensor:
        """Reconstruct the full weight tensor from all levels
        
        Returns
        -------
        torch.Tensor
            Reconstructed full weight tensor
        """
        assert self._decomposed, "Decomposition must be performed first"
        assert self._original_shape is not None, "Original shape not stored"
        
        full_weight = torch.zeros(self._original_shape, device=self._get_device())
        
        for res, factor in self.factors.items():
            recon_level = self._reconstruct_level(factor)
            recon_full = self._upsample_to_original(recon_level, self._original_shape, self._spatial_dims)
            full_weight = full_weight + recon_full
        
        return full_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using decomposed factors directly
        
        This method contracts the input with decomposed factors
        without reconstructing the full weight tensor, which
        is the main advantage of MHF for memory efficiency.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor in Fourier space
            
        Returns
        -------
        torch.Tensor
            Output after contraction with decomposed weights
        """
        assert self._decomposed, "Decomposition must be performed first"
        
        # For each resolution level, compute the contribution
        # and accumulate the result
        output = None
        
        for res, factor in self.factors.items():
            # Contract with this level's factors
            # The exact contraction depends on factorization type
            if self.factorization_type == "tucker":
                # Tucker: core × factors along each dimension
                contrib = self._contract_tucker(x, factor)
            elif self.factorization_type == "cp":
                # CP: weighted outer product of factors
                contrib = self._contract_cp(x, factor)
            elif self.factorization_type == "tt":
                # Tensor Train: sequential contraction
                contrib = self._contract_tt(x, factor)
            else:
                raise ValueError(f"Unknown factorization type: {self.factorization_type}")
            
            if output is None:
                output = contrib
            else:
                output = output + contrib
        
        return output
    
    def _downsample_to_resolution(
        self,
        weight: torch.Tensor,
        target_res: int,
        spatial_dims: List[int],
    ) -> torch.Tensor:
        """Downsample weight to target resolution"""
        # Implementation depends on the interpolation method
        # For frequency space, we can just crop
        return self._crop_spatial(weight, target_res, spatial_dims)
    
    def _upsample_to_original(
        self,
        weight: torch.Tensor,
        target_shape: Tuple[int, ...],
        spatial_dims: List[int],
    ) -> torch.Tensor:
        """Upsample weight to original resolution by padding with zeros"""
        return self._pad_spatial(weight, target_shape, spatial_dims)
    
    def _crop_spatial(
        self,
        weight: torch.Tensor,
        target_res: int,
        spatial_dims: List[int],
    ) -> torch.Tensor:
        """Crop spatial dimensions to target resolution"""
        slices = list(slice(None) for _ in weight.shape)
        for dim in spatial_dims:
            slices[dim] = slice(0, target_res)
        return weight[tuple(slices)]
    
    def _pad_spatial(
        self,
        weight: torch.Tensor,
        target_shape: Tuple[int, ...],
        spatial_dims: List[int],
    ) -> torch.Tensor:
        """Pad spatial dimensions to target shape with zeros"""
        pad = []
        for dim in reversed(range(len(weight.shape))):
            if dim in spatial_dims:
                diff = target_shape[dim] - weight.shape[dim]
                pad.extend([0, diff])
            else:
                pad.extend([0, 0])
        
        return nn.functional.pad(weight, pad)
    
    def _factorize(self, tensor: torch.Tensor, rank: int) -> Any:
        """Perform tensor factorization on the given tensor"""
        if self.factorization_type == "tucker":
            return tl.decomposition.tucker(tensor, rank=rank)
        elif self.factorization_type == "cp":
            return tl.decomposition.cp(tensor, rank=rank)
        elif self.factorization_type == "tt":
            return tl.decomposition.tt(tensor, rank=rank)
        else:
            raise ValueError(f"Unknown factorization type: {self.factorization_type}")
    
    def _reconstruct_level(self, factor: Any) -> torch.Tensor:
        """Reconstruct a single level from its factors"""
        if self.factorization_type == "tucker":
            core, factors = factor
            return tl.tucker_to_tensor((core, factors))
        elif self.factorization_type == "cp":
            weights, factors = factor
            return tl.cp_to_tensor((weights, factors))
        elif self.factorization_type == "tt":
            return tl.tt_to_tensor(factor)
        else:
            raise ValueError(f"Unknown factorization type: {self.factorization_type}")
    
    def _get_rank_for_level(self, level: int) -> Union[int, Tuple[int, ...]]:
        """Get rank for a specific level"""
        if isinstance(self.ranks, int):
            return self.ranks
        elif isinstance(self.ranks, list):
            return self.ranks[level]
        elif isinstance(self.ranks, dict):
            return self.ranks.get(str(level), self.ranks.get("default", 8))
        else:
            raise ValueError(f"Invalid ranks type: {type(self.ranks)}")
    
    def _get_device(self) -> torch.device:
        """Get the device of the decomposition"""
        first_factor = next(iter(self.factors.values()))
        if self.factorization_type == "tucker":
            core, _ = first_factor
            return core.device
        elif self.factorization_type == "cp":
            _, factors = first_factor
            return factors[0].device
        elif self.factorization_type == "tt":
            return first_factor[0].device
        else:
            return torch.device("cpu")
    
    # Contraction methods for different factorization types
    # These are used during forward pass without full reconstruction
    
    def _contract_tucker(self, x: torch.Tensor, factor: Any) -> torch.Tensor:
        """Contract input with Tucker decomposition factors"""
        # Implementation follows the strategy in neuralop spectral convolution
        core, factors = factor
        return tl.contract(tucker, x)
    
    def _contract_cp(self, x: torch.Tensor, factor: Any) -> torch.Tensor:
        """Contract input with CP decomposition factors"""
        weights, factors = factor
        return tl.cp_contraction(x, weights, factors)
    
    def _contract_tt(self, x: torch.Tensor, factor: Any) -> torch.Tensor:
        """Contract input with Tensor Train decomposition factors"""
        return tl.tt_contract(x, factor)
    
    def count_params(self) -> int:
        """Count number of parameters in decomposition"""
        total = 0
        for res, factor in self.factors.items():
            if self.factorization_type == "tucker":
                core, factors_list = factor
                total += core.numel()
                for f in factors_list:
                    total += f.numel()
            elif self.factorization_type == "cp":
                weights, factors_list = factor
                total += weights.numel()
                for f in factors_list:
                    total += f.numel()
            elif self.factorization_type == "tt":
                for core in factor:
                    total += core.numel()
        return total
    
    def is_decomposed(self) -> bool:
        """Check if decomposition is complete"""
        return self._decomposed
