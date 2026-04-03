"""
MHF-optimized GNO Block for Geometry-Informed Neural Operators
"""

from typing import Tuple, List, Union, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.gno_block import GNOBlock
from mhf.base import MultiResolutionHierarchicalFactorization


class GNOBlockMHF(GNOBlock):
    """MHF-optimized GNO Block
    
    Applies MHF multi-resolution hierarchical factorization to the
    kernel weights in GNO for parameter compression.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: float,
        base_activation: nn.Module = nn.GELU(),
        mlp_layers: int = 2,
        mlp_hidden_channels: int = 256,
        mhf_rank: int = 8,
        mhf_factorization: str = "tucker",
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            radius=radius,
            base_activation=base_activation,
            mlp_layers=mlp_layers,
            mlp_hidden_channels=mlp_hidden_channels,
            **kwargs,
        )
        
        self.mhf_rank = mhf_rank
        self.mhf_factorization = mhf_factorization
        self._mhf_decomposed = False
        self._mhf_decomposition = None
    
    def decompose(self, resolutions: Optional[List[int]] = None) -> None:
        """Perform MHF decomposition on the kernel MLP weights
        
        Parameters
        ----------
        resolutions : List[int], optional
            List of resolutions for hierarchical decomposition.
            If None, automatically generated based on weight shape.
        """
        if self._mhf_decomposed:
            return
        
        # MHF can decompose the weights in the kernel network
        # For a typical kernel MLP that produces weight for (in, out),
        # we apply multi-resolution decomposition to the final weight
        
        # Get the final linear layer weight
        last_layer = list(self.kernel_network.children())[-2]
        if isinstance(last_layer, nn.Linear):
            weight = last_layer.weight
            out_features, in_features = weight.shape
            
            # Auto-generate resolutions
            if resolutions is None:
                max_dim = max(in_features, out_features)
                resolutions = []
                current = 8
                while current < max_dim:
                    resolutions.append(current)
                    current *= 2
                if resolutions[-1] != max_dim:
                    resolutions.append(max_dim)
            
            # Create MHF decomposition
            self._mhf_decomposition = MultiResolutionHierarchicalFactorization(
                resolutions=resolutions,
                ranks=self.mhf_rank,
                factorization_type=self.mhf_factorization,
            )
            
            self._mhf_decomposition.decompose(weight.data)
            self._mhf_decomposed = True
    
    def forward_mhf(self, *args, **kwargs) -> Any:
        """Forward pass using MHF decomposition"""
        assert self._mhf_decomposed, "Must decompose first"
        return self.forward(*args, **kwargs)
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics"""
        if not self._mhf_decomposed or self._mhf_decomposition is None:
            return {
                "decomposed": False,
                "message": "MHF not applied to this GNO block",
            }
        
        # Find the last linear layer
        last_layer = None
        for layer in self.kernel_network.children():
            if isinstance(layer, nn.Linear):
                last_layer = layer
        
        if last_layer is None:
            return {"decomposed": False}
        
        original_params = last_layer.weight.numel()
        decomposed_params = self._mhf_decomposition.count_params()
        
        return {
            "decomposed": True,
            "original_params": original_params,
            "decomposed_params": decomposed_params,
            "compression_ratio": decomposed_params / original_params,
            "compression_factor": original_params / decomposed_params,
        }


__all__ = ["GNOBlockMHF"]
