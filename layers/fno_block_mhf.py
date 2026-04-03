"""
MHF-optimized FNO Blocks
"""

from typing import Tuple, List, Union, Literal, Optional

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.channel_mlp import ChannelMLP
from mhf.spectral_mhf import SpectralConvMHF


class FNOBlocksMHF(nn.Module):
    """MHF-optimized FNO Blocks
    
    Drop-in replacement for FNOBlocks with MHF optimization for all
    spectral convolution layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        n_layers: int,
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        factorization: str = "tucker",
        mhf_implementation: str = "factorized",
        non_linearity: nn.Module = F.gelu,
        norm: Optional[Literal["ada_in", "group_norm", "instance_norm"]] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        separable: bool = False,
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.resolution_scaling_factor = resolution_scaling_factor
        
        # Store MHF parameters
        self.mhf_rank = mhf_rank
        self.mhf_resolutions = mhf_resolutions
        self.mhf_factorization = factorization
        self.mhf_implementation = mhf_implementation
        
        # Create a spectral conv for each layer
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = SpectralConvMHF(
                in_channels=in_channels,
                out_channels=out_channels,
                n_modes=n_modes,
                mhf_rank=mhf_rank,
                mhf_resolutions=mhf_resolutions,
                factorization=factorization,
                implementation=mhf_implementation,
                separable=separable,
                complex_data=complex_data,
            )
            self.convs.append(conv)
        
        # Skip connections
        self.skip_connections = nn.ModuleList()
        if fno_skip == "linear":
            for _ in range(n_layers):
                self.skip_connections.append(nn.Conv1d(in_channels, out_channels, 1))
        elif fno_skip == "identity":
            assert in_channels == out_channels
            for _ in range(n_layers):
                self.skip_connections.append(nn.Identity())
        
        # Channel MLP
        self.channel_mlps = nn.ModuleList()
        if use_channel_mlp:
            for _ in range(n_layers):
                self.channel_mlps.append(ChannelMLP(
                    in_channels=out_channels,
                    dropout=channel_mlp_dropout,
                    expansion=channel_mlp_expansion,
                    skip_connection=channel_mlp_skip,
                ))
        
        # Normalization
        self.norm = norm
        if norm == "ada_in":
            self.ada_in_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.ada_in_layers.append(nn.Linear(in_channels, 2 * out_channels))
        elif norm in ["group_norm", "instance_norm"]:
            self.norm_layers = nn.ModuleList()
            for _ in range(n_layers):
                if norm == "group_norm":
                    self.norm_layers.append(nn.GroupNorm(1, out_channels))
                else:
                    self.norm_layers.append(nn.InstanceNorm2d(out_channels))
        
        self.fno_skip = fno_skip
        self.non_linearity = non_linearity
    
    def decompose_all(self) -> None:
        """Perform MHF decomposition on all conv layers"""
        for conv in self.convs:
            conv.decompose()
    
    def forward(
        self,
        x: torch.Tensor,
        ada_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass"""
        for i in range(self.n_layers):
            x = self.convs[i](x)
            
            if self.fno_skip == "linear":
                # Reshape for 1D conv
                shape = x.shape
                x_reshaped = x.reshape(shape[0], shape[1], -1)
                x = x + self.skip_connections[i](x_reshaped).reshape(shape)
            
            if self.norm == "ada_in":
                assert ada_in is not None, "ada_in must be provided for ada_in norm"
                x = self._apply_ada_in(x, self.ada_in_layers[i](ada_in))
            elif self.norm in ["group_norm", "instance_norm"]:
                x = self.norm_layers[i](x)
            
            x = self.non_linearity(x)
            
            if len(self.channel_mlps) > i:
                x = self.channel_mlps[i](x)
        
        return x
    
    def _apply_ada_in(self, x: torch.Tensor, ada_params: torch.Tensor) -> torch.Tensor:
        """Apply adaptive instance normalization"""
        bs = x.shape[0]
        out_channels = ada_params.shape[1] // 2
        gamma = ada_params[:, :out_channels]
        beta = ada_params[:, out_channels:]
        
        # Add dimensions for broadcasting
        for _ in range(len(x.shape) - 2):
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        # Compute mean and std over spatial dimensions
        spatial_dims = list(range(2, len(x.shape)))
        mean = x.mean(dim=spatial_dims, keepdim=True)
        std = x.std(dim=spatial_dims, keepdim=True)
        
        return gamma * (x - mean) / (std + 1e-8) + beta


__all__ = ["FNOBlocksMHF"]
