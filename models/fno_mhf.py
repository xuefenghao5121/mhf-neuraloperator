"""
MHF-optimized Fourier Neural Operator (FNO)
"""

from functools import partialmethod
from typing import Tuple, List, Union, Literal, Optional

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models.fno import FNO
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.padding import DomainPadding
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.complex import ComplexValued
from neuralop.models.base_model import BaseModel

from mhf.spectral_mhf import SpectralConvMHF
from mhf.base import BaseMHF


class MHFNOBlocks(nn.Module):
    """MHF-optimized FNO Blocks
    
    Drop-in replacement for FNOBlocks that uses MHF-optimized spectral convolution.
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
        spectral_conv: nn.Module = SpectralConvMHF,
        **kwargs,
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
            conv = spectral_conv(
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
            # AdaIN normalization
            self.ada_in_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.ada_in_layers.append(nn.Linear(in_channels, 2 * out_channels))
        elif norm == "group_norm":
            self.norm_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.norm_layers.append(nn.GroupNorm(1, out_channels))
        elif norm == "instance_norm":
            self.norm_layers = nn.ModuleList()
            for _ in range(n_layers):
                self.norm_layers.append(nn.InstanceNorm2d(out_channels))
        
        self.fno_skip = fno_skip
        self.non_linearity = non_linearity
    
    def decompose_all(self) -> None:
        """Perform MHF decomposition on all layers"""
        for conv in self.convs:
            if isinstance(conv, SpectralConvMHF):
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
                x = x + self.skip_connections[i](x.reshape(x.shape[0], x.shape[1], -1)).reshape_as(x)
            
            if self.norm == "ada_in":
                assert ada_in is not None, "ada_in must be provided for ada_in norm"
                x = self._apply_ada_in(x, self.ada_in_layers[i](ada_in))
            elif self.norm in ["group_norm", "instance_norm"]:
                x = self.norm_layers[i](x)
            
            x = self.non_linearity(x)
            
            if hasattr(self, "channel_mlps") and len(self.channel_mlps) > 0:
                x = self.channel_mlps[i](x)
        
        return x
    
    def _apply_ada_in(self, x: torch.Tensor, ada_params: torch.Tensor) -> torch.Tensor:
        """Apply adaptive instance normalization"""
        bs = x.shape[0]
        out_channels = ada_params.shape[1] // 2
        gamma = ada_params[:, :out_channels].unsqueeze(2).unsqueeze(3)
        beta = ada_params[:, out_channels:].unsqueeze(2).unsqueeze(3)
        
        # Compute mean and std
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        
        return gamma * (x - mean) / (std + 1e-8) + beta


class MHFNO(FNO, BaseMHF):
    """MHF-optimized N-Dimensional Fourier Neural Operator
    
    This is a drop-in replacement for the original FNO class in neuraloperator,
    with the added option of MHF (Multi-Resolution Hierarchical Factorization)
    optimization for parameter compression.
    
    All original API parameters are preserved, additional MHF parameters are added.
    
    Parameters
    ----------
    All parameters are the same as FNO, plus:
    
    mhf_rank : Union[int, List[int]], default=8
        Rank(s) for MHF factorization. If list, specifies rank per resolution level.
    mhf_resolutions : List[int], optional
        Resolutions for MHF hierarchy. If None, automatically generated based on n_modes.
    mhf_factorization : str, default="tucker"
        Tensor factorization type: "cp", "tucker", or "tt"
    mhf_implementation : str, default="factorized"
        - "factorized": Use decomposed factors directly for forward pass (most memory efficient)
        - "reconstructed": Reconstruct full weight from factors (slower, but for comparison)
    """
    
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2.0,
        projection_channel_ratio: Number = 2.0,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Optional[Literal["ada_in", "group_norm", "instance_norm"]] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        domain_padding: Optional[Union[Number, List[Number]]] = None,
        # MHF specific parameters
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        mhf_factorization: str = "tucker",
        mhf_implementation: str = "factorized",
    ):
        # Initialize BaseMHF
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=mhf_resolutions or [],
            factorization=mhf_factorization,
        )
        
        # Initialize FNO
        FNO.__init__(
            self,
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=resolution_scaling_factor,
            domain_padding=domain_padding,
        )
        
        # Replace the FNO blocks with MHF-optimized blocks
        self.fno_blocks = MHFNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            mhf_rank=mhf_rank,
            mhf_resolutions=mhf_resolutions,
            factorization=mhf_factorization,
            mhf_implementation=mhf_implementation,
            non_linearity=non_linearity,
            norm=norm,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=resolution_scaling_factor,
            spectral_conv=SpectralConvMHF,
        )
    
    def decompose(self) -> None:
        """Perform MHF decomposition on all spectral conv layers"""
        self.fno_blocks.decompose_all()
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct all full weights from decomposition"""
        for conv in self.fno_blocks.convs:
            full_weight = conv.recompose()
            with torch.no_grad():
                conv.weight.copy_(full_weight)
    
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass guaranteed to use MHF decomposition"""
        assert self._decomposed, "Must call decompose() before using forward_mhf"
        return self.forward(x)
    
    def get_compression_stats(self) -> dict:
        """Get overall compression statistics across all layers"""
        total_original = 0
        total_decomposed = 0
        
        for conv in self.fno_blocks.convs:
            if isinstance(conv, SpectralConvMHF):
                orig, decomp = conv.count_parameters()
                total_original += orig
                total_decomposed += decomp
        
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


class MHFTFNO(MHFNO):
    """MHF-optimized Tensorized Fourier Neural Operator
    
    TFNO in neuraloperator already uses tensor factorization, this class
    adds MHF multi-resolution hierarchical factorization on top for
    additional compression.
    """
    
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        mhf_rank: Union[int, List[int]] = 4,
        mhf_resolutions: Optional[List[int]] = None,
        mhf_factorization: str = "cp",
        mhf_implementation: str = "factorized",
        **kwargs,
    ):
        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            mhf_rank=mhf_rank,
            mhf_resolutions=mhf_resolutions,
            mhf_factorization=mhf_factorization,
            mhf_implementation=mhf_implementation,
            **kwargs
        )


@classmethod
def from_original(cls, original_fno: FNO, mhf_rank: int = 8, factorization: str = "tucker") -> MHFNO:
    """Create MHFNO from an already-trained original FNO
    
    Parameters
    ----------
    original_fno : FNO
        Trained original FNO model from neuraloperator
    mhf_rank : int, default=8
        Rank for MHF factorization
    factorization : str, default="tucker"
        Factorization type
        
    Returns
    -------
    MHFNO
        MHF-optimized model with weights copied from original
    """
    # Extract parameters from original
    n_modes = original_fno.n_modes
    in_channels = original_fno.in_channels
    out_channels = original_fno.out_channels
    hidden_channels = original_fno.hidden_channels
    n_layers = len(original_fno.fno_blocks.convs)
    
    # Create MHF model
    mhf_model = cls(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        mhf_rank=mhf_rank,
        mhf_factorization=factorization,
    )
    
    # Copy projection and lifting layers
    with torch.no_grad():
        mhf_model.lifting.weight.copy_(original_fno.lifting.weight)
        mhf_model.projection.weight.copy_(original_fno.projection.weight)
        
        # Copy conv weights
        for i, (orig_conv, mhf_conv) in enumerate(zip(original_fno.fno_blocks.convs, 
                                                      mhf_model.fno_blocks.convs)):
            mhf_conv.weight.copy_(orig_conv.weight)
    
    # Perform decomposition
    mhf_model.decompose()
    
    return mhf_model


MHFNO.from_original = from_original
