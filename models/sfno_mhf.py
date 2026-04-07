"""
MHF-optimized Spherical Fourier Neural Operator (SFNO)
"""

from typing import Tuple, List, Union, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralop.models.sfno import SFNO
    NEURALOP_SFNO_AVAILABLE = True
except ImportError:
    NEURALOP_SFNO_AVAILABLE = False
    SFNO = None

# Import SphericalConvMHF
import sys
sys.path.insert(0, '.')

try:
    from layers.spherical_mhf import SphericalConvMHF, TORCH_HARMONICS_AVAILABLE
except (ImportError, SyntaxError):
    SphericalConvMHF = None
    TORCH_HARMONICS_AVAILABLE = False

# BaseMHF definition (inline to avoid circular imports)
class BaseMHF(nn.Module):
    """Minimal BaseMHF for avoiding circular imports"""
    def __init__(self, ranks, resolutions, factorization):
        super().__init__()
        self.ranks = ranks
        self.resolutions = sorted(resolutions) if isinstance(resolutions, list) else [resolutions]
        self.factorization = factorization
        self._decomposed = False
    
    def decompose(self) -> None:
        raise NotImplementedError
    
    def recompose(self) -> torch.Tensor:
        raise NotImplementedError
    
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def is_decomposed(self) -> bool:
        return self._decomposed


class MHFSFNOBlocks(nn.Module):
    """MHF-optimized SFNO Blocks
    
    Drop-in replacement for SFNO blocks that uses MHF-optimized spherical spectral convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_shape: Tuple[int, int],
        n_layers: int,
        max_degree: int,
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        factorization: str = "tucker",
        mhf_implementation: str = "factorized",
        grid_type: str = "equiangular",
        norm: str = "ortho",
        non_linearity: nn.Module = F.gelu,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        sfno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        **kwargs,
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.grid_shape = grid_shape
        self.max_degree = max_degree
        
        # Store MHF parameters
        self.mhf_rank = mhf_rank
        self.mhf_resolutions = mhf_resolutions
        self.mhf_factorization = factorization
        self.mhf_implementation = mhf_implementation
        
        # Create spherical conv for each layer
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = SphericalConvMHF(
                in_channels=in_channels,
                out_channels=out_channels,
                grid_shape=grid_shape,
                max_degree=max_degree,
                mhf_rank=mhf_rank,
                mhf_resolutions=mhf_resolutions,
                factorization=factorization,
                implementation=mhf_implementation,
                grid_type=grid_type,
                norm=norm,
            )
            self.convs.append(conv)
        
        # Skip connections
        self.skip_connections = nn.ModuleList()
        if sfno_skip == "linear":
            for _ in range(n_layers):
                self.skip_connections.append(nn.Conv2d(in_channels, out_channels, 1))
        elif sfno_skip == "identity":
            assert in_channels == out_channels
            for _ in range(n_layers):
                self.skip_connections.append(nn.Identity())
        
        # Channel MLP
        from neuralop.layers.channel_mlp import ChannelMLP
        self.channel_mlps = nn.ModuleList()
        if use_channel_mlp:
            for _ in range(n_layers):
                self.channel_mlps.append(ChannelMLP(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    hidden_channels=int(out_channels * channel_mlp_expansion),
                    n_layers=2,
                    dropout=channel_mlp_dropout,
                ))
        
        self.sfno_skip = sfno_skip
        self.non_linearity = non_linearity
    
    def decompose_all(self) -> None:
        """Perform MHF decomposition on all layers"""
        for conv in self.convs:
            if isinstance(conv, SphericalConvMHF):
                conv.decompose()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i in range(self.n_layers):
            x = self.convs[i](x)
            
            if self.sfno_skip == "linear":
                x = x + self.skip_connections[i](x)
            elif self.sfno_skip == "identity":
                x = x + self.skip_connections[i](x)
            
            x = self.non_linearity(x)
            
            if hasattr(self, "channel_mlps") and len(self.channel_mlps) > 0:
                x = self.channel_mlps[i](x)
        
        return x


class MHSFNO(BaseMHF, nn.Module):
    """MHF-optimized Spherical Fourier Neural Operator (SFNO)
    
    SFNO works on spherical domains using spherical harmonics for spectral transforms.
    MHF optimizes spherical spectral convolutions through multi-resolution
    hierarchical factorization of weight tensors.
    
    This implementation provides:
    1. Drop-in replacement for original SFNO with same API
    2. MHF compression for spherical spectral convolution weights
    3. Support for optional CoDA (Cross-Head Attention) optimization
    4. Automatic resolution hierarchy based on spherical harmonic orders
    
    Parameters
    ----------
    grid_shape : Tuple[int, int]
        Spherical grid shape (nlat, nlon)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    hidden_channels : int
        Number of hidden channels
    n_layers : int, default=4
        Number of SFNO layers
    max_degree : int, optional
        Maximum spherical harmonic degree
    mhf_rank : Union[int, List[int]], default=8
        Rank(s) for MHF factorization. If list, specifies rank per resolution level.
    mhf_resolutions : List[int], optional
        Resolutions for MHF hierarchy (spherical harmonic orders). 
        If None, automatically generated.
    mhf_factorization : str, default="tucker"
        Tensor factorization type: "cp", "tucker", or "tt"
    mhf_implementation : str, default="factorized"
        - "factorized": Use decomposed factors directly (most memory efficient)
        - "reconstructed": Reconstruct full weight (for comparison)
    grid_type : str, default="equiangular"
        Type of spherical grid
    use_coda : bool, default=False
        Whether to use CoDA (Cross-Head Attention) optimization
    coda_reduction : int, default=4
        Reduction ratio for CoDA
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        max_degree: Optional[int] = None,
        # MHF parameters
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        mhf_factorization: str = "tucker",
        mhf_implementation: str = "factorized",
        # SFNO parameters
        grid_type: str = "equiangular",
        norm: str = "ortho",
        hard_cutoff: bool = False,
        non_linearity: nn.Module = F.gelu,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        sfno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        # CoDA parameters
        use_coda: bool = False,
        coda_reduction: int = 4,
    ):
        BaseMHF.__init__(
            self,
            ranks=mhf_rank,
            resolutions=mhf_resolutions or [],
            factorization=mhf_factorization,
        )
        nn.Module.__init__(self)
        
        # Store parameters
        self.grid_shape = grid_shape
        self.nlat, self.nlon = grid_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.max_degree = max_degree if max_degree is not None else self.nlat // 2
        
        # Lifting network: from in_channels to hidden_channels
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # Projection network: from hidden_channels to out_channels
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
        
        # MHF-optimized SFNO blocks
        self.sfno_blocks = MHFSFNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            grid_shape=grid_shape,
            n_layers=n_layers,
            max_degree=self.max_degree,
            mhf_rank=mhf_rank,
            mhf_resolutions=mhf_resolutions,
            factorization=mhf_factorization,
            mhf_implementation=mhf_implementation,
            grid_type=grid_type,
            norm=norm,
            non_linearity=non_linearity,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            sfno_skip=sfno_skip,
        )
        
        # CoDA (Cross-Head Attention) for MHF optimization
        self.use_coda = use_coda
        if use_coda:
            try:
                from mhf.coda import CrossHeadAttention
                self.coda = CrossHeadAttention(
                    n_heads=n_layers,
                    channels_per_head=hidden_channels,
                    reduction=coda_reduction,
                )
            except ImportError:
                print("Warning: CoDA not available, falling back without CoDA")
                self.use_coda = False
                self.coda = None
        
        self._decomposed = False
    
    def decompose(self) -> None:
        """Perform MHF decomposition on all spherical spectral conv layers"""
        self.sfno_blocks.decompose_all()
        self._decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct all full weights from decomposition"""
        for conv in self.sfno_blocks.convs:
            if isinstance(conv, SphericalConvMHF):
                full_weight = conv.recompose()
                with torch.no_grad():
                    conv.weight.copy_(full_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MHF-optimized SFNO
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, in_channels, nlat, nlon]
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, out_channels, nlat, nlon]
        """
        # Lifting
        x = self.lifting(x)
        
        # SFNO blocks
        x = self.sfno_blocks(x)
        
        # CoDA (optional)
        if self.use_coda and self.coda is not None:
            x_reshaped = x.unsqueeze(1).repeat(1, self.n_layers, 1, 1, 1)
            x = self.coda(x_reshaped).mean(dim=1)
        
        # Projection
        x = self.projection(x)
        
        return x
    
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass guaranteed to use MHF decomposition"""
        assert self._decomposed, "Must call decompose() before using forward_mhf"
        return self.forward(x)
    
    def get_compression_stats(self) -> dict:
        """Get overall compression statistics across all layers"""
        total_original = 0
        total_decomposed = 0
        
        for conv in self.sfno_blocks.convs:
            if isinstance(conv, SphericalConvMHF):
                orig, decomp = conv.count_parameters()
                total_original += orig
                total_decomposed += decomp
        
        lifting_params = self.lifting.weight.numel() + self.lifting.bias.numel()
        projection_params = self.projection.weight.numel() + self.projection.bias.numel()
        
        total_original += lifting_params + projection_params
        total_decomposed += lifting_params + projection_params
        
        if total_original == 0:
            return {
                "decomposed": self._decomposed,
                "total_original_params": 0,
                "total_decomposed_params": 0,
                "overall_compression_ratio": 1.0,
                "overall_compression_factor": 1.0,
                "use_coda": self.use_coda,
            }
        
        return {
            "decomposed": self._decomposed,
            "total_original_params": total_original,
            "total_decomposed_params": total_decomposed,
            "overall_compression_ratio": total_decomposed / total_original,
            "overall_compression_factor": total_original / total_decomposed,
            "use_coda": self.use_coda,
        }
    
    @classmethod
    def from_original(
        cls,
        original_sfno,
        mhf_rank: int = 8,
        mhf_factorization: str = "tucker",
    ) -> "MHSFNO":
        """Create MHSFNO from an already-trained original SFNO
        
        Parameters
        ----------
        original_sfno
            Trained original SFNO model from neuraloperator
        mhf_rank : int, default=8
            Rank for MHF factorization
        mhf_factorization : str, default="tucker"
            Factorization type
            
        Returns
        -------
        MHSFNO
            MHF-optimized model with weights copied from original
        """
        in_channels = original_sfno.in_channels
        out_channels = original_sfno.out_channels
        hidden_channels = original_sfno.hidden_channels
        n_layers = original_sfno.n_layers
        grid_shape = getattr(original_sfno, 'grid_shape', (original_sfno.nlat, original_sfno.nlon))
        max_degree = getattr(original_sfno, 'max_degree', None)
        
        mhf_model = cls(
            grid_shape=grid_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            max_degree=max_degree,
            mhf_rank=mhf_rank,
            mhf_factorization=mhf_factorization,
        )
        
        with torch.no_grad():
            mhf_model.lifting.weight.copy_(original_sfno.lifting.weight)
            mhf_model.lifting.bias.copy_(original_sfno.lifting.bias)
            mhf_model.projection.weight.copy_(original_sfno.projection.weight)
            mhf_model.projection.bias.copy_(original_sfno.projection.bias)
        
        if hasattr(original_sfno, 'sfno_blocks') and hasattr(original_sfno.sfno_blocks, 'convs'):
            for i, orig_conv in enumerate(original_sfno.sfno_blocks.convs):
                if i < len(mhf_model.sfno_blocks.convs):
                    if hasattr(orig_conv, 'weight'):
                        mhf_model.sfno_blocks.convs[i].weight.copy_(orig_conv.weight)
        
        mhf_model.decompose()
        
        return mhf_model


# Export with standard naming convention
MHFSFNO = MHSFNO

__all__ = ["MHSFNO", "MHFSFNO", "TORCH_HARMONICS_AVAILABLE"]
