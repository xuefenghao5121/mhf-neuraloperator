"""
Final comparison script with correct einsum indexing
"""

import sys
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorly as tl
tl.set_backend('pytorch')

from tltorch.factorized_tensors.core import FactorizedTensor

# Add project root to path
sys.path.insert(0, '/home/huawei/.openclaw/workspace')
sys.path.insert(0, '/home/huawei/Desktop/home/xuefenghao/workspace/MHF-NeuralOperator')

from neuralop.models import TFNO
from neuralop.layers.spectral_convolution import SpectralConv

# Direct import from our MHF
sys.path.insert(0, '/home/huawei/Desktop/home/xuefenghao/workspace/MHF-NeuralOperator/mhf')
from base import BaseMHF, MHFMetadata, MultiResolutionHierarchicalFactorization
from coda import CrossHeadAttention


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_forward_time(model: nn.Module, input_tensor: torch.Tensor, n_runs: int = 100) -> Dict[str, float]:
    """Measure forward pass time"""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    start_time = time.time()
    for _ in range(n_runs):
        _ = model(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    return {
        'total_time_s': total_time,
        'avg_time_ms': (total_time * 1000) / n_runs,
        'throughput_samples_per_sec': n_runs / total_time,
    }


# Our complete MHF-optimized Spectral Conv
class MHF_SpectralConv_TFNO(nn.Module):
    """
    MHF-optimized Spectral Convolution for TFNO
    
    TFNO already uses Tucker factorization on the spectral weights.
    This class further applies MHF: multi-resolution hierarchical factorization
    to achieve additional parameter compression.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        factorization: str = "tucker",
        implementation: str = "factorized",
        n_heads: int = 1,
        use_coda: bool = False,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = len(n_modes)
        self.factorization = factorization
        self.implementation = implementation
        self.n_heads = n_heads
        self.use_coda = use_coda
        
        # Original TFNO spectral convolution (already factorized)
        self.original_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            **kwargs
        )
        
        self.n_modes = n_modes
        
        # MHF hierarchical decomposition
        self.mhf = None
        self._mhf_decomposed = False
        self._metadata = None
        
        if mhf_resolutions is None:
            # Auto-generate hierarchical resolutions
            mhf_resolutions = self._auto_generate_resolutions(n_modes)
        self.mhf_resolutions = sorted(mhf_resolutions)
        self.mhf_rank = mhf_rank
        
        # Create MHF decomposition
        self.mhf = MultiResolutionHierarchicalFactorization(
            resolutions=self.mhf_resolutions,
            ranks=mhf_rank,
            factorization_type=factorization,
        )
        
        # CoDA (Cross-head Attention)
        if use_coda:
            if out_channels % n_heads != 0:
                raise ValueError(
                    f"out_channels ({out_channels}) must be divisible by n_heads ({n_heads})"
                )
            channels_per_head = out_channels // n_heads
            self.coda = CrossHeadAttention(
                n_heads=n_heads,
                channels_per_head=channels_per_head,
                reduction=coda_reduction,
                dropout=coda_dropout,
            )
        else:
            self.coda = None
    
    def _auto_generate_resolutions(self, n_modes: Tuple[int, ...]) -> List[int]:
        """Auto-generate hierarchical resolutions based on n_modes"""
        max_res = max(n_modes)
        resolutions = []
        current = 4
        while current <= max_res:
            resolutions.append(current)
            current *= 2
        if resolutions[-1] != max_res:
            resolutions.append(max_res)
        return resolutions
    
    def decompose(self) -> None:
        """Perform MHF decomposition on top of existing TFNO factorization
        
        This should be called after initial training to further compress the model.
        """
        # Get the full weight tensor from the existing TFNO factorization
        if hasattr(self.original_conv, 'weight'):
            if isinstance(self.original_conv.weight, FactorizedTensor):
                # Reconstruct full weight from existing factorization
                full_weight = self.original_conv.weight.to_tensor()
            else:
                full_weight = self.original_conv.weight
            
            # Spatial dimensions are after in/out channels
            spatial_dims = list(range(2, 2 + self.dim))
            
            # Perform MHF decomposition
            self.mhf.decompose(full_weight, spatial_dims=spatial_dims)
            
            # Create metadata
            original_params = full_weight.numel()
            decomposed_params = self.mhf.count_params()
            
            self._metadata = MHFMetadata(
                original_shape=tuple(full_weight.shape),
                decomposed_shape=None,
                resolutions=self.mhf_resolutions,
                ranks=self.mhf_rank,
                factorization_type=self.factorization,
                original_num_params=original_params,
                decomposed_num_params=decomposed_params,
                decomposed=True,
            )
            
            self._mhf_decomposed = True
    
    def recompose(self) -> torch.Tensor:
        """Reconstruct full weight tensor from MHF decomposition"""
        assert self._mhf_decomposed, "Decompose must be called first"
        return self.mhf.reconstruct()
    
    def transform(self, x, output_shape=None):
        """Delegate transform to original convolution"""
        return self.original_conv.transform(x, output_shape=output_shape)
    
    @property
    def n_modes(self):
        """Delegate n_modes to original convolution"""
        return self.original_conv.n_modes
    
    @n_modes.setter
    def n_modes(self, value):
        """Delegate n_modes setter to original convolution"""
        self.original_conv.n_modes = value
    
    @property
    def complex_data(self):
        """Delegate complex_data to original convolution"""
        return self.original_conv.complex_data
    
    @property
    def order(self):
        """Delegate order to original convolution"""
        return self.original_conv.order
    
    @property
    def max_n_modes(self):
        """Delegate max_n_modes to original convolution"""
        return self.original_conv.max_n_modes
    
    @property
    def fno_block_precision(self):
        """Delegate fno_block_precision to original convolution"""
        return self.original_conv.fno_block_precision
    
    @property
    def fft_norm(self):
        """Delegate fft_norm to original convolution"""
        return self.original_conv.fft_norm
    
    @property
    def enforce_hermitian_symmetry(self):
        """Delegate enforce_hermitian_symmetry to original convolution"""
        return self.original_conv.enforce_hermitian_symmetry
    
    @property
    def bias(self):
        """Delegate bias to original convolution"""
        return self.original_conv.bias
    
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None) -> torch.Tensor:
        """Forward pass
        
        If MHF has been decomposed, use MHF factors directly.
        Otherwise, fall back to original TFNO factorization.
        """
        if self._mhf_decomposed and self.implementation == "factorized":
            # Use MHF decomposed weights directly
            # This follows the same FFT processing as original SpectralConv
            # but uses MHF for the contraction
            
            batchsize, channels, *mode_sizes = x.shape
            
            fft_size = list(mode_sizes)
            if not self.original_conv.complex_data:
                fft_size[-1] = fft_size[-1] // 2 + 1
            fft_dims = list(range(-self.dim, 0))
            
            # FFT
            if self.original_conv.fno_block_precision == "half":
                x = x.half()
            
            if self.original_conv.complex_data:
                x = torch.fft.fftn(x, norm=self.original_conv.fft_norm, dim=fft_dims)
                dims_to_fft_shift = fft_dims
            else:
                x = torch.fft.rfftn(x, norm=self.original_conv.fft_norm, dim=fft_dims)
                dims_to_fft_shift = fft_dims[:-1]
            
            if self.dim > 1:
                x = torch.fft.fftshift(x, dim=dims_to_fft_shift)
            
            if self.original_conv.fno_block_precision == "mixed":
                x = x.chalf()
            
            if self.original_conv.fno_block_precision in ["half", "mixed"]:
                out_dtype = torch.chalf
            else:
                out_dtype = torch.cfloat
            out_fft = torch.zeros(
                [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
            )
            
            # Get the right modes - follows original SpectralConv
            starts = [
                (max_modes - min(size, n_mode))
                for (size, n_mode, max_modes) in zip(
                    fft_size, self.original_conv.n_modes, self.original_conv.max_n_modes
                )
            ]
            
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels
            if not self.original_conv.complex_data:
                slices_w += [
                    slice(start // 2, -start // 2) if start else slice(start, None)
                    for start in starts[:-1]
                ]
                slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
            else:
                slices_w += [
                    slice(start // 2, -start // 2) if start else slice(start, None)
                    for start in starts
                ]
            slices_w = tuple(slices_w)
            
            # For MHF decomposition with multiple resolution levels,
            # we loop over each resolution level in the decomposition and add all contributions
            # For each level, ALL spatial dimensions use the current resolution level
            for res, factor in self.mhf.factors.items():
                # For this resolution level, EVERY spatial dimension gets cropped to this resolution
                slices_x = [slice(None), slice(None)]  # Batch_size, channels
                
                for dim_size in fft_size:
                    # For each spatial dimension, use current resolution
                    kept_modes = res
                    center = dim_size // 2
                    negative_freqs = kept_modes // 2
                    positive_freqs = kept_modes // 2 + kept_modes % 2
                    slices_x += [slice(center - negative_freqs, center + positive_freqs)]
                
                slices_x = tuple(slices_x)
                
                # Get the correctly sliced x at this resolution level
                x_at_res = x[slices_x]
                # Directly contract using this level's factor - we don't need to loop again because factor already is at this resolution
                if self.mhf.factorization_type == "tucker":
                    contrib = self.mhf._contract_tucker(x_at_res, factor)
                elif self.mhf.factorization_type == "cp":
                    contrib = self.mhf._contract_cp(x_at_res, factor)
                elif self.mhf.factorization_type == "tt":
                    contrib = self.mhf._contract_tt(x_at_res, factor)
                else:
                    raise ValueError(f"Unknown factorization type: {self.mhf.factorization_type}")
                
                out_fft[slices_x] = contrib
            
            # After processing all resolution levels, proceed with inverse FFT
            if self.original_conv.order > 1:
                out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
            
            if self.original_conv.complex_data:
                x = torch.fft.ifftn(out_fft, s=mode_sizes, norm=self.original_conv.fft_norm, dim=fft_dims)
            else:
                if self.original_conv.enforce_hermitian_symmetry:
                    out_fft = torch.fft.ifftn(out_fft, s=mode_sizes[:-1], dim=fft_dims[:-1], norm=self.original_conv.fft_norm)
                    out_fft[..., 0].imag.zero_()
                    if mode_sizes[-1] % 2 == 0:
                        out_fft[..., -1].imag.zero_()
                    x = torch.fft.irfft(out_fft, n=mode_sizes[-1], dim=fft_dims[-1], norm=self.original_conv.fft_norm)
                else:
                    x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.original_conv.fft_norm)
            
            # Add bias
            if self.original_conv.bias is not None:
                x = x + self.original_conv.bias
        else:
            # Fall back to original convolution
            x = self.original_conv(x, output_shape=output_shape)
        
        # Apply CoDA if enabled
        if self.use_coda and self.coda is not None:
            if x.dim() == 3:  # 1D: [B, C, L]
                B, C, L = x.shape
                x_heads = x.view(B, self.n_heads, C // self.n_heads, L)
                x_heads = self.coda(x_heads)
                x = x_heads.view(B, C, L)
            elif x.dim() == 4:  # 2D: [B, C, H, W]
                B, C, H, W = x.shape
                x_heads = x.view(B, self.n_heads, C // self.n_heads, H, W)
                x_heads = self.coda(x_heads)
                x = x_heads.view(B, C, H, W)
            elif x.dim() == 5:  # 3D: [B, C, H, W, D]
                B, C, H, W, D = x.shape
                x_heads = x.view(B, self.n_heads, C // self.n_heads, H, W, D)
                x_heads = self.coda(x_heads)
                x = x_heads.view(B, C, H, W, D)
        
        return x
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        if not self._mhf_decomposed:
            # Get original parameter count
            if isinstance(self.original_conv.weight, FactorizedTensor):
                original_params = sum(p.numel() for p in self.original_conv.weight.parameters())
            else:
                original_params = self.original_conv.weight.numel()
            
            return {
                "mhf_decomposed": False,
                "original_factorized_params": original_params,
                "mhf_params": None,
                "compression_ratio": None,
                "compression_factor": None,
            }
        
        if self._metadata is not None:
            # Get original TFNO parameter count
            if isinstance(self.original_conv.weight, FactorizedTensor):
                original_tfno_params = sum(p.numel() for p in self.original_conv.weight.parameters())
            else:
                original_tfno_params = self.original_conv.weight.numel()
            
            return {
                "mhf_decomposed": True,
                "original_dense_params": self._metadata.original_num_params,
                "original_tfno_params": original_tfno_params,
                "mhf_params": self._metadata.decomposed_num_params,
                "compression_ratio_vs_dense": self._metadata.compression_ratio,
                "compression_factor_vs_dense": self._metadata.compression_factor,
                "compression_ratio_vs_tfno": self._metadata.decomposed_num_params / original_tfno_params,
                "compression_factor_vs_tfno": original_tfno_params / max(1, self._metadata.decomposed_num_params),
            }


from neuralop.models.fno import FNO

class MHF_TFNO(FNO):
    """
    MHF-optimized Tensorized Fourier Neural Operator
    
    This class extends the original TFNO with:
    1. Multi-resolution hierarchical factorization (MHF) on spectral weights
    2. Cross-head attention (CoDA) for improved information flow across heads
    
    TFNO already applies Tucker factorization to compress the spectral weights.
    MHF further compresses these by hierarchical multi-resolution decomposition.
    
    Parameters
    ----------
    n_modes : Tuple[int, ...]
        Number of Fourier modes along each dimension
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    hidden_channels : int
        Number of hidden channels
    n_layers : int, default=4
        Number of FNO layers
    mhf_rank : Union[int, Dict[str, int]], default=8
        Rank(s) for MHF factorization at each resolution level
    mhf_resolutions : List[int], optional
        Resolutions for MHF hierarchy. If None, auto-generated.
    mhf_factorization : str, default="tucker"
        Tensor factorization type for MHF: "cp", "tucker", or "tt"
    n_heads : int, default=4
        Number of heads for CoDA cross-head attention
    use_coda : bool, default=True
        Whether to enable Cross-head Attention (CoDA)
    coda_reduction : int, default=4
        Channel reduction ratio in CoDA FFN
    coda_dropout : float, default=0.0
        Dropout probability for CoDA
    """
    
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: float = 2.0,
        projection_channel_ratio: float = 2.0,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Optional[Literal["ada_in", "group_norm", "instance_norm"]] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Optional[Union[float, List[float]]] = None,
        domain_padding: Optional[Union[float, List[float]]] = None,
        # MHF specific parameters
        mhf_rank: Union[int, Dict[str, int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        mhf_factorization: str = "tucker",
        n_heads: int = 4,
        use_coda: bool = True,
        coda_reduction: int = 4,
        coda_dropout: float = 0.0,
        rank: float = 0.1,
        factorization: str = "Tucker",
    ):
        # Initialize original FNO (which becomes TFNO with the default factorization)
        super().__init__(
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
            rank=rank,
            factorization=factorization,
        )
        
        # MHF parameters storage
        self.mhf_rank = mhf_rank
        self.mhf_resolutions = mhf_resolutions
        self.mhf_factorization = mhf_factorization
        self.n_heads = n_heads
        self.use_coda = use_coda
        self.coda_reduction = coda_reduction
        self.coda_dropout = coda_dropout
        self._mhf_decomposed = False
        self._metadata = None
        
        # Replace all spectral convolutions in FNOBlocks with MHF-optimized versions
        original_convs = self.fno_blocks.convs
        
        # Get common kwargs from first original convolution
        first_conv = original_convs[0]
        replace_kwargs = {
            "rank": getattr(first_conv, 'rank', 0.1),
            "fixed_rank_modes": getattr(first_conv, 'fixed_rank_modes', False),
            "implementation": getattr(first_conv, 'implementation', 'factorized'),
            "decomposition_kwargs": getattr(first_conv, 'decomposition_kwargs', {}),
            "separable": getattr(first_conv, 'separable', False),
            "complex_data": getattr(first_conv, 'complex_data', False),
            "max_n_modes": getattr(first_conv, 'max_n_modes', n_modes),
            "fno_block_precision": getattr(first_conv, 'fno_block_precision', 'full'),
            "fft_norm": getattr(first_conv, 'fft_norm', 'forward'),
            "init_std": getattr(first_conv, 'init_std', 'auto'),
        }
        
        # Replace each convolution
        new_convs = nn.ModuleList()
        for i, orig_conv in enumerate(original_convs):
            in_ch = orig_conv.in_channels
            out_ch = orig_conv.out_channels
            
            # Create MHF-optimized convolution
            mhf_conv = MHF_SpectralConv_TFNO(
                in_channels=in_ch,
                out_channels=out_ch,
                n_modes=n_modes,
                mhf_rank=self.mhf_rank,
                mhf_resolutions=self.mhf_resolutions,
                factorization=self.mhf_factorization,
                n_heads=self.n_heads,
                use_coda=self.use_coda,
                coda_reduction=self.coda_reduction,
                coda_dropout=self.coda_dropout,
                **replace_kwargs
            )
            
            # Copy weights from original convolution
            if hasattr(orig_conv, 'weight'):
                with torch.no_grad():
                    mhf_conv.original_conv.weight = orig_conv.weight
                if hasattr(orig_conv, 'bias') and orig_conv.bias is not None:
                    mhf_conv.original_conv.bias = orig_conv.bias
            
            new_convs.append(mhf_conv)
        
        # Replace the convs in fno_blocks
        self.fno_blocks.convs = new_convs
    
    def decompose(self) -> None:
        """
        Perform MHF decomposition on all spectral convolutions
        
        This should be called after training the model with the original
        factorization to further compress the parameters.
        """
        total_original_dense = 0
        total_original_tfno = 0
        total_mhf = 0
        
        # Decompose each convolution
        for conv in self.fno_blocks.convs:
            conv.decompose()
            stats = conv.get_compression_stats()
            
            if stats["mhf_decomposed"]:
                total_original_dense += stats.get("original_dense_params", 0)
                total_original_tfno += stats.get("original_tfno_params", 0)
                total_mhf += stats.get("mhf_params", 0)
        
        # Aggregate metadata
        if total_original_tfno > 0 and total_mhf > 0:
            self._metadata = {
                "n_layers": self.n_layers,
                "total_original_dense_params": total_original_dense,
                "total_original_tfno_params": total_original_tfno,
                "total_mhf_params": total_mhf,
                "compression_ratio_vs_dense": total_mhf / total_original_dense,
                "compression_factor_vs_dense": total_original_dense / max(1, total_mhf),
                "compression_ratio_vs_tfno": total_mhf / total_original_tfno,
                "compression_factor_vs_tfno": total_original_tfno / max(1, total_mhf),
                "mhf_factorization": self.mhf_factorization,
                "use_coda": self.use_coda,
                "n_heads": self.n_heads if self.use_coda else None,
            }
        
        self._mhf_decomposed = True
    
    def recompose(self) -> None:
        """Reconstruct all full weights from MHF decomposition"""
        for conv in self.fno_blocks.convs:
            if hasattr(conv, 'recompose'):
                full_weight = conv.recompose()
                # Update the original convolution weight
                if hasattr(conv.original_conv, 'weight'):
                    with torch.no_grad():
                        if isinstance(conv.original_conv.weight, nn.Parameter):
                            conv.original_conv.weight.data.copy_(full_weight)
    
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using MHF decomposition directly
        
        Required by BaseMHF abstract interface.
        """
        # The forward pass already uses MHF when decomposed
        return self.forward(x)
    
    def get_compression_stats(self) -> Optional[Dict[str, Any]]:
        """Get overall compression statistics"""
        if not self._mhf_decomposed:
            # Count parameters without MHF decomposition
            total_params = count_parameters(self)
            return {
                "mhf_decomposed": False,
                "total_params": total_params,
            }
        
        return self._metadata


def print_model_stats(name: str, model: nn.Module, after_decompose: bool = False):
    """Print model statistics"""
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Check if model has get_compression_stats
    if hasattr(model, 'get_compression_stats'):
        stats = model.get_compression_stats()
        if stats is not None:
            if after_decompose and stats.get('mhf_decomposed', False):
                if 'total_original_dense_params' in stats:
                    print(f"Original dense params: {stats['total_original_dense_params']:,}")
                if 'total_original_tfno_params' in stats:
                    print(f"Original TFNO params: {stats['total_original_tfno_params']:,}")
                if 'total_mhf_params' in stats:
                    print(f"MHF decomposed params: {stats['total_mhf_params']:,}")
                if 'compression_ratio_vs_tfno' in stats:
                    print(f"Compression ratio vs original TFNO: {stats['compression_ratio_vs_tfno']:.4f}")
                    print(f"Compression factor vs original TFNO: {stats['compression_factor_vs_tfno']:.2f}x")
                if 'compression_ratio_vs_dense' in stats:
                    print(f"Compression ratio vs dense: {stats['compression_ratio_vs_dense']:.4f}")
                    print(f"Compression factor vs dense: {stats['compression_factor_vs_dense']:.2f}x")
    
    print()


def compare_models(args):
    """Compare different model configurations"""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    if args.dim == 2:
        n_modes = (args.n_modes, args.n_modes)
        input_shape = (args.batch_size, args.in_channels, args.resolution, args.resolution)
    elif args.dim == 3:
        n_modes = (args.n_modes, args.n_modes, args.n_modes)
        input_shape = (args.batch_size, args.in_channels, args.resolution, args.resolution, args.resolution)
    else:  # 1D
        n_modes = (args.n_modes,)
        input_shape = (args.batch_size, args.in_channels, args.resolution)
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {args.dim}D")
    print(f"  n_modes: {n_modes}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Input resolution: {args.resolution}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  TFNO rank: {args.tfno_rank}")
    
    # Create dummy input
    x = torch.randn(*input_shape)
    
    results = []
    
    # =============
    # 1. Original TFNO
    # =============
    print("\n" + "-"*60)
    print("1. Original TFNO (from neuraloperator)")
    print("-"*60)
    
    original_tfno = TFNO(
        n_modes=n_modes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        rank=args.tfno_rank,
        factorization="Tucker",
    )
    original_tfno = original_tfno.to(device)
    original_params = count_parameters(original_tfno)
    print_model_stats("Original TFNO", original_tfno)
    
    # Measure forward time
    time_stats = measure_forward_time(original_tfno, x.clone(), n_runs=args.n_runs)
    print(f"Average forward pass time: {time_stats['avg_time_ms']:.3f} ms")
    print(f"Throughput: {time_stats['throughput_samples_per_sec']:.2f} samples/sec")
    
    results.append({
        'name': 'Original TFNO',
        'params': original_params,
        'avg_time_ms': time_stats['avg_time_ms'],
        'compression_ratio': 1.0,
        'compression_factor': 1.0,
    })
    
    # =============
    # 2. MHF-TFNO (without CoDA)
    # =============
    print("\n" + "-"*60)
    print("2. MHF-TFNO (Multi-Resolution Hierarchical Factorization)")
    print("-"*60)
    
    # Auto-generate resolutions based on n_modes
    max_mode = max(n_modes)
    mhf_resolutions = []
    current = 4
    while current <= max_mode:
        mhf_resolutions.append(current)
        current *= 2
    if mhf_resolutions[-1] != max_mode:
        mhf_resolutions.append(max_mode)
    
    print(f"MHF resolutions: {mhf_resolutions}")
    print(f"MHF rank: {args.mhf_rank}")
    print(f"MHF factorization: {args.mhf_factorization}")
    print(f"Use CoDA: False")
    
    mhf_tfno = MHF_TFNO(
        n_modes=n_modes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        mhf_rank=args.mhf_rank,
        mhf_resolutions=mhf_resolutions,
        mhf_factorization=args.mhf_factorization,
        use_coda=False,
        rank=args.tfno_rank,
        factorization="Tucker",
    )
    mhf_tfno = mhf_tfno.to(device)
    
    # Before decomposition
    print("\nBefore MHF decomposition:")
    before_params = count_parameters(mhf_tfno)
    print_model_stats("MHF-TFNO (before decompose)", mhf_tfno)
    
    # Perform MHF decomposition
    print("Performing MHF decomposition...")
    mhf_tfno.decompose()
    print("Decomposition complete.")
    
    # After decomposition
    print("\nAfter MHF decomposition:")
    after_params = count_parameters(mhf_tfno)
    print_model_stats("MHF-TFNO (after decompose)", mhf_tfno, after_decompose=True)
    
    # Measure forward time
    time_stats = measure_forward_time(mhf_tfno, x.clone(), n_runs=args.n_runs)
    print(f"\nAverage forward pass time: {time_stats['avg_time_ms']:.3f} ms")
    print(f"Throughput: {time_stats['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Get compression stats
    stats = mhf_tfno.get_compression_stats()
    if stats:
        compression_factor = stats.get('compression_factor_vs_tfno', 1.0)
        compression_ratio = stats.get('compression_ratio_vs_tfno', 1.0)
    else:
        if original_params > 0:
            compression_factor = original_params / max(1, after_params)
            compression_ratio = after_params / original_params
        else:
            compression_factor = 1.0
            compression_ratio = 1.0
    
    results.append({
        'name': 'MHF-TFNO',
        'params': after_params,
        'avg_time_ms': time_stats['avg_time_ms'],
        'compression_ratio': compression_ratio,
        'compression_factor': compression_factor,
    })
    
    # =============
    # 3. MHF-TFNO + CoDA
    # =============
    print("\n" + "-"*60)
    print("3. MHF-TFNO + CoDA (with Cross-Head Attention)")
    print("-"*60)
    
    print(f"MHF resolutions: {mhf_resolutions}")
    print(f"MHF rank: {args.mhf_rank}")
    print(f"MHF factorization: {args.mhf_factorization}")
    print(f"Use CoDA: True")
    print(f"Number of heads: {args.n_heads}")
    print(f"CoDA reduction: {args.coda_reduction}")
    
    mhf_tfno_coda = MHF_TFNO(
        n_modes=n_modes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        mhf_rank=args.mhf_rank,
        mhf_resolutions=mhf_resolutions,
        mhf_factorization=args.mhf_factorization,
        use_coda=True,
        n_heads=args.n_heads,
        coda_reduction=args.coda_reduction,
        coda_dropout=args.coda_dropout,
        rank=args.tfno_rank,
        factorization="Tucker",
    )
    mhf_tfno_coda = mhf_tfno_coda.to(device)
    
    print("\nBefore MHF decomposition:")
    before_coda_params = count_parameters(mhf_tfno_coda)
    print_model_stats("MHF-TFNO + CoDA (before decompose)", mhf_tfno_coda)
    
    # Perform MHF decomposition
    print("Performing MHF decomposition...")
    mhf_tfno_coda.decompose()
    print("Decomposition complete.")
    
    print("\nAfter MHF decomposition:")
    after_coda_params = count_parameters(mhf_tfno_coda)
    print_model_stats("MHF-TFNO + CoDA (after decompose)", mhf_tfno_coda, after_decompose=True)
    
    # Measure forward time
    time_stats = measure_forward_time(mhf_tfno_coda, x.clone(), n_runs=args.n_runs)
    print(f"\nAverage forward pass time: {time_stats['avg_time_ms']:.3f} ms")
    print(f"Throughput: {time_stats['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Get compression stats
    stats = mhf_tfno_coda.get_compression_stats()
    if stats:
        compression_factor = stats.get('compression_factor_vs_tfno', 1.0)
        compression_ratio = stats.get('compression_ratio_vs_tfno', 1.0)
    else:
        if original_params > 0:
            compression_factor = original_params / max(1, after_coda_params)
            compression_ratio = after_coda_params / original_params
        else:
            compression_factor = 1.0
            compression_ratio = 1.0
    
    results.append({
        'name': 'MHF-TFNO + CoDA',
        'params': after_coda_params,
        'avg_time_ms': time_stats['avg_time_ms'],
        'compression_ratio': compression_ratio,
        'compression_factor': compression_factor,
    })
    
    # =============
    # Summary
    # =============
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Configuration: {args.dim}D, modes={args.n_modes}, hidden={args.hidden_channels}, layers={args.n_layers}")
    print()
    
    print(f"{'Model':<20} {'Parameters':>12} {'Compression':>12} {'Factor':>10} {'Avg Time (ms)':>15}")
    print("-"*70)
    for res in results:
        cf = res['compression_factor']
        cf_str = f"{cf:.2f}x" if cf > 1 else "1.00x"
        cr = f"{res['compression_ratio']:.4f}" if res['compression_ratio'] != 1.0 else "1.0000"
        print(f"{res['name']:<20} {res['params']:>12,} {cr:>12} {cf_str:>10} {res['avg_time_ms']:>15.3f}")
    print()
    
    # Save results to file
    with open('tfno_mhf_comparison_results.txt', 'w') as f:
        f.write(f"Configuration:\n")
        f.write(f"  dim: {args.dim}\n")
        f.write(f"  n_modes: {args.n_modes}\n")
        f.write(f"  in_channels: {args.in_channels}\n")
        f.write(f"  out_channels: {args.out_channels}\n")
        f.write(f"  hidden_channels: {args.hidden_channels}\n")
        f.write(f"  n_layers: {args.n_layers}\n")
        f.write(f"  resolution: {args.resolution}\n")
        f.write(f"  batch_size: {args.batch_size}\n")
        f.write(f"  tfno_rank: {args.tfno_rank}\n")
        f.write(f"  mhf_rank: {args.mhf_rank}\n")
        f.write(f"  mhf_factorization: {args.mhf_factorization}\n")
        f.write(f"  n_heads: {args.n_heads}\n")
        f.write(f"  coda_reduction: {args.coda_reduction}\n")
        f.write("\n")
        f.write(f"{'Model':<20} {'Parameters':>12} {'Compression':>12} {'Factor':>10} {'Avg Time (ms)':>15}\n")
        f.write("-"*70 + "\n")
        for res in results:
            cf = res['compression_factor']
            cf_str = f"{cf:.2f}x" if cf > 1 else "1.00x"
            cr = f"{res['compression_ratio']:.4f}" if res['compression_ratio'] != 1.0 else "1.0000"
            f.write(f"{res['name']:<20} {res['params']:>12,} {cr:>12} {cf_str:>10} {res['avg_time_ms']:>15.3f}\n")
    
    print(f"Results saved to: tfno_mhf_comparison_results.txt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare original TFNO with MHF-optimized TFNO'
    )
    parser.add_argument('--dim', type=int, default=2, choices=[1, 2, 3],
                        help='Problem dimension (default: 2)')
    parser.add_argument('--n-modes', type=int, default=16,
                        help='Number of Fourier modes (default: 16)')
    parser.add_argument('--in-channels', type=int, default=1,
                        help='Number of input channels (default: 1)')
    parser.add_argument('--out-channels', type=int, default=1,
                        help='Number of output channels (default: 1)')
    parser.add_argument('--hidden-channels', type=int, default=64,
                        help='Number of hidden channels (default: 64)')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of FNO layers (default: 4)')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Input resolution (default: 64)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for timing (default: 1)')
    parser.add_argument('--tfno-rank', type=float, default=0.1,
                        help='TFNO rank for Tucker decomposition (default: 0.1)')
    parser.add_argument('--mhf-rank', type=int, default=8,
                        help='MHF rank for factorization (default: 8)')
    parser.add_argument('--mhf-factorization', type=str, default='tucker',
                        choices=['cp', 'tucker', 'tt'],
                        help='MHF factorization type (default: tucker)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of heads for CoDA (default: 4)')
    parser.add_argument('--coda-reduction', type=int, default=4,
                        help='CoDA channel reduction (default: 4)')
    parser.add_argument('--coda-dropout', type=float, default=0.0,
                        help='CoDA dropout (default: 0.0)')
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Number of forward runs for timing (default: 20)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')
    
    args = parser.parse_args()
    compare_models(args)


if __name__ == '__main__':
    main()
