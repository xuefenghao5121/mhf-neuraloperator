"""
Example: MHF-optimized FNO for Darcy Flow

This example shows how to use MHF optimization with FNO.
You can compare the original FNO with MHF-FNO in terms of:
- Parameter count
- Memory usage
- Accuracy
"""

import torch
import torch.nn as nn
import numpy as np

from mhf.models import MHFNO
from neuralop.models import FNO


def compare_parameter_counts():
    """Compare parameter counts between original FNO and MHF-FNO"""
    
    print("=" * 60)
    print("Comparing original FNO vs MHF-optimized FNO")
    print("=" * 60)
    
    # Model configuration
    config = {
        "n_modes": (16, 16),
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": 64,
        "n_layers": 4,
    }
    
    # Original FNO
    original_fno = FNO(**config)
    
    # Count original parameters
    original_params = sum(p.numel() for p in original_fno.parameters())
    print(f"\nOriginal FNO:")
    print(f"  Total parameters: {original_params:,}")
    
    # MHF FNO with different ranks
    for mhf_rank in [4, 8, 16]:
        mhf_fno = MHFNO(
            **config,
            mhf_rank=mhf_rank,
            mhf_factorization="tucker",
        )
        mhf_fno.decompose()
        
        stats = mhf_fno.get_compression_stats()
        
        print(f"\nMHF-FNO (rank={mhf_rank}):")
        print(f"  Original spectral params: {stats['total_original_params']:,}")
        print(f"  Compressed spectral params: {stats['total_decomposed_params']:,}")
        print(f"  Overall compression ratio: {stats['overall_compression_ratio']:.3f}")
        print(f"  Compression factor: {stats['overall_compression_factor']:.2f}x")
        
        total_mhf_params = sum(p.numel() for p in mhf_fno.parameters())
        # Subtract the original dense weights which are kept but not used in factorized mode
        print(f"  Total model parameters with MHF: {total_mhf_params:,}")
        print(f"  Total model compression: {total_mhf_params / original_params:.3f}")
    
    print("\n" + "=" * 60)


def forward_pass_speed_test():
    """Simple forward pass speed test"""
    print("\nSpeed test on a random input:")
    
    config = {
        "n_modes": (16, 16),
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": 64,
        "n_layers": 4,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Create models
    original_fno = FNO(**config).to(device)
    mhf_fno = MHFNO(**config, mhf_rank=8).to(device)
    mhf_fno.decompose()
    
    # Random input
    batch_size = 8
    x = torch.randn(batch_size, 1, 64, 64).to(device)
    
    # Warmup
    for _ in range(10):
        _ = original_fno(x)
        _ = mhf_fno(x)
    
    # Speed test original
    torch.cuda.synchronize() if device.type == "cuda" else None
    import time
    start = time.time()
    for _ in range(100):
        _ = original_fno(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    original_time = (time.time() - start) / 100
    
    # Speed test MHF
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(100):
        _ = mhf_fno(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    mhf_time = (time.time() - start) / 100
    
    print(f"  Original FNO: {original_time * 1000:.2f} ms/forward")
    print(f"  MHF-FNO: {mhf_time * 1000:.2f} ms/forward")
    print(f"  Speedup: {original_time / mhf_time:.2f}x")


def loading_from_original_checkpoint():
    """Example showing how to load an already-trained original FNO and compress it with MHF"""
    print("\n" + "=" * 60)
    print("Example: Compressing a pre-trained original FNO")
    print("=" * 60)
    
    # Create and save a random original FNO
    config = {
        "n_modes": (16, 16),
        "in_channels": 1,
        "out_channels": 1,
        "hidden_channels": 64,
        "n_layers": 4,
    }
    
    original_fno = FNO(**config)
    
    # Simulate saving a checkpoint
    print("\nSaving original FNO checkpoint...")
    torch.save(original_fno.state_dict(), "/tmp/original_fno.pt")
    
    # Now load and compress
    from mhf.factory import load_original_and_compress
    
    print("Loading original and compressing with MHF...")
    compressed_model = load_original_and_compress(
        checkpoint_path="/tmp/original_fno.pt",
        model_name="fno",
        model_kwargs=config,
        mhf_rank=8,
        factorization="tucker",
        output_path="/tmp/mhf_fno.pt",
    )
    
    stats = compressed_model.get_compression_stats()
    print(f"\nCompression result:")
    print(f"  Compression factor: {stats['overall_compression_factor']:.2f}x")
    print(f"  Saved compressed checkpoint to /tmp/mhf_fno.pt")
    
    # Check forward pass works
    x = torch.randn(1, 1, 32, 32)
    output = compressed_model(x)
    print(f"\nForward pass OK, output shape: {output.shape}")
    
    print("\nDone!")


if __name__ == "__main__":
    compare_parameter_counts()
    forward_pass_speed_test()
    loading_from_original_checkpoint()
