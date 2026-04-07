"""
MHF-SFNO Example: Spherical Fourier Neural Operator with MHF Optimization

This example demonstrates how to use MHF-optimized SFNO for spherical data tasks.

Requirements:
    pip install torch-harmonics
"""

import torch
import numpy as np
from models.sfno_mhf import MHSFNO, TORCH_HARMONICS_AVAILABLE

def check_dependencies():
    """Check if required dependencies are available"""
    if not TORCH_HARMONICS_AVAILABLE:
        print("⚠️  Warning: torch-harmonics is not installed.")
        print("   Install it with: pip install torch-harmonics")
        print("   Without it, SFNO will not be functional.")
        return False
    print("✓ torch-harmonics is available")
    return True


def create_spherical_data(batch_size=1, channels=3, nlat=64, nlon=128):
    """Create synthetic spherical data for demonstration
    
    In practice, this would be real spherical data like:
    - Weather/climate data on Earth's sphere
    - Cosmic microwave background radiation
    - Planetary atmospheric data
    
    Parameters
    ----------
    batch_size : int
        Number of samples
    channels : int
        Number of channels (e.g., temperature, pressure, humidity)
    nlat : int
        Number of latitude points
    nlon : int
        Number of longitude points
    
    Returns
    -------
    torch.Tensor
        Spherical data of shape [B, C, nlat, nlon]
    """
    # Create random spherical-like data with some structure
    # Use spherical harmonics-like patterns (simplified)
    x = torch.randn(batch_size, channels, nlat, nlon)
    
    # Add some spatial structure
    for b in range(batch_size):
        for c in range(channels):
            # Create a latitude-dependent pattern
            lat_pattern = torch.sin(torch.linspace(0, np.pi, nlat)).unsqueeze(1)
            # Create a longitude-dependent pattern
            lon_pattern = torch.cos(torch.linspace(0, 2*np.pi, nlon)).unsqueeze(0)
            # Combine
            pattern = lat_pattern @ lon_pattern
            x[b, c] += 0.5 * pattern
    
    return x


def example_basic_sfno_mhf():
    """Basic MHF-SFNO example"""
    print("\n" + "="*60)
    print("Example 1: Basic MHF-SFNO")
    print("="*60)
    
    if not check_dependencies():
        return
    
    # Model parameters
    grid_shape = (64, 128)  # (nlat, nlon) - typical for Earth data
    in_channels = 3
    out_channels = 3
    hidden_channels = 64
    n_layers = 4
    max_degree = 20  # Maximum spherical harmonic degree
    
    # Create MHF-optimized SFNO model
    model = MHSFNO(
        grid_shape=grid_shape,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        max_degree=max_degree,
        # MHF parameters
        mhf_rank=8,
        mhf_factorization="tucker",
        mhf_implementation="factorized",
    )
    
    print(f"\nModel created:")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Channels: {in_channels} -> {out_channels}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Layers: {n_layers}")
    print(f"  Max spherical harmonic degree: {max_degree}")
    
    # Create test data
    x = create_spherical_data(batch_size=2, channels=in_channels, 
                          nlat=grid_shape[0], nlon=grid_shape[1])
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass (without MHF decomposition)
    print("\nForward pass (dense weights)...")
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Perform MHF decomposition
    print("\nPerforming MHF decomposition...")
    model.decompose()
    
    # Get compression statistics
    stats = model.get_compression_stats()
    print(f"\nCompression statistics:")
    print(f"  Decomposed: {stats['decomposed']}")
    print(f"  Original params: {stats['total_original_params']:,}")
    print(f"  Decomposed params: {stats['total_decomposed_params']:,}")
    print(f"  Compression ratio: {stats['overall_compression_ratio']:.4f}")
    print(f"  Compression factor: {stats['overall_compression_factor']:.2f}x")
    
    # Forward pass with MHF decomposition
    print("\nForward pass (MHF-decomposed weights)...")
    with torch.no_grad():
        out_mhf = model.forward_mhf(x)
    print(f"Output shape: {out_mhf.shape}")
    
    # Compare outputs
    diff = torch.max(torch.abs(out - out_mhf)).item()
    print(f"\nMax difference between dense and MHF outputs: {diff:.2e}")


def example_sfno_with_coda():
    """MHF-SFNO with CoDA optimization"""
    print("\n" + "="*60)
    print("Example 2: MHF-SFNO with CoDA Optimization")
    print("="*60)
    
    if not check_dependencies():
        return
    
    # Create model with CoDA enabled
    model = MHSFNO(
        grid_shape=(64, 128),
        in_channels=3,
        out_channels=3,
        hidden_channels=64,
        n_layers=4,
        max_degree=20,
        # MHF parameters
        mhf_rank=8,
        mhf_factorization="tucker",
        # CoDA parameters
        use_coda=True,
        coda_reduction=4,
    )
    
    print("\nModel created with CoDA enabled")
    
    # Perform MHF decomposition
    model.decompose()
    
    # Get statistics
    stats = model.get_compression_stats()
    print(f"\nCompression statistics:")
    print(f"  Compression factor: {stats['overall_compression_factor']:.2f}x")
    print(f"  CoDA enabled: {stats['use_coda']}")


def example_multi_resolution():
    """MHF-SFNO with custom multi-resolution hierarchy"""
    print("\n" + "="*60)
    print("Example 3: Custom Multi-Resolution Hierarchy")
    print("="*60)
    
    if not check_dependencies():
        return
    
    # Custom resolution hierarchy based on spherical harmonic orders
    # Each resolution corresponds to (degree + 1)^2 coefficients
    mhf_resolutions = [9, 25, 81, 225, 441]  # Orders: 2, 4, 8, 14, 20
    
    model = MHSFNO(
        grid_shape=(64, 128),
        in_channels=3,
        out_channels=3,
        hidden_channels=64,
        n_layers=4,
        max_degree=20,
        # MHF parameters with custom resolutions
        mhf_rank=[4, 6, 8, 12, 16],  # Different ranks per resolution
        mhf_resolutions=mhf_resolutions,
        mhf_factorization="tucker",
    )
    
    print(f"\nCustom resolution hierarchy:")
    for i, res in enumerate(mhf_resolutions):
        print(f"  Level {i}: {res} coefficients (degree ~{int(np.sqrt(res))-1})")
    
    model.decompose()
    
    stats = model.get_compression_stats()
    print(f"\nCompression factor: {stats['overall_compression_factor']:.2f}x")


def example_compression_comparison():
    """Compare different MHF configurations"""
    print("\n" + "="*60)
    print("Example 4: Compression Comparison")
    print("="*60)
    
    if not check_dependencies():
        return
    
    configs = [
        {"name": "Low Rank (4)", "rank": 4},
        {"name": "Medium Rank (8)", "rank": 8},
        {"name": "High Rank (16)", "rank": 16},
    ]
    
    print(f"\n{'Config':<20} {'Original':>12} {'Compressed':>12} {'Factor':>10}")
    print("-" * 60)
    
    for config in configs:
        model = MHSFNO(
            grid_shape=(64, 128),
            in_channels=3,
            out_channels=3,
            hidden_channels=64,
            n_layers=4,
            max_degree=20,
            mhf_rank=config["rank"],
            mhf_factorization="tucker",
        )
        
        model.decompose()
        stats = model.get_compression_stats()
        
        print(f"{config['name']:<20} "
              f"{stats['total_original_params']:>12,} "
              f"{stats['total_decomposed_params']:>12,} "
              f"{stats['overall_compression_factor']:>10.2f}x")


def main():
    """Run all examples"""
    print("\nMHF-SFNO (Spherical Fourier Neural Operator with MHF)")
    print("="*60)
    
    try:
        # Run examples
        example_basic_sfno_mhf()
        example_sfno_with_coda()
        example_multi_resolution()
        example_compression_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("  Make sure all dependencies are installed:")
        print("  pip install torch-harmonics")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
