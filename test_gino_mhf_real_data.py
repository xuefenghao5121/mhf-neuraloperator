"""
Test GINO MHF on real Darcy Flow dataset
=======================================

This script tests MHF-GINO against baseline GINO on a real dataset (Darcy Flow).
Compares:
- Accuracy (Loss)
- Parameter count
- Inference latency
- Training speed
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import datasets from neuraloperator
from neuralop.data.datasets.darcy import load_darcy_flow_small
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

# Import baseline GINO and MHF-GINO
from neuralop.models.gino import GINO as BaselineGINO
from models import MHF_GINO

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Test MHF-GINO on real Darcy Flow dataset')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Geometric dimension (2 or 3)')
    parser.add_argument('--hidden-channels', type=int, default=64,
                        help='Hidden channels in FNO')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of FNO layers')
    parser.add_argument('--n-modes', type=int, default=16,
                        help='Number of Fourier modes per dimension')
    parser.add_argument('--mhf-rank', type=int, default=8,
                        help='MHF rank for factorization')
    parser.add_argument('--factorization', type=str, default='tucker',
                        choices=['tucker', 'cp', 'tt'],
                        help='Tensor factorization type')
    parser.add_argument('--in-radius', type=float, default=0.033,
                        help='Input GNO radius')
    parser.add_argument('--out-radius', type=float, default=0.033,
                        help='Output GNO radius')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--benchmark-runs', type=int, default=50,
                        help='Number of runs for inference benchmark')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--data-root', type=str, 
                        default='/home/huawei/.openclaw/workspace/neuraloperator/neuralop/data/datasets/data',
                        help='Path to Darcy data')
    return parser.parse_args()


def convert_darcy_to_gino_format(batch, device):
    """Convert Darcy Flow data to GINO format.
    
    GINO expects:
    - input_geom: (batch, n_input_points, dim) - input point coordinates
    - latent_queries: (batch, *latent_grid_shape, dim) - latent grid coordinates
    - output_queries: (batch, n_output_points, dim) - output point coordinates
    - x: (batch, n_input_points, in_channels) - input features
    - y: (batch, n_output_points, out_channels) - output 
    
    Note: GNO expects coordinates WITHOUT batch dimension for neighbor search
    """
    x = batch['x']
    y = batch['y']
    batch_size = x.shape[0]
    
    # Darcy x shape: (batch, 1, h, w)
    # We need to convert grid to point cloud format
    if x.dim() == 4:  # (batch, channel, h, w)
        h, w = x.shape[2], x.shape[3]
    else:
        h, w = x.shape[1], x.shape[2]
    
    # Create grid coordinates
    if args.dim == 2:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device),
            torch.linspace(0, 1, w, device=device),
            indexing='ij'
        )
        # GNO expects (n_points, dim) - no batch dimension for coordinates
        input_geom = torch.stack([grid_x, grid_y], dim=-1)
        input_geom = input_geom.reshape(-1, 2)
        
        # Output queries are same as input for Darcy Flow
        output_queries = input_geom.clone()
        
        # Latent grid queries - GINO expects (latent_h, latent_w, dim)
        latent_h, latent_w = 2 * args.n_modes, 2 * args.n_modes
        ly, lx = torch.meshgrid(
            torch.linspace(0, 1, latent_h, device=device),
            torch.linspace(0, 1, latent_w, device=device),
            indexing='ij'
        )
        latent_queries = torch.stack([lx, ly], dim=-1)
    
    # Flatten input features: (batch, 1, h, w) → (batch, h*w, 1)
    x = x.permute(0, 2, 3, 1).reshape(batch_size, h*w, 1)
    # Flatten output
    y = y.permute(0, 2, 3, 1).reshape(batch_size, h*w, 1)
    
    x = x.to(device)
    y = y.to(device)
    
    return input_geom, latent_queries, output_queries, x, y


def count_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        input_geom, latent_queries, output_queries, x, y = convert_darcy_to_gino_format(batch, device)
        
        optimizer.zero_grad()
        output = model(input_geom, latent_queries, output_queries, x=x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / n_batches


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_geom, latent_queries, output_queries, x, y = convert_darcy_to_gino_format(batch, device)
            output = model(input_geom, latent_queries, output_queries, x=x)
            loss = criterion(output, y)
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def benchmark_inference(model, sample_batch, device, n_runs):
    """Benchmark inference latency"""
    input_geom, latent_queries, output_queries, x, _ = convert_darcy_to_gino_format(sample_batch, device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_geom, latent_queries, output_queries, x=x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_geom, latent_queries, output_queries, x=x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = (time.time() - start_time) / n_runs
    
    return elapsed


def main(args):
    print("=" * 80)
    print("Testing Baseline GINO vs MHF-GINO on real Darcy Flow dataset")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Dimension: {args.dim}D")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  FNO layers: {args.n_layers}")
    print(f"  Fourier modes: {args.n_modes} per dimension")
    print(f"  MHF rank: {args.mhf_rank}")
    print(f"  Factorization: {args.factorization}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  Data root: {args.data_root}")
    print("=" * 80)
    
    # Load real Darcy Flow dataset
    print("\n[1] Loading Darcy Flow dataset...")
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000,
        n_tests=[100],
        batch_size=args.batch_size,
        test_batch_sizes=[args.batch_size],
        data_root=args.data_root,
        test_resolutions=[16],
        encode_input=False,
        encode_output=True,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loaders[16])}")
    
    # Get one sample for inference benchmark
    sample_batch = next(iter(train_loader))
    
    # Prepare n_modes based on dimension
    if args.dim == 2:
        fno_n_modes = (args.n_modes, args.n_modes)
    else:
        fno_n_modes = (args.n_modes, args.n_modes, args.n_modes)
    
    # Create baseline GINO
    print("\n[2] Creating Baseline GINO...")
    baseline_model = BaselineGINO(
        in_channels=1,
        out_channels=1,
        fno_hidden_channels=args.hidden_channels,
        fno_n_layers=args.n_layers,
        fno_n_modes=fno_n_modes,
        gno_coord_dim=args.dim,
        in_gno_radius=args.in_radius,
        out_gno_radius=args.out_radius,
    ).to(args.device)
    
    baseline_params = count_parameters(baseline_model)
    print(f"  Baseline GINO parameters: {baseline_params:,}")
    
    # Create MHF-GINO
    print("\n[3] Creating MHF-GINO...")
    mhf_model = MHF_GINO(
        in_channels=1,
        out_channels=1,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        n_modes=fno_n_modes,
        in_radius=args.in_radius,
        out_radius=args.out_radius,
        gno_coord_dim=args.dim,
        mhf_rank=args.mhf_rank,
        mhf_factorization=args.factorization,
    ).to(args.device)
    
    # Training setup
    criterion = nn.MSELoss()
    
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.learning_rate)
    mhf_optimizer = torch.optim.Adam(mhf_model.parameters(), lr=args.learning_rate)
    
    # Train baseline
    print("\n[4] Training Baseline GINO...")
    baseline_train_losses = []
    baseline_start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(baseline_model, train_loader, baseline_optimizer, criterion, args.device)
        baseline_train_losses.append(train_loss)
        print(f"  Epoch {epoch+1}/{args.epochs}: loss = {train_loss:.6f}")
    
    baseline_total_time = time.time() - baseline_start_time
    baseline_avg_time_per_epoch = baseline_total_time / args.epochs
    
    # Evaluate baseline
    baseline_test_loss = evaluate(baseline_model, test_loaders[16], criterion, args.device)
    print(f"\n  Baseline training complete:")
    print(f"    Total training time: {baseline_total_time:.2f}s")
    print(f"    Avg time per epoch: {baseline_avg_time_per_epoch:.2f}s")
    print(f"    Final train loss: {baseline_train_losses[-1]:.6f}")
    print(f"    Test loss: {baseline_test_loss:.6f}")
    
    # Train MHF-GINO (full weights before decomposition)
    print("\n[5] Training MHF-GINO (full weights)...")
    mhf_train_losses = []
    mhf_start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(mhf_model, train_loader, mhf_optimizer, criterion, args.device)
        mhf_train_losses.append(train_loss)
        print(f"  Epoch {epoch+1}/{args.epochs}: loss = {train_loss:.6f}")
    
    mhf_total_time = time.time() - mhf_start_time
    mhf_avg_time_per_epoch = mhf_total_time / args.epochs
    
    # After training, perform MHF decomposition
    print("\n  Performing MHF decomposition after training...")
    mhf_model.decompose()
    mhf_params_after_decompose = count_parameters(mhf_model)
    print(f"  MHF-GINO parameters after decomposition: {mhf_params_after_decompose:,}")
    
    compression_ratio = mhf_params_after_decompose / baseline_params
    compression_factor = baseline_params / mhf_params_after_decompose
    print(f"  Compression ratio: {compression_ratio:.2%}")
    print(f"  Compression factor: {compression_factor:.2f}x")
    
    # Fine-tune after decomposition (to recover accuracy)
    print("\n  Fine-tuning after decomposition (5 epochs with reduced lr)...")
    mhf_optimizer_finetune = torch.optim.Adam(mhf_model.parameters(), lr=args.learning_rate * 0.1)
    mhf_train_losses_finetune = []
    mhf_start_time_finetune = time.time()
    for epoch in range(5):
        train_loss = train_one_epoch(mhf_model, train_loader, mhf_optimizer_finetune, criterion, args.device)
        mhf_train_losses_finetune.append(train_loss)
        print(f"    Epoch {epoch+1}/5: loss = {train_loss:.6f}")
    
    mhf_total_time_finetune = time.time() - mhf_start_time_finetune
    
    # Evaluate MHF after decomposition and fine-tuning
    mhf_test_loss = evaluate(mhf_model, test_loaders[16], criterion, args.device)
    print(f"\n  MHF-GINO after decomposition + fine-tuning:")
    print(f"    Total training time (full+fine): {mhf_total_time + mhf_total_time_finetune:.2f}s")
    print(f"    Avg time per epoch (full): {mhf_avg_time_per_epoch:.2f}s")
    print(f"    Final train loss (after fine-tune): {mhf_train_losses_finetune[-1]:.6f}")
    print(f"    Test loss (after fine-tune): {mhf_test_loss:.6f}")
    
    # Benchmark inference
    print("\n[6] Benchmarking inference latency...")
    baseline_inference_time = benchmark_inference(baseline_model, sample_batch, args.device, args.benchmark_runs)
    mhf_inference_time = benchmark_inference(mhf_model, sample_batch, args.device, args.benchmark_runs)
    
    inference_speedup = baseline_inference_time / mhf_inference_time
    
    print(f"  Baseline average inference: {baseline_inference_time * 1000:.2f} ms")
    print(f"  MHF-GINO average inference: {mhf_inference_time * 1000:.2f} ms")
    print(f"  Inference speedup: {inference_speedup:.2f}x")
    
    # Calculate training speedup (including fine-tuning)
    total_mhf_time = mhf_total_time + mhf_total_time_finetune
    training_speedup = baseline_total_time / total_mhf_time
    print(f"  Training speedup: {training_speedup:.2f}x")
    
    # Calculate accuracy retention
    accuracy_retention = baseline_test_loss / mhf_test_loss if mhf_test_loss > baseline_test_loss else mhf_test_loss / baseline_test_loss
    print(f"  Accuracy retention: {accuracy_retention:.2%}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'Baseline GINO':<20} {'MHF-GINO':<20} {'Ratio (MHF/Baseline)':<20}")
    print("-" * 80)
    print(f"{'Parameters':<25} {baseline_params:<20,} {mhf_params_after_decompose:<20,} {compression_ratio:<20.2%}")
    print(f"{'Compression Factor':<25} {'-':<20} {'-':<20} {compression_factor:<20.2f}x")
    print(f"{'Total Train Time (s)':<25} {baseline_total_time:<20.2f} {total_mhf_time:<20.2f} {total_mhf_time/baseline_total_time:<20.2f}")
    print(f"{'Avg Epoch Time (s)':<25} {baseline_avg_time_per_epoch:<20.2f} {mhf_avg_time_per_epoch:<20.2f} {mhf_avg_time_per_epoch/baseline_avg_time_per_epoch:<20.2f}")
    print(f"{'Inference Latency (ms)':<25} {baseline_inference_time*1000:<20.2f} {mhf_inference_time*1000:<20.2f} {mhf_inference_time/baseline_inference_time:<20.2f}")
    print(f"{'Test Loss':<25} {baseline_test_loss:<20.6f} {mhf_test_loss:<20.6f} {accuracy_retention*100:<20.2f}%")
    print("=" * 80)
    
    print(f"\nTraining speedup: {training_speedup:.2f}x")
    print(f"Inference speedup: {inference_speedup:.2f}x")
    
    # Save results
    results = {
        'baseline_params': baseline_params,
        'mhf_params_after_decompose': mhf_params_after_decompose,
        'compression_ratio': compression_ratio,
        'compression_factor': compression_factor,
        'baseline_total_time': baseline_total_time,
        'mhf_total_time': mhf_total_time,
        'baseline_avg_epoch_time': baseline_avg_time_per_epoch,
        'mhf_avg_epoch_time': mhf_avg_time_per_epoch,
        'baseline_inference_time': baseline_inference_time,
        'mhf_inference_time': mhf_inference_time,
        'baseline_test_loss': baseline_test_loss,
        'mhf_test_loss': mhf_test_loss,
        'training_speedup': training_speedup,
        'inference_speedup': inference_speedup,
        'accuracy_retention': accuracy_retention,
        'config': vars(args),
    }
    
    import json
    with open('gino_mhf_real_data_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('gino_mhf_real_data_report.txt', 'w') as f:
        f.write("GINO MHF Real Dataset Test Report (Darcy Flow 16x16)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Dimension: {args.dim}D\n")
        f.write(f"  Hidden channels: {args.hidden_channels}\n")
        f.write(f"  FNO layers: {args.n_layers}\n")
        f.write(f"  Fourier modes: {args.n_modes}\n")
        f.write(f"  MHF rank: {args.mhf_rank}\n")
        f.write(f"  Factorization: {args.factorization}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs} (full) + 5 (fine-tune after decompose)\n")
        f.write(f"  Device: {args.device}\n")
        f.write("\n")
        f.write("Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Baseline GINO:\n")
        f.write(f"  Parameters: {baseline_params:,}\n")
        f.write(f"  Total train time: {baseline_total_time:.2f}s\n")
        f.write(f"  Avg epoch time: {baseline_avg_time_per_epoch:.2f}s\n")
        f.write(f"  Inference latency: {baseline_inference_time * 1000:.2f}ms\n")
        f.write(f"  Test loss: {baseline_test_loss:.6f}\n")
        f.write("\n")
        f.write(f"MHF-GINO (after decomposition + fine-tuning):\n")
        f.write(f"  Parameters: {mhf_params_after_decompose:,}\n")
        f.write(f"  Total train time: {total_mhf_time:.2f}s ({mhf_total_time:.2f} + {mhf_total_time_finetune:.2f} fine)\n")
        f.write(f"  Avg epoch time: {mhf_avg_time_per_epoch:.2f}s\n")
        f.write(f"  Inference latency: {mhf_inference_time * 1000:.2f}ms\n")
        f.write(f"  Test loss: {mhf_test_loss:.6f}\n")
        f.write("\n")
        f.write(f"Summary:\n")
        f.write(f"  Compression ratio: {compression_ratio:.2%}\n")
        f.write(f"  Compression factor: {compression_factor:.2f}x\n")
        f.write(f"  Training speedup: {training_speedup:.2f}x\n")
        f.write(f"  Inference speedup: {inference_speedup:.2f}x\n")
        f.write(f"  Accuracy retention: {accuracy_retention:.2%}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nResults saved to:")
    print(f"  - gino_mhf_real_data_results.json")
    print(f"  - gino_mhf_real_data_report.txt")
    print("\nDone!")


if __name__ == '__main__':
    args = parse_args()
    # Fix argparse typo above
    if not hasattr(args, 'out_radius'):
        args.out_radius = args.out__radius
    main(args)
