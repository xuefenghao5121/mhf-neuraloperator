#!/usr/bin/env python3
"""
Quick test: FNO on Darcy with small dataset
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, "/home/huawei/.openclaw/workspace/neuraloperator")
torch
from torch.utils.data import DataLoader

import torch.nn as nn

from neuralop.data.datasets import DarcyDataset
from neuralop.models import FNO
from models.fno_mhf import MHFNO
from neuralop.losses import LpLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load small subset of dataset
print("\nLoading dataset...")
try:
    # Use a small subset for quick testing
    train_loader, DataLoader(
        list(range(10)),
        batch_size=4,
        shuffle=False
    )
    
    test_loader = DataLoader(
        list(range(5)),
        batch_size=4,
        shuffle=False,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Create models
print("\nCreating models...")
try:
    # Baseline FNO
    baseline = FNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=2,
    ).to(device)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline parameters: {baseline_params:,}")
    
    # MHF-FNO
    mhf = MHFNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=2,
        mhf_rank=4,
        mhf_resolutions=[4, 8, 16],
    ).to(device)
    
 mhf.decompose()
    mhf_params = sum(p.numel() for p in mhf.parameters())
    print(f"MHF parameters (after decompose): {mhf_params:,}")
    
 compression_stats = mhf.get_compression_stats()
    print(f"Compression factor: {compression_stats['overall_compression_factor']:.2f}x")
    
except Exception as e:
    print(f"Error creating models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Quick training
print("\nTraining baseline...")
optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
criterion = LpLoss(d=2, p=2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    train_loss = train_epoch(baseline, train_loader, optimizer, criterion)
    scheduler.step()
    
    print(f"Epoch {epoch+1}: {train_loss:.6f}")
    
    # Evaluate
    test_loss = evaluate(baseline, test_loader, criterion)
    print(f"Test loss: {test_loss:.6f}")
    
    # Save baseline
    torch.save(baseline.state_dict(), project_root / "results" / "baseline_best.pt")
    print(f"✅ Baseline saved")

 return

    
    # Train MHF
    print("\nTraining MHF...")
    mhf.train()
    optimizer = torch.optim.Adam(mhf.parameters(), lr=1e-3)
            criterion = LpLoss(d=2, p=2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            
 for epoch in range(10):
                train_loss = train_epoch(mhf, train_loader, optimizer, criterion)
                scheduler.step()
                
                print(f"Epoch {epoch+1}: {train_loss:.6f}")
            
 # Evaluate
    test_loss = evaluate(mhf, test_loader, criterion)
            print(f"MHF test loss: {test_loss:.6f}")
    
    # Save MHF
    torch.save(mhf.state_dict(), project_root / "results" / "mhf_best.pt")
    print(f"✅ MHF saved")
    
    # Compare results
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"{'Variant':<15} {'Params':<12} {'Test Loss':<15}")
    print("-" * 60)
    print(f"{'Baseline':<15} {baseline_params:>12} {baseline_test_loss:>12.6f}")
            print(f"{'MHF':<15} {mhf_params:>12} {mhf_test_loss:>12.6f}")
            compression = {compression_stats['overall_compression_factor']:.2f}x")
            print(f"\nCompression ratio: {compression_stats['overall_compression_ratio']:.3f}")
            print(f"Parameter reduction: {(1 - compression_stats['overall_compression_ratio']) * 100:.2f}%")
    
 print(f"\n✅ Quick test completed successfully!")
    print("="*60)

    
    sys.exit(0)

if __name__ == "__main__":
    main()
