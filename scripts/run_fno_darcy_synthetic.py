#!/usr/bin/env python3
"""
Run FNO on synthetic data to verify framework (no real dataset loading)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, "/home/huawei/.openclaw/workspace/neuraloperator")

from models.fno_mhf import MHFNO
 from neuralop.models import FNO
 from neuralop.losses import LpLoss

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print(f"Using device: {device}")
 print(f"\n{'='*60}")
 print("Creating synthetic data...")
 
 x_train = torch.randn(10, 1, 16, 16).to(device)
 y_train = torch.randn(10, 1, 16, 16).to(device)
 x_test = torch.randn(5, 1, 16, 16).to(device)
 y_test = torch.randn(5, 1, 16, 16).to(device)
 print(f"Train data: {x_train.shape}")
 print(f"Test data: {x_test.shape}")
 
 # Create models
 print(f"\n{'='*60}")
 print("Creating models...")
 baseline = FNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=2,
    ).to(device)
 baseline_params = sum(p.numel() for p in baseline.parameters())
 print(f"Baseline params: {baseline_params:,}")
    
 mhf = MHFNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=2,
        mhf_rank=4,
        mhf_resolutions=[5, 8, 16],
    ).to(device)
 mhf.decompose()
 mhf_params = sum(p.numel() for p in mhf.parameters())
 print(f"MHF params (after decompose): {mhf_params:,}")
 stats = mhf.get_compression_stats()
 print(f"Compression factor: {stats['overall_compression_factor']:.2f}x")
 print(f"Compression ratio: {stats['overall_compression_ratio']:.3f}")
    
 # Training functions
 def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
 def evaluate(model, loader, criterion):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(loader)
    
 # Train baseline
 print(f"\n{'='*60}")
 print("Training baseline...")
 optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
 criterion = LpLoss(d=2, p=2)
 
 for epoch in range(10):
        train_loss = train_epoch(baseline, [(x_train, y_train)], optimizer, criterion)
        print(f"Epoch {epoch+1}: {train_loss:.6f}")
    
 test_loss = evaluate(baseline, [(x_test, y_test)], criterion)
        print(f"Test loss: {test_loss:.6f}")
    
 print(f"✅ Baseline test completed")
    
 # Train MHf
 print(f"\n{'='*60}")
 print("Training MHF...")
 optimizer = torch.optim.Adam(mhf.parameters(), lr=1e-3)
 criterion = LpLoss(d=2, p=2)
    
 for epoch in range(10):
        train_loss = train_epoch(mhf, [(x_train, y_train)], optimizer, criterion)
        print(f"Epoch {epoch+1}: {train_loss:.6f}")
    
 test_loss = evaluate(mhf, [(x_test, y_test)], criterion)
        print(f"Test loss: {test_loss:.6f}")
    
 print(f"✅ MHF test completed")
    
 # Compare results
 print(f"\n{'='*60}")
 print("Results Summary")
 print(f"{'='*60}")
 print(f"{'Variant':<15} {'Params':<12} {'Test Loss':<15}")
 print(f"{'='*60}")
 print(f"{'Baseline':<15} {baseline_params:>12} {baseline_test_loss:.6f}")
 print(f"{'MHF':<15} {mhf_params:>12} {mhf_test_loss:.6f}")
 print(f"\nCompression factor: {stats['overall_compression_factor']:.2f}x")
 print(f"Compression ratio: {stats['overall_compression_ratio']:.3f}")
 print(f"Parameter reduction: {(1 - stats['overall_compression_ratio']) * 100:.2f}%")
    
 print(f"\n✅ Test completed successfully!")
