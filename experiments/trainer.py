"""
Experiment trainer for MHF-NeuralOperator

Trains and evaluates three variants:
1. Baseline: Original neural operator
2. MHF: Multi-resolution hierarchical factorization
3. MHF+Attention: MHF with cross-head attention
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from neuralop.training import trainer
from neuralop.losses import LpLoss

from .config import ExperimentConfig


class ExperimentTrainer:
    """Trainer for comparing baseline vs MHF vs MHF+Attention"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"Experiment: {config.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
    
    def load_dataset(self) -> Tuple[DataLoader, List[DataLoader]]:
        """Load training and test datasets"""
        from neuralop.data.datasets import DarcyDataset
        
        # For now, we focus on Darcy dataset
        # Other datasets can be added later
        if self.config.dataset_name == "darcy":
            train_loader, test_loaders, _ = DarcyDataset(
                root_dir="/home/huawei/.openclaw/workspace/neuraloperator/neuralop/data/datasets/data",
                n_train=self.config.n_train,
                n_tests=[self.config.n_test] * len(self.config.test_resolutions),
                batch_size=self.config.batch_size,
                test_batch_sizes=[self.config.batch_size] * len(self.config.test_resolutions),
                train_resolution=self.config.train_resolution,
                test_resolutions=self.config.test_resolutions,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
        
        return train_loader, test_loaders
    
    def create_model(self, variant: str) -> nn.Module:
        """Create model for specified variant
        
        Args:
            variant: "baseline", "mhf", or "mhf_attention"
        """
        # Import the appropriate model based on config
        if self.config.model_name == "FNO":
            if variant == "baseline":
                from neuralop.models import FNO
                model = FNO(
                    n_modes=self.config.n_modes,
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    hidden_channels=self.config.hidden_channels,
                    n_layers=self.config.n_layers,
                )
            else:  # MHF or MHF+Attention
                from models.fno_mhf import MHFNO
                model = MHFNO(
                    n_modes=self.config.n_modes,
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    hidden_channels=self.config.hidden_channels,
                    n_layers=self.config.n_layers,
                    mhf_rank=self.config.mhf_rank,
                    mhf_resolutions=self.config.mhf_resolutions,
                    mhf_factorization=self.config.mhf_factorization,
                )
                if variant == "mhf":
                    model.decompose()
                elif variant == "mhf_attention":
                    # TODO: Add cross-head attention
                    model.decompose()
        
        elif self.config.model_name == "UNO":
            if variant == "baseline":
                from neuralop.models import UNO
                model = UNO(
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    hidden_channels=self.config.hidden_channels,
                    n_layers=self.config.n_layers,
                    # UNO has specific parameters
                    uno_out_channels=[self.config.hidden_channels] * self.config.n_layers,
                    uno_n_modes=self.config.n_modes[0],
                )
            else:
                from models.uno_mhf import MHFUNO
                model = MHFUNO(
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    hidden_channels=self.config.hidden_channels,
                    n_layers=self.config.n_layers,
                    uno_out_channels=[self.config.hidden_channels] * self.config.n_layers,
                    uno_n_modes=self.config.n_modes[0],
                    mhf_rank=self.config.mhf_rank,
                    mhf_resolutions=self.config.mhf_resolutions,
                    mhf_factorization=self.config.mhf_factorization,
                )
                if variant in ["mhf", "mhf_attention"]:
                    model.decompose()
        
        elif self.config.model_name == "GINO":
            if variant == "baseline":
                from neuralop.models import GINO
                model = GINO(
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    hidden_channels=self.config.hidden_channels,
                    n_layers=self.config.n_layers,
                )
            else:
                from models.gino_mhf import MHFGINO
                model = MHFGINO(
                    in_channels=self.config.in_channels,
                    out_channels=self.config.out_channels,
                    hidden_channels=self.config.hidden_channels,
                    n_layers=self.config.n_layers,
                    mhf_rank=self.config.mhf_rank,
                    mhf_resolutions=self.config.mhf_resolutions,
                    mhf_factorization=self.config.mhf_factorization,
                )
                if variant in ["mhf", "mhf_attention"]:
                    model.decompose()
        
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        model = model.to(self.device)
        return model
    
    def count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loaders: List[DataLoader],
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate model on all test sets"""
        model.eval()
        results = {}
        
        for i, test_loader in enumerate(test_loaders):
            res = self.config.test_resolutions[i]
            total_loss = 0.0
            n_batches = 0
            
            for batch in test_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            results[f"test_loss_res{res}"] = avg_loss
        
        return results
    
    @torch.no_grad()
    def measure_inference_time(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        n_runs: int = 100,
    ) -> float:
        """Measure average inference time in milliseconds"""
        model.eval()
        x = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Measure
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_runs):
            _ = model(x)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        avg_time_ms = (elapsed / n_runs) * 1000
        
        return avg_time_ms
    
    def run_variant(self, variant: str, train_loader: DataLoader, test_loaders: List[DataLoader]) -> Dict:
        """Run training and evaluation for one variant
        
        Returns:
            Dictionary with all metrics for this variant
        """
        print(f"\n{'='*60}")
        print(f"Running variant: {variant}")
        print(f"{'='*60}")
        
        # Create model
        model = self.create_model(variant)
        n_params = self.count_parameters(model)
        print(f"Parameters: {n_params:,}")
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        criterion = LpLoss(d=2, p=2)
        
        # Training history
        history = {
            "train_loss": [],
            "test_metrics": [],
        }
        
        best_test_loss = float("inf")
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            
            history["train_loss"].append(train_loss)
            
            # Evaluate periodically
            if (epoch + 1) % self.config.eval_interval == 0 or epoch == self.config.max_epochs - 1:
                test_results = self.evaluate(model, test_loaders, criterion)
                history["test_metrics"].append({
                    "epoch": epoch + 1,
                    **test_results,
                })
                
                # Track best test loss
                current_test_loss = test_results[f"test_loss_res{self.config.test_resolutions[0]}"]
                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    # Save best model
                    torch.save(model.state_dict(), self.output_dir / f"{variant}_best.pt")
                
                print(f"Epoch {epoch+1}/{self.config.max_epochs} - Train Loss: {train_loss:.6f} - "
                      f"Test Loss (res{self.config.test_resolutions[0]}): {current_test_loss:.6f}")
        
        # Measure inference time
        sample_batch = next(iter(test_loaders[0]))
        input_shape = sample_batch[0].shape
        inference_time = self.measure_inference_time(model, input_shape)
        print(f"Inference time: {inference_time:.2f} ms")
        
        # Compile results
        results = {
            "variant": variant,
            "n_params": n_params,
            "inference_time_ms": inference_time,
            "best_test_loss": best_test_loss,
            "history": history,
        }
        
        # Add compression stats for MHF variants
        if variant in ["mhf", "mhf_attention"] and hasattr(model, "get_compression_stats"):
            stats = model.get_compression_stats()
            results["compression_stats"] = stats
            print(f"Compression factor: {stats['overall_compression_factor']:.2f}x")
        
        # Save results
        with open(self.output_dir / f"{variant}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_experiment(self) -> Dict:
        """Run complete experiment comparing all variants
        
        Returns:
            Dictionary with results for all variants
        """
        print(f"\n{'='*80}")
        print(f"Starting experiment: {self.config.experiment_name}")
        print(f"{'='*80}\n")
        
        # Load dataset
        train_loader, test_loaders = self.load_dataset()
        print(f"Train batches: {len(train_loader)}")
        for i, test_loader in enumerate(test_loaders):
            print(f"Test batches (res {self.config.test_resolutions[i]}): {len(test_loader)}")
        
        # Run all variants
        all_results = {}
        
        # 1. Baseline
        all_results["baseline"] = self.run_variant("baseline", train_loader, test_loaders)
        
        # 2. MHF
        all_results["mhf"] = self.run_variant("mhf", train_loader, test_loaders)
        
        # 3. MHF+Attention
        all_results["mhf_attention"] = self.run_variant("mhf_attention", train_loader, test_loaders)
        
        # Save combined results
        with open(self.output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Experiment complete: {self.config.experiment_name}")
        print(f"{'='*80}\n")
        
        print(f"{'Variant':<20} {'Params':>12} {'Inference(ms)':>15} {'Best Test Loss':>15}")
        print("-" * 70)
        for variant, results in all_results.items():
            print(f"{variant:<20} {results['n_params']:>12,} {results['inference_time_ms']:>15.2f} "
                  f"{results['best_test_loss']:>15.6f}")
        
        return all_results
