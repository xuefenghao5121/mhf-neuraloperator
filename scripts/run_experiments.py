#!/usr/bin/env python3
"""
Run experiments for MHF-NeuralOperator

Usage:
    python scripts/run_experiments.py --model FNO --dataset darcy
    python scripts/run_experiments.py --all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.config import EXPERIMENT_CONFIGS, ExperimentConfig
from experiments.trainer import ExperimentTrainer


def run_single_experiment(experiment_name: str):
    """Run a single experiment by name"""
    if experiment_name not in EXPERIMENT_CONFIGS:
        print(f"Error: Unknown experiment '{experiment_name}'")
        print(f"Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")
        return
    
    config = EXPERIMENT_CONFIGS[experiment_name]
    trainer = ExperimentTrainer(config)
    results = trainer.run_experiment()
    
    print(f"\n✅ Experiment completed: {experiment_name}")
    print(f"Results saved to: {trainer.output_dir}")


def run_all_experiments():
    """Run all configured experiments"""
    print(f"\n{'='*80}")
    print(f"Running ALL experiments")
    print(f"{'='*80}\n")
    
    total = len(EXPERIMENT_CONFIGS)
    completed = 0
    failed = []
    
    for i, (name, config) in enumerate(EXPERIMENT_CONFIGS.items(), 1):
        print(f"\n[{i}/{total}] Running: {name}")
        try:
            run_single_experiment(name)
            completed += 1
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed.append((name, str(e)))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{total}")
    
    if failed:
        print(f"\nFailed experiments:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print(f"\n✅ All experiments completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run MHF-NeuralOperator experiments")
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Name of specific experiment to run"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name (e.g., FNO, UNO, GINO)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset name (e.g., darcy, burgers)"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all configured experiments"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiments"
    )
    
    # Custom experiment parameters
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=200, help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mhf-rank", type=int, default=8, help="MHF rank")
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("Available experiments:")
        for name in sorted(EXPERIMENT_CONFIGS.keys()):
            config = EXPERIMENT_CONFIGS[name]
            print(f"  - {name}: {config.model_name} on {config.dataset_name}")
        return
    
    # Run all experiments
    if args.all:
        run_all_experiments()
        return
    
    # Run specific experiment by name
    if args.experiment:
        run_single_experiment(args.experiment)
        return
    
    # Run custom experiment with specified model and dataset
    if args.model and args.dataset:
        experiment_name = f"{args.model.lower()}_{args.dataset}"
        
        # Create custom config
        config = ExperimentConfig(
            model_name=args.model.upper(),
            dataset_name=args.dataset,
            n_train=args.n_train,
            n_test=args.n_test,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            learning_rate=args.lr,
            mhf_rank=args.mhf_rank,
            experiment_name=experiment_name,
        )
        
        trainer = ExperimentTrainer(config)
        results = trainer.run_experiment()
        
        print(f"\n✅ Custom experiment completed")
        return
    
    # No arguments provided
    parser.print_help()


if __name__ == "__main__":
    main()
