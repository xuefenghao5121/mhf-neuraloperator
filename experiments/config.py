"""
Experiment configurations for MHF-NeuralOperator
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment
    
    An experiment compares three variants:
    1. Baseline: Original neural operator
    2. MHF: Multi-resolution hierarchical factorization
    3. MHF+Attention: MHF with cross-head attention
    """
    
    # Model configuration
    model_name: str  # e.g., "FNO", "UNO", "GINO"
    dataset_name: str  # e.g., "darcy", "navier_stokes", "burgers"
    
    # Dataset configuration
    train_resolution: int = 16
    test_resolutions: List[int] = field(default_factory=lambda: [16, 32])
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 32
    
    # Model architecture
    n_modes: Tuple[int, ...] = (16, 16)
    in_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 64
    n_layers: int = 4
    
    # MHF parameters
    mhf_rank: int = 8
    mhf_resolutions: List[int] = field(default_factory=lambda: [4, 8, 16])
    mhf_factorization: str = "tucker"
    
    # Training parameters
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Evaluation
    eval_interval: int = 10
    
    # Output
    output_dir: str = "results"
    experiment_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "train_resolution": self.train_resolution,
            "test_resolutions": self.test_resolutions,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "batch_size": self.batch_size,
            "n_modes": self.n_modes,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "mhf_rank": self.mhf_rank,
            "mhf_resolutions": self.mhf_resolutions,
            "mhf_factorization": self.mhf_factorization,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eval_interval": self.eval_interval,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
        }


# Standard experiment configurations for different operators

EXPERIMENT_CONFIGS = {
    # FNO experiments
    "fno_darcy": ExperimentConfig(
        model_name="FNO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="fno_darcy",
    ),
    
    # UNO experiments
    "uno_darcy": ExperimentConfig(
        model_name="UNO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="uno_darcy",
    ),
    
    # GINO experiments
    "gino_darcy": ExperimentConfig(
        model_name="GINO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="gino_darcy",
    ),
    
    # SFNO experiments (spherical data)
    "sfno_airfoil": ExperimentConfig(
        model_name="SFNO",
        dataset_name="airfoil",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="sfno_airfoil",
    ),
    
    # CODANO experiments
    "codano_darcy": ExperimentConfig(
        model_name="CODANO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="codano_darcy",
    ),
    
    # RNO experiments
    "rno_darcy": ExperimentConfig(
        model_name="RNO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="rno_darcy",
    ),
    
    # LocalNO experiments
    "localno_darcy": ExperimentConfig(
        model_name="LocalNO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="localno_darcy",
    ),
    
    # OTNO experiments
    "otno_darcy": ExperimentConfig(
        model_name="OTNO",
        dataset_name="darcy",
        train_resolution=16,
        test_resolutions=[16, 32],
        n_train=1000,
        n_test=200,
        batch_size=32,
        n_modes=(16, 16),
        hidden_channels=64,
        n_layers=4,
        mhf_rank=8,
        mhf_resolutions=[4, 8, 16],
        experiment_name="otno_darcy",
    ),
}
