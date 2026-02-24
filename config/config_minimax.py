"""
Configuration - Minimax Version
==============================
Central configuration for all training parameters.
Sanitized version for public release.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Set ROCm environment variables FIRST
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"


@dataclass
class PathConfig:
    """Paths configuration."""
    base_dir: Path = Path("./project")
    model_dir: Path = field(default_factory=lambda: Path("./project/model"))
    dataset_path: Path = field(default_factory=lambda: Path("./project/data/dataset.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("./project/output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./project/checkpoints"))
    logs_dir: Path = field(default_factory=lambda: Path("./project/logs"))
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration."""
    dtype: str = "bfloat16"
    max_length: int = 640
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True


@dataclass  
class DataConfig:
    """Dataset configuration."""
    train_split: float = 0.95
    val_split: float = 0.05
    shuffle: bool = True
    seed: int = 42
    num_workers: int = 8
    
    # ChatML special tokens (Qwen3)
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Batch settings
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = field(init=False)
    
    # Learning rate
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    
    # Regularization
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Training duration
    num_epochs: int = 3
    
    # Checkpointing
    save_steps: int = 250
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Optimizer
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Misc
    seed: int = 42
    dataloader_pin_memory: bool = False
    
    def __post_init__(self):
        self.effective_batch_size = self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  Model: {self.paths.model_dir}\n"
            f"  Dataset: {self.paths.dataset_path} ({self.data.train_split*100:.0f}/{self.data.val_split*100:.0f} split)\n"
            f"  Batch: {self.training.per_device_batch_size} x {self.training.gradient_accumulation_steps} = {self.training.effective_batch_size}\n"
            f"  LR: {self.training.learning_rate}, Epochs: {self.training.num_epochs}\n"
            f"  Max Length: {self.model.max_length}, Dtype: {self.model.dtype}\n"
            f")"
        )


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


if __name__ == "__main__":
    cfg = get_config()
    print(cfg)
    print(f"\nOutput directory: {cfg.paths.output_dir}")
    print(f"Checkpoints: {cfg.paths.checkpoint_dir}")
