"""
LoRA Training Configuration - Minimax Version
=============================================
Extends the base configuration with LoRA-specific parameters.
Sanitized version for public release.
"""

from dataclasses import dataclass, field
from typing import List
import os
import sys
from pathlib import Path

# Add parent directory to path to import base config
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_minimax import Config, get_config as get_base_config

@dataclass
class LoraConfig:
    r: int = 64  # Rank
    lora_alpha: int = 128 # Alpha (usually 2x Rank)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
@dataclass
class ConfigLoRA(Config):
    lora: LoraConfig = field(default_factory=LoraConfig)

def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    try:
        return int(str(v).strip())
    except Exception as e:
        raise ValueError(f"Invalid integer for {name}: {v!r}") from e

def _env_bool(name: str) -> bool | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean for {name}: {v!r} (expected 1/0/true/false)")

def get_lora_config() -> ConfigLoRA:
    base = get_base_config()
    
    # Override defaults if needed for LoRA specifics
    # For LoRA, we might want a higher learning rate than full fine-tuning
    base.training.learning_rate = 2e-4 
    base.training.per_device_batch_size = 4 # LoRA uses less VRAM, potentially increase batch
    base.training.gradient_accumulation_steps = 4

    # Optional env overrides (lets you tune VRAM utilization without editing files).
    bs = _env_int("SMART_LORA_BATCH_SIZE") or _env_int("SMART_BATCH_SIZE")
    ga = _env_int("SMART_LORA_GRAD_ACCUM") or _env_int("SMART_GRAD_ACCUM")
    max_len = _env_int("SMART_LORA_MAX_LENGTH") or _env_int("SMART_MAX_LENGTH")
    # Common toggle used across scripts.
    grad_ckpt = _env_bool("SMART_GRAD_CHECKPOINT")
    if grad_ckpt is None:
        grad_ckpt = _env_bool("SMART_LORA_GRAD_CHECKPOINTING")
    if grad_ckpt is None:
        grad_ckpt = _env_bool("SMART_GRAD_CHECKPOINTING")

    if bs is not None:
        base.training.per_device_batch_size = bs
    if ga is not None:
        base.training.gradient_accumulation_steps = ga
    if max_len is not None:
        base.model.max_length = max_len
    if grad_ckpt is not None:
        base.model.gradient_checkpointing = bool(grad_ckpt)

    save_steps = _env_int("SMART_LORA_SAVE_STEPS") or _env_int("SMART_SAVE_STEPS")
    eval_steps = _env_int("SMART_LORA_EVAL_STEPS") or _env_int("SMART_EVAL_STEPS")
    logging_steps = _env_int("SMART_LORA_LOGGING_STEPS") or _env_int("SMART_LOGGING_STEPS")
    if save_steps is not None:
        base.training.save_steps = save_steps
    if eval_steps is not None:
        base.training.eval_steps = eval_steps
    if logging_steps is not None:
        base.training.logging_steps = logging_steps

    # TrainingConfig.__post_init__ already ran in get_base_config(); keep derived fields consistent.
    base.training.effective_batch_size = (
        base.training.per_device_batch_size * base.training.gradient_accumulation_steps
    )
    
    # Optional path overrides (lets the launcher pick dataset/model without editing config.py)
    dataset_override = os.environ.get("SMART_DATASET_PATH")
    if dataset_override:
        base.paths.dataset_path = Path(dataset_override)
    model_override = os.environ.get("SMART_MODEL_DIR")
    if model_override:
        base.paths.model_dir = Path(model_override)
    
    return ConfigLoRA(
        paths=base.paths,
        model=base.model,
        data=base.data,
        training=base.training,
        lora=LoraConfig()
    )

if __name__ == "__main__":
    cfg = get_lora_config()
    print(cfg)
