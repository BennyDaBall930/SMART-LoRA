"""
SMART LoRA Training
Physics-inspired regularization for efficient fine-tuning.
"""

import os

# Set ROCm environment variables early. The launcher can override these.
# If you run this script directly (without `run_lora.ps1`), setting these before importing torch matters.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
import time
import math
import json
import re
import torch
import logging
import random
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, TaskType, get_peft_model
from peft import LoraConfig as PeftLoraConfig

# Add parent dir for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config_lora_smart import get_lora_config
from data_processor_smart import ZEngineerDataset, load_jsonl, create_splits as create_base_splits

# Import Smart Components
from smart_components_smart import (
    EntropicRegularizer,
    HolographicDepthRegularizer,
    DifferentiableTopologyRegularizer,
    # ManifoldRegularizer is NOT used for LoRA
)

# On Windows, Triton/Inductor cache paths can exceed MAX_PATH because kernel filenames are long.
def _maybe_set_short_inductor_cache_dir() -> None:
    if os.name != "nt":
        return
    if os.environ.get("SMART_COMPILE", "0") != "1":
        return
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return

    try:
        drive_root = Path(__file__).anchor 
        if not drive_root:
            return
        cache_root = Path(drive_root) / "ti_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root)
    except Exception:
        return

# Custom Handler to route logs through tqdm
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(logging.INFO)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def _resolve_run_dirs(config):
    ckpt_root_env = os.environ.get("SMART_CHECKPOINT_ROOT")
    checkpoint_root = Path(ckpt_root_env) if ckpt_root_env else (config.paths.base_dir / "checkpoints_lora")

    run_dir_env = os.environ.get("SMART_RUN_DIR")
    if run_dir_env:
        run_dir = Path(run_dir_env)
    else:
        default_id = time.strftime("run_lora_%Y%m%d_%H%M%S")
        run_id = os.environ.get("SMART_RUN_ID") or f"{default_id}_{os.getpid()}"
        if not run_id.startswith("run_"):
            run_id = f"run_{run_id}"
        run_dir = checkpoint_root / run_id

    run_ckpt_dir = run_dir / "checkpoints"
    run_logs_dir = run_dir / "logs"

    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_root, run_dir, run_ckpt_dir, run_logs_dir

def _write_run_manifest(run_dir: Path, checkpoint_root: Path, run_ckpt_dir: Path, run_logs_dir: Path, config) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": "LoRA",
        "lora_config": str(config.lora),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "checkpoints_dir": str(run_ckpt_dir.resolve()),
        "logs_dir": str(run_logs_dir.resolve()),
        "model_dir": str(Path(config.paths.model_dir).resolve()),
        "dataset_path": str(Path(config.paths.dataset_path).resolve()),
        "model": {k: getattr(config.model, k, None) for k in vars(config.model)},
        "data": {k: getattr(config.data, k, None) for k in vars(config.data)},
        "training": {k: getattr(config.training, k, None) for k in vars(config.training)},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

device = torch.device("cuda")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(32)

def _format_progress(step: int, total_steps: int) -> str:
    if total_steps <= 0: return f"{step}"
    pct = (step / total_steps) * 100.0
    return f"{step}/{total_steps} ({pct:.2f}%)"

def _format_smart_metrics(metrics: dict) -> str:
    base = metrics.get('base', 0.0)
    total = metrics.get('total', 0.0)
    ent = metrics.get('ent', 0.0)
    holo = metrics.get('holo', 0.0)
    topo = metrics.get('topo', 0.0)
    
    return (
        f"Loss: {total:.4f} (Base: {base:.4f}) | "
        f"Smart [ Ent: {ent:.4f} | Holo: {holo:.4f} | Topo: {topo:.4f} ]"
    )

def _format_vram() -> str:
    if not torch.cuda.is_available():
        return "VRAM: n/a"
    try:
        gb = 1024.0**3
        alloc = torch.cuda.memory_allocated() / gb
        reserved = torch.cuda.memory_reserved() / gb
        peak = torch.cuda.max_memory_allocated() / gb
        return f"VRAM {alloc:.1f}G alloc | {reserved:.1f}G reserved | {peak:.1f}G peak"
    except Exception:
        return "VRAM: n/a"

def _use_eager_causal_lm_loss() -> bool:
    """Avoid compiling the fused log_softmax/nll_loss path on Windows+ROCm where Triton can crash."""
    v = os.environ.get("SMART_EAGER_LOSS")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    if os.environ.get("SMART_COMPILE", "0") == "1" and os.name == "nt" and getattr(torch.version, "hip", None):
        return True
    return False

def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _parse_checkpoint_config_str(config_str: str | None) -> dict | None:
    """Best-effort parse of legacy `training_state.pt` config string."""
    if not isinstance(config_str, str) or not config_str:
        return None

    def _re_int(field: str) -> int | None:
        m = re.search(rf"{re.escape(field)}=(\d+)", config_str)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _re_float(field: str) -> float | None:
        m = re.search(rf"{re.escape(field)}=([0-9.eE+-]+)", config_str)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    bs = _re_int("per_device_batch_size")
    ga = _re_int("gradient_accumulation_steps")
    max_len = _re_int("max_length")
    lr = _re_float("learning_rate")

    if bs is None and ga is None and max_len is None and lr is None:
        return None

    snap = {"training": {}, "model": {}}
    if bs is not None:
        snap["training"]["per_device_batch_size"] = bs
    if ga is not None:
        snap["training"]["gradient_accumulation_steps"] = ga
    if lr is not None:
        snap["training"]["learning_rate"] = lr
    if max_len is not None:
        snap["model"]["max_length"] = max_len
    return snap

def _is_torch_compile_failure(exc: BaseException) -> bool:
    try:
        from torch._inductor.exc import InductorError
        if isinstance(exc, InductorError):
            return True
    except Exception:
        pass
    try:
        from torch._dynamo.exc import BackendCompilerFailed
        if isinstance(exc, BackendCompilerFailed):
            return True
    except Exception:
        pass

    msg = str(exc) or repr(exc)
    lower = msg.lower()
    if "torch._inductor" in lower or "inductorerror" in lower:
        return True
    if "triton compilation failed" in lower or "passmanager::run failed" in lower:
        return True
    if "ttgir" in lower and "pipeline failed" in lower:
        return True
    return False

def _disable_torch_compile_and_unwrap(model, entropic_reg=None, holo_depth_reg=None, topo_reg=None, exc: BaseException | None = None):
    if exc is not None:
        logging.error("torch.compile execution failed; disabling compilation and retrying eagerly. Error: %s", exc)

    os.environ["SMART_COMPILE"] = "0"

    def _unwrap(m):
        return m._orig_mod if hasattr(m, "_orig_mod") else m

    model = _unwrap(model)
    if entropic_reg is not None:
        entropic_reg = _unwrap(entropic_reg)
    if holo_depth_reg is not None:
        holo_depth_reg = _unwrap(holo_depth_reg)
    if topo_reg is not None:
        topo_reg = _unwrap(topo_reg)

    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception:
        pass

    return model, entropic_reg, holo_depth_reg, topo_reg

def _causal_lm_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError(f"Unexpected shapes: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

def evaluate(model, val_loader, entropic_reg=None, holo_depth_reg=None, topo_reg=None):
    model_was_training = model.training
    regs = [entropic_reg, holo_depth_reg, topo_reg]
    regs_were_training = [r.training if r is not None else None for r in regs]

    model.eval()
    for r in regs:
        if r is not None:
            r.eval()
    totals = {'base': 0.0, 'ent': 0.0, 'holo': 0.0, 'topo': 0.0, 'total': 0.0}
    count = 0
    logging.info("Running Validation...")
     
    try:
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                lengths = None
                if "attention_mask" in batch:
                    try:
                        lengths = batch["attention_mask"].sum(dim=-1).tolist()
                    except Exception:
                        lengths = None
                inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = inputs.get("labels")
                model_inputs = inputs

                eager_loss = _use_eager_causal_lm_loss()
                if eager_loss and labels is not None:
                    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(**model_inputs, output_hidden_states=True)

                    if eager_loss and labels is not None:
                        base_loss = _causal_lm_loss_from_logits(outputs.logits, labels)
                    else:
                        base_loss = outputs.loss

                    mask = inputs.get('attention_mask')
                    hidden_states = getattr(outputs, "hidden_states", None)

                    l_ent = torch.tensor(0.0, device=base_loss.device)
                    if entropic_reg is not None and hidden_states:
                        l_ent, _, _ = entropic_reg(outputs.logits, hidden_states[-1], mask)

                    l_holo = torch.tensor(0.0, device=base_loss.device)
                    if holo_depth_reg is not None and hidden_states:
                        l_holo = 0.05 * holo_depth_reg(hidden_states, mask)

                    l_topo = torch.tensor(0.0, device=base_loss.device)
                    if topo_reg is not None and hidden_states:
                        l_topo = topo_reg(hidden_states[-1], mask, lengths=lengths)

                    total_loss = base_loss + l_ent + l_holo + l_topo

                totals['base'] += base_loss.item()
                totals['ent'] += l_ent.item()
                totals['holo'] += l_holo.item()
                totals['topo'] += l_topo.item()
                totals['total'] += total_loss.item()
                count += 1

        if count <= 0:
            avg = totals
        else:
            avg = {k: v / count for k, v in totals.items()}
        return avg
    finally:
        if model_was_training:
            model.train()
        for r, was_training in zip(regs, regs_were_training):
            if r is None or was_training is None:
                continue
            if was_training:
                r.train()
            else:
                r.eval()

def safe_evaluate(model, val_loader, entropic_reg=None, holo_depth_reg=None, topo_reg=None):
    try:
        avg = evaluate(model, val_loader, entropic_reg, holo_depth_reg, topo_reg)
        return avg, model, entropic_reg, holo_depth_reg, topo_reg
    except Exception as e:
        if os.environ.get("SMART_COMPILE", "0") == "1" and _is_torch_compile_failure(e):
            model, entropic_reg, holo_depth_reg, topo_reg = _disable_torch_compile_and_unwrap(
                model,
                entropic_reg,
                holo_depth_reg,
                topo_reg,
                exc=e,
            )
            avg = evaluate(model, val_loader, entropic_reg, holo_depth_reg, topo_reg)
            return avg, model, entropic_reg, holo_depth_reg, topo_reg
        raise

def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    run_ckpt_dir: Path,
    run_dir: Path,
    config,
    step: int,
    entropic_reg=None,
    holo_depth_reg=None,
    topo_reg=None,
    epoch: int | None = None,
    batch_idx: int | None = None,
    tag: str | None = None,
):
    ckpt_name = f"checkpoint-{tag}" if tag else f"checkpoint-{step}"
    ckpt_path = run_ckpt_dir / ckpt_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving LoRA checkpoint to {ckpt_path}...")
    
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    save_model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    
    def _unwrap_state_dict(m):
        m = m._orig_mod if hasattr(m, "_orig_mod") else m
        return m.state_dict()

    effective_bs = getattr(config.training, "effective_batch_size", None)
    if effective_bs is None:
        effective_bs = int(config.training.per_device_batch_size) * int(config.training.gradient_accumulation_steps)

    state = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': int(step),
        'tag': tag,
        'config': str(config),
        'config_snapshot': {
            "paths": {
                "model_dir": str(getattr(config.paths, "model_dir", "")),
                "dataset_path": str(getattr(config.paths, "dataset_path", "")),
            },
            "model": {
                "dtype": str(getattr(config.model, "dtype", "")),
                "max_length": int(getattr(config.model, "max_length", 0) or 0),
            },
            "data": {
                "seed": int(getattr(config.data, "seed", 0) or 0),
                "shuffle": bool(getattr(config.data, "shuffle", False)),
            },
            "training": {
                "per_device_batch_size": int(getattr(config.training, "per_device_batch_size", 0) or 0),
                "gradient_accumulation_steps": int(getattr(config.training, "gradient_accumulation_steps", 0) or 0),
                "effective_batch_size": int(effective_bs or 0),
                "learning_rate": float(getattr(config.training, "learning_rate", 0.0) or 0.0),
                "warmup_ratio": float(getattr(config.training, "warmup_ratio", 0.0) or 0.0),
                "weight_decay": float(getattr(config.training, "weight_decay", 0.0) or 0.0),
            },
            "lora": {
                "r": int(getattr(config.lora, "r", 0) or 0),
                "lora_alpha": int(getattr(config.lora, "lora_alpha", 0) or 0),
                "lora_dropout": float(getattr(config.lora, "lora_dropout", 0.0) or 0.0),
                "bias": str(getattr(config.lora, "bias", "")),
                "target_modules": list(getattr(config.lora, "target_modules", []) or []),
            },
        },
        'epoch': None if epoch is None else int(epoch),
        'batch_idx': None if batch_idx is None else int(batch_idx),
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if entropic_reg is not None:
        state['entropic_reg_state_dict'] = _unwrap_state_dict(entropic_reg)
    if holo_depth_reg is not None:
        state['holo_depth_reg_state_dict'] = _unwrap_state_dict(holo_depth_reg)
    if topo_reg is not None:
        state['topo_reg_state_dict'] = _unwrap_state_dict(topo_reg)
    torch.save(state, ckpt_path / "training_state.pt")
    
    last_ckpt_path = run_dir / "last_checkpoint.txt"
    last_ckpt_path.write_text(str(ckpt_path.resolve()), encoding="utf-8")

def _fix_compiled_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def maybe_resume_state(resume_dir, optimizer, scheduler, entropic_reg=None, holo_depth_reg=None, topo_reg=None):
    state_path = resume_dir / "training_state.pt"
    if not state_path.exists():
        logging.warning(f"Resume requested but training_state.pt not found in {resume_dir}")
        return {"step": 0, "epoch": None, "batch_idx": None}

    logging.info(f"Loading training state from {state_path}...")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])

    def _load_module_state(mod, key):
        if mod is None or key not in state:
            return
        target = mod._orig_mod if hasattr(mod, "_orig_mod") else mod
        sd = _fix_compiled_keys(state[key])
        target.load_state_dict(sd, strict=False)

    _load_module_state(entropic_reg, 'entropic_reg_state_dict')
    _load_module_state(holo_depth_reg, 'holo_depth_reg_state_dict')
    _load_module_state(topo_reg, 'topo_reg_state_dict')

    rng_state = state.get("rng_state")
    if isinstance(rng_state, dict):
        try:
            if "python" in rng_state and rng_state["python"] is not None:
                random.setstate(rng_state["python"])
        except Exception as exc:
            logging.warning("Failed restoring python RNG state: %s", exc)
        try:
            if "numpy" in rng_state and rng_state["numpy"] is not None:
                np.random.set_state(rng_state["numpy"])
        except Exception as exc:
            logging.warning("Failed restoring numpy RNG state: %s", exc)
        try:
            if "torch" in rng_state and rng_state["torch"] is not None:
                torch.set_rng_state(rng_state["torch"])
        except Exception as exc:
            logging.warning("Failed restoring torch RNG state: %s", exc)
        try:
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        except Exception as exc:
            logging.warning("Failed restoring cuda RNG state: %s", exc)

    for optim_state in optimizer.state.values():
        for k, v in optim_state.items():
            if torch.is_tensor(v):
                optim_state[k] = v.to(device)

    step = state.get('step', 0)
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = int(scheduler.state_dict().get('last_epoch', 0))
        logging.warning("Non-numeric checkpoint step; falling back to scheduler.last_epoch=%s", step)

    epoch = state.get("epoch", None)
    try:
        epoch = None if epoch is None else int(epoch)
    except (TypeError, ValueError):
        epoch = None

    batch_idx = state.get("batch_idx", None)
    try:
        batch_idx = None if batch_idx is None else int(batch_idx)
    except (TypeError, ValueError):
        batch_idx = None

    config_str = state.get("config")
    if not isinstance(config_str, str):
        config_str = None

    config_snapshot = state.get("config_snapshot")
    if not isinstance(config_snapshot, dict):
        config_snapshot = None

    return {
        "step": step,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "config_str": config_str,
        "config_snapshot": config_snapshot,
    }

def train_lora():
    config = get_lora_config()
    _maybe_set_short_inductor_cache_dir()

    checkpoint_root, run_dir, run_ckpt_dir, run_logs_dir = _resolve_run_dirs(config)
    configure_logging(run_logs_dir / "training_lora_log.txt")
    _write_run_manifest(run_dir, checkpoint_root, run_ckpt_dir, run_logs_dir, config)

    if not Path(config.paths.model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {config.paths.model_dir}")
    if not Path(config.paths.dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {config.paths.dataset_path}")

    set_seed(config.data.seed)

    logging.info("="*60)
    logging.info(" SMART LoRA Training (Minimized Version)")
    logging.info(f" DEVICE: {torch.cuda.get_device_name(0)}")
    logging.info(f" MODEL:   {config.paths.model_dir}")
    logging.info(f" TARGETS: {config.lora.target_modules}")
    logging.info(f" RANK:    {config.lora.r}")
    logging.info(
        f" BATCH:   {config.training.per_device_batch_size} x {config.training.gradient_accumulation_steps} = {getattr(config.training, 'effective_batch_size', config.training.per_device_batch_size * config.training.gradient_accumulation_steps)}"
    )
    logging.info(f" MAX_LEN: {config.model.max_length}")
    logging.info("="*60)
    
    # 1. Load Model (+ optional adapter resume)
    resume_env = os.environ.get("SMART_RESUME_PATH")
    resume_dir = Path(resume_env) if resume_env else None

    if resume_dir is None:
        last_ckpt_file = run_dir / "last_checkpoint.txt"
        if last_ckpt_file.exists():
            candidate = Path(last_ckpt_file.read_text(encoding="utf-8").strip())
            if candidate.exists():
                resume_dir = candidate

    tokenizer_load_dir = resume_dir if (resume_dir and resume_dir.exists()) else Path(config.paths.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.paths.model_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    use_grad_ckpt = _env_flag("SMART_GRAD_CHECKPOINT", bool(getattr(config.model, "gradient_checkpointing", False)))
    if use_grad_ckpt:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False
    else:
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        base_model.config.use_cache = False

    if resume_dir and resume_dir.exists():
        logging.info(f"Resuming LoRA adapter from checkpoint dir: {resume_dir}")
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    else:
        if resume_dir:
            logging.warning(f"Resume checkpoint not found: {resume_dir}")
        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
        )
        model = get_peft_model(base_model, peft_config)

    model.print_trainable_parameters()
    model.to(device)

    trainable_in_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_in_model <= 0:
        raise RuntimeError(
            "LoRA injection produced zero trainable parameters. "
            "This usually means `target_modules` didn't match the model's module names."
        )
    
    # 3. Initialize Smart Components
    hidden_size = model.config.hidden_size
    num_hidden_layers = model.config.num_hidden_layers
    
    logging.info(f"Initializing Smart Components (H={hidden_size}, L={num_hidden_layers})...")
    entropic_reg = EntropicRegularizer(hidden_size, entropy_scale=0.01).to(device)
    holo_depth_reg = HolographicDepthRegularizer(num_hidden_layers, hidden_size).to(device)
    topo_reg = DifferentiableTopologyRegularizer(hidden_size, connectivity_weight=0.1, hole_weight=0.1).to(device)

    train_entropic_reg = _env_flag("SMART_TRAIN_ENTROPIC_REG", False)
    train_topo_reg = _env_flag("SMART_TRAIN_TOPO_REG", False)
    if not train_entropic_reg:
        for p in entropic_reg.parameters():
            p.requires_grad = False
        entropic_reg.eval()
    if not train_topo_reg:
        for p in topo_reg.parameters():
            p.requires_grad = False
        topo_reg.eval()
    logging.info("SMART regs trainable: entropic=%s topo=%s", train_entropic_reg, train_topo_reg)

    # Optional Torch Compile (Triton)
    if os.environ.get("SMART_COMPILE", "0") == "1":
        logging.info("Compiling with torch.compile (Backend: inductor)...")
        compile_mode = os.environ.get("SMART_COMPILE_MODE", "default")

        is_windows_rocm = (os.name == "nt" and getattr(torch.version, "hip", None))
        compile_model = _env_flag("SMART_COMPILE_MODEL", False if is_windows_rocm else True)
        compile_smart_regs = _env_flag("SMART_COMPILE_SMART_REGS", False if is_windows_rocm else True)
        compile_entropic = _env_flag("SMART_COMPILE_ENTROPIC", False if is_windows_rocm else compile_smart_regs)
        compile_holo = _env_flag("SMART_COMPILE_HOLO", compile_smart_regs)
        compile_topo = _env_flag("SMART_COMPILE_TOPO", compile_smart_regs)

        if not any([compile_model, compile_entropic, compile_holo, compile_topo]):
            logging.info("Compilation disabled by config. Running eager.")
        else:
            try:
                if compile_model:
                    model = torch.compile(model, mode=compile_mode, backend="inductor")
                if compile_entropic:
                    entropic_reg = torch.compile(entropic_reg, mode=compile_mode, backend="inductor")
                if compile_holo:
                    holo_depth_reg = torch.compile(holo_depth_reg, mode=compile_mode, backend="inductor")
                if compile_topo:
                    topo_reg = torch.compile(topo_reg, mode=compile_mode, backend="inductor")

                logging.info(
                    "Compilation enabled (Mode: %s). model=%s entropic=%s holo=%s topo=%s",
                    compile_mode,
                    compile_model,
                    compile_entropic,
                    compile_holo,
                    compile_topo,
                )
            except Exception as e:
                logging.error(f"Compilation failed: {e}")
                logging.warning("Falling back to eager mode.")
    
    # 4. Setup Data
    all_data = load_jsonl(config.paths.dataset_path)
    train_data, val_data = create_base_splits(all_data, train_ratio=config.data.train_split, seed=config.data.seed)
    
    train_dataset = ZEngineerDataset(train_data, tokenizer, config.model.max_length)
    val_dataset = ZEngineerDataset(val_data, tokenizer, config.model.max_length)
    
    train_loader_kwargs = dict(
        batch_size=config.training.per_device_batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        pin_memory=config.training.dataloader_pin_memory,
        persistent_workers=(config.data.num_workers > 0),
    )
    val_loader_kwargs = dict(
        batch_size=config.training.per_device_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.training.dataloader_pin_memory,
        persistent_workers=(config.data.num_workers > 0),
    )
    if config.data.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["prefetch_factor"] = 2

    train_gen = torch.Generator()
    if config.data.shuffle:
        train_loader_kwargs["generator"] = train_gen

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    # 5. Optimization
    accum_steps = config.training.gradient_accumulation_steps
    steps_per_epoch = max(1, int(math.ceil(len(train_loader) / float(accum_steps))))
    total_steps = steps_per_epoch * config.training.num_epochs
    
    # Group parameters: LoRA Params + Smart Reg Params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    ent_params = [p for p in entropic_reg.parameters() if p.requires_grad]
    holo_params = [p for p in holo_depth_reg.parameters() if p.requires_grad]
    topo_params = [p for p in topo_reg.parameters() if p.requires_grad]

    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found on the model.")

    optimizer = torch.optim.AdamW(
        [
            {'params': model.parameters()},
            {'params': entropic_reg.parameters(), 'lr': config.training.learning_rate * 10},
            {'params': holo_depth_reg.parameters(), 'lr': config.training.learning_rate},
            {'params': topo_reg.parameters(), 'lr': config.training.learning_rate},
        ],
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * config.training.warmup_ratio), 
        num_training_steps=total_steps
    )
    
    logging.info(f"Total Global Steps: {total_steps}")
    
    # 6. Training Loop
    global_step = 0

    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    running_aux = {
        'ent': torch.zeros((), device=device, dtype=torch.float32),
        'holo': torch.zeros((), device=device, dtype=torch.float32),
        'topo': torch.zeros((), device=device, dtype=torch.float32),
    }
    log_window = {'base': 0.0, 'ent': 0.0, 'holo': 0.0, 'topo': 0.0}
    log_window_steps = 0

    model.train()
    if train_entropic_reg:
        entropic_reg.train()
    else:
        entropic_reg.eval()
    holo_depth_reg.train()
    if train_topo_reg:
        topo_reg.train()
    else:
        topo_reg.eval()

    optimizer.zero_grad(set_to_none=True)

    resume_info = None
    if resume_dir and resume_dir.exists():
        resume_info = maybe_resume_state(
            resume_dir,
            optimizer,
            scheduler,
            entropic_reg,
            holo_depth_reg,
            topo_reg,
        )

    global_step = int(resume_info["step"]) if resume_info else 0

    if resume_info:
        saved_snapshot = resume_info.get("config_snapshot")
        if not isinstance(saved_snapshot, dict):
            saved_snapshot = _parse_checkpoint_config_str(resume_info.get("config_str"))

        if isinstance(saved_snapshot, dict):
            saved_training = saved_snapshot.get("training") if isinstance(saved_snapshot.get("training"), dict) else {}
            saved_model_cfg = saved_snapshot.get("model") if isinstance(saved_snapshot.get("model"), dict) else {}

            saved_bs = saved_training.get("per_device_batch_size")
            saved_ga = saved_training.get("gradient_accumulation_steps")
            saved_lr = saved_training.get("learning_rate")
            saved_max_len = saved_model_cfg.get("max_length")

            if any(v is not None for v in (saved_bs, saved_ga, saved_max_len, saved_lr)):
                logging.info(
                    "Checkpoint config (from training_state.pt): bs=%s accum=%s max_len=%s lr=%s",
                    saved_bs,
                    saved_ga,
                    saved_max_len,
                    saved_lr,
                )

            try:
                cur_bs = int(config.training.per_device_batch_size)
            except Exception:
                cur_bs = None
            try:
                cur_ga = int(config.training.gradient_accumulation_steps)
            except Exception:
                cur_ga = None
            try:
                cur_max_len = int(config.model.max_length)
            except Exception:
                cur_max_len = None

            if saved_bs is not None and cur_bs is not None and int(saved_bs) != int(cur_bs):
                logging.warning(
                    "Resume config mismatch: checkpoint per_device_batch_size=%s, current=%s. "
                    "Set SMART_LORA_BATCH_SIZE=%s to match the checkpoint.",
                    saved_bs,
                    cur_bs,
                    saved_bs,
                )
            if saved_ga is not None and cur_ga is not None and int(saved_ga) != int(cur_ga):
                logging.warning(
                    "Resume config mismatch: checkpoint gradient_accumulation_steps=%s, current=%s. "
                    "Set SMART_LORA_GRAD_ACCUM=%s to match the checkpoint.",
                    saved_ga,
                    cur_ga,
                    saved_ga,
                )
            if saved_max_len is not None and cur_max_len is not None and int(saved_max_len) != int(cur_max_len):
                logging.warning(
                    "Resume config mismatch: checkpoint max_length=%s, current=%s. "
                    "Set SMART_LORA_MAX_LENGTH=%s to match the checkpoint.",
                    saved_max_len,
                    cur_max_len,
                    saved_max_len,
                )

    start_epoch = 0
    resume_epoch = None
    resume_start_batch_in_epoch = 0
    num_batches = len(train_loader)
    if global_step > 0:
        ckpt_epoch = resume_info.get("epoch") if resume_info else None
        ckpt_batch = resume_info.get("batch_idx") if resume_info else None
        if ckpt_epoch is not None and ckpt_batch is not None:
            start_epoch = ckpt_epoch
            resume_epoch = start_epoch
            resume_start_batch_in_epoch = ckpt_batch
            if resume_start_batch_in_epoch >= num_batches:
                start_epoch = start_epoch + 1
                resume_epoch = None
                resume_start_batch_in_epoch = 0
        else:
            start_epoch = global_step // steps_per_epoch
            resume_epoch = start_epoch
            resume_start_batch_in_epoch = min((global_step % steps_per_epoch) * accum_steps, num_batches)

        start_epoch = min(max(int(start_epoch), 0), config.training.num_epochs)

    last_seen_epoch = None
    last_seen_batch_idx = None

    if global_step == 0:
        logging.info("Running Initial Validation Baseline...")
        val_avg, model, entropic_reg, holo_depth_reg, topo_reg = safe_evaluate(
            model, val_loader, entropic_reg, holo_depth_reg, topo_reg
        )
        logging.info(f"VAL Step {_format_progress(global_step, total_steps)} | {_format_smart_metrics(val_avg)}")
    else:
        logging.info(
            "Resuming at step=%d (epoch=%s batch=%s).",
            global_step,
            resume_epoch,
            resume_start_batch_in_epoch,
        )
        resume_validate = _env_flag("SMART_LORA_RESUME_VALIDATE", _env_flag("SMART_RESUME_VALIDATE", False))
        if resume_validate:
            logging.info("Running Resume Validation Sanity Check...")
            val_avg, model, entropic_reg, holo_depth_reg, topo_reg = safe_evaluate(
                model, val_loader, entropic_reg, holo_depth_reg, topo_reg
            )
            logging.info(f"VAL (Resume) Step {_format_progress(global_step, total_steps)} | {_format_smart_metrics(val_avg)}")
        else:
            logging.info("Skipping Resume Validation to accelerate startup.")

    last_log_time = time.time()
    tokens_since_log = 0
    tokens_in_accum_group = 0
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    for epoch in range(start_epoch, config.training.num_epochs):
        if global_step >= total_steps:
            break

        if config.data.shuffle:
            train_gen.manual_seed(int(config.data.seed) + int(epoch))

        logging.info(f"Starting Epoch {epoch+1}/{config.training.num_epochs}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        if resume_epoch is not None and epoch == resume_epoch and resume_start_batch_in_epoch > 0:
            logging.info(
                "Fast-forwarding dataloader: Skipping %d batches to resume position...",
                resume_start_batch_in_epoch,
            )
        announced_resume = False

        num_batches = len(train_loader)
        remainder = num_batches % accum_steps
        if remainder == 0:
            remainder = accum_steps

        for batch_idx, batch in enumerate(pbar, start=1):
            if global_step >= total_steps:
                break

            if resume_epoch is not None and epoch == resume_epoch and resume_start_batch_in_epoch > 0:
                if batch_idx <= resume_start_batch_in_epoch:
                    if batch_idx % 1000 == 0:
                        pbar.set_description(f"Skipping {batch_idx}/{resume_start_batch_in_epoch}")
                    continue
                elif not announced_resume:
                    logging.info(f"Resuming training at Epoch {epoch+1}, Batch {batch_idx}")
                    announced_resume = True

            last_seen_epoch = epoch
            last_seen_batch_idx = batch_idx

            try:
                micro_bs = int(batch["input_ids"].shape[0])
                micro_seq = int(batch["input_ids"].shape[1])
                tokens_in_accum_group += micro_bs * micro_seq
            except Exception:
                pass

            lengths = None
            if "attention_mask" in batch:
                try:
                    lengths = batch["attention_mask"].sum(dim=-1).tolist()
                except Exception:
                    lengths = None

            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = inputs.get("labels")
            model_inputs = inputs
            eager_loss = _use_eager_causal_lm_loss()
            if eager_loss and labels is not None:
                model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

            mask = inputs.get('attention_mask')
            group_size = remainder if (batch_idx > (num_batches - remainder)) else accum_steps

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                try:
                    outputs = model(**model_inputs, output_hidden_states=True)

                    if eager_loss and labels is not None:
                        base_loss = _causal_lm_loss_from_logits(outputs.logits, labels)
                    else:
                        base_loss = outputs.loss

                    l_ent, _, _ = entropic_reg(outputs.logits, outputs.hidden_states[-1], mask)
                    l_holo = 0.05 * holo_depth_reg(outputs.hidden_states, mask)
                    l_topo = topo_reg(outputs.hidden_states[-1], mask, lengths=lengths)
                except Exception as e:
                    if os.environ.get("SMART_COMPILE", "0") == "1" and _is_torch_compile_failure(e):
                        model, entropic_reg, holo_depth_reg, topo_reg = _disable_torch_compile_and_unwrap(
                            model, entropic_reg, holo_depth_reg, topo_reg, exc=e
                        )
                        outputs = model(**model_inputs, output_hidden_states=True)

                        if eager_loss and labels is not None:
                            base_loss = _causal_lm_loss_from_logits(outputs.logits, labels)
                        else:
                            base_loss = outputs.loss

                        l_ent, _, _ = entropic_reg(outputs.logits, outputs.hidden_states[-1], mask)
                        l_holo = 0.05 * holo_depth_reg(outputs.hidden_states, mask)
                        l_topo = topo_reg(outputs.hidden_states[-1], mask, lengths=lengths)
                    else:
                        raise

                total_loss = base_loss + l_ent + l_holo + l_topo
                loss = total_loss / float(group_size)

            loss.backward()

            running_loss += base_loss.detach().float()
            running_aux['ent'] += l_ent.detach().float()
            running_aux['holo'] += l_holo.detach().float()
            running_aux['topo'] += l_topo.detach().float()

            should_step = (batch_idx % accum_steps == 0) or (batch_idx == num_batches)
            if not should_step:
                continue

            params_to_clip = []
            params_to_clip.extend(lora_params)
            params_to_clip.extend(ent_params)
            params_to_clip.extend(holo_params)
            params_to_clip.extend(topo_params)
            torch.nn.utils.clip_grad_norm_(params_to_clip, config.training.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            tokens_since_log += tokens_in_accum_group
            tokens_in_accum_group = 0
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else config.training.learning_rate

            inv_group = 1.0 / float(group_size)
            step_metrics = {
                'base': (running_loss * inv_group).item(),
                'ent': (running_aux['ent'] * inv_group).item(),
                'holo': (running_aux['holo'] * inv_group).item(),
                'topo': (running_aux['topo'] * inv_group).item(),
            }

            running_loss.zero_()
            for k in running_aux:
                running_aux[k].zero_()

            for k in log_window:
                log_window[k] += step_metrics.get(k, 0.0)
            log_window_steps += 1

            if global_step % config.training.logging_steps == 0 and log_window_steps > 0:
                now = time.time()
                dt = max(now - last_log_time, 1e-6)
                tps = float(tokens_since_log) / dt

                avg_metrics = {k: v / log_window_steps for k, v in log_window.items()}
                avg_metrics['total'] = sum(avg_metrics.values())

                pbar.set_description(f"Ep {epoch+1} | {tps:.1f} t/s | {_format_smart_metrics(avg_metrics)}")
                logging.info(
                    f"Step {global_step} | LR {current_lr:.2e} | {tps:.1f} t/s | {_format_smart_metrics(avg_metrics)} | {_format_vram()}"
                )

                last_log_time = now
                tokens_since_log = 0
                log_window = {k: 0.0 for k in log_window}
                log_window_steps = 0

            if global_step % config.training.eval_steps == 0:
                val_avg, model, entropic_reg, holo_depth_reg, topo_reg = safe_evaluate(
                    model, val_loader, entropic_reg, holo_depth_reg, topo_reg
                )
                logging.info(f"VAL Step {global_step} | {_format_smart_metrics(val_avg)}")

            if global_step % config.training.save_steps == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    run_ckpt_dir,
                    run_dir,
                    config,
                    global_step,
                    entropic_reg,
                    holo_depth_reg,
                    topo_reg,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )

    if global_step > 0:
        val_avg, model, entropic_reg, holo_depth_reg, topo_reg = safe_evaluate(
            model, val_loader, entropic_reg, holo_depth_reg, topo_reg
        )
        logging.info(f"VAL Step {_format_progress(global_step, total_steps)} | {_format_smart_metrics(val_avg)}")
        save_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            run_ckpt_dir,
            run_dir,
            config,
            global_step,
            entropic_reg,
            holo_depth_reg,
            topo_reg,
            epoch=last_seen_epoch,
            batch_idx=last_seen_batch_idx,
            tag="final",
        )

if __name__ == "__main__":
    train_lora()
