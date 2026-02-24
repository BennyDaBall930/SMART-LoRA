# SMART LoRA: Physics-Inspired Regularization for Efficient Fine-Tuning

A novel approach to fine-tuning large language models using physics and mathematics-inspired regularizers combined with LoRA (Low-Rank Adaptation).

## Overview

SMART LoRA (Structured Manifold-Aware Regularized Training) combines four distinct regularization techniques with LoRA adapters:

1. **Entropic Regularizer** — Prevents mode collapse and encourages prediction diversity via entropy-scaled knowledge mass penalties
2. **Holographic Depth Regularizer** — Enforces layer-wise information density profiles inspired by AdS/CFT correspondence
3. **Differentiable Topology Regularizer** — Ensures latent space connectivity and prevents topological holes
4. **Manifold Regularizer** — Constrains weights to the doubly-stochastic manifold (full fine-tune only)

## Key Features

- **Parameter Efficient**: All regularization gradients flow through the frozen base + LoRA delta — only adapter weights are trained
- **Physics-Inspired**: Draws from information theory, holography, topology, and optimal transport
- **Trainable vs. Frozen Regulators**: Entropic and topology regularizers can be frozen (LoRA default) or trainable (full fine-tune), controlled by environment flags
- **Fault-Tolerant**: Robust checkpoint resumption with configuration mismatch detection and dataloader fast-forwarding
- **ROCm Compatible**: Tested on AMD Radeon 8060S with ROCm

## Repository Structure

```
SMART-LoRA/
├── README.md
├── .gitignore
├── PAPER/
│   └── paper.md                              # Full research paper
├── src/
│   ├── train_lora_smart.py                   # Main training script
│   ├── smart_components_smart.py             # SMART regularizer implementations
│   ├── config_lora_smart.py                  # LoRA training configuration
│   └── data_processor_smart.py               # Data loading and preprocessing
├── config/
│   ├── config_smart.py                       # Full fine-tune configuration
│   └── env_example                           # Environment variable template
└── benchmarks/
    ├── comprehensive_benchmark_results.json   # Full benchmark data
    ├── lora_benchmark.json                    # LoRA benchmark summary
    ├── lora_benchmark_full.json               # Detailed LoRA benchmarks
    ├── lora_benchmark_full_samples.md         # Qualitative generation samples
    └── lora_benchmark_report.md               # Benchmark report
```

> **Note:** Training data, model weights, logs, and draft papers are excluded from this repository via `.gitignore`.

## Requirements

- Python 3.10+
- PyTorch with ROCm (AMD) or CUDA (NVIDIA)
- Transformers
- PEFT
- tqdm

## Installation

```bash
git clone https://github.com/BennyDaBall930/SMART-LoRA.git
cd SMART-LoRA
pip install torch transformers peft tqdm
```

## Configuration

Copy `config/env_example` to `.env` and adjust for your hardware:

```bash
# AMD ROCm settings
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
HSA_OVERRIDE_GFX_VERSION=11.5.1

# SMART regularizer controls
SMART_TRAIN_ENTROPIC_REG=0    # 0=frozen (LoRA default), 1=trainable
SMART_TRAIN_TOPO_REG=0        # 0=frozen (LoRA default), 1=trainable
```

## Training

```bash
cd src
python train_lora_smart.py
```

Environment variables can override config values at runtime:
- `SMART_RESUME_PATH` — Resume from a specific checkpoint directory
- `SMART_LORA_BATCH_SIZE` — Override per-device batch size
- `SMART_LORA_GRAD_ACCUM` — Override gradient accumulation steps
- `SMART_COMPILE` — Enable `torch.compile` (set to `1`)

## Results

Training over 6,000+ steps demonstrated:
- **Base loss convergence**: 2.95 → 1.21 (59% reduction)
- **Throughput**: ~485 tokens/second sustained on AMD Radeon 8060S
- **Qualitative benchmarks**: Multiple checkpoints scored 100/100 on diverse generative tasks

See [`benchmarks/`](benchmarks/) for detailed results and [`PAPER/paper.md`](PAPER/paper.md) for the full research paper.

## The SMART Regularizers

### Entropic Regularizer
Prevents mode collapse by penalizing over-confident predictions. Uses a learned knowledge mass estimator (2-layer MLP) to scale the entropy penalty. Critical hyperparameter: `entropy_scale=0.01` (100× reduction from naïve default was essential for stability).

### Holographic Depth Regularizer
Inspired by AdS/CFT correspondence. Enforces that deeper layers maintain decreasing information density following a `1/(i+1)` target profile. Uses `eigvalsh` on Gram matrices for efficient eigenvalue-based entropy computation.

### Differentiable Topology Regularizer
Combines connectivity loss (via `torch.cdist` pairwise distances) and hole avoidance loss (via random triplet sampling) to maintain latent space structure.

### Manifold Regularizer
Constrains weight matrices toward doubly-stochastic manifold via Sinkhorn-inspired variance minimization. Only active during full model fine-tuning (excluded when base weights are frozen in LoRA mode).

## Paper

The full research paper is available at [`PAPER/paper.md`](PAPER/paper.md), including:
- Detailed methodology for all four regularizers
- Trainable vs. non-trainable regularizer mode analysis
- 13-checkpoint convergence tables from training logs
- Qualitative benchmark evaluations
- The "holographic compression event" phenomenon

## Citation

```bibtex
@article{smart-lora-2026,
  title={SMART-LoRA: Physics-Inspired Regularization for Efficient Fine-Tuning of Large Language Models},
  year={2026},
  url={https://github.com/BennyDaBall930/SMART-LoRA}
}
```

## Acknowledgments

- [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [PEFT library](https://github.com/huggingface/peft) (Hugging Face)
- AdS/CFT correspondence, topological data analysis, and optimal transport theory
