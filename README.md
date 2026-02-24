# SMART LoRA: Physics-Inspired Regularization for Efficient Fine-Tuning

A novel approach to fine-tuning large language models using physics and mathematics-inspired regularizers combined with LoRA (Low-Rank Adaptation).

## Overview

SMART LoRA combines four distinct regularization techniques:

1. **Entropic Regularizer** - Prevents mode collapse and encourages prediction diversity
2. **Holographic Depth Regularizer** - Enforces layer-wise information density (AdS/CFT inspired)
3. **Differentiable Topology Regularizer** - Ensures latent space stays connected
4. **Manifold Regularizer** - Constrains weights to doubly-stochastic manifold

## Key Features

- **Parameter Efficient**: Uses LoRA for memory-efficient fine-tuning
- **Physics-Inspired**: Draws from information theory, holography, topology, and optimal transport
- **Innovative Combination**: First system to combine all four regularizers with LoRA

## Directory Structure

```
SMART-LoRA/
├── src/                    # Source code
│   ├── train_lora_minimax.py
│   ├── smart_components_minimax.py
│   ├── config_lora_minimax.py
│   └── data_processor_minimax.py
├── config/                 # Configuration files
│   ├── config_minimax.py
│   └── env_example
├── logs/                   # Training logs
├── benchmarks/             # Benchmark results
├── papers/                 # Research papers (PDFs and notes)
├── data/                   # Dataset examples
└── PAPER/                  # Paper writing
    ├── drafts_outlines/
    ├── research_and_citations/
    └── final_draft/
```

## Requirements

- PyTorch with ROCm (AMD GPU) or CUDA (NVIDIA GPU)
- Transformers library
- PEFT library
- AMD GPU (tested on ROCm) or NVIDIA GPU

## Installation

```bash
# Clone the repository
# Install dependencies
pip install torch transformers peft
```

## Configuration

Copy `config/env_example` to `.env` and adjust values:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
HSA_OVERRIDE_GFX_VERSION=11.5.1
# ... other settings
```

## Training

```bash
cd src
python train_lora_minimax.py
```

## Regularizers

### Entropic Regularizer
Prevents mode collapse by penalizing over-confident predictions. Uses a knowledge mass estimator to scale the entropy penalty appropriately.

### Holographic Depth Regularizer  
Inspired by the AdS/CFT correspondence, enforces that deeper layers should have decreasing information density (1/depth profile).

### Differentiable Topology Regularizer
Uses triplet loss to ensure the latent space remains connected without topological holes.

### Manifold Regularizer
Projects weight matrices toward the doubly-stochastic manifold using Sinkhorn projections.

## Results

See `benchmarks/` for detailed benchmark results showing the effectiveness of SMART regularizers.

## Paper

See `PAPER/` for paper drafts and outlines.

## License

[To be determined]

## Acknowledgments

- LoRA (Hu et al., 2021)
- PEFT library (Hugging Face)
- Various physics and mathematics inspirations

## Citation

[To be completed]
