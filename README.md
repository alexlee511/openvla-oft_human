# OpenVLA-OFT with Humanized Demonstrations

This project extends [OpenVLA-OFT](https://github.com/moojink/openvla-oft) to support **humanized robot demonstrations** — trajectories that exhibit more human-like motion characteristics while maintaining task success. The model is fine-tuned on LIBERO simulation benchmarks using absolute joint position control instead of end-effector position control.

## Overview

- **Base model**: OpenVLA-OFT (Vision-Language-Action model with optimized fine-tuning)
- **Key modification**: Fine-tuning with humanized demonstrations using absolute joint position actions (7 DOF + gripper)
- **Supported task suites**: LIBERO-10, LIBERO-Goal, LIBERO-Spatial, LIBERO-Object (both with and without no-op filtering)
- **Action space**: 8-dimensional absolute joint positions (7 joint angles + 1 gripper), action chunks of 8 steps
- **Proprioceptive state**: 8-dimensional (7 joint positions + 1 gripper width)

## Repository Structure

```
├── prismatic/                  # Core model code (VLA, backbones, training)
│   └── vla/
│       ├── constants.py        # Platform-specific constants (LIBERO/ALOHA/Bridge)
│       └── datasets/rlds/oxe/
│           ├── configs.py      # Dataset configurations (humanized entries added)
│           ├── transforms.py   # Data transforms for each dataset
│           └── materialize.py  # Dataset materialization utilities
├── vla-scripts/                # Fine-tuning and deployment scripts
├── experiments/robot/libero/   # LIBERO evaluation and data processing
│   └── A_npz_to_hdf5.py       # Convert humanized NPZ trajectories to HDF5
├── rlds_dataset_builder/       # RLDS dataset builders for TensorFlow Datasets
│   ├── LIBERO_10_humanized/    # Builder for humanized LIBERO-10
│   ├── LIBERO_Goal_humanized/  # Builder for humanized LIBERO-Goal
│   ├── LIBERO_Object_humanized/# Builder for humanized LIBERO-Object
│   └── LIBERO_Spatial_humanized/# Builder for humanized LIBERO-Spatial
├── LIBERO/                     # LIBERO benchmark (code only, datasets excluded)
└── modified_libero_rlds/       # Modified RLDS metadata (not tracked, local only)
```

## System Requirements

**Inference:**
- 1 GPU with ~16 GB VRAM for LIBERO simulation benchmark tasks

**Training:**
- 1–8 GPUs with 27–80 GB VRAM, depending on the desired training setup (bfloat16). See the [original project FAQ](https://openvla-oft.github.io/#train-compute) for details.

## Quick Start

1. Set up a conda environment (see [SETUP.md](SETUP.md)).

2. Fine-tune on a humanized dataset:
```bash
# See vla-scripts/finetune.py and LIBERO.md for full training instructions
```

3. Evaluate on LIBERO:
```bash
# See experiments/robot/libero/run_libero_eval.py
```

## Data Pipeline

1. **Record humanized joint angle trajectories** (NPZ format)
2. **Replay in LIBERO simulation** to render images and compute states → cleaned HDF5
3. **Convert HDF5 to RLDS** using the builders in `rlds_dataset_builder/`
4. **Fine-tune OpenVLA-OFT** on the resulting RLDS datasets

## Installation

See [SETUP.md](SETUP.md) for conda environment setup instructions.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning and evaluation on LIBERO simulation benchmarks.

See [ALOHA.md](ALOHA.md) for fine-tuning and evaluation on real-world ALOHA robot tasks.

## Acknowledgments

This project is built on top of [OpenVLA-OFT](https://github.com/moojink/openvla-oft) by Kim et al.

## Citation

If you use this code in your work, please cite the original OpenVLA-OFT paper:

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
