# DiffKD-Yolo

DiffKD-Yolo is a **knowledge distillation (KD)** project for **low-light object detection** built on the **Ultralytics YOLO** ecosystem. A frozen **low-light image enhancement (LLIE) teacher** (default: Retinexformer) guides a **YOLO student** through **feature distillation** with the **DiffKD** module.

## Overview

Main training pipeline:

1. Load the **YOLO student** from a checkpoint or default configuration.
2. Build the **teacher** from the config and load its checkpoint (frozen).
3. Hook feature layers between the teacher and the student.
4. Compute **feature loss** with `DiffKD` to bring the student closer to the teacher.
5. Train the detector using the custom trainer in `engine/`.

## Features

- Detection training with **knowledge distillation**.
- All experiment settings (teacher, hooked layers, loss weights, training) in one YAML config.
- Self-contained teacher implementations in `models/teachers/` (no external repo imports).
- Integrated **DiffKD** module in `models/modules/diffkd.py`.
- Compatible with the **Ultralytics** training workflow.

## Project Structure

```text
DiffKD-Yolo/
├── configs/
│   └── distill/
│       └── retinexformer_yolov10s.yaml   # Experiment config
├── engine/
│   ├── distill.py                        # Feature hooks + DiffKD feature loss
│   └── trainer.py                        # Custom DetectionTrainer
├── models/
│   ├── modules/
│   │   ├── diffkd.py                     # DiffKD module
│   │   ├── feature_kd.py                 # Plain feature KD (adapter + MSE)
│   │   └── lce.py                        # Light Condition Encoder (from Salience-DETR)
│   └── teachers/
│       ├── __init__.py                   # Teacher registry: build_teacher(type, **args)
│       ├── retinexformer.py              # Retinexformer (ICCV 2023)
│       ├── irformer.py                   # IRFormer
│       └── mbllen.py                     # MBLLEN
├── tools/
│   ├── train.py                          # Training entry point
│   └── inspect_model.py                  # Print module names / output channels
├── utils/
│   └── config.py                         # YAML loading + CLI overrides
├── pretrained/                           # Teacher checkpoints (not tracked)
├── data/                                 # Dataset configs / data
└── experiments/
    └── runs/                             # Training outputs (not tracked)
```

## Requirements

- Python 3.10+ recommended
- PyTorch
- Ultralytics
- NumPy, PyYAML
- einops (only for the IRFormer teacher)

If you are using CUDA, install the PyTorch build that matches your GPU and driver setup.

## Installation

```bash
git clone https://github.com/VAL-Lab-HY/DiffKD-Yolo.git
cd DiffKD-Yolo
conda create -n diffkd python=3.10 -y && conda activate diffkd

# PyTorch matching your CUDA version (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the Retinexformer LOL-v2-real checkpoint from the [official repo](https://github.com/caiyuanhao1998/Retinexformer) and place it at `pretrained/LOL_v2_real.pth`.

## Dataset Setup

The dataset config uses the Ultralytics format, set via `train.data` in the experiment config:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names: ["class_name"]
```

## Training

Run from the repository root:

```bash
python tools/train.py configs/distill/retinexformer_yolov10s.yaml
```

Override any config entry with dotted `key=value` pairs:

```bash
python tools/train.py configs/distill/retinexformer_yolov10s.yaml --opts train.epochs=100 train.batch=8 distill.loss_weight=0.2
```

The resolved config is saved to `<save_dir>/config.yaml` for reproducibility.

## Configuration

| Section | Description |
|---|---|
| `student.model` | Student checkpoint or YAML (e.g. `yolov10s.pt`) |
| `teacher.type` | Key in `models/teachers/__init__.py:TEACHERS` |
| `teacher.checkpoint` | Teacher weights (BasicSR `params`, `state_dict`, raw state dict, or Ultralytics checkpoint) |
| `teacher.input_size` | Teacher input is resized to `(S, S)`; omit to feed the student-sized batch |
| `teacher.args` | Constructor arguments of the teacher |
| `distill.method` | `diffkd` (default) or `feature` (FitNets-style 1x1 adapter + MSE, `models/modules/feature_kd.py`) |
| `distill.loss_weight` | Global weight of the KD loss |
| `distill.layers` | List of `{teacher, student, teacher_channels?, student_channels?, weight}` hook pairs |
| `train` | Passed verbatim to Ultralytics (`data`, `epochs`, `batch`, `imgsz`, `project`, `name`, ...) |

Use `tools/inspect_model.py` to find module names and channels when choosing `distill.layers`:

```bash
python tools/inspect_model.py --model yolov10s.pt --filter model.4
```

To add a teacher, put its implementation in `models/teachers/` and register it in `TEACHERS`.

## Author

- Full name: Nguyen Van Dat
- Email: nvdat1601@gmai.com
