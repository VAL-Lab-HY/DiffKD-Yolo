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
│   │   ├── fgd.py                        # FGD: Focal and Global KD (CVPR 2022)
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

For the local ExDark dataset, convert the original `ExDark/` images and
`ExDark_Annno/` bbGt labels with:

```bash
python tools/prepare_exdark.py
```

This preserves the originals and copies images into `data/exdark/images/{train,val,test}`,
with normalized YOLO labels in `data/exdark/labels/{train,val,test}`. Class IDs 0–11
follow Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table.
The checked-in `data/exdark_imageclasslist.txt` comes from the
[official ExDark experimental split](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/imageclasslist.txt).
The converter matches filenames ignoring case, clips boxes to image bounds, and
records split counts and corrected boxes in `data/exdark/conversion_report.json`.
The official split contains 3,000 training, 1,800 validation, and 2,563 test images.
Conversion preserves all 23,710 boxes, clipping 50 boxes at image boundaries.
One image with EXIF rotation is exported losslessly as PNG without the rotation
metadata so its raw pixels remain aligned with the original annotations.
It refuses to overwrite an existing destination. The generated `data.yaml` uses
an absolute dataset path; update `path` if moving the dataset to another machine.
The Retinexformer config uses `data/exdark/data.yaml`.

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
| `distill.method` | `diffkd` (default), `feature` (FitNets-style 1x1 adapter + MSE, `models/modules/feature_kd.py`) or `fgd` (Focal and Global KD, `models/modules/fgd.py`) |
| `distill.args` | Optional constructor arguments of the KD module (e.g. FGD `temp`, `alpha_fgd`, `beta_fgd`, `gamma_fgd`, `lambda_fgd`) |
| `distill.loss_weight` | Global weight of the KD loss |
| `distill.layers` | List of `{teacher, student, teacher_channels?, student_channels?, weight}` hook pairs |
| `train` | Passed verbatim to Ultralytics (`data`, `epochs`, `batch`, `imgsz`, `project`, `name`, ...) |

Use `tools/inspect_model.py` to find module names and channels when choosing `distill.layers`:

```bash
python tools/inspect_model.py --model yolov10s.pt --filter model.4
```

To add a teacher, put its implementation in `models/teachers/` and register it in `TEACHERS`.

## FGD (Focal and Global Knowledge Distillation)

`distill.method: fgd` ports [FGD](https://arxiv.org/abs/2111.11837) (Yang et al., CVPR 2022) from
the [official mmdet code](https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py) to plain PyTorch:

- **Focal**: feature MSE weighted by the teacher's spatial / channel attention, split into a
  foreground part (GT boxes, each weighted by 1 / box area) and a background part (`alpha_fgd`,
  `beta_fgd`), plus an L1 loss between student and teacher attention maps (`gamma_fgd`).
- **Global**: MSE between GcBlock-enhanced student and teacher features (`lambda_fgd`).

The GT boxes come from the Ultralytics batch (`bboxes` + `batch_idx`). FGD losses are per-image
means, so the trainer multiplies them by the batch size to match the batch-summed Ultralytics
detection loss. With `distill.darkness_weight`, the LCE darkness map re-weights the fg / bg masks.

```bash
python tools/train.py configs/distill/yolo26m_yolo26s_p345_fgd.yaml
```

The defaults follow the paper's RetinaNet config; tune `distill.args` / `distill.loss_weight` if the
FGD loss dominates or vanishes next to the detection loss.

## Visualize YOLOv10s before/after Retinexformer distillation

Use `--mode training` to compare the actual YOLO student before and after KD
training on the same image. This mode accepts regular Ultralytics checkpoints;
it does not require the teacher checkpoint or saved KD adapters.

```bash
python tools/visualize_features.py \
  --mode training \
  --config configs/distill/retinexformer_yolov10s.yaml \
  --image /path/to/image.jpg \
  --before-student yolov10s.pt \
  --after-student experiments/runs/retinexformer_yolov10s_320_lce/weights/best.pt \
  --layers model.4 model.6 model.8 \
  --device cpu \
  --output experiments/feature_maps_retinexformer
```

Here `model.4`, `model.6`, and `model.8` select backbone P3/P4/P5 in YOLOv10s.
If `--layers` is omitted, the script uses the student layers in `distill.layers`:
the current Retinexformer config selects **P2/P3/P5**, not P3/P4/P5. P4 may still
change through training even though it has no direct KD loss in that config.
`--before-student` defaults to `student.model` in the config; pass the actual
initial weights used for your run if they differ. `--after-student` must point to
the detector trained with Retinexformer, not the Retinexformer `.pth` file.
Config selects modules and input size; it cannot establish how supplied weights
were trained.

Each row of `comparison.png` contains input, before-distillation student, and
after-distillation student. Before/after share a heatmap scale per layer. The
script also writes individual heatmaps, overlays, full tensors in `features.npz`,
and `metadata.json`. Checkpoints containing the project's LCE retain their
learned modulation; the comparison therefore includes the full training change.
You can also use `--transfer-checkpoint transfer_best.pt` instead of `--after-student`.

For the separate **before/after adapter** view, use `--mode transfer` (default)
with the same Retinexformer config and a matching `--transfer-checkpoint`.
This loads Retinexformer via `teacher.type` / `teacher.args` and supports BasicSR
`params`, `params_ema`, `state_dict`, or raw teacher state dictionaries.

## Visualize P3 / P4 / P5 feature transfer

For YOLO26m → YOLO26s, the three neck outputs feeding Detect are `model.16`
(P3/8), `model.19` (P4/16), and `model.22` (P5/32). At input size 640 their
grids are 80×80, 40×40, and 20×20. Train all three using:

```bash
python tools/train.py configs/distill/yolo26m_yolo26s_p345_feature.yaml
```

Training now also saves `weights/transfer_last.pt` and `weights/transfer_best.pt`.
Each contains the exact student, KD adapter weights and layer mapping together.
These use the current training student, whereas Ultralytics `best.pt` normally
uses EMA weights. Existing regular `best.pt` / `last.pt` files do not contain
the transfer modules and cannot reconstruct the learned adapter.

Pass any local image to visualize the learned transfer:

```bash
python tools/visualize_features.py \
  --image /path/to/image.jpg \
  --transfer-checkpoint experiments/runs/yolo26m_yolo26s_p345_feature/weights/transfer_best.pt \
  --device cpu \
  --output experiments/feature_maps
```

Use `--teacher /path/to/teacher.pt` if the teacher checkpoint has moved. Use
`--device cuda:0` for CUDA. The teacher must match the teacher used in training.
The transfer checkpoint supplies the student, method, input size and layer mapping.
The original P3-only config remains available; its transfer checkpoints visualize
only its two configured P3 pairs.

To try the visualization immediately without training:

```bash
python tools/visualize_features.py \
  --image /path/to/image.jpg \
  --allow-random-transfer \
  --output experiments/feature_maps_demo
```

This demo loads the configured YOLO checkpoints and initializes transfer modules
randomly. Its figures are labeled **RANDOM TRANSFER - UNTRAINED** and do not show
learned knowledge transfer.

Outputs include `comparison.png` (one row per layer, columns: input, student
before transfer, student after transfer, teacher target), individual heatmaps and
overlays, `features.npz` with full BCHW tensors and reduced activation maps, and
`metadata.json` with shapes and after/target MSE. The image is letterboxed and
converted from BGR to RGB in [0, 1] before inference; overlays use this same padded
image for spatial alignment. Hooks are removed after each forward.

“Before/after” means **before/after the KD transfer module on the same forward**,
not before/after training. The teacher target follows the existing KD code:
spatial resizing if needed, channel normalization for FeatureKD, or the teacher
autoencoder for DiffKD. After/target share one heatmap scale; the original student
uses its own scale because its channels and feature space may differ. Heatmaps
reduce channels with `mean(abs(feature))`; they are activation visualizations,
not class-specific saliency maps. DiffKD sampling uses `--seed` (default 0).

## Author

- Full name: Nguyen Van Dat
- Email: nvdat1601@gmai.com
