"""Visualize one image before/after distillation training or a KD transfer module."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to one input image")
    parser.add_argument("--mode", choices=("transfer", "training"), default="transfer")
    parser.add_argument("--config", default=str(ROOT / "configs/distill/yolo26m_yolo26s_p345_feature.yaml"))
    parser.add_argument("--transfer-checkpoint", help="transfer_last.pt or transfer_best.pt from training")
    parser.add_argument("--student", help="Student YOLO checkpoint (random-transfer demo only)")
    parser.add_argument("--before-student", help="YOLO weights before KD training; defaults to config student.model")
    parser.add_argument("--after-student", help="YOLO best.pt/last.pt after KD training (training mode)")
    parser.add_argument("--layers", nargs="+", help="Student modules to compare, e.g. model.4 model.6 model.8 (training mode)")
    parser.add_argument("--teacher", help="Override teacher checkpoint path")
    parser.add_argument("--allow-random-transfer", action="store_true", help="Explicitly allow an untrained adapter demo")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, or mps")
    parser.add_argument("--output", default="experiments/feature_maps")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity in overlays")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "training":
        if not args.after_student and not args.transfer_checkpoint:
            parser.error("Training comparison requires --after-student best.pt or --transfer-checkpoint.")
        if args.after_student and args.transfer_checkpoint:
            parser.error("Choose --after-student or --transfer-checkpoint, not both.")
        if args.student or args.allow_random_transfer:
            parser.error("Training comparison uses --before-student and --after-student; omit demo arguments.")
    elif args.before_student or args.after_student or args.layers:
        parser.error("--before-student, --after-student and --layers require --mode training.")
    elif not args.transfer_checkpoint and not args.allow_random_transfer:
        parser.error("Pass --transfer-checkpoint for learned transfer, or --allow-random-transfer for a demo.")
    if args.transfer_checkpoint and args.student:
        parser.error("--transfer-checkpoint already contains the matching student; omit --student.")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.imgsz is not None and args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    return args


def capture(model, names, image):
    """Capture exact configured module outputs without fusing or changing model mode."""
    import torch

    modules = dict(model.named_modules())
    missing = sorted(set(names) - modules.keys())
    if missing:
        raise ValueError(f"Modules not found: {missing}. Inspect the model with tools/inspect_model.py.")
    features, handles = {}, []

    def hook(name):
        def save(module, inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError(f"{name} must output a BCHW tensor")
            features[name] = output.detach().clone()
        return save

    try:
        for name in dict.fromkeys(names):
            handles.append(modules[name].register_forward_hook(hook(name)))
        model(image)
    finally:
        for handle in handles:
            handle.remove()
    if set(names) - features.keys():
        raise ValueError("Some configured modules did not run on this image")
    return features


def activation(feature):
    """Mean absolute activation across channels; avoids signed-channel cancellation."""
    import numpy as np

    value = feature[0].float().abs().mean(0).cpu().numpy()
    if not np.isfinite(value).all():
        raise ValueError("Feature map contains NaN or infinity")
    return value


def panel(image, title, shape):
    import cv2
    import numpy as np

    strip = np.full((64, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(strip, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(strip, shape, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate((strip, image), axis=0)


def write_image(path, image):
    import cv2

    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def visualize_training(args, cfg):
    """Compare actual detector activations across checkpoints; no KD adapter is needed."""
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    before_path = args.before_student or cfg["student"]["model"]
    before_model = YOLO(before_path).model.to(args.device).float().eval()
    if args.transfer_checkpoint:
        ckpt = torch.load(args.transfer_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "kd_modules" not in ckpt:
            raise ValueError("Expected transfer_*.pt; use --after-student for a regular YOLO checkpoint.")
        after_model = ckpt["model"]
        default_layers = ckpt["layers"]
        default_size = ckpt["imgsz"]
    else:
        after_model = YOLO(args.after_student).model
        default_layers = cfg.get("distill", {}).get("layers", [])
        default_size = cfg.get("train", {}).get("imgsz", 640)
    after_model = after_model.to(args.device).float().eval()
    names = args.layers or list(dict.fromkeys(layer["student"] for layer in default_layers))
    if not names:
        raise ValueError("No student layers selected; pass --layers or set distill.layers in config.")
    stride = max(int(before_model.stride.max()), int(after_model.stride.max()), 32)
    size = math.ceil((args.imgsz or default_size) / stride) * stride
    original = cv2.imread(args.image)
    if original is None:
        raise ValueError(f"Cannot read image: {args.image}")
    image = LetterBox(new_shape=(size, size), auto=False, stride=stride)(image=original)
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(args.device).float() / 255.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, rows, arrays = [], [], {}
    with torch.inference_mode():
        before_features = capture(before_model, names, tensor)
        after_features = capture(after_model, names, tensor)
        write_image(output / "input.png", image)
        for i, name in enumerate(names):
            before, after = before_features[name], after_features[name]
            if before.shape != after.shape:
                raise ValueError(f"{name}: before {list(before.shape)} != after {list(after.shape)}. Use matching student architectures.")
            stride_y, stride_x = size / before.shape[2], size / before.shape[3]
            pyramid = f"P{round(math.log2(stride_y))}" if stride_y == stride_x and stride_y in (4, 8, 16, 32) else "feature"
            label = f"{pyramid}_{i}"
            values = {"before_distill": before, "after_distill": after}
            maps = {key: activation(value) for key, value in values.items()}
            scale = max(*(float(value.max()) for value in maps.values()), 1e-12)
            panels = [panel(image, f"{label} input", name)]
            for key, value in values.items():
                gray = np.uint8(np.clip(maps[key] / scale, 0, 1) * 255)
                heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
                overlay = cv2.addWeighted(image, 1 - args.alpha, heat, args.alpha, 0)
                panels.append(panel(overlay, f"{label} {key} (shared scale)", str(list(value.shape))))
                write_image(output / f"{label}_{key}_heatmap.png", heat)
                write_image(output / f"{label}_{key}_overlay.png", overlay)
                arrays[f"{label}_{key}"] = value.float().cpu().numpy()
                arrays[f"{label}_{key}_activation"] = maps[key]
            row = np.concatenate(panels, axis=1)
            write_image(output / f"{label}_comparison.png", row)
            rows.append(row)
            records.append({"label": label, "student_layer": name, "shape": list(before.shape), "shared_scale_max": scale})
    write_image(output / "comparison.png", np.concatenate(rows, axis=0))
    np.savez_compressed(output / "features.npz", **arrays)
    metadata = {
        "mode": "training", "config": str(Path(args.config).resolve()),
        "image": str(Path(args.image).resolve()), "before_student": str(before_path),
        "after_student": args.after_student or args.transfer_checkpoint,
        "teacher_type_from_config": cfg.get("teacher", {}).get("type"),
        "input_size": size, "layers": records, "seed": args.seed,
        "reduction": "mean(abs(feature), channel); shared before/after scale per layer",
        "note": "Actual student activations before/after training, including any LCE saved in the checkpoint. Teacher and KD adapters are not executed.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved student before/after distillation comparison to {output.resolve()}")


def main():
    args = get_args()
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    from utils.config import load_config

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    if args.mode == "training":
        visualize_training(args, cfg)
        return
    from engine.distill import build_kd_module

    checkpoint = None
    if args.transfer_checkpoint:
        checkpoint = torch.load(args.transfer_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "kd_modules" not in checkpoint:
            raise ValueError("Expected a transfer_*.pt checkpoint. Regular best.pt/last.pt do not contain KD adapters.")
    layers = checkpoint["layers"] if checkpoint else cfg["distill"]["layers"]
    method = checkpoint["method"] if checkpoint else cfg["distill"].get("method", "diffkd")
    kd_args = (checkpoint.get("kd_args") if checkpoint else cfg["distill"].get("args")) or {}
    teacher_cfg = checkpoint["teacher_cfg"] if checkpoint else cfg["teacher"]
    teacher_path = args.teacher or (checkpoint["teacher_path"] if checkpoint else teacher_cfg["checkpoint"])
    if teacher_cfg.get("type"):
        from models.teachers import build_teacher
        from tools.inspect_teacher import load_checkpoint

        teacher = build_teacher(teacher_cfg["type"], **teacher_cfg.get("args", {}))
        load_checkpoint(teacher, teacher_path)
    else:
        teacher = YOLO(teacher_path).model
    teacher = teacher.to(args.device).float().eval()
    student = checkpoint["model"] if checkpoint else YOLO(args.student or cfg["student"]["model"]).model
    if checkpoint is None and cfg["student"].get("lce"):
        from models.modules.lce import attach_lce, find_lce

        if find_lce(student) is None:
            lce_cfg = {key: value for key, value in cfg["student"]["lce"].items() if key != "loss_weight"}
            attach_lce(student, **lce_cfg)
    student = student.to(args.device).float().eval()
    stride = max(int(student.stride.max()), int(teacher.stride.max()) if hasattr(teacher, "stride") else 32, 32)
    requested_size = args.imgsz or (checkpoint["imgsz"] if checkpoint else cfg.get("train", {}).get("imgsz", 640))
    size = math.ceil(requested_size / stride) * stride
    original = cv2.imread(args.image)
    if original is None:
        raise ValueError(f"Cannot read image: {args.image}")
    image = LetterBox(new_shape=(size, size), auto=False, stride=stride)(image=original)
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(args.device).float() / 255.0
    teacher_size = checkpoint["teacher_input_size"] if checkpoint else teacher_cfg.get("input_size")
    teacher_input = tensor
    if teacher_size:
        teacher_input = F.interpolate(tensor, size=(teacher_size, teacher_size), mode="bilinear", align_corners=False, antialias=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = "TRAINED TRANSFER" if checkpoint else "RANDOM TRANSFER - UNTRAINED"
    print(status)
    records, rows, arrays = [], [], {}
    with torch.inference_mode():
        student_features = capture(student, [layer["student"] for layer in layers], tensor)
        teacher_features = capture(teacher, [layer["teacher"] for layer in layers], teacher_input)
        kd_modules = torch.nn.ModuleList([
            build_kd_module(
                method, student_features[layer["student"]].shape[1], teacher_features[layer["teacher"]].shape[1], **kd_args
            )
            for layer in layers
        ]).to(args.device).float().eval()
        if checkpoint:
            kd_modules.load_state_dict(checkpoint["kd_modules"], strict=True)
        write_image(output / "input.png", image)
        for i, (layer, kd) in enumerate(zip(layers, kd_modules)):
            before = student_features[layer["student"]]
            teacher_raw = teacher_features[layer["teacher"]]
            target = teacher_raw
            if target.shape[2:] != before.shape[2:]:
                target = F.interpolate(target, size=before.shape[2:], mode="bilinear", align_corners=False, antialias=True)
            after, target, loss, _ = kd(before, target)
            stride_y, stride_x = size / before.shape[2], size / before.shape[3]
            pyramid = f"P{round(math.log2(stride_y))}" if stride_y == stride_x and stride_y in (4, 8, 16, 32) else "feature"
            label = f"{pyramid}_{i}"
            values = {"before": before, "after": after, "target": target}
            maps = {key: activation(value) for key, value in values.items()}
            # Shared after/target range: these tensors are in the same transfer space.
            shared_max = max(float(maps["after"].max()), float(maps["target"].max()), 1e-12)
            panels = [panel(image, f"{label} input", status)]
            for key, value in values.items():
                scale = max(float(maps[key].max()), 1e-12) if key == "before" else shared_max
                gray = np.uint8(np.clip(maps[key] / scale, 0, 1) * 255)
                heat = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
                overlay = cv2.addWeighted(image, 1 - args.alpha, heat, args.alpha, 0)
                title = f"{label} {key}" + (" (own scale)" if key == "before" else " (shared scale)")
                panels.append(panel(overlay, title, str(list(value.shape))))
                write_image(output / f"{label}_{key}_heatmap.png", heat)
                write_image(output / f"{label}_{key}_overlay.png", overlay)
                arrays[f"{label}_{key}"] = value.float().cpu().numpy()
                arrays[f"{label}_{key}_activation"] = maps[key]
            row = np.concatenate(panels, axis=1)
            write_image(output / f"{label}_comparison.png", row)
            rows.append(row)
            records.append({
                "label": label, "student_layer": layer["student"], "teacher_layer": layer["teacher"],
                "shapes": {key: list(value.shape) for key, value in values.items()},
                "teacher_raw_shape": list(teacher_raw.shape), "kd_loss": float(loss),
                "after_target_mse": float(F.mse_loss(after.float(), target.float())),
                "before_max": float(maps["before"].max()), "after_target_scale_max": shared_max,
            })
    write_image(output / "comparison.png", np.concatenate(rows, axis=0))
    np.savez_compressed(output / "features.npz", **arrays)
    metadata = {
        "image": str(Path(args.image).resolve()), "transfer_status": status, "method": method,
        "transfer_checkpoint": args.transfer_checkpoint, "teacher": str(teacher_path),
        "input_size": size, "seed": args.seed, "layers": records,
        "reduction": "mean(abs(feature), channel)",
        "target": "Teacher target after FeatureKD normalization or DiffKD autoencoder and spatial alignment",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved comparison.png, per-layer heatmaps/overlays, features.npz and metadata.json to {output.resolve()}")


if __name__ == "__main__":
    main()
