import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.teachers import build_teacher  # noqa: E402
from utils.config import load_config  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(description="Load checkpoint teacher, in các module và output size tương ứng")

    parser.add_argument("--config", type=str, default="configs/distill/retinexformer_yolov10s.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Mặc định lấy teacher.checkpoint trong config.")
    parser.add_argument("--imgsz", type=int, default=None, help="Mặc định lấy teacher.input_size trong config.")
    parser.add_argument("--depth", type=int, default=6, help="Độ sâu tên module, vd 4 -> 'body.0.denoiser.bottleneck'. 0 = in tất cả.")
    parser.add_argument("--filter", type=str, default=None, help="Chỉ in module có tên chứa chuỗi này, vd 'denoiser'.")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def load_checkpoint(model, path):
    """Load BasicSR (``params``), ``state_dict`` hoặc raw state_dict với strict=True."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params", "params_ema", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    model.load_state_dict(ckpt, strict=True)


def shape_of(out):
    """Trả về shape (bỏ batch) của output (tensor, list/tuple hoặc dict lồng nhau)."""
    if isinstance(out, torch.Tensor):
        return list(out.shape[1:])
    if isinstance(out, (list, tuple)):
        return [shape_of(o) for o in out]
    if isinstance(out, dict):
        return {k: shape_of(v) for k, v in out.items()}
    return type(out).__name__


def main():
    args = get_args()
    cfg = load_config(args.config)
    teacher_cfg = cfg["teacher"]
    checkpoint = args.checkpoint or teacher_cfg["checkpoint"]
    imgsz = args.imgsz or teacher_cfg.get("input_size") or 640
    kd_layers = {layer["teacher"] for layer in cfg.get("distill", {}).get("layers", [])}

    model = build_teacher(teacher_cfg["type"], **teacher_cfg.get("args", {}))
    load_checkpoint(model, checkpoint)
    model = model.float().eval().to(args.device)

    records = []
    handles = []
    for name, module in model.named_modules():
        if not name:
            continue
        if args.depth > 0 and len(name.split(".")) > args.depth and name not in kd_layers:
            continue
        if args.filter and args.filter not in name:
            continue

        def hook(m, inp, out, name=name):
            n_params = sum(p.numel() for p in m.parameters())
            records.append((name, type(m).__name__, shape_of(out), n_params))

        handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        out = model(torch.rand(1, 3, imgsz, imgsz, device=args.device))

    for h in handles:
        h.remove()

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Teacher: {teacher_cfg['type']} | checkpoint: {checkpoint} (loaded, strict=True)")
    print(f"Input: [1, 3, {imgsz}, {imgsz}] | output: {list(out.shape)} | params: {n_total:,}\n")
    print(f"{'name':<45} {'type':<28} {'params':>10}  out_shape (C, H, W)")
    print("-" * 110)
    for name, typ, shape, n_params in records:
        mark = "  ← KD" if name in kd_layers else ""
        print(f"{name:<45} {typ:<28} {n_params:>10,}  {shape}{mark}")

    missing = kd_layers - {name for name, *_ in records}
    if missing and not args.filter:
        print(f"\nWARNING: distill.layers không tồn tại trong teacher: {sorted(missing)}")


if __name__ == "__main__":
    main()
