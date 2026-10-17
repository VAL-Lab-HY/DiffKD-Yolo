import argparse

import torch

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(description="In các module của model và output size tương ứng")

    parser.add_argument("--model", type=str, default="yolo26m.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--depth", type=int, default=2, help="Độ sâu tên module, vd 2 -> 'model.4'. 0 = in tất cả.")
    parser.add_argument("--filter", type=str, default=None, help="Chỉ in module có tên chứa chuỗi này, vd 'model.6'.")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def channels_of(out):
    """Trả về số channel của output (tensor, list/tuple hoặc dict lồng nhau)."""
    if isinstance(out, torch.Tensor):
        return out.shape[1] if out.dim() > 1 else out.shape[0]
    if isinstance(out, (list, tuple)):
        return [channels_of(o) for o in out]
    if isinstance(out, dict):
        return {k: channels_of(v) for k, v in out.items()}
    return type(out).__name__


def main():
    args = get_args()
    model = YOLO(args.model).model.float().eval().to(args.device)

    records = []
    handles = []
    for name, module in model.named_modules():
        if not name:
            continue
        if args.depth > 0 and len(name.split(".")) > args.depth:
            continue
        if args.filter and args.filter not in name:
            continue

        def hook(m, inp, out, name=name):
            n_params = sum(p.numel() for p in m.parameters())
            records.append((name, type(m).__name__, channels_of(out), n_params))

        handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device))

    for h in handles:
        h.remove()

    print(f"Model: {args.model} | input: [1, 3, {args.imgsz}, {args.imgsz}]\n")
    print(f"{'name':<40} {'type':<20} {'params':>12}  out_channels")
    print("-" * 90)
    for name, typ, channels, n_params in records:
        print(f"{name:<40} {typ:<20} {n_params:>12,}  {channels}")


if __name__ == "__main__":
    main()
