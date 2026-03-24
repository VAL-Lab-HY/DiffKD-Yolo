import argparse


def get_args():
    parser = argparse.ArgumentParser(description="KD Training: IRFormer → YOLOv10s")

    # Paths
    parser.add_argument("--data", type=str, required=True,  help="Path to data.yaml")
    parser.add_argument("--teacher", type=str, required=True,  help="IRFormer checkpoint (.pth / .pt)")
    parser.add_argument("--model", type=str, default="yolov10s.pt", help="Student weights")
    parser.add_argument("--project", type=str, default="runs/diffkd")
    parser.add_argument("--name", type=str, default="run1")

    # Training
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--batch", type=int,   default=32)
    parser.add_argument("--imgsz", type=int,   default=640)
    parser.add_argument("--device", type=str,   default="0")
    parser.add_argument("--workers", type=int,   default=8)

    # Optimizer
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--optimizer", type=str,   default="auto",
                        choices=["auto", "SGD", "AdamW", "Adam"])

    # KD
    parser.add_argument("--teacher-name", type=str,   default="irformer")
    parser.add_argument("--student-name", type=str,   default="yolov10s")
    parser.add_argument("--kd-loss-weight", type=float, default=0.5)
    parser.add_argument("--ae-channels", type=int,   default=16)
    parser.add_argument("--kd-tau", type=float, default=1.0)
    parser.add_argument("--use-ae", action="store_true", default=True)

    # Misc
    parser.add_argument("--resume", type=str,  default="")
    parser.add_argument("--save-period", type=int,  default=5)
    parser.add_argument("--plots", action="store_true", default=True)

    return parser.parse_args()


def build_overrides(args) -> dict:
    """Chuyển argparse namespace → dict overrides cho DetectionTrainer."""
    overrides = {
        # ── Paths & model ──────────────────────────────────────────
        "model":   args.model,
        "data":    args.data,
        "project": args.project,
        "name":    args.name,

        # ── Training ───────────────────────────────────────────────
        "epochs":         args.epochs,
        "batch":          args.batch,
        "imgsz":          args.imgsz,
        "device":         args.device,
        "workers":        args.workers,

        # ── Optimizer ──────────────────────────────────────────────
        "lr0":            args.lr0,
        "weight_decay":   args.weight_decay,
        "momentum":       args.momentum,
        "warmup_epochs":  args.warmup_epochs,
        "optimizer":      args.optimizer,

        # ── KD (BaseTrainer.pop() những key này) ───────────────────
        "teacher":        args.teacher,           # str path → BaseTrainer load
        "teacher_name":   args.teacher_name,      # key trong KD_MODULES
        "student_name":   args.student_name,      # key trong KD_MODULES
        "kd_loss_weight": args.kd_loss_weight,
        "kd_loss_kwargs": {
            "ae_channels": args.ae_channels,
            "use_ae":      args.use_ae,
            "tau":         args.kd_tau,
        },

        # ── Misc ───────────────────────────────────────────────────
        "resume":      args.resume or False,
        "save_period": args.save_period,
        "plots":       args.plots,
        "save":        True,
    }
    # Bỏ resume nếu không truyền
    if not overrides["resume"]:
        overrides.pop("resume")
    return overrides