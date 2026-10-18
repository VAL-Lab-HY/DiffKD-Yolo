import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.trainer import DetectionTrainer  # noqa: E402
from utils.config import apply_overrides, load_config, save_config  # noqa: E402

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="KD Training")
    parser.add_argument("config", type=str, help="Path to config file, e.g. configs/distill/retinexformer_yolov10s.yaml")
    parser.add_argument(
        "--opts",
        nargs="+",
        default=[],
        help="Override config entries with dotted key=value pairs, e.g. --opts train.epochs=100 train.batch=8",
    )
    return parser.parse_args()


def build_overrides(cfg) -> dict:
    """Config dict → overrides for DetectionTrainer."""
    return {
        "model": cfg["student"]["model"],
        "lce": cfg["student"].get("lce"),
        "teacher": cfg["teacher"],
        "distill": cfg.get("distill", {}),
        **cfg.get("train", {}),
    }


def main():
    args = get_args()
    cfg = apply_overrides(load_config(args.config), args.opts)

    logger.info("=" * 60)
    logger.info("KD Training")
    logger.info(f"Config   : {args.config}")
    logger.info(f"Teacher  : {cfg['teacher'].get('type')} ({cfg['teacher'].get('checkpoint')})")
    logger.info(f"Student  : {cfg['student']['model']}")
    logger.info("=" * 60)

    trainer = DetectionTrainer(overrides=build_overrides(cfg))
    save_config(cfg, Path(trainer.save_dir) / "config.yaml")
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
