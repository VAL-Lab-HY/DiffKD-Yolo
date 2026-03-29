import argparse
import logging

from engine.trainer import DetectionTrainer

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="KD Training: IRFormer → YOLOv10s")

    # KD
    parser.add_argument("--model", type=str, default="yolov10s.pt")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher model checkpoint.")

    return parser.parse_args()


def build_overrides(args) -> dict:
    """Chuyển argparse namespace → dict overrides cho DetectionTrainer."""
    overrides = {
        "model": args.model,             # str path → BaseTrainer load
        "teacher": args.teacher,           # str path → BaseTrainer load
    }
    # Bỏ resume nếu không truyền
    if not overrides["resume"]:
        overrides.pop("resume")
    return overrides


def main():
    args = get_args()
    overrides = build_overrides(args)

    logger.info("=" * 60)
    logger.info("KD Training  |  IRFormer → YOLOv10s")
    logger.info(f"Teacher  : {args.teacher}")
    logger.info(f"Student  : {args.model}")
    logger.info("=" * 60)

    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()