import argparse
import logging

from engine.trainer import DetectionTrainer

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="KD Training")

    parser.add_argument("--model", type=str, default="yolov10s.pt")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher model checkpoint.")
    parser.add_argument("--data", type=str, default="data/coco.yaml", help="Path to dataset config file.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)

    return parser.parse_args()


def build_overrides(args) -> dict:
    """Chuyển argparse namespace → dict overrides cho DetectionTrainer."""
    overrides = {
        "model": args.model,    
        "teacher": args.teacher,
        "data": args.data,    
        "epochs": args.epochs,
        "batch": args.batch,
    }
    return overrides


def main():
    args = get_args()
    overrides = build_overrides(args)

    logger.info("=" * 60)
    logger.info("KD Training")
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