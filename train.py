import logging

from utils.args import get_args, build_overrides
from engine.trainer import DetectionTrainer

logger = logging.getLogger(__name__)


def main():
    args = get_args()
    overrides = build_overrides(args)

    logger.info("=" * 60)
    logger.info("KD Training  |  IRFormer → YOLOv10s")
    logger.info(f"Teacher  : {args.teacher}")
    logger.info(f"Student  : {args.model}")
    logger.info(f"KD weight: {args.kd_loss_weight}  |  method: {args.kd_method}")
    logger.info("=" * 60)

    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()