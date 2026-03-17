"""
Training script: Knowledge Distillation
Teacher : IRFormer
Student : YOLOv10s
Task    : Object Detection (COCO / custom)
"""

import argparse
import logging
from pathlib import Path

from engine.trainer import YOLOV10KDTrainer

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='KD Training: IRFormer → YOLOv10s')

    # paths
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to dataset root')
    parser.add_argument('--teacher-ckpt', type=str, required=True, 
                        help='IRFormer pretrained weights')
    parser.add_argument('--student-ckpt', type=str, default='', 
                        help='YOLOv10s pretrained weights (optional)')
    parser.add_argument('--save-dir', type=str, default='runs/diffkd', 
                        help='Checkpoint & log output dir')

    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.937)

    # scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'onecycle'])
    parser.add_argument('--warmup-epochs',type=int, default=3)

    # KD
    parser.add_argument('--kd-method', type=str, default='diffkd')
    parser.add_argument('--ori-loss-weight', type=float, default=1.0)
    parser.add_argument('--kd-loss-weight', type=float, default=0.5)
    parser.add_argument('--ae-channels', type=int, default=16)
    parser.add_argument('--kd-tau',  type=float, default=1.0)

    # checkpoint
    parser.add_argument('--resume', type=str, default='', 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-period',type=int, default=1,  
                        help='Save checkpoint every N epochs')

    # logging
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--project', type=str, default='kd-irformer-yolo')
    parser.add_argument('--run-name', type=str, default='')

    return parser.parse_args()


def main():
    args = dict(
        # --- Paths ---
        model='yolov10s.pt',      # Student checkpoint
        data='my_data.yaml',       # File yaml dataset (train/test/nc/names)
        project='diffkd-irformer-yolo',
        name='run1',
        
        # --- Hyperparameters ---
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,                 # Index của GPU
        lr0=0.01,                 # Ultralytics dùng lr0 thay vì lr
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # --- KD Specific (Sẽ được truyền vào YOLOV10KDTrainer) ---
        teacher_ckpt='weights/irformer_pretrained.pt', # Path tới IRFormer
        kd_loss_weight=0.5,
        
        # --- Logging ---
        plots=True,               # Vẽ biểu đồ kết quả
        save=True,
    )

    # Khởi tạo trainer custom
    trainer = YOLOV10KDTrainer(overrides=args)
    
    # Bắt đầu huấn luyện
    # Ultralytics tự động gọi train_one_epoch, validate, save_checkpoint bên trong
    trainer.train()

if __name__ == '__main__':
    main()