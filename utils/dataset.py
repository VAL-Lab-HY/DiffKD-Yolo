"""
dataset/llvip.py
────────────────────────────────────────────────────────────────────────────
LLVIP Dataset loader cho 2 bước training:

  Bước 1 (distill) : LLVIPDistillDataset
    - Trả về cặp (rgb, ir) — không cần annotations
    - rgb → YOLO student input
    - ir  → IRFormer teacher input

  Bước 2 (finetune): LLVIPDetectDataset
    - Trả về (rgb, boxes, labels)
    - Chỉ dùng RGB + annotations
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

class NightAugment:
    """
    Augmentation nhẹ cho ảnh ban đêm.
    Tránh thay đổi brightness/contrast quá mạnh vì đây là nighttime images.
    """
    def __init__(self, img_size: int = 640, flip_prob: float = 0.5):
        self.img_size  = img_size
        self.flip_prob = flip_prob

    def __call__(self, rgb: Image.Image,
                 boxes: Optional[np.ndarray] = None
                 ) -> Tuple:
        """
        rgb   : PIL Image
        boxes : (N, 4) xyxy normalized [0,1] hoặc None
        """
        # 1. Resize
        rgb = TF.resize(rgb, (self.img_size, self.img_size))

        # 2. Random horizontal flip
        if random.random() < self.flip_prob:
            rgb = TF.hflip(rgb)
            if boxes is not None and len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]

        # 3. ToTensor + normalize [0,1]
        rgb = TF.to_tensor(rgb)   # (3, H, W), float32 [0,1]

        return rgb, boxes


class BasicTransform:
    """Resize + ToTensor, không augment — dùng cho val/test và IR images."""
    def __init__(self, img_size: int = 640):
        self.img_size = img_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = TF.resize(img, (self.img_size, self.img_size))
        return TF.to_tensor(img)   # (3, H, W) hoặc (1, H, W)


# ══════════════════════════════════════════════════════════════════════════════
#  BƯỚC 1: Distillation Dataset  (RGB + IR pairs, no annotations)
# ══════════════════════════════════════════════════════════════════════════════

class LLVIPDistillDataset(Dataset):
    """
    Dataset cho Bước 1 — DiffKD Distillation.

    Không cần annotations vì distillation chỉ dùng features,
    không dùng ground-truth boxes.

    Trả về:
        rgb : (3, H, W) float32 [0,1]  — nighttime RGB → YOLO student
        ir  : (3, H, W) float32 [0,1]  — IR thermal    → IRFormer teacher

    LLVIP folder structure:
        data_root/visible/train/*.jpg    ← RGB
        data_root/infrared/train/*.jpg   ← IR
    """

    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 640, augment: bool = True):
        self.img_size = img_size
        self.augment  = augment

        rgb_dir = Path(data_root) / 'visible'  / split
        ir_dir  = Path(data_root) / 'infrared' / split

        # Lấy danh sách file — LLVIP dùng cùng tên file cho RGB và IR
        rgb_files = sorted(rgb_dir.glob('*.jpg')) + sorted(rgb_dir.glob('*.png'))
        self.samples = []
        for rgb_path in rgb_files:
            ir_path = ir_dir / rgb_path.name
            if ir_path.exists():
                self.samples.append((rgb_path, ir_path))

        assert len(self.samples) > 0, \
            f"Không tìm thấy ảnh trong {rgb_dir}. Kiểm tra đường dẫn LLVIP."

        self.rgb_tf = NightAugment(img_size, flip_prob=0.5) if augment \
                      else BasicTransform(img_size)
        self.ir_tf  = BasicTransform(img_size)

        print(f"[LLVIPDistillDataset] {split}: {len(self.samples)} pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rgb_path, ir_path = self.samples[idx]

        rgb_img = Image.open(rgb_path).convert('RGB')
        ir_img  = Image.open(ir_path).convert('RGB')

        if self.augment:
            # Áp dụng cùng flip cho cả RGB lẫn IR
            flip = random.random() < 0.5
            rgb_img = TF.resize(rgb_img, (self.img_size, self.img_size))
            ir_img  = TF.resize(ir_img,  (self.img_size, self.img_size))
            if flip:
                rgb_img = TF.hflip(rgb_img)
                ir_img  = TF.hflip(ir_img)
            rgb = TF.to_tensor(rgb_img)
            ir  = TF.to_tensor(ir_img)
        else:
            rgb = self.rgb_tf(rgb_img)
            ir  = self.ir_tf(ir_img)

        return {'rgb': rgb, 'ir': ir}


# ══════════════════════════════════════════════════════════════════════════════
#  BƯỚC 2: Detection Fine-tune Dataset  (RGB + annotations)
# ══════════════════════════════════════════════════════════════════════════════

class LLVIPDetectDataset(Dataset):
    """
    Dataset cho Bước 2 — Detection Fine-tuning.

    Chỉ dùng RGB images + COCO-format annotations.
    IR images KHÔNG cần thiết ở bước này.

    LLVIP annotations (COCO format):
        {
          "images": [{"id": 1, "file_name": "010001.jpg", ...}],
          "annotations": [{"image_id": 1, "bbox": [x,y,w,h], "category_id": 1}],
          "categories": [{"id": 1, "name": "person"}]
        }

    Trả về dict:
        rgb    : (3, H, W) float32
        boxes  : (N, 4) xyxy normalized [0,1]
        labels : (N,) long — tất cả 0 (pedestrian)
        mask   : (max_gt,) bool
        img_id : int
    """

    MAX_GT = 100   # max ground-truth per image (padding)

    def __init__(self, data_root: str, split: str = 'train',
                 img_size: int = 640, augment: bool = True):
        self.img_size = img_size
        self.augment  = augment
        self.rgb_dir  = Path(data_root) / 'visible' / split
        ann_file      = Path(data_root) / 'Annotations' / f'{split}.json'

        assert ann_file.exists(), f"Annotation file not found: {ann_file}"

        with open(ann_file) as f:
            coco = json.load(f)

        # Build image id → file name map
        self.id2file = {img['id']: img['file_name'] for img in coco['images']}

        # Build image id → list of annotations
        self.id2anns: Dict[int, List] = {img['id']: [] for img in coco['images']}
        for ann in coco['annotations']:
            self.id2anns[ann['image_id']].append(ann)

        # Chỉ giữ images có ít nhất 1 annotation
        self.img_ids = [
            img_id for img_id in self.id2file
            if len(self.id2anns[img_id]) > 0
        ]

        # Image sizes (cần để convert bbox)
        self.id2size = {img['id']: (img['width'], img['height'])
                        for img in coco['images']}

        self.augment_fn = NightAugment(img_size, flip_prob=0.5) if augment \
                          else None
        self.basic_tf   = BasicTransform(img_size)

        print(f"[LLVIPDetectDataset] {split}: {len(self.img_ids)} images")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id   = self.img_ids[idx]
        filename = self.id2file[img_id]
        W, H     = self.id2size[img_id]

        # Load RGB image
        img_path = self.rgb_dir / filename
        rgb_img  = Image.open(img_path).convert('RGB')

        # Parse annotations → xyxy normalized
        anns  = self.id2anns[img_id]
        boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']   # COCO format: xywh pixels
            # Convert → xyxy normalized
            x1 = x / W;       y1 = y / H
            x2 = (x + w) / W; y2 = (y + h) / H
            x1, x2 = max(0., x1), min(1., x2)
            y1, y2 = max(0., y1), min(1., y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        boxes = np.array(boxes, dtype=np.float32) if boxes else \
                np.zeros((0, 4), dtype=np.float32)

        # Augmentation
        if self.augment and self.augment_fn is not None:
            rgb, boxes = self.augment_fn(rgb_img, boxes)
        else:
            rgb    = self.basic_tf(rgb_img)
            # boxes không đổi

        # Pad boxes → (MAX_GT, 4)
        n     = len(boxes)
        n_pad = self.MAX_GT
        boxes_pad  = np.zeros((n_pad, 4), dtype=np.float32)
        labels_pad = np.zeros(n_pad, dtype=np.int64)
        mask_pad   = np.zeros(n_pad, dtype=bool)

        if n > 0:
            n_fill = min(n, n_pad)
            boxes_pad[:n_fill]  = boxes[:n_fill]
            labels_pad[:n_fill] = 0       # class 0 = pedestrian
            mask_pad[:n_fill]   = True

        return {
            'rgb':    rgb,                                          # (3,H,W)
            'boxes':  torch.from_numpy(boxes_pad),                  # (MAX_GT,4)
            'labels': torch.from_numpy(labels_pad),                 # (MAX_GT,)
            'mask':   torch.from_numpy(mask_pad),                   # (MAX_GT,)
            'img_id': torch.tensor(img_id, dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def collate_distill(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate cho LLVIPDistillDataset."""
    return {
        'rgb': torch.stack([b['rgb'] for b in batch]),
        'ir':  torch.stack([b['ir']  for b in batch]),
    }


def collate_detect(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate cho LLVIPDetectDataset."""
    return {
        'rgb':    torch.stack([b['rgb']    for b in batch]),
        'boxes':  torch.stack([b['boxes']  for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'mask':   torch.stack([b['mask']   for b in batch]),
        'img_id': torch.stack([b['img_id'] for b in batch]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_distill_loader(cfg: dict, split: str = 'train'):
    """DataLoader cho Bước 1."""
    from torch.utils.data import DataLoader
    ds = LLVIPDistillDataset(
        data_root=cfg['data_root'],
        split=split,
        img_size=cfg.get('img_size', 640),
        augment=(split == 'train'),
    )
    return DataLoader(
        ds,
        batch_size=cfg['step1']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=cfg['step1'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_distill,
        drop_last=(split == 'train'),
    )


def build_detect_loader(cfg: dict, split: str = 'train'):
    """DataLoader cho Bước 2."""
    from torch.utils.data import DataLoader
    ds = LLVIPDetectDataset(
        data_root=cfg['data_root'],
        split=split,
        img_size=cfg.get('img_size', 640),
        augment=(split == 'train'),
    )
    return DataLoader(
        ds,
        batch_size=cfg['step2']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=cfg['step2'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_detect,
        drop_last=(split == 'train'),
    )
