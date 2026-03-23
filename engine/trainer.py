import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from models.losses.kd_loss import KDLoss 


class DiffKDTrainer():
    pass


class YOLOV10KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        
        # Tách custom args ra
        self.teacher_ckpt = overrides.pop("teacher_ckpt", None)
        self.kd_loss_weight = overrides.pop("kd_loss_weight", 1.0)

        super().__init__(cfg, overrides, _callbacks)
        # Khởi tạo Teacher và DiffKD ở đây hoặc trong get_model
        self.teacher = None
        self.kd_loss_fn = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Khởi tạo Student và chèn Teacher vào cùng quy trình"""
        model = super().get_model(cfg, weights, verbose)
        
        # 1. Khởi tạo Teacher (IRFormer)
        # Giả sử bạn truyền path teacher qua overrides hoặc lấy từ args
        teacher_ckpt = self.teacher_ckpt
        
        from models.irformer import Model as IRFormer
        self.teacher = IRFormer(in_nc=3, out_nc=3, base_nf=16).to(self.device)
        ckpt = torch.load(teacher_ckpt, map_location=self.device)
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        self.teacher.load_state_dict(state_dict, strict=False)        
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 2. Khởi tạo KDLoss Wrapper
        # ori_loss lúc này chính là v8DetectionLoss (sẽ được gán sau trong criterion)
        self.kd_loss_fn = KDLoss(
            student=model,
            teacher=self.teacher,
            student_name='yolov10s',
            teacher_name='irformer',
            ori_loss=None, # Sẽ gán trong phương thức criterion
            kd_method='diffkd',
            kd_loss_weight=self.kd_loss_weight,
            kd_loss_kwargs={'ae_channels': 16, 'use_ae': True}
        )
        return model

    def criterion(self, preds, batch):
        if self.kd_loss_fn.ori_loss is None:
            self.kd_loss_fn.ori_loss = v8DetectionLoss(self.model)

        total_loss, loss_items = self.kd_loss_fn(batch['img'], batch)

        return total_loss, loss_items 