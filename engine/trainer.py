import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from models.losses.kd_loss import KDLoss 


class YOLOV10KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # Khởi tạo Teacher và DiffKD ở đây hoặc trong get_model
        self.teacher = None
        self.kd_loss_fn = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Khởi tạo Student và chèn Teacher vào cùng quy trình"""
        model = super().get_model(cfg, weights, verbose)
        
        # 1. Khởi tạo Teacher (IRFormer)
        # Giả sử bạn truyền path teacher qua overrides hoặc lấy từ args
        teacher_ckpt = self.args.teacher_ckpt if hasattr(self.args, 'teacher_ckpt') else 'irformer.pt'
        
        from models.irformer import Model as IRFormer
        self.teacher = IRFormer(in_nc=3, out_nc=3, base_nf=16).to(self.device)
        self.teacher.load_state_dict(torch.load(teacher_ckpt, map_location=self.device))
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
            kd_loss_weight=self.args.kd_loss_weight,
            kd_loss_kwargs={'ae_channels': 16, 'use_ae': True}
        )
        return model

    def preprocess_batch(self, batch):
        """Xử lý batch trước khi đưa vào mô hình"""
        batch = super().preprocess_batch(batch)
        # Tạo thêm ảnh 256 cho Teacher
        batch['img_teacher'] = F.interpolate(batch['img'], size=(256, 256), 
                                            mode='bilinear', align_corners=False)
        return batch

    def criterion(self, preds, batch):
        """Ghi đè hàm tính Loss để chèn KD"""
        if self.kd_loss_fn.ori_loss is None:
            # Khởi tạo loss gốc của YOLO nếu chưa có
            self.kd_loss_fn.ori_loss = v8DetectionLoss(self.model)

        # Chạy Teacher forward bằng ảnh đã resize trong preprocess_batch
        with torch.no_grad():
            _ = self.teacher(batch['img_teacher'])

        # Tính toán tổng Loss (YOLO + DiffKD) qua class KDLoss đã viết
        # KDLoss sẽ lấy đặc trưng qua Hook đã đăng ký trong __init__ của nó
        total_loss = self.kd_loss_fn(batch['img'], batch['cls']) # batch['cls'] tùy thuộc vào dataset format
        
        # Ultralytics yêu cầu return về (loss_sum, loss_items_tensor)
        return total_loss, torch.zeros(3, device=self.device)