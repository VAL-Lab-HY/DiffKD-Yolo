import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics.models.yolov10 import YOLOv10
from ultralytics.utils.loss import v10Loss
from ultralytics.models.yolov10.val import YOLOv10Validator
from models.diffkd.diffkd import DiffKD

# Giả định bạn đã có các module DiffKD từ file của mình
# from .diffkd_modules import DiffKD 

# --- 1. CƠ CHẾ HOOK ĐỂ LẤY FEATURE ---
class FeatureExtractor:
    """Hứng đặc trưng từ các tầng Backbone của YOLO"""
    def __init__(self, model, layers_idx=[0, 1, 2]):
        self.features = {}
        self.hooks = []
        for i in layers_idx:
            # Truy cập vào mảng .model của Ultralytics
            hook = model.model[i].register_forward_hook(self.get_hook(f'layer_{i}'))
            self.hooks.append(hook)
            
    def get_hook(self, name):
        def hook(module, input, output):
            # Nếu output là tuple (thường thấy ở một số bản YOLO), lấy phần tử đầu
            self.features[name] = output[0] if isinstance(output, tuple) else output
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()

# --- 2. HÀM TRAIN TỔNG HỢP (DETECTION + DISTILLATION) ---
def train_one_epoch(student, teacher, diffkd_dict, loader, optimizer, criterion, device, alpha=0.1, beta=0.05):
    student.train()
    teacher.eval()
    diffkd_dict.train()
    
    # Đăng ký Hook cho 3 block đầu (0, 1, 2)
    s_extractor = FeatureExtractor(student, layers_idx=[0, 1, 2])
    t_extractor = FeatureExtractor(teacher, layers_idx=[0, 1, 2])
    
    pbar = tqdm(loader, desc="Training")
    total_loss = 0

    for batch in pbar:
        # Chuẩn bị dữ liệu
        imgs = batch['img'].to(device).float() / 255.0
        targets = batch 

        # Forward Teacher (Đã pretrain trên IR/Hồng ngoại)
        with torch.no_grad():
            _ = teacher(imgs)

        # Forward Student (YOLOv10s hoặc YOLOv12s)
        preds = student(imgs)
        
        # 1. Detection Loss (Cốt lõi của YOLO)
        det_loss, loss_items = criterion(preds, targets)

        # 2. Distillation Loss (Truyền tri thức từ Teacher)
        kd_loss_total = 0
        diff_loss_total = 0
        
        for i in range(3):
            key = f'layer_{i}'
            s_f = s_extractor.features[key]
            t_f = t_extractor.features[key]

            # Khớp size nếu Teacher và Student khác độ phân giải feature map
            if s_f.shape[-2:] != t_f.shape[-2:]:
                t_f = F.interpolate(t_f, size=s_f.shape[-2:], mode='bilinear', align_corners=False)

            # Forward qua module DiffKD bạn đã viết
            # refined_s: feature student sau khi được lọc qua diffusion theo phong cách teacher
            refined_s, _, ddim_l, rec_l = diffkd_dict[f'backbone_{i}'](s_f, t_f)
            
            # Ép Student backbone tạo ra đặc trưng giống bản "refined"
            kd_loss_total += F.mse_loss(s_f, refined_s.detach())
            diff_loss_total += ddim_l
            if rec_l is not None:
                diff_loss_total += rec_l

        # Tổng hợp Loss
        loss = det_loss + (alpha * kd_loss_total) + (beta * diff_loss_total)

        # Cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            "Det": f"{det_loss.item():.2f}", 
            "KD": f"{kd_loss_total.item():.4f}",
            "mAP_sim": "calculating..." # mAP sẽ được tính ở hàm validate
        })

    # Gỡ Hooks sau mỗi epoch để tránh tràn bộ nhớ
    s_extractor.remove()
    t_extractor.remove()
    
    return total_loss / len(loader)

# --- 3. HÀM TÍNH mAP (VALIDATION) ---
def validate_model(model, val_loader, args):
    # Sử dụng Validator chuẩn của Ultralytics để tính mAP50, mAP50-95
    validator = YOLOv10Validator(args=args)
    stats = validator(model=model, dataloader=val_loader)
    # stats.results_dict chứa các chỉ số mAP
    return stats.results_dict['metrics/mAP50(B)'], stats.results_dict['metrics/mAP50-95(B)']


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. Khởi tạo Models
    student = YOLOv10("yolov10s.pt").to(device)
    teacher = torch.load("irformer_pretrained.pth").to(device) # Model đã học IR
    
    # B. Khởi tạo 3 module DiffKD (Số kênh ví dụ: 64, 128, 128)
    diffkd_layers = nn.ModuleDict({
        'backbone_0': DiffKD(student_channels=64,  teacher_channels=64),
        'backbone_1': DiffKD(student_channels=128, teacher_channels=128),
        'backbone_2': DiffKD(student_channels=128, teacher_channels=128)
    }).to(device)

    # C. Loss & Optimizer
    criterion = v10Loss(student) 
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(diffkd_layers.parameters()), 
        lr=1e-4
    )

    # D. Dataloader (Cần config file data.yaml cho ảnh tối)
    # train_loader, val_loader = ...

    # E. Vòng lặp Training
    best_map = 0
    for epoch in range(100):
        avg_loss = train_one_epoch(student, teacher, diffkd_layers, train_loader, optimizer, criterion, device)
        
        # Sau mỗi epoch tính mAP
        map50, map50_95 = validate_model(student, val_loader, args=None)
        
        print(f"Epoch {epoch} | mAP50: {map50:.4f} | Loss: {avg_loss:.4f}")

        # Lưu model tốt nhất dựa trên mAP
        if map50 > best_map:
            best_map = map50
            torch.save(student.state_dict(), "best_distilled_yolo.pth")
# --- 4. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    main()