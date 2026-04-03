import torch
from torch import nn
import torch.nn.functional as F

class DiffKD(nn.Module):
    def __init__(
        self,
        student_channels,
        teacher_channels,
        kernel_size=3,
        inference_steps=3,   # 🔥 giảm steps
        num_train_timesteps=1000,
        use_ae=False,
        ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps

        # ===== AE (optional, nhẹ hơn) =====
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = nn.Sequential(
                nn.Conv2d(teacher_channels, ae_channels, 1),
                nn.ReLU(),
                nn.Conv2d(ae_channels, teacher_channels, 1)
            )
        else:
            self.ae = None

        # ===== align channel =====
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)

        # ===== diffusion model (nhẹ hơn) =====
        self.model = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
        )

        # ❌ bỏ BatchNorm → dùng GroupNorm
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.GroupNorm(8, teacher_channels)
        )

    def normalize(self, x):
        # 🔥 ổn định hơn var()
        mean = x.mean(dim=(2,3), keepdim=True)
        std = x.std(dim=(2,3), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, student_feat, teacher_feat):
        # ===== normalize nhẹ =====
        student_feat = self.normalize(student_feat)
        teacher_feat = self.normalize(teacher_feat)

        # ===== align =====
        student_feat = self.trans(student_feat)

        # ===== AE (optional) =====
        if self.ae is not None:
            rec = self.ae(teacher_feat)
            rec_loss = F.l1_loss(rec, teacher_feat)  # 🔥 L1 ổn định hơn MSE
            teacher_feat = rec.detach()
        else:
            rec_loss = None

        # ===== diffusion (simple residual) =====
        noise_pred = self.model(student_feat)
        refined_feat = student_feat + noise_pred  # 🔥 residual stable

        # ===== clamp để tránh explode =====
        refined_feat = torch.clamp(refined_feat, -5, 5)

        refined_feat = self.proj(refined_feat)

        # ===== KD loss (cosine + MSE) =====
        kd_loss = self.kd_loss(refined_feat, teacher_feat)

        return refined_feat, teacher_feat, kd_loss, rec_loss

    def kd_loss(self, s, t):
        # combine loss (rất hiệu quả cho detection)
        
        # cosine loss
        cos = 1 - F.cosine_similarity(
            s.flatten(1), t.flatten(1), dim=1
        ).mean()

        # mse (nhẹ thôi)
        mse = F.mse_loss(s, t)

        return 0.7 * cos + 0.3 * mse