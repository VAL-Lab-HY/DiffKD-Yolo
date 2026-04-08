import torch
from torch import nn
import torch.nn.functional as F


class DiffKD(nn.Module):
    def __init__(
        self,
        student_channels,
        teacher_channels,
        num_train_timesteps=500,
        beta_start=1e-4,
        beta_end=0.02,
        use_ae=False,
        ae_channels=None,
    ):
        super().__init__()

        self.use_ae = use_ae
        self.num_train_timesteps = num_train_timesteps

        # ===== noise schedule =====
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # ===== timestep embedding =====
        self.time_embed = nn.Embedding(num_train_timesteps, teacher_channels)

        # ===== AE =====
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = nn.Sequential(
                nn.Conv2d(teacher_channels, ae_channels, 1),
                nn.ReLU(),
                nn.Conv2d(ae_channels, teacher_channels, 1),
            )
        else:
            self.ae = None

        # ===== align =====
        self.s_proj = nn.Conv2d(student_channels, teacher_channels, 1)

        # ===== diffusion model (conditioned) =====
        self.model = nn.Sequential(
            nn.Conv2d(2 * teacher_channels, teacher_channels, 3, padding=1),
            nn.GroupNorm(8, teacher_channels),
            nn.ReLU(),
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
        )

        # ===== output refine =====
        self.out_proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.GroupNorm(8, teacher_channels)
        )

    def normalize(self, x):
        mean = x.mean(dim=(2,3), keepdim=True)
        std = x.std(dim=(2,3), keepdim=True) + 1e-6
        return (x - mean) / std

    def forward(self, student_feat, teacher_feat):
        B = student_feat.shape[0]
        device = student_feat.device

        # ===== normalize =====
        student_feat = self.normalize(student_feat)
        teacher_feat = self.normalize(teacher_feat)

        # ===== align student =====
        student_feat = self.s_proj(student_feat)

        # ===== AE =====
        if self.ae is not None:
            rec = self.ae(teacher_feat)
            rec_loss = F.l1_loss(rec, teacher_feat)
            teacher_feat = rec.detach()
        else:
            rec_loss = None

        # ===== timestep =====
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        alpha_bar = self.alphas_cumprod[t].view(B, 1, 1, 1)

        # ===== noise =====
        noise = torch.randn_like(teacher_feat)

        # ===== forward diffusion =====
        noisy_feat = torch.sqrt(alpha_bar) * teacher_feat + torch.sqrt(1 - alpha_bar) * noise

        # ===== timestep embedding =====
        t_embed = self.time_embed(t).view(B, -1, 1, 1)

        # ===== CONDITIONING (🔥 quan trọng nhất) =====
        model_input = torch.cat([noisy_feat + t_embed, student_feat], dim=1)

        # ===== predict noise =====
        noise_pred = self.model(model_input)

        # ===== diffusion loss =====
        diff_loss = F.mse_loss(noise_pred, noise)

        # ===== refine student =====
        refined_feat = student_feat - noise_pred
        refined_feat = self.out_proj(refined_feat)

        refined_feat = torch.clamp(refined_feat, -5, 5)

        # ===== KD loss =====
        kd_loss = self.kd_loss(refined_feat, teacher_feat)

        return refined_feat, teacher_feat, kd_loss + 0.1 * diff_loss, rec_loss

    def kd_loss(self, s, t):
        cos = 1 - F.cosine_similarity(s.flatten(1), t.flatten(1), dim=1).mean()
        mse = F.mse_loss(s, t)
        return 0.7 * cos + 0.3 * mse