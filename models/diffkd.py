import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
    

class DiffusionModel(nn.Module):
    """Dự đoán noise, có điều kiện theo timestep."""

    def __init__(self, channels_in: int, kernel_size: int = 3):
        super().__init__()

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, channels_in),
            nn.SiLU(),
            nn.Linear(channels_in, channels_in),
        )

        self.net = nn.Sequential(
            Bottleneck(channels_in, channels_in),
            Bottleneck(channels_in, channels_in),
            nn.Conv2d(channels_in, channels_in, 1),
            nn.BatchNorm2d(channels_in),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return self.net(x + t_emb)


class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.feat = nn.Sequential(
            Bottleneck(channels, channels, reduction=8),
            nn.AdaptiveAvgPool2d(1)
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)


class DDIMScheduler:
    """
    Lịch nhiễu tuyến tính + bước denoising DDIM.
    Chỉ giữ lại những gì DiffKD thực sự dùng.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise NotImplementedError(f"beta_schedule '{beta_schedule}' chưa hỗ trợ.")

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)   # ᾱ_t

    # forward process
    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,) -> torch.Tensor:
        acp = self.alphas_cumprod.to(original.device)
        sqrt_alpha = acp[timesteps] ** 0.5
        sqrt_one_minus = (1 - acp[timesteps]) ** 0.5

        # reshape để broadcast với (B, C, H, W)
        def expand(v):
            for _ in range(original.ndim - 1):
                v = v.unsqueeze(-1)
            return v

        return expand(sqrt_alpha) * original + expand(sqrt_one_minus) * noise

    def step(self, noise_pred: torch.Tensor, t: int, x_t: torch.Tensor, prev_t: int,) -> torch.Tensor:
        acp = self.alphas_cumprod.to(x_t.device)
        alpha_t    = acp[t]
        alpha_prev = acp[prev_t] if prev_t >= 0 else torch.tensor(1.0)

        # Dự đoán x_0
        pred_x0 = (x_t - (1 - alpha_t) ** 0.5 * noise_pred) / (alpha_t ** 0.5 + 1e-8)
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-1, 1)

        # DDIM deterministic step
        x_prev = (alpha_prev ** 0.5) * pred_x0 + ((1 - alpha_prev) ** 0.5) * noise_pred
        return x_prev

    def set_timesteps(self, num_inference_steps: int):
        step_size = self.num_train_timesteps // num_inference_steps
        self.timesteps = list(
            range(self.num_train_timesteps - 1, -1, -step_size)
        )[:num_inference_steps]


class DDIMPipeline:
    """
    Denoising pipeline: bắt đầu từ noisy student feature,
    dần dần refine về phía teacher distribution.
    """

    def __init__(
        self,
        model: DiffusionModel,
        scheduler: DDIMScheduler,
        noise_adapter: NoiseAdapter,
    ):
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        shape,                       # (C, H, W)
        feat: torch.Tensor,          # student feature đã align
        num_inference_steps: int = 5,
        proj: nn.Module = None,
    ) -> torch.Tensor:

        self.scheduler.set_timesteps(num_inference_steps)

        # Mức nhiễu thích hợp cho từng sample
        noise_level = self.noise_adapter(feat)            # (B, 1)
        noise = torch.randn((batch_size, *shape), device=device, dtype=dtype)

        # Khởi tạo: trộn student feat + noise theo noise_level
        nl = noise_level.view(batch_size, 1, 1, 1)
        x = torch.sqrt(1 - nl) * feat + torch.sqrt(nl) * noise

        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.model(x, t_tensor)

            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.scheduler.step(noise_pred, t, x, prev_t)

        return x


class DiffKD(nn.Module):
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        kernel_size: int = 3,
        inference_steps: int = 5,
        num_train_timesteps: int = 500,
        use_ae: bool = False,
        ae_channels: int = None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps

        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        # align student → teacher dim
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)

        # diffusion model
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            clip_sample=False,
            beta_schedule="linear",
        )

        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)

        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)

        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor):
        # 1. Align student
        student_feat = self.trans(student_feat)

        # 2. Auto encode
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # 3. Denoise student feature (inference)
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj,
        )
        refined_feat = self.proj(refined_feat)

        # 4. Train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)

        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(gt_feat)
        bs = gt_feat.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device
        ).long()
        noisy = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy, timesteps)
        return F.mse_loss(noise_pred, noise)
    
    def forward(self, student_feat, teacher_feat):
        student_feat = self.trans(student_feat)

        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj,
        )
        refined_feat = self.proj(refined_feat)

        ddim_loss = self.ddim_loss(teacher_feat)

        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def kd_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        cos_loss = 1 - F.cosine_similarity(
            student.flatten(1), teacher.flatten(1), dim=1
        ).mean()
        mse_loss = F.mse_loss(student, teacher)
        return 0.7 * cos_loss + 0.3 * mse_loss