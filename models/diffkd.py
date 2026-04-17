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

    def forward(self, x):
        out = self.block(x)
        return out + x 
    

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


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps=500,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        self.num_train_timesteps = num_train_timesteps

        # noise schedule
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t

    def add_noise(self, x0, noise, timesteps):
        """
        x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
        """
        acp = self.alphas_cumprod.to(x0.device)

        alpha_bar = acp[timesteps].view(-1, 1, 1, 1)

        return (
            torch.sqrt(alpha_bar) * x0 +
            torch.sqrt(1 - alpha_bar) * noise
        )

    def step(self, noise_pred, t, x_t, t_prev):
        """
        x_{t-1} = sqrt(alpha_prev) * x0_pred + sqrt(1 - alpha_prev) * noise_pred
        """

        acp = self.alphas_cumprod.to(x_t.device)

        alpha_t = acp[t]
        alpha_prev = acp[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)

        # predict x0 
        x0_pred = (
            x_t - torch.sqrt(1 - alpha_t) * noise_pred
        ) / (torch.sqrt(alpha_t) + 1e-8)

        # compute x_{t-1} 
        x_prev = (
            torch.sqrt(alpha_prev) * x0_pred +
            torch.sqrt(1 - alpha_prev) * noise_pred
        )

        return x_prev

    def set_timesteps(self, num_inference_steps):
        step = self.num_train_timesteps // num_inference_steps

        self.timesteps = list(
            range(self.num_train_timesteps - 1, -1, -step)
        )[:num_inference_steps]


class DDIMPipeline:
    def __init__(
        self,
        model: DiffusionModel,
        scheduler: DDIMScheduler,
        noise_adapter: NoiseAdapter = None,
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
        shape,                      # (C, H, W)
        feat: torch.Tensor,         # student feature
        num_inference_steps: int = 5,
        proj: nn.Module = None,
    ) -> torch.Tensor:

        # set timesteps 
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # init noise 
        noise = torch.randn((batch_size, *shape), device=device, dtype=dtype)

        if self.noise_adapter is not None:
            nl = self.noise_adapter(feat).view(batch_size, 1, 1, 1)
            x = torch.sqrt(1 - nl) * feat + torch.sqrt(nl) * noise
        else:
            # fallback
            x = feat

        # DDIM loop 
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # predict noise
            noise_pred = self.model(x, t_tensor)

            # lấy timestep trước
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            x = self.scheduler.step(noise_pred, t, x, prev_t)

        # optional projection
        if proj is not None:
            x = proj(x)

        return x


class DiffKD(nn.Module):
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        kernel_size: int = 3,
        inference_steps: int = 5,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.diffusion_inference_steps = inference_steps

        # align student → teacher dim
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)

        # diffusion model
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
        )

        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)

        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)

        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm2d(teacher_channels),
        )

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

        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=None,
        )

        refined_feat = self.proj(refined_feat)

        diff_loss = self.ddim_loss(teacher_feat)

        return refined_feat, teacher_feat, diff_loss
