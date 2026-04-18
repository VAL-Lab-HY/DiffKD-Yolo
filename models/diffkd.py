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
    def __init__(self, channels_in: int, kernel_size: int = 3):
        super().__init__()
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
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        return self.net(x + t_emb)


class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3, num_train_timesteps=500):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps

        self.feat = nn.Sequential(
            Bottleneck(channels, channels, reduction=8),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        t = torch.sigmoid(self.pred(x))      
        t = (t * self.num_train_timesteps - 1).long().squeeze(1) 
        return t


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1),
            nn.BatchNorm2d(latent_channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)


class DDIMScheduler:
    def __init__(self, num_train_timesteps=500, beta_start=1e-4, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, x0, noise, timesteps):
        acp = self.alphas_cumprod.to(x0.device)
        alpha_bar = acp[timesteps].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def step(self, noise_pred, t, x_t, t_prev):
        acp = self.alphas_cumprod.to(x_t.device)
        alpha_t = acp[t]
        alpha_prev = acp[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / (torch.sqrt(alpha_t) + 1e-8)
        x_prev = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
        return x_prev

    def set_timesteps(self, num_inference_steps):
        step = self.num_train_timesteps // num_inference_steps
        self.timesteps = list(range(self.num_train_timesteps - 1, -1, -step))[:num_inference_steps]
        
    def add_noise_diff2(self, x0, noise, timesteps):
        assert timesteps.shape[0] == x0.shape[0], \
            f"batch mismatch: timesteps {timesteps.shape} vs x0 {x0.shape}"
        
        acp = self.alphas_cumprod.to(x0.device)
        timesteps = timesteps.clamp(0, len(acp) - 1).long()
        alpha_bar = acp[timesteps].view(-1, 1, 1, 1) 
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise


class DDIMPipeline:
    def __init__(self, model, scheduler, noise_adapter=None):
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter

    def __call__(self, batch_size, device, dtype, shape, feat,
                 num_inference_steps=5, proj=None):
        noise = torch.randn((batch_size, *shape), device=device, dtype=dtype)

        if self.noise_adapter is not None:
            timesteps = self.noise_adapter(feat)
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            image = feat

        self.scheduler.set_timesteps(num_inference_steps * 2)
        for t in self.scheduler.timesteps[len(self.scheduler.timesteps) // 2:]:
            noise_pred = self.model(image, t.to(device))
            image = self.scheduler.step(noise_pred, t, image)

        if proj is not None:
            image = proj(image)

        return image


class DiffKD(nn.Module):
    def __init__(self, student_channels, teacher_channels, kernel_size=3,
                 inference_steps=5, num_train_timesteps=500,
                 use_ae=False, ae_channels=None):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps

        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
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

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor):
        # 1. Align student channels
        student_feat = self.trans(student_feat)

        # 2. AutoEncoder (optional)
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            ae_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            ae_loss = None

        # 3. Denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=None,  # FIX Bug 1: không truyền proj vào pipeline
        )
        refined_feat = self.proj(refined_feat)  # apply đúng 1 lần

        # 4. Train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)
        kd_loss = F.mse_loss(refined_feat, teacher_feat) + ddim_loss

        return refined_feat, teacher_feat, kd_loss, ae_loss

    def ddim_loss(self, gt_feat: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(gt_feat)
        bs = gt_feat.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device
        ).long()
        noisy = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy, timesteps)
        return F.mse_loss(noise_pred, noise)