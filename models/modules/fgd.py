"""FGD: Focal and Global Knowledge Distillation for Detectors (Yang et al., CVPR 2022).

Port of ``mmdet/distillation/losses/fgd.py`` from https://github.com/yzd-v/FGD without mmcv/mmdet.
Differences from the original:

- GT boxes come in the Ultralytics format (normalized ``xyxy`` + ``batch_idx``) instead of pixel
  boxes + ``img_metas``; the foreground mask is built vectorized instead of a per-box Python loop.
- Same interface as ``DiffKD`` / ``FeatureKD``: returns ``(student_feat, teacher_feat, loss, None)``.
- Optional ``weight`` (B, 1, H, W) (e.g. the LCE darkness map) re-weights the focal fg / bg masks.
- ``targets=None`` (e.g. in tools/visualize_features.py) treats the whole map as background.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _global_context_block(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels // 2, kernel_size=1),
        nn.LayerNorm([channels // 2, 1, 1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels // 2, channels, kernel_size=1),
    )


class FGD(nn.Module):
    """Focal (fg/bg-separated, attention-weighted) + global (GcBlock relation) feature distillation.

    Args:
        student_channels: channels of the student feature map.
        teacher_channels: channels of the teacher feature map (student is aligned by a 1x1 conv).
        temp: temperature of the spatial / channel attention.
        alpha_fgd: weight of the foreground feature loss.
        beta_fgd: weight of the background feature loss.
        gamma_fgd: weight of the attention (mask) loss.
        lambda_fgd: weight of the global relation loss.
    """

    def __init__(
        self,
        student_channels,
        teacher_channels,
        temp=0.5,
        alpha_fgd=0.001,
        beta_fgd=0.0005,
        gamma_fgd=0.001,
        lambda_fgd=0.000005,
    ):
        super().__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        self.align = (
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
            if student_channels != teacher_channels
            else None
        )
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = _global_context_block(teacher_channels)
        self.channel_add_conv_t = _global_context_block(teacher_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in (self.conv_mask_s, self.conv_mask_t):
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        # zero-init the last conv: the GcBlocks start as identity (as in the original)
        for block in (self.channel_add_conv_s, self.channel_add_conv_t):
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def forward(self, student_feat, teacher_feat, weight=None, targets=None):
        """
        Args:
            student_feat: (B, Cs, H, W) student feature.
            teacher_feat: (B, Ct, H, W) teacher feature (same spatial size).
            weight: optional (B, 1, H, W) spatial weight multiplied into the fg / bg masks.
            targets: optional ``(boxes, batch_idx)``; ``boxes`` (M, 4) normalized ``xyxy`` in [0, 1]
                relative to the input image, ``batch_idx`` (M,) the image index of each box.
        """
        assert student_feat.shape[-2:] == teacher_feat.shape[-2:], "teacher / student spatial sizes differ"
        if self.align is not None:
            student_feat = self.align(student_feat)
        preds_s = student_feat.float()
        preds_t = teacher_feat.float()
        n, _, h, w = preds_s.shape

        s_att_t, c_att_t = self.get_attention(preds_t, self.temp)
        s_att_s, c_att_s = self.get_attention(preds_s, self.temp)

        mask_fg, mask_bg = self.get_masks(targets, n, h, w, preds_s.device)
        if weight is not None:
            weight = weight.float().reshape(n, h, w)
            mask_fg = mask_fg * weight
            mask_bg = mask_bg * weight

        fg_loss, bg_loss = self.get_fea_loss(preds_s, preds_t, mask_fg, mask_bg, c_att_t, s_att_t)
        mask_loss = self.get_mask_loss(c_att_s, c_att_t, s_att_s, s_att_t)
        rela_loss = self.get_rela_loss(preds_s, preds_t)

        loss = (
            self.alpha_fgd * fg_loss
            + self.beta_fgd * bg_loss
            + self.gamma_fgd * mask_loss
            + self.lambda_fgd * rela_loss
        )
        return student_feat, teacher_feat, loss, None

    @staticmethod
    def get_attention(preds, temp):
        """Spatial (B, H, W) and channel (B, C) attention of ``|preds|``, scaled to mean 1."""
        n, c, h, w = preds.shape
        value = preds.abs()
        fea_map = value.mean(dim=1)  # B, H, W
        s_attention = h * w * F.softmax((fea_map / temp).view(n, -1), dim=1).view(n, h, w)
        channel_map = value.mean(dim=(2, 3))  # B, C
        c_attention = c * F.softmax(channel_map / temp, dim=1)
        return s_attention, c_attention

    @staticmethod
    def get_masks(targets, n, h, w, device):
        """Foreground mask = 1 / box area (smallest box wins on overlap), background = normalized complement."""
        mask_fg = torch.zeros(n, h * w, device=device)
        if targets is not None and len(targets[0]):
            boxes, batch_idx = targets
            boxes = boxes.to(device).float()
            batch_idx = batch_idx.to(device).long().view(-1)
            wmin = torch.floor(boxes[:, 0] * w)
            hmin = torch.floor(boxes[:, 1] * h)
            wmax = torch.ceil(boxes[:, 2] * w)
            hmax = torch.ceil(boxes[:, 3] * h)
            area = 1.0 / ((hmax + 1 - hmin) * (wmax + 1 - wmin))  # (M,)

            ys = torch.arange(h, device=device, dtype=torch.float32)
            xs = torch.arange(w, device=device, dtype=torch.float32)
            in_y = (ys[None] >= hmin[:, None]) & (ys[None] <= hmax[:, None])  # M, H
            in_x = (xs[None] >= wmin[:, None]) & (xs[None] <= wmax[:, None])  # M, W
            box_masks = (in_y[:, :, None] & in_x[:, None, :]).view(-1, h * w) * area[:, None]  # M, H*W
            mask_fg.scatter_reduce_(0, batch_idx[:, None].expand_as(box_masks), box_masks, reduce="amax")

        mask_bg = (mask_fg <= 0).float()
        mask_bg = mask_bg / mask_bg.sum(dim=1, keepdim=True).clamp(min=1.0)
        return mask_fg.view(n, h, w), mask_bg.view(n, h, w)

    @staticmethod
    def get_fea_loss(preds_s, preds_t, mask_fg, mask_bg, c_t, s_t):
        """Focal loss: MSE weighted by the teacher spatial / channel attention, split into fg / bg."""
        n = preds_s.shape[0]
        att = s_t[:, None] * c_t[:, :, None, None]  # B, C, H, W
        diff = (preds_s - preds_t).pow(2) * att
        fg_loss = (diff * mask_fg[:, None]).sum() / n
        bg_loss = (diff * mask_bg[:, None]).sum() / n
        return fg_loss, bg_loss

    @staticmethod
    def get_mask_loss(c_s, c_t, s_s, s_t):
        n = c_s.shape[0]
        return (c_s - c_t).abs().sum() / n + (s_s - s_t).abs().sum() / n

    def spatial_pool(self, x, conv_mask):
        n, c, h, w = x.shape
        context_mask = F.softmax(conv_mask(x).view(n, 1, h * w), dim=2)  # B, 1, HW
        context = torch.bmm(x.view(n, c, h * w), context_mask.transpose(1, 2))  # B, C, 1
        return context.view(n, c, 1, 1)

    def get_rela_loss(self, preds_s, preds_t):
        """Global loss: MSE between GcBlock-enhanced student and teacher features."""
        out_s = preds_s + self.channel_add_conv_s(self.spatial_pool(preds_s, self.conv_mask_s))
        out_t = preds_t + self.channel_add_conv_t(self.spatial_pool(preds_t, self.conv_mask_t))
        return F.mse_loss(out_s, out_t, reduction="sum") / preds_s.shape[0]
