import torch
import torch.nn.functional as F
from torch import nn


class FeatureKD(nn.Module):
    """FitNets-style feature KD: 1x1 conv adapter on the student feature + MSE to the teacher feature.

    Same interface as ``DiffKD`` so ``FeatureLoss`` can swap them: returns
    ``(student_feat, teacher_feat, kd_loss, aux_loss)`` with ``aux_loss = None``.
    """

    def __init__(self, student_channels, teacher_channels, normalize=True):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )
        # LLIE teacher features have a very different scale than detector features: standardizing the
        # teacher per channel gives a target the BN-ended adapter can match
        self.normalize = normalize

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, weight: torch.Tensor = None):
        """``weight``: optional (B, 1, H, W) spatial weight of the MSE (e.g. darkness map)."""
        student_feat = self.adapter(student_feat)
        teacher_feat = teacher_feat.float()
        if self.normalize:
            mean = teacher_feat.mean(dim=(2, 3), keepdim=True)
            std = teacher_feat.std(dim=(2, 3), keepdim=True)
            teacher_feat = (teacher_feat - mean) / (std + 1e-6)

        if weight is None:
            kd_loss = F.mse_loss(student_feat.float(), teacher_feat)
        else:
            kd_loss = (weight * (student_feat.float() - teacher_feat).pow(2)).mean()
        return student_feat, teacher_feat, kd_loss, None
