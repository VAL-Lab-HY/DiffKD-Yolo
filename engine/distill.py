from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from models.modules.diffkd import DiffKD
from ultralytics.utils import LOGGER


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, device=None, kd_weight=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        n = len(channels_s)
        if isinstance(kd_weight, (list, tuple)):
            assert len(kd_weight) == n, (
                f"kd_weight length ({len(kd_weight)}) phải khớp số layer ({n})"
            )
            self.kd_weights = list(kd_weight)
        else:
            self.kd_weights = [kd_weight] * n

        self.diffkd = nn.ModuleList(
            [
                DiffKD(
                    student_channels=s,
                    teacher_channels=t,
                    use_ae=(t > s),
                ).to(self.device)
                for s, t in zip(channels_s, channels_t)
            ]
        )

    def forward(self, y_s, y_t, darkness=None):
        """``darkness``: optional (B, 1, H, W) map in [0, 1]; darker regions get a larger KD weight."""
        if len(y_s) != len(y_t):
            y_t = y_t[-len(y_s):]

        total_loss = 0.0

        for i, (s, t) in enumerate(zip(y_s, y_t)):
            if s.shape[2:] != t.shape[2:]:
                # antialias avoids aliasing when the teacher map is downsampled (e.g. 4x for P5)
                t = F.interpolate(t, size=s.shape[2:], mode="bilinear", align_corners=False, antialias=True)

            weight = None
            if darkness is not None:
                weight = F.adaptive_avg_pool2d(darkness.float(), s.shape[2:])
                weight = weight / (weight.mean() + 1e-6)  # mean weight 1 keeps the KD loss scale

            _, _, kd_loss, ae_loss = self.diffkd[i](s, t.detach(), weight=weight)

            loss = self.kd_weights[i] * kd_loss
            if ae_loss is not None:
                loss = loss + ae_loss

            if not torch.isfinite(loss):
                LOGGER.warning(f"layer {i} NaN → zero")
                loss = torch.zeros_like(loss)

            total_loss += loss

        return total_loss


class DistillationTrainer:
    DEFAULT_LAYERS = ["6"]

    def __init__(
        self,
        student,
        teacher,
        layers=None,
        device=None,
        num_classes=2,
        teacher_layer_names=None,
        student_layer_names=None,
        teacher_channels=None,
        student_channels=None,
        layer_weights=None,
    ):
        self.layers = layers or self.DEFAULT_LAYERS
        self.student = student
        self.teacher = teacher
        self._handles = []
        # pair index → feature; filled by hooks so pairing follows the config order, not the forward order
        self.student_outputs, self.teacher_outputs = {}, {}
        self.student_logits, self.teacher_logits = [], []
        self.layer_weights = layer_weights

        if device is None:
            device = next(student.parameters()).device
        self.device = torch.device(device)

        self._teacher_layer_names = teacher_layer_names
        self._student_layer_names = student_layer_names
        self._teacher_channels_override = teacher_channels
        self._student_channels_override = student_channels

        self.channels_s, self.channels_t = [], []
        self.teacher_modules, self.student_modules = [], []
        self._find_layers()
        self.loss_fn = FeatureLoss(
            channels_s=self.channels_s,
            channels_t=self.channels_t,
            kd_weight=self.layer_weights if self.layer_weights is not None else 1.0,
            device=self.device,
        ).to(self.device)

    def _find_layers(self):
        if self._teacher_layer_names or self._student_layer_names:
            self._find_layers_by_name()
        else:
            self._find_layers_by_cv2()

    def _find_layers_by_cv2(self):
        def _collect(model, target_list, channel_list, tag=""):
            found = []
            for name, module in model.named_modules():
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "model" and parts[1] in self.layers and "cv2" in name:
                    m = module.conv if hasattr(module, "conv") else module
                    found.append((name, m))

            seen = {}
            for name, m in found:
                idx = name.split(".")[1]
                seen[idx] = (name, m)

            for idx in sorted(seen.keys(), key=lambda x: int(x)):
                n, m = seen[idx]
                ch = self._infer_out_channels(m)
                channel_list.append(ch)
                target_list.append(m)
                LOGGER.info(f"  ✓ [{tag}] Hooked layer {n} | Channels: {ch}")

        _collect(self.teacher, self.teacher_modules, self.channels_t, "TEACHER")
        _collect(self.student, self.student_modules, self.channels_s, "STUDENT")

    def _infer_out_channels(self, module):
        # Ultralytics C2f-like blocks: output = cv2 (the first Conv2d found below would be cv1)
        if hasattr(module, "cv2") and hasattr(module.cv2, "conv"):
            return module.cv2.conv.out_channels
        if hasattr(module, "out_channels"):
            return module.out_channels
        if hasattr(module, "num_features"):
            return module.num_features
        if hasattr(module, "conv") and hasattr(module.conv, "out_channels"):
            return module.conv.out_channels

        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m.out_channels
            if isinstance(m, nn.BatchNorm2d):
                return m.num_features
        return None

    def get_loss(self, darkness=None):
        n = len(self.student_modules)
        if len(self.teacher_outputs) != n or len(self.student_outputs) != n:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        feat_loss = self.loss_fn(
            y_s=[self.student_outputs[i] for i in range(n)],
            y_t=[self.teacher_outputs[i] for i in range(n)],
            darkness=darkness,
        )
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        self.student_logits.clear()
        self.teacher_logits.clear()
        return feat_loss

    def _find_layers_by_name(self):
        def _collect(model, target_names, module_list, channel_list, channels_override):
            named = dict(model.named_modules())
            for i, name in enumerate(target_names):
                if name not in named:
                    raise RuntimeError(
                        f"DistillationTrainer: module '{name}' not found. "
                        f"First 30 modules: {list(named.keys())[:30]}"
                    )
                module = named[name]
                module_list.append(module)
                ch = (
                    channels_override[i]
                    if channels_override is not None
                    and i < len(channels_override)
                    and channels_override[i] is not None
                    else self._infer_out_channels(module)
                )
                channel_list.append(ch)
                LOGGER.info(f"  ✓ hooked '{name}' → out_channels={ch}")

        LOGGER.info(
            f"DistillationTrainer: hooking by name | teacher={self._teacher_layer_names} | student={self._student_layer_names}"
        )
        _collect(
            self.teacher,
            self._teacher_layer_names,
            self.teacher_modules,
            self.channels_t,
            self._teacher_channels_override,
        )
        _collect(
            self.student,
            self._student_layer_names,
            self.student_modules,
            self.channels_s,
            self._student_channels_override,
        )
        assert len(self.channels_s) == len(self.channels_t), (
            f"teacher layers ({len(self.channels_t)}) != student layers ({len(self.channels_s)})"
        )
        LOGGER.info(
            f"DistillationTrainer: hooked {len(self.channels_s)} pair(s) | student_ch={self.channels_s} | teacher_ch={self.channels_t}"
        )

    def register_hooks(self):
        self.remove_hooks()
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        self.student_logits.clear()
        self.teacher_logits.clear()

        def _first_tensor(out):
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, (list, tuple)):
                for item in out:
                    feat = _first_tensor(item)
                    if feat is not None:
                        return feat
            if isinstance(out, dict):
                for item in out.values():
                    feat = _first_tensor(item)
                    if feat is not None:
                        return feat
            return None

        def _make_hook(storage, idx, detach=False):
            def hook(m, inp, out):
                feat = _first_tensor(out)
                if feat is None:
                    return
                storage[idx] = feat.detach() if detach else feat

            return hook

        for i, (t_mod, s_mod) in enumerate(zip(self.teacher_modules, self.student_modules)):
            self._handles.append(t_mod.register_forward_hook(_make_hook(self.teacher_outputs, i, detach=True)))
            self._handles.append(s_mod.register_forward_hook(_make_hook(self.student_outputs, i, detach=False)))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
