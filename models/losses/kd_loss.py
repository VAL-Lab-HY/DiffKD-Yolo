import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .diffkd import DiffKD

import logging
logger = logging.getLogger()


KD_MODULES = {
    'irformer': dict(modules=['transformer.2'], channels=[16]),
    'yolov10s': dict(modules=['model.4'], channels=[128]),
    'yolov12s': dict(modules=['model.4'], channels=[128]),
}


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return loss
    

class KDLoss():
    def __init__(
        self,
        student,
        teacher,
        student_name,
        teacher_name,
        ori_loss,
        kd_method='diffkd',
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        kd_loss_kwargs={}
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        
        # Tự động xác định device
        self.device = next(student.parameters()).device

        self._teacher_out = {}
        self._student_out = {}

        if kd_method == 'diffkd':
            ae_channels = kd_loss_kwargs.get('ae_channels', 16) # Để 16 cho nhẹ vì teacher chỉ 16
            use_ae = kd_loss_kwargs.get('use_ae', True)
            tau = kd_loss_kwargs.get('tau', 1)

            student_modules = KD_MODULES[student_name]['modules']
            student_channels = KD_MODULES[student_name]['channels']
            teacher_modules = KD_MODULES[teacher_name]['modules']
            teacher_channels = KD_MODULES[teacher_name]['channels']
            
            kernel_sizes = [1 if tm == '' else 3 for tm in teacher_modules]

            self.diff = nn.ModuleDict()
            self.kd_loss = nn.ModuleDict()
            
            for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
                # FIX: Thay '.' bằng '_' để làm key cho ModuleDict
                m_key = tm.replace('.', '_') if tm != '' else 'logits'
                
                self.diff[m_key] = DiffKD(sc, tc, kernel_size=ks, use_ae=(ks!=1) and use_ae, ae_channels=ae_channels)
                self.kd_loss[m_key] = nn.MSELoss() if ks != 1 else KLDivergence(tau=tau)
            
            self.diff.to(self.device)
            self.kd_loss.to(self.device)
            self.student._diff = self.diff
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        for sm, tm in zip(student_modules, teacher_modules):
            self._register_forward_hook(student, sm, teacher=False)
            self._register_forward_hook(teacher, tm, teacher=True)
            
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules
        teacher.eval()
        self._iter = 0

    def compute_kd_loss(self):
        kd_loss = 0

        for tm, sm in zip(self.teacher_modules, self.student_modules):
            m_key = tm.replace('.', '_') if tm != '' else 'logits'

            s_feat = self._reshape_BCHW(self._student_out[sm])
            t_feat = self._reshape_BCHW(self._teacher_out[tm])

            s_feat = F.normalize(s_feat, dim=1)
            t_feat = F.normalize(t_feat, dim=1)

            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)

            s_feat_refined, t_feat_target, diff_loss, ae_loss = self.diff[m_key](s_feat, t_feat)

            kd_loss_item = self.kd_loss[m_key](s_feat_refined, t_feat_target)

            kd_loss += kd_loss_item + diff_loss
            if ae_loss is not None:
                kd_loss += ae_loss

        self._teacher_out = {}
        self._student_out = {}
        self._iter += 1

        return kd_loss
    
    def _register_forward_hook(self, model, name, teacher=False):
        if name == '':
            model.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            if module is None:
                raise ValueError(f"Module '{name}' not found in model.")
            module.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))

    def _forward_hook(self, module, input, output, name, teacher=False):
        # Nếu là tuple/list lấy phần tử đầu tiên là feature map.
        if isinstance(output, (tuple, list)):
            out = output[0]
        else:
            out = output
            
        if teacher:
            self._teacher_out[name] = out.detach() # Detach teacher để nhẹ memory
        else:
            self._student_out[name] = out

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x
    
    def __call__(self, preds, batch):
        with torch.no_grad():
            _ = self.teacher(batch['img'])
        
        # Tính detection loss
        det_loss, loss_items = self.ori_loss(preds, batch)

        # KD loss (DiffKD)
        kd_loss = self.compute_kd_loss()

        total_loss = self.ori_loss_weight * det_loss + self.kd_loss_weight * kd_loss

        return total_loss, loss_items