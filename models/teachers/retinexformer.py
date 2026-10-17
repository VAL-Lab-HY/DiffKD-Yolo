"""Retinexformer (Cai et al., ICCV 2023), self-contained port of
https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/models/archs/RetinexFormer_arch.py

Module/parameter names are kept identical to the original so BasicSR checkpoints
(e.g. LOL_v2_real.pth, key ``params``) load with ``strict=True``.
"""

import torch
import torch.nn.functional as F
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        """img: b,3,h,w → illu_fea: b,c,h,w ; illu_map: b,3,h,w"""
        mean_c = img.mean(dim=1, keepdim=True)
        x = self.conv1(torch.cat([img, mean_c], dim=1))
        illu_fea = self.depth_conv(x)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention (attention across channels)."""

    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def _split_heads(self, t):
        # b n (h d) -> b h n d
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, -1).transpose(1, 2)

    def forward(self, x_in, illu_fea_trans):
        """x_in, illu_fea_trans: b,h,w,c → out: b,h,w,c"""
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v, illu_attn = map(self._split_heads, (q_inp, k_inp, v_inp, illu_fea_trans.flatten(1, 2)))
        v = v * illu_attn
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale  # A = K^T * Q
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """x: b,h,w,c → out: b,h,w,c"""
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    """Illumination-Guided Attention Block."""

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([IG_MSA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))])
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, illu_fea):
        """x, illu_fea: b,c,h,w → out: b,c,h,w"""
        x = x.permute(0, 2, 3, 1)
        illu_fea_trans = illu_fea.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea_trans) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class Denoiser(nn.Module):
    """U-shaped corruption restorer guided by illumination features."""

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super().__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # feature downsample
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # illumination downsample
                    ]
                )
            )
            dim_level *= 2

        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                    ]
                )
            )
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """x: b,c,h,w (H, W divisible by 2**level) ; illu_fea: b,c,h,w → out: b,c,h,w"""
        fea = self.embedding(x)

        fea_encoder, illu_fea_list = [], []
        for igab, fea_downsample, illu_fea_downsample in self.encoder_layers:
            fea = igab(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = fea_downsample(fea)
            illu_fea = illu_fea_downsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (fea_upsample, fusion, igab) in enumerate(self.decoder_layers):
            fea = fea_upsample(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = igab(fea, illu_fea_list[self.level - 1 - i])

        return self.mapping(fea) + x


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks)

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img  # lit-up image
        return self.denoiser(input_img, illu_fea)


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super().__init__()
        self.stage = stage
        self.body = nn.Sequential(
            *[
                RetinexFormer_Single_Stage(
                    in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks
                )
                for _ in range(stage)
            ]
        )

    def forward(self, x):
        """x: b,3,h,w in [0, 1] → enhanced image b,3,h,w"""
        return self.body(x)
