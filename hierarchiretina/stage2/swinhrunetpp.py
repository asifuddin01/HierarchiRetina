"""SwinHRUNetPP: dual-encoder (SwinV2-B + high-resolution CNN) vessel segmentation network.

Architecture (input 512 x 512, Section "SwinHRUNetPP (vessels)" of the paper):

* SwinV2-Base encoder (window 8, ImageNet-1k weights): s0..s3 with 128/256/512/1024 channels at
  128^2 / 64^2 / 32^2 / 16^2.
* High-resolution CNN branch (trained from scratch, ``hr_base=64``): h0..h2 with 128/256/512
  channels at 128^2 / 64^2 / 32^2.
* Fusion: F1 = conv[s1, up(h2)] (192 ch, 64^2), F0 = conv[s0, up(h1)] (128 ch, 128^2).
* Bottleneck ResBlock 1024 -> 512 on s3; attention-gated decoder blocks
  D4 (skip s2, 256 ch), D3 (F1, 192 ch), D2 (F0, 128 ch), D1 (h0, 64 ch).
* Vessel refinement (dilations 1/2/4 + SE), topology block (1x15 / 15x1), x4 up-sampling, 1x1 head.
* Deep supervision: auxiliary 1x1 heads on D4, D3, D2 (returned only in training mode).

Module attribute names and nesting are identical to the original training notebook so that the
released checkpoints load with ``strict=True``.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Final configuration reported in the paper.
SWIN_MODEL_NAME = "swinv2_base_window8_256"
IMG_SIZE = 512
HR_BASE_CH = 64
DEC_CHS = (512, 256, 192, 128, 64)
DROPOUT_P = 0.10


class ConvBNReLU(nn.Module):
    """Conv -> BatchNorm -> (optional) ReLU."""

    def __init__(self, ic, oc, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        layers = [nn.Conv2d(ic, oc, k, s, p, groups=g, bias=False), nn.BatchNorm2d(oc)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel re-weighting."""

    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).view(x.shape[0], -1, 1, 1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with optional SE and spatial dropout, plus a (projected) identity."""

    def __init__(self, ic, oc, use_se=True, drop_p=0.0):
        super().__init__()
        self.c1 = ConvBNReLU(ic, oc)
        self.c2 = nn.Sequential(nn.Conv2d(oc, oc, 3, 1, 1, bias=False), nn.BatchNorm2d(oc))
        self.se = SEBlock(oc) if use_se else nn.Identity()
        self.drop = nn.Dropout2d(drop_p) if drop_p > 0 else nn.Identity()
        self.skip = (nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc))
                     if ic != oc else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.drop(self.se(self.c2(self.c1(x)))) + self.skip(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al.) that re-weights a skip feature ``x`` using ``g``."""

    def __init__(self, g_ch, x_ch):
        super().__init__()
        mid = max(g_ch // 2, 4)
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, mid, 1, bias=False), nn.BatchNorm2d(mid))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, mid, 1, bias=False), nn.BatchNorm2d(mid))
        self.psi = nn.Sequential(nn.Conv2d(mid, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.Wg(g) + self.Wx(x), inplace=True))


class HRBranch(nn.Module):
    """High-resolution CNN branch: 7x7/2 stem, three strided stages -> h0, h1, h2."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, base * 2, k=7, s=2, p=3)
        self.d1 = ConvBNReLU(base * 2, base * 2, s=2)
        self.r1 = ResBlock(base * 2, base * 2, use_se=False)
        self.d2 = ConvBNReLU(base * 2, base * 4, s=2)
        self.r2 = ResBlock(base * 4, base * 4)
        self.d3 = ConvBNReLU(base * 4, base * 8, s=2)
        self.r3 = ResBlock(base * 8, base * 8)
        self.out_chs = [base * 2, base * 4, base * 8]

    def forward(self, x):
        s = self.stem(x)
        h0 = self.r1(self.d1(s))
        h1 = self.r2(self.d2(h0))
        h2 = self.r3(self.d3(h1))
        return h0, h1, h2


class SwinEncoderWrapper(nn.Module):
    """timm SwinV2 ``features_only`` encoder returning NCHW feature maps.

    The channel counts and memory layout (timm Swin returns NHWC) are detected with one dummy
    forward pass at construction time; ``img_size`` must equal the training/inference input size.
    """

    def __init__(self, model_name, pretrained=True, img_size=512):
        super().__init__()
        self.enc = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                     out_indices=(0, 1, 2, 3), img_size=img_size)
        self.enc.eval()
        with torch.no_grad():
            feats = self.enc(torch.zeros(1, 3, img_size, img_size))
        self.out_chs: list[int] = []
        self._nhwc: list[bool] = []
        for f in feats:
            d = f.shape
            if d[1] == d[2]:
                self.out_chs.append(d[3])
                self._nhwc.append(True)
            else:
                self.out_chs.append(d[1])
                self._nhwc.append(False)

    def forward(self, x):
        feats = self.enc(x)
        return [f.permute(0, 3, 1, 2).contiguous() if nh else f
                for f, nh in zip(feats, self._nhwc)]


class FeatureFusion(nn.Module):
    """Concatenate a Swin feature with the up-sampled CNN feature and project with a 3x3 conv."""

    def __init__(self, ca, cb, out):
        super().__init__()
        self.conv = ConvBNReLU(ca + cb, out)

    def forward(self, a, b):
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, a.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([a, b], 1))


class DecoderBlock(nn.Module):
    """Resize to the skip, gate the skip with an attention gate, SE residual block."""

    def __init__(self, in_c, skip_c, out_c, drop_p=0.0):
        super().__init__()
        self.attn = AttentionGate(g_ch=in_c, x_ch=skip_c)
        self.res = ResBlock(in_c + skip_c, out_c, use_se=True, drop_p=drop_p)

    def forward(self, x, skip):
        x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([x_up, self.attn(g=x_up, x=skip)], dim=1))


class VesselRefinementBlock(nn.Module):
    """Parallel 3x3 convolutions with dilation 1, 2, 4, fused and re-weighted by SE."""

    def __init__(self, ch):
        super().__init__()
        self.d1 = ConvBNReLU(ch, ch)
        self.d2 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False),
                                nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.d4 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=4, dilation=4, bias=False),
                                nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.fuse = ConvBNReLU(ch * 3, ch)
        self.se = SEBlock(ch)
        self.proj = ConvBNReLU(ch, ch)

    def forward(self, x):
        return self.proj(self.se(self.fuse(torch.cat([self.d1(x), self.d2(x), self.d4(x)], 1))))


class TopologyRefinementBlock(nn.Module):
    """Depthwise 3x3 conv followed by parallel elongated 1x15 and 15x1 convolutions."""

    def __init__(self, ch):
        super().__init__()
        self.dw = ConvBNReLU(ch, ch, k=3, g=ch)
        self.horiz = nn.Sequential(nn.Conv2d(ch, ch, (1, 15), padding=(0, 7), bias=False),
                                   nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.vert = nn.Sequential(nn.Conv2d(ch, ch, (15, 1), padding=(7, 0), bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.fuse = ConvBNReLU(ch * 2, ch)

    def forward(self, x):
        d = self.dw(x)
        return self.fuse(torch.cat([self.horiz(d), self.vert(d)], 1))


class SwinHRUNetPP(nn.Module):
    """SwinHRUNetPP vessel segmenter.

    In training mode ``forward`` returns ``(main, aux_d4, aux_d3, aux_d2)`` logits, all at input
    resolution (deep supervision); in eval mode it returns the main logit map only.
    """

    def __init__(self, model_name: str = SWIN_MODEL_NAME, pretrained: bool = True,
                 img_size: int = IMG_SIZE, hr_base: int = HR_BASE_CH,
                 dec_chs: tuple[int, ...] = DEC_CHS, drop_p: float = DROPOUT_P):
        super().__init__()
        dc = dec_chs
        self.swin = SwinEncoderWrapper(model_name, pretrained, img_size)
        sc = self.swin.out_chs
        self.hr = HRBranch(in_ch=3, base=hr_base)
        hc = self.hr.out_chs
        self.fuse1 = FeatureFusion(sc[1], hc[2], dc[2])
        self.fuse0 = FeatureFusion(sc[0], hc[1], dc[3])
        self.bn = ResBlock(sc[3], dc[0])
        self.dec4 = DecoderBlock(dc[0], sc[2], dc[1], drop_p=drop_p)
        self.dec3 = DecoderBlock(dc[1], dc[2], dc[2], drop_p=drop_p)
        self.dec2 = DecoderBlock(dc[2], dc[3], dc[3], drop_p=drop_p)
        self.dec1 = DecoderBlock(dc[3], hc[0], dc[4], drop_p=drop_p)
        self.vr = VesselRefinementBlock(dc[4])
        self.tr = TopologyRefinementBlock(dc[4])
        self.head_main = nn.Conv2d(dc[4], 1, 1)
        self.head_a1 = nn.Conv2d(dc[1], 1, 1)
        self.head_a2 = nn.Conv2d(dc[2], 1, 1)
        self.head_a3 = nn.Conv2d(dc[3], 1, 1)

    def forward(self, x):
        s0, s1, s2, s3 = self.swin(x)
        h0, h1, h2 = self.hr(x)
        bt = self.bn(s3)
        d4 = self.dec4(bt, s2)
        f1 = self.fuse1(s1, h2)
        d3 = self.dec3(d4, f1)
        f0 = self.fuse0(s0, h1)
        d2 = self.dec2(d3, f0)
        d1 = self.dec1(d2, h0)
        out = self.tr(self.vr(d1))
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        main = self.head_main(out)
        if self.training:
            def up(t):
                return F.interpolate(t, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return main, up(self.head_a1(d4)), up(self.head_a2(d3)), up(self.head_a3(d2))
        return main


def init_head_bias(model: SwinHRUNetPP, vessel_ratio: float) -> float:
    """Initialise all four output-head biases to the log-odds of the vessel prior.

    ``vessel_ratio`` is the fraction of vessel pixels inside the FOV (clamped below at 0.01, as in
    the original notebook). Returns the bias value used.
    """
    r = max(float(vessel_ratio), 0.01)
    bias = math.log(r / (1 - r))
    for head in (model.head_main, model.head_a1, model.head_a2, model.head_a3):
        nn.init.constant_(head.bias, bias)
    return bias


def build_swinhrunetpp(pretrained: bool = True, img_size: int = IMG_SIZE) -> SwinHRUNetPP:
    """Build SwinHRUNetPP with the configuration reported in the paper (110.2 M parameters)."""
    return SwinHRUNetPP(model_name=SWIN_MODEL_NAME, pretrained=pretrained, img_size=img_size,
                        hr_base=HR_BASE_CH, dec_chs=DEC_CHS, drop_p=DROPOUT_P)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters (trainable and frozen)."""
    return sum(p.numel() for p in model.parameters())
