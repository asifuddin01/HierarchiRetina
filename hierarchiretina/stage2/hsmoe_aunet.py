"""HSMoE-AUNet: Hierarchical Sparse Mixture-of-Experts Attention U-Net (Stage II lesions).

One configurable implementation of the four lesion models (MA, HE, EX, CWS). The four
original notebooks differ only in (i) the pool of expert *types* that is cycled inside every
MoE block, (ii) the number of experts per block, and (iii) training / inference settings.
All of these live in :data:`LESION_CONFIGS`; :func:`build_hsmoe_aunet` builds the model.

Module attribute names and nesting are identical to the original notebook classes, so the
released per-lesion ``best_model.pth`` checkpoints load with ``strict=True``.

Architecture (input 1024x1024):
    ConvNeXt-Small encoder -> f0..f3 (96/192/384/768 ch at /4, /8, /16, /32)
    skip_i   = CBAM -> SparseMoE (64/128/256 ch)
    bottle   = SparseMoE (768 -> 512) -> global multi-head self-attention (8 heads)
    dec3..1  = upsample, 1x1 reduce, concat skip, SparseMoE, CBAM (256/128/64 ch)
    dec0     = upsample + plain 3x3 conv (32 ch); head upsamples to full resolution
    aux heads (sigmoid) at the dec3 (/16) and dec2 (/8) levels for deep supervision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# timm tag resolved by ``timm.create_model('convnext_small', pretrained=True)`` in the
# original runs (training logs: "Loading pretrained weights ... convnext_small.in12k_ft_in1k").
ENCODER_NAME = "convnext_small.in12k_ft_in1k"
N_FROZEN_ENCODER_TENSORS = 20  # first 20 encoder parameter tensors are frozen


# =============================================================================
# Expert modules
# =============================================================================
class ConvExpert(nn.Module):
    """Two 3x3 convolutions with a residual projection (dots, precise edges)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class DilatedExpert(nn.Module):
    """Dilated 3x3 convolution (dilation 2): lesion neighbourhood context."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class DWSExpert(nn.Module):
    """Depthwise-separable 5x5 convolution (texture)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class LargeKernelExpert(nn.Module):
    """7x7 large-kernel convolution (large bright regions)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class MultiScaleExpert(nn.Module):
    """1x1 expand, parallel depthwise 3x3/5x5/7x7 branches, 1x1 fuse (HE, EX, CWS)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.dw3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.dw5 = nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch, bias=False)
        self.dw7 = nn.Conv2d(out_ch, out_ch, 7, padding=3, groups=out_ch, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.expand(x)
        m = torch.cat([self.dw3(y), self.dw5(y), self.dw7(y)], dim=1)
        return self.act(self.fuse(m) + self.skip(x))


class BrightSpotExpert(nn.Module):
    """Morphological top-hat expert (max-pool minus min-pool, 7x7) for bright EX and CWS.

    The top-hat response is blended with the raw feature through a learnable scalar gate.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.kernel_size = kernel_size
        self.pad = pad
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.expand(x)
        max_p = F.max_pool2d(y, self.kernel_size, stride=1, padding=self.pad)
        min_p = -F.max_pool2d(-y, self.kernel_size, stride=1, padding=self.pad)
        top_hat = max_p - min_p
        g = torch.sigmoid(self.gate)
        y = g * top_hat + (1 - g) * y
        return self.act(self.refine(y) + self.skip(x))


class EdgeEnhancingExpert(nn.Module):
    """Difference-of-Gaussians edge expert (sharp-edged EX).

    The depthwise blurs are initialised as 3x3 / 5x5 binomial kernels. Note: as in the
    original, :meth:`HSMoEAUNet._init_weights` later re-initialises every Conv2d, so this
    initialisation is overwritten when the expert is part of the full model.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.blur_s = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.blur_l = nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch, bias=False)
        with torch.no_grad():
            k3 = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
            k5 = torch.tensor([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
            self.blur_s.weight.copy_(k3.view(1, 1, 3, 3).expand(out_ch, 1, 3, 3).clone())
            self.blur_l.weight.copy_(k5.view(1, 1, 5, 5).expand(out_ch, 1, 5, 5).clone())
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.expand(x)
        edges = self.blur_s(y) - self.blur_l(y)
        return self.act(self.fuse(edges) + self.skip(x))


class SoftRegionExpert(nn.Module):
    """Low-pass expert: average pooling at 3/5/7 fused by 1x1 (diffuse CWS)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.pool3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.pool5 = nn.AvgPool2d(5, stride=1, padding=2)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.expand(x)
        smooth = torch.cat([self.pool3(y), self.pool5(y), self.pool7(y)], dim=1)
        return self.act(self.fuse(smooth) + self.skip(x))


class TextureExpert(nn.Module):
    """Local mean and local mean-absolute-deviation (5x5) expert (CWS).

    MAD is used instead of variance because ``E[x^2]-E[x]^2`` overflows in FP16.
    """

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 5):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.ksize = ksize
        self.pool = nn.AvgPool2d(ksize, stride=1, padding=ksize // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.expand(x)
        mean = self.pool(y)
        mad = self.pool((y - mean).abs())
        feat = torch.cat([mean, mad], dim=1)
        return self.act(self.fuse(feat) + self.skip(x))


EXPERT_REGISTRY: dict[str, type[nn.Module]] = {
    "conv": ConvExpert,
    "dilated": DilatedExpert,
    "dws": DWSExpert,
    "large_kernel": LargeKernelExpert,
    "multiscale": MultiScaleExpert,
    "bright_spot": BrightSpotExpert,
    "edge_enhancing": EdgeEnhancingExpert,
    "soft_region": SoftRegionExpert,
    "texture": TextureExpert,
}
BASE_EXPERT_TYPES = ("conv", "dilated", "dws", "large_kernel")


def _resolve_expert_types(expert_types: Sequence[str | type]) -> tuple[type[nn.Module], ...]:
    return tuple(EXPERT_REGISTRY[t] if isinstance(t, str) else t for t in expert_types)


# =============================================================================
# Router and sparse MoE block
# =============================================================================
class NoisyTopKGate(nn.Module):
    """Per-image router: GAP -> Linear(C, max(C/4, N)) -> GELU -> Linear(., N).

    Gaussian noise (std ``noise_eps``) is added to the logits in training; the top-k logits
    are normalised with a softmax.
    """

    def __init__(self, in_ch: int, num_experts: int, k: int = 2, noise_eps: float = 1e-2):
        super().__init__()
        self.k, self.num_experts, self.noise_eps = k, num_experts, noise_eps
        hidden = max(in_ch // 4, num_experts)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x):
        logits = self.gate(x)                                       # (B, E)
        if self.training and self.noise_eps > 0:
            logits = logits + torch.randn_like(logits) * self.noise_eps
        top_w, top_idx = torch.topk(logits, self.k, dim=-1)        # (B, k)
        top_w = F.softmax(top_w, dim=-1)
        return top_w, top_idx, logits


class SparseMoEBlock(nn.Module):
    """Heterogeneous sparse MoE: y = BN(sum_{i in top-k} g_i E_i(x) + W_s x).

    Expert ``i`` has type ``expert_types[i % len(expert_types)]``. All experts are evaluated
    (sparse in the mixture, not in compute). Returns ``(y, load_balance_loss)`` with
    ``load_balance_loss = N * sum_i mean_softmax_prob_i * selection_fraction_i``.
    """

    def __init__(self, in_ch: int, out_ch: int, num_experts: int = 8, k: int = 2,
                 noise_eps: float = 1e-2,
                 expert_types: Sequence[str | type] = BASE_EXPERT_TYPES):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        types = _resolve_expert_types(expert_types)
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            cls = types[i % len(types)]
            kwargs = {"dilation": 2} if cls is DilatedExpert else {}
            self.experts.append(cls(in_ch, out_ch, **kwargs))
        self.gate = NoisyTopKGate(in_ch, num_experts, k, noise_eps)
        self.proj_skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                          if in_ch != out_ch else nn.Identity())
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        B, _, H, W = x.shape
        top_w, top_idx, gate_logits = self.gate(x)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)   # (B, E, C, H, W)
        out_ch = expert_outs.shape[2]

        idx = top_idx[:, :, None, None, None].expand(B, self.k, out_ch, H, W)
        selected = expert_outs.gather(1, idx)                             # (B, k, C, H, W)
        output = (selected * top_w[:, :, None, None, None]).sum(dim=1)
        output = self.norm(output + self.proj_skip(x))

        importance = F.softmax(gate_logits, dim=-1).mean(0)               # (E,)
        hot = F.one_hot(top_idx.reshape(-1), self.num_experts).float()
        load = hot.view(B, self.k, self.num_experts).mean(0).mean(0)      # (E,)
        aux_loss = self.num_experts * (importance * load).sum()
        return output, aux_loss


# =============================================================================
# Attention modules
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(ch, ch // r), nn.ReLU(),
                                nn.Linear(ch // r, ch))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.sig((self.fc(self.avg(x)) + self.fc(self.max(x))).unsqueeze(-1).unsqueeze(-1))
        return x * a


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx = x.max(1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional block attention module: channel then spatial attention."""

    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class GlobalSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention over flattened spatial tokens, with residual."""

    def __init__(self, ch: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_d = ch // heads
        self.scale = self.head_d ** -0.5
        self.norm = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch, bias=False)
        self.proj = nn.Linear(ch, ch, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        xf = x.flatten(2).transpose(1, 2)                                 # (B, N, C)
        residual = xf
        qkv = self.qkv(self.norm(xf))
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out) + residual
        return out.transpose(1, 2).reshape(B, C, H, W)


# =============================================================================
# U-Net blocks
# =============================================================================
class SkipProcessor(nn.Module):
    """CBAM followed by a sparse MoE block on an encoder skip feature."""

    def __init__(self, in_ch: int, out_ch: int, num_experts: int = 8, k: int = 2,
                 expert_types: Sequence[str | type] = BASE_EXPERT_TYPES):
        super().__init__()
        self.cbam = CBAM(in_ch)
        self.moe = SparseMoEBlock(in_ch, out_ch, num_experts, k, expert_types=expert_types)

    def forward(self, x):
        return self.moe(self.cbam(x))


class DecoderBlock(nn.Module):
    """Upsample x2, 1x1 halve channels, concat skip, sparse MoE, CBAM."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_experts: int = 8, k: int = 2,
                 expert_types: Sequence[str | type] = BASE_EXPERT_TYPES):
        super().__init__()
        self.up_proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, 1, bias=False),
            nn.BatchNorm2d(in_ch // 2), nn.GELU(),
        )
        self.moe = SparseMoEBlock(in_ch // 2 + skip_ch, out_ch, num_experts, k,
                                  expert_types=expert_types)
        self.cbam = CBAM(out_ch)

    def forward(self, x, skip):
        x = torch.cat([self.up_proj(x), skip], dim=1)
        x, aux = self.moe(x)
        return self.cbam(x), aux


class AuxHead(nn.Module):
    """Deep-supervision head; returns probabilities (sigmoid applied)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2), nn.GELU(),
            nn.Conv2d(in_ch // 2, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class HSMoEAUNet(nn.Module):
    """Hierarchical Sparse MoE Attention U-Net.

    ``forward`` returns ``(logits, aux3_prob, aux2_prob, moe_loss)``: main-head logits at input
    resolution, the two auxiliary probability maps (/16 and /8), and the load-balancing loss
    summed over the seven MoE blocks and multiplied by 0.01.
    """

    ENC_CH = [96, 192, 384, 768]
    SKIP_CH = [64, 128, 256]
    BOTTLE_CH = 512
    DEC_CH = [256, 128, 64, 32]
    MOE_LOSS_WEIGHT = 0.01

    def __init__(self, expert_types: Sequence[str | type] = BASE_EXPERT_TYPES,
                 skip_experts: Sequence[int] = (6, 8, 10), bottle_experts: int = 12,
                 dec_experts: Sequence[int] = (10, 8, 6), pretrained: bool = True, k: int = 2,
                 encoder_name: str = ENCODER_NAME):
        super().__init__()
        self.SKIP_EXPERTS = list(skip_experts)
        self.BOTTLE_EXPERTS = bottle_experts
        self.DEC_EXPERTS = list(dec_experts)
        et = tuple(expert_types)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained,
                                         features_only=True, out_indices=(0, 1, 2, 3))
        for param in list(self.encoder.parameters())[:N_FROZEN_ENCODER_TENSORS]:
            param.requires_grad = False

        self.skip0 = SkipProcessor(self.ENC_CH[0], self.SKIP_CH[0], skip_experts[0], k, et)
        self.skip1 = SkipProcessor(self.ENC_CH[1], self.SKIP_CH[1], skip_experts[1], k, et)
        self.skip2 = SkipProcessor(self.ENC_CH[2], self.SKIP_CH[2], skip_experts[2], k, et)

        self.bottle_moe = SparseMoEBlock(self.ENC_CH[3], self.BOTTLE_CH, bottle_experts, k,
                                         expert_types=et)
        self.bottle_attn = GlobalSelfAttention(self.BOTTLE_CH, heads=8)

        self.dec3 = DecoderBlock(self.BOTTLE_CH, self.SKIP_CH[2], self.DEC_CH[0],
                                 dec_experts[0], k, et)
        self.dec2 = DecoderBlock(self.DEC_CH[0], self.SKIP_CH[1], self.DEC_CH[1],
                                 dec_experts[1], k, et)
        self.dec1 = DecoderBlock(self.DEC_CH[1], self.SKIP_CH[0], self.DEC_CH[2],
                                 dec_experts[2], k, et)
        self.dec0_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.DEC_CH[2], self.DEC_CH[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.DEC_CH[3]), nn.GELU(),
        )

        self.output_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.output_conv = nn.Sequential(
            nn.Conv2d(self.DEC_CH[3], 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

        self.aux_head3 = AuxHead(self.DEC_CH[0])
        self.aux_head2 = AuxHead(self.DEC_CH[1])
        self._init_weights()

    def _init_weights(self):
        # Kept exactly as in the original notebooks: the loop visits *all* modules, including
        # the timm encoder. ConvNeXt's Conv2d (stem, depthwise, downsample) and Linear (MLP)
        # weights are therefore re-initialised after the pretrained weights were loaded; only
        # its LayerNorm weights and layer-scale (gamma) parameters keep pretrained values.
        # This is what produced the released checkpoints and is preserved for fidelity.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)

        s0, a_s0 = self.skip0(f0)
        s1, a_s1 = self.skip1(f1)
        s2, a_s2 = self.skip2(f2)

        b, a_b = self.bottle_moe(f3)
        b = self.bottle_attn(b)

        d3, a_d3 = self.dec3(b, s2)
        d2, a_d2 = self.dec2(d3, s1)
        d1, a_d1 = self.dec1(d2, s0)
        d0 = self.dec0_up(d1)

        out = self.output_conv(self.output_up(d0))
        aux3 = self.aux_head3(d3)
        aux2 = self.aux_head2(d2)
        moe_aux_loss = (a_s0 + a_s1 + a_s2 + a_b + a_d3 + a_d2 + a_d1) * self.MOE_LOSS_WEIGHT
        return out, aux3, aux2, moe_aux_loss


# =============================================================================
# Per-lesion configuration (values as run in the four training notebooks)
# =============================================================================
@dataclass
class LesionLossConfig:
    """Loss = FTL + Dice + bce_w*BCE [+ boundary] [+ region coherence] [+ optic disc]
    + aux_w * mean(FTL+Dice of the two aux heads) + MoE load balance.

    If ``phase_epochs=(e1, e2)`` (CWS only): epoch < e1 uses FTL(gamma_warmup) + Dice + BCE;
    e1 <= epoch < e2 adds the optic-disc term; epoch >= e2 adds boundary and region terms and
    switches the main FTL to ``ftl_gamma``. Aux heads then always use ``ftl_gamma_warmup``.
    """

    ftl_alpha: float            # weight on false negatives
    ftl_beta: float             # weight on false positives
    ftl_gamma: float
    bce_w: float = 0.1
    bce_pos_weight: float | None = None
    boundary: str | None = None       # 'sobel' (HE) | 'dual_edge' (EX) | 'soft' (CWS)
    boundary_w: float = 0.0
    region_coherence_w: float = 0.0
    optic_disc_w: float = 0.0
    optic_disc_open_ksize: int = 15
    ftl_w: float = 1.0
    dice_w: float = 1.0
    aux_w: float = 0.3
    phase_epochs: tuple[int, int] | None = None
    ftl_gamma_warmup: float | None = None


@dataclass
class LesionConfig:
    """Everything that differs between the four lesion notebooks."""

    name: str
    full_name: str
    # --- data ---
    min_lesion_pixels: int            # mask dropped if positive px (original resolution) <= this
    mask_strip_suffixes: tuple[str, ...]  # removed from mask stems to pair them with images
    pixel_col: str                    # manifest column with positive pixels after cropping
    expected_split_sizes: tuple[int, int, int]  # (train, val, test) of the paper run
    # --- model ---
    expert_types: tuple[str, ...]
    skip_experts: tuple[int, int, int]
    bottle_experts: int
    dec_experts: tuple[int, int, int]
    # --- loss ---
    loss: LesionLossConfig
    # --- optimisation ---
    lr: float
    scheduler: str                    # 'plateau' | 'cosine_restarts' | 'cosine'
    scheduler_kwargs: dict = field(default_factory=dict)
    weight_decay: float = 1e-4
    batch_size: int = 2               # DataLoader batch size actually used (see README)
    accum_steps: int = 4
    max_epochs: int = 400
    patience: int = 40
    grad_clip: float = 1.0
    nan_guard: bool = False
    # --- operating point / evaluation ---
    threshold: float = 0.5            # deployed probability threshold
    threshold_note: str = ""
    sweep_grid: tuple[float, float, float] = (0.2, 0.7, 0.05)  # np.arange(start, stop, step)
    sweep_tta: bool = False
    tta: bool = False                 # H/V flip averaging at test and deployment
    postprocess: dict | None = None   # CWS: {'close_ksize': 5, 'min_area': 40}
    lesion_iou: float = 0.3           # lesion-level matching IoU in the paper table
    # --- deployment (Stage I-routed images) ---
    inference_crop: str = "circle"    # 'circle' (MA, HE) | 'rect' (EX, CWS)
    inference_resize: str = "stretch"  # 'stretch' (MA, HE) | 'letterbox' (EX, CWS)
    inference_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    mask_suffix: str = "_mask"        # output name: <image stem><mask_suffix>.png
    # --- reference values (paper) ---
    paper_params_m: float = 0.0
    paper_best_epoch: int = 0


LESION_CONFIGS: dict[str, LesionConfig] = {
    "MA": LesionConfig(
        name="MA", full_name="Microaneurysms",
        min_lesion_pixels=10, mask_strip_suffixes=("_MA", "_ma"), pixel_col="ma_pixels",
        expected_split_sizes=(1232, 353, 175),
        expert_types=BASE_EXPERT_TYPES,
        skip_experts=(6, 8, 10), bottle_experts=12, dec_experts=(10, 8, 6),
        loss=LesionLossConfig(ftl_alpha=0.3, ftl_beta=0.7, ftl_gamma=2.0, bce_w=0.1),
        lr=1e-4, scheduler="plateau",
        scheduler_kwargs=dict(mode="max", factor=0.5, patience=15, min_lr=1e-7),
        batch_size=4, grad_clip=1.0,
        threshold=0.01,
        threshold_note=("Manual: validation Dice and recall keep rising as the threshold "
                        "falls (wide sweep 0.001-0.5); 0.01 deployed."),
        sweep_grid=(0.001, 0.5, 0.005), sweep_tta=False, tta=False, postprocess=None,
        lesion_iou=0.2,
        inference_crop="circle", inference_resize="stretch",
        inference_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".ppm", ".tif"),
        mask_suffix="_mask", paper_params_m=218.3, paper_best_epoch=262,
    ),
    "HE": LesionConfig(
        name="HE", full_name="Haemorrhages",
        min_lesion_pixels=50,
        mask_strip_suffixes=("_HE", "_he", "_Hemorrhages", "_hemorrhages"), pixel_col="he_pixels",
        expected_split_sizes=(1101, 316, 157),
        expert_types=BASE_EXPERT_TYPES + ("multiscale",),
        skip_experts=(6, 8, 10), bottle_experts=12, dec_experts=(10, 8, 6),
        loss=LesionLossConfig(ftl_alpha=0.5, ftl_beta=0.5, ftl_gamma=4.0 / 3.0, bce_w=0.1,
                              boundary="sobel", boundary_w=0.3),
        lr=1e-4, scheduler="cosine_restarts",
        scheduler_kwargs=dict(T_0=20, T_mult=2, eta_min=1e-7),
        batch_size=2, grad_clip=1.0,
        threshold=0.20, threshold_note="Validation Dice optimum (sweep 0.20-0.70, step 0.05).",
        sweep_grid=(0.20, 0.71, 0.05), sweep_tta=False, tta=False, postprocess=None,
        lesion_iou=0.3,
        inference_crop="circle", inference_resize="stretch",
        mask_suffix="_he_mask", paper_params_m=190.5, paper_best_epoch=128,
    ),
    "EX": LesionConfig(
        name="EX", full_name="Hard exudates",
        min_lesion_pixels=30,
        mask_strip_suffixes=("_EX", "_ex", "_Hard_Exudates", "_hard_exudates", "_HardExudates",
                             "_Exudates", "_exudates"),
        pixel_col="ex_pixels",
        expected_split_sizes=(998, 287, 142),
        expert_types=BASE_EXPERT_TYPES + ("multiscale", "bright_spot", "edge_enhancing"),
        skip_experts=(7, 9, 11), bottle_experts=14, dec_experts=(11, 9, 7),
        loss=LesionLossConfig(ftl_alpha=0.45, ftl_beta=0.55, ftl_gamma=1.0, bce_w=0.1,
                              boundary="dual_edge", boundary_w=0.5,
                              optic_disc_w=0.15, optic_disc_open_ksize=15),
        lr=1e-4, scheduler="cosine", scheduler_kwargs=dict(T_max=400, eta_min=1e-7),
        batch_size=2, grad_clip=1.0,
        threshold=0.325,
        threshold_note="Validation Dice optimum with TTA (sweep 0.20-0.65, step 0.025).",
        sweep_grid=(0.2, 0.65, 0.025), sweep_tta=True, tta=True, postprocess=None,
        lesion_iou=0.3,
        inference_crop="rect", inference_resize="letterbox",
        mask_suffix="_ex_mask", paper_params_m=187.8, paper_best_epoch=199,
    ),
    "CWS": LesionConfig(
        name="CWS", full_name="Cotton-wool spots",
        min_lesion_pixels=20,
        mask_strip_suffixes=("_CWS", "_cws", "_SE", "_se", "_Soft_Exudates", "_soft_exudates",
                             "_SoftExudates", "_Cotton_Wool_Spots", "_cotton_wool_spots",
                             "_CottonWoolSpots"),
        pixel_col="cws_pixels",
        expected_split_sizes=(480, 138, 69),
        expert_types=BASE_EXPERT_TYPES + ("multiscale", "bright_spot", "soft_region",
                                          "texture"),
        skip_experts=(8, 10, 12), bottle_experts=16, dec_experts=(12, 10, 8),
        loss=LesionLossConfig(ftl_alpha=0.55, ftl_beta=0.45, ftl_gamma=4.0 / 3.0, bce_w=0.2,
                              bce_pos_weight=10.0, boundary="soft", boundary_w=0.15,
                              region_coherence_w=0.02, optic_disc_w=0.08,
                              optic_disc_open_ksize=11, phase_epochs=(15, 30),
                              ftl_gamma_warmup=1.0),
        lr=7e-5, scheduler="cosine_restarts",
        scheduler_kwargs=dict(T_0=15, T_mult=2, eta_min=1e-7),
        batch_size=2, grad_clip=0.5, nan_guard=True,
        threshold=0.40,
        threshold_note=("Manual: validation Dice optimum 0.575 (sweep 0.05-0.60, TTA); "
                        "0.40 deployed for higher recall."),
        sweep_grid=(0.05, 0.6, 0.025), sweep_tta=True, tta=True,
        postprocess=dict(close_ksize=5, min_area=40),
        lesion_iou=0.3,
        inference_crop="rect", inference_resize="letterbox",
        mask_suffix="_cws_mask", paper_params_m=192.7, paper_best_epoch=84,
    ),
}
LESIONS: tuple[str, ...] = tuple(LESION_CONFIGS)


def get_lesion_config(lesion: str) -> LesionConfig:
    key = lesion.upper()
    if key not in LESION_CONFIGS:
        raise KeyError(f"Unknown lesion '{lesion}'; expected one of {LESIONS}")
    return LESION_CONFIGS[key]


def build_hsmoe_aunet(lesion: str, pretrained: bool = True,
                      encoder_name: str = ENCODER_NAME) -> HSMoEAUNet:
    """Build the HSMoE-AUNet variant used for ``lesion`` ('MA' | 'HE' | 'EX' | 'CWS')."""
    cfg = get_lesion_config(lesion)
    return HSMoEAUNet(expert_types=cfg.expert_types, skip_experts=cfg.skip_experts,
                      bottle_experts=cfg.bottle_experts, dec_experts=cfg.dec_experts,
                      pretrained=pretrained, k=2, encoder_name=encoder_name)


def load_hsmoe_aunet(lesion: str, checkpoint: str, map_location: str | torch.device = "cpu",
                     strict: bool = True) -> HSMoEAUNet:
    """Build the model (no ImageNet download) and load a ``best_model.pth`` checkpoint.

    Accepts both the notebook format ``{'model_state': ..., 'epoch': ..., 'val_dice': ...}``
    and a bare state_dict.
    """
    model = build_hsmoe_aunet(lesion, pretrained=False)
    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=strict)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
