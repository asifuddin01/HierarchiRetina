"""LG-DRG: Lesion-Guided DR Grader (Stage III).

Two streams, one residual cross-attention gate, two heads:

* RGB stream: ConvNeXt V2-Large (``convnextv2_large.fcmae_ft_in22k_in1k_384``, timm,
  ``features_only``); last stage gives F_img in R^{1536 x 16 x 16} at 512 px input.
* Mask stream (``LesionEncoder``): three stride-2 blocks (3x3 conv, BN, GELU; 32/64/128 ch)
  over the five Stage II masks, trained from scratch.
* ``CrossAttnGate``: mask tokens are queries, image tokens are keys/values (8 heads, width
  256), ``F_gated = F_img + gamma * W_o A`` with a learned scalar gamma initialised to 0.
* BN + global average pooling -> 1536-d vector -> ``GradabilityHead`` (1536-256-1, logit of
  P(gradable)) and ``CORNHead`` (1536-512-3, CORN logits over severity grades 1-4).

Attribute names and nesting are identical to the original notebook class, so the released
``state_dict``s load with ``strict=True``. The inference notebook's copy differed in one
respect only: a ``pretrained`` constructor argument (the training notebook always used
``pretrained=True``). That argument is kept here; it does not touch the parameter set.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

K_SEV = 4  # severity grades 1..4 -> 3 CORN logits


class LesionEncoder(nn.Module):
    """Light CNN over the 5 binary masks -> spatial feature map (stride 8)."""

    def __init__(self, in_ch: int = 5, dims: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        c = in_ch
        layers: list[nn.Module] = []
        for d in dims:
            layers += [nn.Conv2d(c, d, 3, 2, 1), nn.BatchNorm2d(d), nn.GELU()]
            c = d
        self.net = nn.Sequential(*layers)
        self.out_ch = dims[-1]

    def forward(self, x):
        return self.net(x)


class CrossAttnGate(nn.Module):
    """Lesion features (queries) attend over image features (keys/values); residual add
    scaled by the learned scalar ``gamma`` (initialised to 0)."""

    def __init__(self, img_ch: int, les_ch: int, dim: int = 256, heads: int = 8):
        super().__init__()
        self.q = nn.Conv2d(les_ch, dim, 1)
        self.k = nn.Conv2d(img_ch, dim, 1)
        self.v = nn.Conv2d(img_ch, dim, 1)
        self.proj = nn.Conv2d(dim, img_ch, 1)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, img_feat, les_feat):
        B, C, H, W = img_feat.shape
        if les_feat.shape[-2:] != (H, W):
            les_feat = F.interpolate(les_feat, (H, W), mode="bilinear", align_corners=False)
        q, k, v = self.q(les_feat), self.k(img_feat), self.v(img_feat)

        def split(t):  # (B, dim, H, W) -> (B, heads, dim/heads, HW)
            return t.flatten(2).reshape(B, self.heads, -1, H * W)

        q, k, v = split(q), split(k), split(v)
        attn = (q.transpose(-1, -2) @ k * self.scale).softmax(-1)   # (B, heads, HW, HW)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2)        # (B, heads, d, HW)
        out = self.proj(out.reshape(B, -1, H, W))
        return img_feat + self.gamma * out


class CORNHead(nn.Module):
    """Conditional ordinal regression over the 4 severity grades -> 3 binary logits."""

    def __init__(self, in_dim: int, k_sev: int = K_SEV, p: float = 0.3):
        super().__init__()
        self.k_sev = k_sev
        self.head = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(512, k_sev - 1))

    def forward(self, x):
        return self.head(x)  # (B, 3)


class GradabilityHead(nn.Module):
    """Binary logit of P(gradable); Grade 5 (ungradable, refer) when P(gradable) < t_g."""

    def __init__(self, in_dim: int, p: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(256, 1))

    def forward(self, x):
        return self.head(x).squeeze(1)  # (B,)


class LGDRG(nn.Module):
    """Two-head grader: gradability (G5 vs gradable) + CORN ordinal severity (G1..G4).

    ``cfg`` needs ``backbone``, ``n_masks``, ``drop_path`` and ``drop_rate``. Input is
    ``(B, 3 + n_masks, H, W)``; output ``(sev_logits[B,3], grad_logit[B])``.
    """

    def __init__(self, cfg, pretrained: bool = False):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(cfg.backbone, pretrained=pretrained,
                                          features_only=True, drop_path_rate=cfg.drop_path)
        ch = self.backbone.feature_info.channels()[-1]
        self.les = LesionEncoder(cfg.n_masks)
        self.gate = CrossAttnGate(ch, self.les.out_ch)
        self.norm = nn.BatchNorm2d(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sev_head = CORNHead(ch, K_SEV, cfg.drop_rate)
        self.grad_head = GradabilityHead(ch, cfg.drop_rate)

    def forward(self, x):
        img, masks = x[:, :3], x[:, 3:]
        feats = self.backbone(img)[-1]
        lf = self.les(masks)
        g = self.norm(self.gate(feats, lf))
        v = self.pool(g).flatten(1)
        return self.sev_head(v), self.grad_head(v)


def load_lgdrg(path: str | Path, cfg, device="cpu", key: str = "ema") -> LGDRG:
    """Build LG-DRG (no ImageNet download) and load a fold checkpoint strictly.

    The reported models use the EMA weights (``key="ema"``). Checkpoints are trusted local
    files written by ``engine.save_ckpt``, hence ``weights_only=False``.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck[key] if isinstance(ck, dict) and key in ck else ck
    m = LGDRG(cfg, pretrained=False).to(device)
    m.load_state_dict(sd, strict=True)
    return m.eval()
