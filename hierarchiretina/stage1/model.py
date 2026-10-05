"""Stage I classifier: timm backbone + (GeM || attention pooling) + MLP head -> one logit.

Attribute names (``backbone``, ``gem``, ``attn``, ``head``) and the ``head`` indices are those of
the original notebooks, so their checkpoints load with ``strict=True``:

    head.0 Dropout(d)  head.1 LayerNorm(2C)  head.2 Linear(2C, H)  head.3 GELU
    head.4 Dropout(d/2)  head.5 Linear(H, 1)

with C = 1536 for both backbones and H = 1536 (ConvNeXt V2-L) or 768 (SwinV2-L).
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Stage1Config


class GeM(nn.Module):
    """Generalised-mean pooling with a learnable exponent p (initialised at 3)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, C, 1, 1]."""
        return F.avg_pool2d(x.clamp(self.eps).pow(self.p), x.shape[-2:]).pow(1 / self.p)

    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, N, C] -> [B, C]."""
        return seq.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class AttentionPool(nn.Module):
    """Softmax attention over tokens: sum_n a_n x_n with a = softmax(MLP(x))."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C] -> [B, C]."""
        return (torch.softmax(self.attn(x), dim=1) * x).sum(1)


class DRClassifier(nn.Module):
    """Binary DR screening model.

    Args:
        model_name: timm backbone identifier.
        hidden_dim: width of the hidden head layer (head.2).
        dropout: dropout before LayerNorm; dropout/2 before the last Linear.
        pretrained: load timm pre-trained weights for the backbone.
        backbone_img_size: passed to timm (SwinV2 only).
        seq_gem_learnable: pooling of token-shaped ([B,H,W,C] / [B,N,C]) features.
            True  -> GeM with the learnable p (SwinV2 training notebook).
            False -> GeM with p fixed at 3 (ConvNeXt notebooks and the hybrid test-time loader).
            ConvNeXt features are [B,C,H,W] and use the learnable p in both modes.
        input_size: if set, inputs of another size are bicubically resized to it (SwinV2).
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        dropout: float,
        pretrained: bool = False,
        backbone_img_size: int | None = None,
        seq_gem_learnable: bool = False,
        input_size: int | None = None,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=0, global_pool="")
        if backbone_img_size is not None:
            kwargs["img_size"] = backbone_img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.feat_dim = self.backbone.num_features
        self.gem = GeM()
        self.attn = AttentionPool(self.feat_dim)
        cd = self.feat_dim * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(cd),
            nn.Linear(cd, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        self.seq_gem_learnable = seq_gem_learnable
        self.input_size = input_size

    @staticmethod
    def _gem_p3(seq: torch.Tensor) -> torch.Tensor:
        return seq.clamp(1e-6).pow(3).mean(1).pow(1 / 3)

    def pool(self, f: torch.Tensor) -> torch.Tensor:
        """Backbone features -> concatenated [GeM, attention] vector of size 2C.

        The branch order of each original notebook is kept verbatim.
        """
        c = self.feat_dim
        if self.seq_gem_learnable:                              # SwinV2 training notebook
            if f.dim() == 4 and f.shape[1] == c:
                seq = f.flatten(2).transpose(1, 2)
            elif f.dim() == 4 and f.shape[-1] == c:
                seq = f.reshape(f.shape[0], -1, c)
            elif f.dim() == 3:
                seq = f
            else:
                raise ValueError(f"Unexpected feature shape {tuple(f.shape)}")
            return torch.cat([self.gem.forward_seq(seq), self.attn(seq)], dim=1)

        # ConvNeXt notebooks and the hybrid test-time loader
        if f.dim() == 4 and f.shape[-1] == c:                   # [B, H, W, C]
            seq = f.reshape(f.shape[0], -1, c)
            return torch.cat([self._gem_p3(seq), self.attn(seq)], dim=1)
        if f.dim() == 4 and f.shape[1] == c:                    # [B, C, H, W] (ConvNeXt)
            return torch.cat([self.gem(f).flatten(1), self.attn(f.flatten(2).transpose(1, 2))],
                             dim=1)
        if f.dim() == 3:                                        # [B, N, C]
            return torch.cat([self._gem_p3(f), self.attn(f)], dim=1)
        raise ValueError(f"Unexpected feature shape {tuple(f.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None and tuple(x.shape[-2:]) != (self.input_size,) * 2:
            x = F.interpolate(x, (self.input_size, self.input_size), mode="bicubic",
                              align_corners=False)
        return self.head(self.pool(self.backbone.forward_features(x)))

    def get_param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        """AdamW groups: backbone at ``lr_backbone``; GeM, attention pool, head at ``lr_head``."""
        return [
            {"params": list(self.backbone.parameters()), "lr": lr_backbone},
            {"params": list(self.gem.parameters()) + list(self.attn.parameters())
                       + list(self.head.parameters()), "lr": lr_head},
        ]


def build_model(cfg: Stage1Config, pretrained: bool = True, **overrides) -> DRClassifier:
    """Instantiate the classifier of a preset. ``overrides`` replace constructor arguments
    (e.g. ``seq_gem_learnable=False, input_size=None`` reproduces the hybrid test-time loader
    used for the cached SwinV2 baseline predictions)."""
    kwargs = dict(
        model_name=cfg.model_name,
        hidden_dim=cfg.head_hidden,
        dropout=cfg.dropout,
        pretrained=pretrained,
        backbone_img_size=cfg.img_size if cfg.pass_img_size_to_backbone else None,
        seq_gem_learnable=cfg.seq_gem_learnable,
        input_size=cfg.img_size if cfg.resize_input else None,
    )
    kwargs.update(overrides)
    return DRClassifier(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_meta(ckpt_path: str | Path) -> dict:
    """Scalar metadata of a checkpoint (epoch, auc, threshold, ...), memory-mapped when possible.

    For the deployed gate this returns epoch 12, auc 0.9546 and threshold 0.2391715943813324.
    """
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {k: v for k, v in ck.items() if isinstance(v, (int, float, str)) or v is None}


def load_eval_weights(model: nn.Module, ckpt_path: str | Path, strict: bool = True) -> dict:
    """Load a training checkpoint and overwrite parameters with its EMA shadow.

    Returns the checkpoint metadata (epoch, auc, threshold, ...) without the tensors.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=strict)
    shadow = ck.get("ema_shadow")
    if shadow:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in shadow.items():
                if k in params and params[k].shape == v.shape:
                    params[k].copy_(v.to(device=params[k].device, dtype=params[k].dtype))
    model.eval()
    meta = {k: v for k, v in ck.items() if k not in ("model", "optimizer", "scheduler",
                                                     "scaler", "ema_shadow")}
    meta["ema_applied"] = bool(shadow)
    return meta
