"""Stage I loss: w_bce * BCE(label-smoothed, pos_weight) + w_focal * Focal(label-smoothed).

Label smoothing maps y -> y (1 - eps) + eps / 2. The focal term is computed on the smoothed
targets and without pos_weight, exactly as in the notebooks.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Stage1Config


class FocalLoss(nn.Module):
    """Binary focal loss, alpha * (1 - p_t)^gamma * BCE, with p_t = exp(-BCE)."""

    def __init__(self, alpha: float = 0.6, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


class CombinedLoss(nn.Module):
    """Weighted sum of label-smoothed BCE (with ``pos_weight``) and focal loss."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.03,
        bce_weight: float = 0.4,
        focal_weight: float = 0.6,
        focal_alpha: float = 0.6,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        sm = targets * (1 - eps) + 0.5 * eps
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, sm, pos_weight=pw)
        return self.bce_weight * bce + self.focal_weight * self.focal(logits, sm)


def compute_pos_weight(binary_labels: np.ndarray, multiplier: float | None) -> torch.Tensor:
    """pos_weight = (n_neg / n_pos) * multiplier on the training split; 1.0 if multiplier is None.

    The 768-px gate uses multiplier 1.6 (pos_weight 2.833 on its 41,156-image training split).
    """
    labels = np.asarray(binary_labels)
    if multiplier is None:
        return torch.tensor([1.0], dtype=torch.float32)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return torch.tensor([n_neg / max(n_pos, 1) * multiplier], dtype=torch.float32)


def build_criterion(cfg: Stage1Config, train_binary_labels: np.ndarray) -> CombinedLoss:
    """Loss of a preset, with pos_weight computed on the training split."""
    return CombinedLoss(
        pos_weight=compute_pos_weight(train_binary_labels, cfg.pos_weight_x),
        label_smoothing=cfg.label_smoothing,
        bce_weight=cfg.bce_weight,
        focal_weight=cfg.focal_weight,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
    )
