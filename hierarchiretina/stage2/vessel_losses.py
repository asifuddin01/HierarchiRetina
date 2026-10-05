"""FOV-restricted vessel loss (Eq. "vesloss" of the paper).

L_v = 0.25 BCE(pos_weight) + 0.35 Dice + 0.30 focal Tversky + 0.10 clDice, every term evaluated only
inside the field-of-view (FOV) mask, so the bright circular fundus boundary never receives a
"vessel" gradient. With deep supervision the total is
1.0 L_v(main) + 0.4 L_v(aux D4) + 0.2 L_v(aux D3) + 0.1 L_v(aux D2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BCE_WEIGHT = 0.25
DICE_WEIGHT = 0.35
FOCAL_TVERSKY_WEIGHT = 0.30
CLDICE_WEIGHT = 0.10
TVERSKY_ALPHA = 0.45  # weight on false positives
TVERSKY_BETA = 0.55   # weight on false negatives
FOCAL_GAMMA = 1.0
DS_WEIGHTS = (1.0, 0.40, 0.20, 0.10)


class DiceLoss(nn.Module):
    """Soft Dice over all FOV pixels of the batch."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, tgt, fov):
        p = (torch.sigmoid(pred) * fov).view(-1)
        t = (tgt * fov).view(-1)
        return 1 - (2 * (p * t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)


class FocalTverskyLoss(nn.Module):
    """(1 - TI)^gamma with TI = TP / (TP + alpha FP + beta FN), inside the FOV."""

    def __init__(self, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, gamma=FOCAL_GAMMA, smooth=1e-6):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth

    def forward(self, pred, tgt, fov):
        p = (torch.sigmoid(pred) * fov).view(-1)
        t = (tgt * fov).view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        return (1 - (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)) ** self.g


class SoftSkeletonize(nn.Module):
    """Differentiable soft skeleton (iterated min/max-pool thinning)."""

    def __init__(self, iters=3):
        super().__init__()
        self.iters = iters

    def forward(self, x):
        s = x.clone()
        for _ in range(self.iters):
            s = s * (1 - (F.max_pool2d(s, 3, 1, 1) - (-F.max_pool2d(-s, 3, 1, 1))).clamp(0, 1))
        return s.clamp(0, 1)


class clDiceLoss(nn.Module):  # noqa: N801  (name kept from the original notebook)
    """Centre-line Dice (Shit et al.), inside the FOV."""

    def __init__(self, iters=3, smooth=1e-6):
        super().__init__()
        self.skel = SoftSkeletonize(iters)
        self.s = smooth

    def forward(self, pred, tgt, fov):
        p = torch.sigmoid(pred) * fov
        t = tgt * fov
        sp, st = self.skel(p), self.skel(t)
        prec = (st * p).sum() / (p.sum() + self.s)
        rec = (sp * t).sum() / (t.sum() + self.s)
        return 1 - 2 * prec * rec / (prec + rec + self.s)


class FOVMaskedBCE(nn.Module):
    """Weighted BCE averaged over FOV pixels only."""

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.pw = pos_weight

    def forward(self, pred, tgt, fov):
        pw = self.pw.to(pred.device) if self.pw is not None else None
        bce = F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=pw, reduction="none")
        return (bce * fov).sum() / (fov.sum() + 1e-6)


class CombinedVesselLoss(nn.Module):
    """FOV-masked BCE + Dice + focal Tversky + clDice, with deep-supervision weighting.

    ``forward(outputs, tgt, fov)`` accepts either the training-mode tuple
    ``(main, aux_d4, aux_d3, aux_d2)`` or a single logit map.
    """

    def __init__(self, pos_weight: float | None = None, bce_weight: float = BCE_WEIGHT,
                 dice_weight: float = DICE_WEIGHT,
                 focal_tversky_weight: float = FOCAL_TVERSKY_WEIGHT,
                 cldice_weight: float = CLDICE_WEIGHT, tversky_alpha: float = TVERSKY_ALPHA,
                 tversky_beta: float = TVERSKY_BETA, focal_gamma: float = FOCAL_GAMMA,
                 ds_weights=DS_WEIGHTS, label_smoothing: float = 0.0):
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = FOVMaskedBCE(pw)
        self.dice = DiceLoss()
        self.ftv = FocalTverskyLoss(tversky_alpha, tversky_beta, focal_gamma)
        self.cld = clDiceLoss()
        self.wb, self.wd = bce_weight, dice_weight
        self.wf, self.wc = focal_tversky_weight, cldice_weight
        self.ds_w = list(ds_weights)
        self.ls = label_smoothing

    def _single(self, pred, tgt, fov):
        t = tgt * (1 - self.ls) + 0.5 * self.ls if self.ls > 0 else tgt
        return (self.wb * self.bce(pred, t, fov) + self.wd * self.dice(pred, t, fov)
                + self.wf * self.ftv(pred, t, fov) + self.wc * self.cld(pred, t, fov))

    def forward(self, outputs, tgt, fov):
        if isinstance(outputs, (tuple, list)):
            m, a1, a2, a3 = outputs
            return (self.ds_w[0] * self._single(m, tgt, fov)
                    + self.ds_w[1] * self._single(a1, tgt, fov)
                    + self.ds_w[2] * self._single(a2, tgt, fov)
                    + self.ds_w[3] * self._single(a3, tgt, fov))
        return self._single(outputs, tgt, fov)
