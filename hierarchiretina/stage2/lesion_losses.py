"""Losses for the four HSMoE-AUNet lesion models.

All per-lesion recipes are expressed by :class:`LesionLoss` driven by
:class:`~hierarchiretina.stage2.hsmoe_aunet.LesionLossConfig`:

    L = ftl_w * FTL + dice_w * Dice + bce_w * BCE(pos_weight)
        [+ boundary_w * boundary] [+ rc_w * region_coherence] [+ od_w * optic_disc]
        + aux_w * (FTL+Dice on aux3 + FTL+Dice on aux2) / 2 + MoE load balance

MA: no extra term. HE: Sobel boundary (0.3). EX: Sobel+Laplacian boundary (0.5) and optic-disc
suppression (0.15). CWS: three phases (see ``LesionLossConfig``), soft boundary (0.15),
probability-weighted TV (0.02) and optic-disc suppression (0.08, opening kernel 11).
FTL/Dice are computed on the whole batch flattened (batch-pooled), as in the notebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hsmoe_aunet import LesionLossConfig, get_lesion_config


class FocalTverskyLoss(nn.Module):
    """FTL = (1 - TI)^gamma, TI = TP / (TP + alpha*FN + beta*FP) on probabilities."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 2.0,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, pred, target):
        pred, target = pred.view(-1), target.view(-1)
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - ti) ** self.gamma


class DiceLoss(nn.Module):
    """Soft Dice loss on probabilities (batch-pooled)."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred, target = pred.view(-1), target.view(-1)
        inter = (pred * target).sum()
        return 1 - (2 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)


def _sobel_kernels():
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def _edge_dice(e_pred, e_gt, smooth):
    """Soft Dice between per-image max-normalised edge maps."""
    e_pred = e_pred / (e_pred.amax(dim=(2, 3), keepdim=True) + 1e-6)
    e_gt = e_gt / (e_gt.amax(dim=(2, 3), keepdim=True) + 1e-6)
    inter = (e_pred * e_gt).sum()
    denom = e_pred.sum() + e_gt.sum()
    return 1 - (2 * inter + smooth) / (denom + smooth)


class BoundaryLoss(nn.Module):
    """HE: soft Dice between Sobel edge magnitudes of prediction and target."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        kx, ky = _sobel_kernels()
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.smooth = smooth

    def _edges(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred_prob, target):
        return _edge_dice(self._edges(pred_prob), self._edges(target), self.smooth)


class DualEdgeBoundaryLoss(nn.Module):
    """EX: edge Dice on 0.6 * Sobel magnitude + 0.4 * |4-neighbour Laplacian|."""

    def __init__(self, smooth: float = 1e-6, sobel_w: float = 0.6, lap_w: float = 0.4):
        super().__init__()
        kx, ky = _sobel_kernels()
        kl = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("kl", kl.view(1, 1, 3, 3))
        self.sobel_w, self.lap_w, self.smooth = sobel_w, lap_w, smooth

    def _edges(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        sob = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = F.conv2d(x, self.kl, padding=1).abs()
        return self.sobel_w * sob + self.lap_w * lap

    def forward(self, pred_prob, target):
        return _edge_dice(self._edges(pred_prob), self._edges(target), self.smooth)


class SoftBoundaryLoss(nn.Module):
    """CWS: Gaussian blur (5x5, sigma 1.5) of both maps before the Sobel edge Dice."""

    def __init__(self, smooth: float = 1e-6, blur_ksize: int = 5, blur_sigma: float = 1.5):
        super().__init__()
        kx, ky = _sobel_kernels()
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        ax = torch.arange(blur_ksize, dtype=torch.float32) - blur_ksize // 2
        g1 = torch.exp(-(ax ** 2) / (2 * blur_sigma ** 2))
        g1 = g1 / g1.sum()
        g2 = g1.unsqueeze(0) * g1.unsqueeze(1)
        self.register_buffer("blur_kernel", g2.view(1, 1, blur_ksize, blur_ksize))
        self.blur_pad = blur_ksize // 2
        self.smooth = smooth

    def _blur(self, x):
        return F.conv2d(x, self.blur_kernel, padding=self.blur_pad)

    def _edges(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred_prob, target):
        return _edge_dice(self._edges(self._blur(pred_prob)),
                          self._edges(self._blur(target)), self.smooth)


class RegionCoherenceLoss(nn.Module):
    """CWS: total variation weighted by the mean probability of each neighbouring pair.

    Normalised by the total weight, so an all-zero prediction gives 0 (no collapse incentive).
    """

    def forward(self, pred_prob):
        eps = 1e-6
        dx = (pred_prob[:, :, :, 1:] - pred_prob[:, :, :, :-1]).abs()
        wx = (pred_prob[:, :, :, 1:] + pred_prob[:, :, :, :-1]) / 2.0
        dy = (pred_prob[:, :, 1:, :] - pred_prob[:, :, :-1, :]).abs()
        wy = (pred_prob[:, :, 1:, :] + pred_prob[:, :, :-1, :]) / 2.0
        return (dx * wx).sum() / (wx.sum() + eps) + (dy * wy).sum() / (wy.sum() + eps)


class OpticDiscSuppressionLoss(nn.Module):
    """EX/CWS: mean predicted probability inside the optic-disc candidate where GT is 0.

    The candidate is the brightest ``top_pct`` of the (un-normalised) green channel of the
    input, cleaned by a morphological opening (``open_ksize``) that removes small bright
    spots. Computed without gradient; acts as a regulariser.
    """

    def __init__(self, top_pct: float = 0.02, open_ksize: int = 15, weight: float = 1.0):
        super().__init__()
        self.top_pct, self.open_ksize, self.weight = top_pct, open_ksize, weight
        # ImageNet normalisation constants of the green channel
        self.register_buffer("mean_g", torch.tensor(0.456).view(1, 1, 1, 1))
        self.register_buffer("std_g", torch.tensor(0.224).view(1, 1, 1, 1))

    @torch.no_grad()
    def _od_candidate_mask(self, img_norm):
        green = (img_norm[:, 1:2] * self.std_g + self.mean_g).clamp(0, 1)
        B = green.shape[0]
        flat = green.view(B, -1)
        k = max(1, int(flat.shape[1] * self.top_pct))
        thr, _ = flat.kthvalue(flat.shape[1] - k + 1, dim=1)
        bright = (green >= thr.view(B, 1, 1, 1)).float()
        ks, pad = self.open_ksize, self.open_ksize // 2
        eroded = -F.max_pool2d(-bright, ks, stride=1, padding=pad)
        return F.max_pool2d(eroded, ks, stride=1, padding=pad)

    def forward(self, pred_prob, target, img_norm):
        od_mask = self._od_candidate_mask(img_norm)
        penalty_region = od_mask * (1.0 - target)
        loss = (pred_prob * penalty_region).sum() / (penalty_region.sum() + 1e-6)
        return self.weight * loss


_BOUNDARY = {"sobel": BoundaryLoss, "dual_edge": DualEdgeBoundaryLoss, "soft": SoftBoundaryLoss}


class LesionLoss(nn.Module):
    """Combined loss of one lesion model (see module docstring).

    Args to ``forward``:
        pred_logits: main-head logits (B,1,H,W); aux3/aux2: aux probabilities;
        target: binary float mask (B,1,H,W); moe_loss: load-balance term from the model;
        img_norm: the normalised input batch (needed for the optic-disc term);
        epoch: 0-based epoch (needed for the CWS three-phase schedule).
    """

    def __init__(self, cfg: LesionLossConfig):
        super().__init__()
        self.cfg = cfg
        phased = cfg.phase_epochs is not None
        gamma_main_early = cfg.ftl_gamma_warmup if phased else cfg.ftl_gamma
        self.ftl_early = FocalTverskyLoss(cfg.ftl_alpha, cfg.ftl_beta, gamma_main_early)
        self.ftl_late = FocalTverskyLoss(cfg.ftl_alpha, cfg.ftl_beta, cfg.ftl_gamma)
        self.dice = DiceLoss()
        pw = None if cfg.bce_pos_weight is None else torch.tensor(float(cfg.bce_pos_weight))
        self.register_buffer("pos_weight", pw)
        self.boundary = _BOUNDARY[cfg.boundary]() if cfg.boundary else None
        self.rcoh = RegionCoherenceLoss() if cfg.region_coherence_w > 0 else None
        self.od = (OpticDiscSuppressionLoss(top_pct=0.02, open_ksize=cfg.optic_disc_open_ksize)
                   if cfg.optic_disc_w > 0 else None)

    def _phase(self, epoch: int) -> int:
        """1: basics only, 2: + optic disc, 3: all terms. Unphased lesions are always 3."""
        if self.cfg.phase_epochs is None:
            return 3
        e1, e2 = self.cfg.phase_epochs
        return 1 if epoch < e1 else (2 if epoch < e2 else 3)

    def forward(self, pred_logits, aux3, aux2, target, moe_loss, img_norm=None, epoch: int = 0):
        c = self.cfg
        phase = self._phase(epoch)
        pred_prob = pred_logits.sigmoid()

        ftl_main = self.ftl_late if (c.phase_epochs is None or phase == 3) else self.ftl_early
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=self.pos_weight)
        main = (c.ftl_w * ftl_main(pred_prob, target)
                + c.dice_w * self.dice(pred_prob, target)
                + c.bce_w * bce)
        if phase == 3:
            if self.boundary is not None:
                main = main + c.boundary_w * self.boundary(pred_prob, target)
            if self.rcoh is not None:
                main = main + c.region_coherence_w * self.rcoh(pred_prob)
        if self.od is not None and phase >= 2 and img_norm is not None:
            main = main + c.optic_disc_w * self.od(pred_prob, target, img_norm)

        # Aux heads: the warm-up FTL for phased (CWS) models, otherwise the single FTL.
        ftl_aux = self.ftl_early
        t3 = F.interpolate(target, size=aux3.shape[2:], mode="nearest")
        t2 = F.interpolate(target, size=aux2.shape[2:], mode="nearest")
        a3 = ftl_aux(aux3, t3) + self.dice(aux3, t3)
        a2 = ftl_aux(aux2, t2) + self.dice(aux2, t2)
        return main + c.aux_w * (a3 + a2) / 2.0 + moe_loss

    @property
    def needs_image(self) -> bool:
        return self.od is not None


def build_lesion_loss(lesion: str) -> LesionLoss:
    """Loss of the given lesion model ('MA' | 'HE' | 'EX' | 'CWS')."""
    return LesionLoss(get_lesion_config(lesion).loss)
