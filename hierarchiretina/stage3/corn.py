"""CORN algebra for LG-DRG: 5-way probability vector, its exact inversion, and decoding.

The model emits three CORN logits and one gradability logit. ``lgdrg_predict`` maps them to
a 5-way vector over grades 1..5::

    p_k = sigmoid(sev_logit_k), cum = cumprod(p)          (k = 1..3)
    psev = [1-p1, p1(1-p2), p1 p2 (1-p3), p1 p2 p3]
    full[0..3] = psev * P(gradable),   full[4] = 1 - P(gradable)

The map is invertible (``corn_from_full``), so averaged ensemble vectors and saved CSVs can be
decoded again with tuned thresholds without re-running the network.

Two discrete decodes exist:

* ``decode_consecutive`` (paper): rank-consistent CORN rank, counts leading thresholds passed
  and stops at the first failure. Used with the OOF-fitted thresholds t=(0.38, 0.60, 0.19),
  t_g=0.10 for every reported test number.
* ``decode_count``: counts every threshold passed; with t=0.5 this equals the rank inside
  ``lgdrg_predict`` (used for validation/OOF metrics during training and for the ablation).
"""
from __future__ import annotations

import numpy as np
import torch

from .model import K_SEV

PAPER_T = (0.38, 0.60, 0.19)
PAPER_TG = 0.10


def lgdrg_predict(sev_logits: torch.Tensor, grad_logits: torch.Tensor, k_sev: int = K_SEV,
                  grad_thresh: float = 0.5):
    """Return ``(pred_idx[B] in 0..4, full[B,5])`` as in the training notebook.

    ``pred_idx`` = 4 (Grade 5) when P(gradable) < ``grad_thresh``, else the count of CORN
    sigmoids above 0.5.
    """
    p_gradable = torch.sigmoid(grad_logits)
    p_ungrad = 1 - p_gradable
    probs = torch.sigmoid(sev_logits)
    cum = torch.cumprod(probs, 1)
    B = sev_logits.size(0)
    psev = torch.zeros(B, k_sev, device=sev_logits.device)
    psev[:, 0] = 1 - cum[:, 0]
    for k in range(1, k_sev - 1):
        psev[:, k] = cum[:, k - 1] - cum[:, k]
    psev[:, k_sev - 1] = cum[:, k_sev - 2]
    psev = psev.clamp(min=1e-9)
    psev = psev / psev.sum(1, keepdim=True)
    full = torch.zeros(B, k_sev + 1, device=sev_logits.device)
    full[:, :k_sev] = psev * p_gradable.unsqueeze(1)
    full[:, k_sev] = p_ungrad
    full = full / full.sum(1, keepdim=True)
    sev_rank = (probs > 0.5).sum(1)
    pred = torch.where(p_gradable < grad_thresh, torch.full_like(sev_rank, k_sev), sev_rank)
    return pred, full


def corn_from_full(P5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert ``lgdrg_predict``: ``P5 (N,5) -> (p1, p2, p3, p_gradable)`` each ``(N,)``."""
    P5 = np.asarray(P5, float)
    if P5.ndim == 1:
        P5 = P5[None, :]
    pu = np.clip(P5[:, 4], 0, 1 - 1e-9)
    pg = np.clip(1 - pu, 1e-9, 1)
    ps = np.clip(P5[:, :4] / pg[:, None], 1e-12, None)
    ps = ps / ps.sum(1, keepdims=True)
    p1 = np.clip(1 - ps[:, 0], 1e-9, 1 - 1e-9)
    s23 = np.clip(ps[:, 2] + ps[:, 3], 1e-12, None)
    p2 = np.clip(s23 / p1, 1e-9, 1 - 1e-9)
    p3 = np.clip(ps[:, 3] / s23, 1e-9, 1 - 1e-9)
    return p1, p2, p3, pg


def full_from_corn(p1, p2, p3, p_gradable) -> np.ndarray:
    """Forward map (numpy), used to verify the inversion."""
    cum = np.stack([p1, p1 * p2, p1 * p2 * p3], 1)
    psev = np.stack([1 - cum[:, 0], cum[:, 0] - cum[:, 1], cum[:, 1] - cum[:, 2], cum[:, 2]], 1)
    psev = np.clip(psev, 1e-9, None)
    psev = psev / psev.sum(1, keepdims=True)
    full = np.concatenate([psev * p_gradable[:, None], (1 - p_gradable)[:, None]], 1)
    return full / full.sum(1, keepdims=True)


def decode_consecutive(p1, p2, p3, pg, t=PAPER_T, tg=PAPER_TG) -> np.ndarray:
    """Grades 1..5: Grade 5 if ``pg < tg``, else 1 + number of *leading* thresholds passed."""
    a = p1 > t[0]
    b = a & (p2 > t[1])
    c = b & (p3 > t[2])
    rank = a.astype(int) + b.astype(int) + c.astype(int)
    return np.where(pg < tg, 5, rank + 1).astype(int)


def decode_count(p1, p2, p3, pg, t=(0.5, 0.5, 0.5), tg=0.5) -> np.ndarray:
    """Grades 1..5 counting *every* threshold passed (``lgdrg_predict`` rule at t=0.5)."""
    rank = (p1 > t[0]).astype(int) + (p2 > t[1]).astype(int) + (p3 > t[2]).astype(int)
    return np.where(pg < tg, 5, rank + 1).astype(int)


def decode_full(P5, t=PAPER_T, tg=PAPER_TG) -> np.ndarray:
    """Convenience: saved/averaged 5-way vectors -> grades 1..5 with the paper decode."""
    return decode_consecutive(*corn_from_full(P5), t=t, tg=tg)
