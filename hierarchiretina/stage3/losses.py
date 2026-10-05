"""LG-DRG losses: weighted gradability BCE on all images + CORN severity loss on gradable ones."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import K_SEV


def corn_loss_sev(logits: torch.Tensor, y_sev: torch.Tensor, k_sev: int = K_SEV) -> torch.Tensor:
    """Ordinal loss over severity 0..k_sev-1; ``logits`` is ``(n, k_sev-1)``.

    Each task k is a BCE of ``y_sev > k`` on a conditioning subset, averaged over tasks.

    Note (kept exactly as trained): the subset for task k >= 1 is ``y_sev >= k - 1``. Standard
    CORN conditions task k on ``y_sev >= k`` (i.e. ``y > k-1``). With 4 grades this means task 1
    is trained on all gradable images and task 2 on ``y_sev >= 1``. The released checkpoints
    were trained with this rule, so it is preserved here.
    """
    loss = 0.0
    n_terms = 0
    for k in range(k_sev - 1):
        sel = torch.ones_like(y_sev, dtype=torch.bool) if k == 0 else (y_sev >= (k - 1))
        if sel.sum() == 0:
            continue
        loss = loss + F.binary_cross_entropy_with_logits(logits[sel, k], (y_sev[sel] > k).float())
        n_terms += 1
    return loss / max(n_terms, 1)


def lgdrg_loss(sev_logits, grad_logits, y_sev, gradable, pos_weight: float, k_sev: int = K_SEV):
    """Total loss = BCE(gradable; pos_weight) on all images + CORN loss on gradable images.

    The BCE target is 1 = gradable, 0 = ungradable (Grade 5); ``pos_weight`` (0.045 =
    n_ungradable / n_gradable) therefore down-weights the gradable majority.
    Returns ``(total, l_grad.detach(), l_sev.detach())``.
    """
    pw = torch.tensor([pos_weight], device=grad_logits.device)
    l_grad = F.binary_cross_entropy_with_logits(grad_logits, gradable.float(), pos_weight=pw)
    mask = gradable.bool()
    if mask.sum() > 0:
        l_sev = corn_loss_sev(sev_logits[mask], y_sev[mask], k_sev)
    else:
        l_sev = sev_logits.sum() * 0.0
    return l_grad + l_sev, l_grad.detach(), l_sev.detach()
