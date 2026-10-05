"""Metrics for Stage III and the cascade.

Severity QWK (paper definition, "oof_match"): Grade 5 is a quality category, not a severity
level. Images whose *true* grade is 5 are removed and a *predicted* Grade 5 is mapped to 4.
Stage A (grader alone) scores grades 1..4; Stage B (end-to-end) scores grades 0..4.

``qwk`` is the dependency-free kappa used by the final evaluation script that produced every
reported test number. It fixes the label set explicitly; ``sklearn.metrics.cohen_kappa_score``
gives the same value whenever every label in the set occurs in ``y_true`` or ``y_pred``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, recall_score, roc_auc_score

UNGRADABLE, MAX_SEV = 5, 4


# ----------------------------------------------------------------------------- training-time
def macro_auc_5(y_true: np.ndarray, prob5: np.ndarray) -> float:
    """Mean one-vs-rest AUC over the 5 grades (model-selection metric); skips absent classes."""
    aucs = []
    for k in range(prob5.shape[1]):
        yk = (y_true == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        aucs.append(roc_auc_score(yk, prob5[:, k]))
    return float(np.mean(aucs)) if aucs else 0.0


def qwk_sev_idx(y_idx: np.ndarray, p_idx: np.ndarray) -> float:
    """Severity QWK on 0-indexed labels (0..4 = grade 1..5), as in the training notebook:
    true index 4 dropped, predicted 4 clipped to 3."""
    m = y_idx < 4
    if m.sum() == 0:
        return float("nan")
    return float(cohen_kappa_score(y_idx[m], np.clip(p_idx[m], 0, 3), weights="quadratic"))


# ----------------------------------------------------------------------------- final script
def qwk(a, b, labels) -> float:
    """Quadratic-weighted kappa over an explicit, ordered label set."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    L = np.asarray(sorted(labels))
    k = len(L)
    ia, ib = np.searchsorted(L, a), np.searchsorted(L, b)
    O = np.bincount(ia * k + ib, minlength=k * k).reshape(k, k).astype(float)
    i, j = np.indices((k, k))
    W = (i - j) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def qwk_sev(yt, yp, include_healthy: bool = True) -> tuple[float, int]:
    """Paper severity QWK on grade labels (0..5). ``include_healthy=False`` -> Stage A (1..4).

    Returns ``(kappa, n_scored)``.
    """
    yt = np.asarray(yt, int)
    yp = np.asarray(yp, int)
    keep = yt != UNGRADABLE
    if not include_healthy:
        keep &= yt > 0
    lo = 0 if include_healthy else 1
    yt, yp = yt[keep], np.clip(yp[keep], lo, MAX_SEV)
    return qwk(yt, yp, range(lo, MAX_SEV + 1)), int(len(yt))


def qwk_severity_modes(yt, yp, include_healthy: bool = True) -> dict[str, tuple[float, int]]:
    """QWK under the three definitions compared in the paper's protocol section.

    ``oof_match`` (reported): drop true 5, map predicted 5 -> 4.
    ``as_published``: Grade 5 left on the ordinal axis (inflates kappa).
    ``strict``: drop every row whose true or predicted grade is 5.
    """
    yt = np.asarray(yt, int)
    yp = np.asarray(yp, int)
    lo = 0 if include_healthy else 1
    base = np.ones(len(yt), bool) if include_healthy else yt > 0
    out = {"oof_match": qwk_sev(yt, yp, include_healthy)}
    m = base
    out["as_published"] = (qwk(yt[m], yp[m], range(lo, 6)), int(m.sum()))
    m = base & (yt != UNGRADABLE) & (yp != UNGRADABLE)
    out["strict"] = (qwk(yt[m], yp[m], range(lo, MAX_SEV + 1)), int(m.sum()))
    return out


def auc(y, s) -> float:
    """Rank-based ROC AUC (Mann-Whitney U, average ranks for ties)."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    r = pd.Series(s).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_f1(yt, yp, labels) -> float:
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    f = []
    for c in labels:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def bootstrap_ci(fn, n: int, rng: np.random.Generator, B: int = 1000) -> tuple[float, float]:
    """Percentile 95% CI of ``fn(idx)`` over ``B`` image-level resamples of size ``n``.

    ``rng`` is shared across calls so that the sequence of resamples, and therefore the CIs,
    are reproducible only when calls are made in the same order (see ``cascade``).
    """
    vals = []
    for _ in range(B):
        v = fn(rng.integers(0, n, n))
        if v == v:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def binary_metrics(t, p) -> dict[str, float]:
    t = np.asarray(t, bool)
    p = np.asarray(p, bool)
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    tn, fp = int((~t & ~p).sum()), int((~t & p).sum())
    return {"sens": tp / (tp + fn), "spec": tn / (tn + fp), "ppv": tp / (tp + fp),
            "npv": tn / (tn + fn), "n_pos": int(t.sum())}


# ----------------------------------------------------------------------------- benchmarks
def five_class_metrics(y_true, y_pred) -> dict[str, float] | None:
    """Five-class (grades 0-4) benchmark subset: true Grade 5 removed (training Cell 13).

    ``acc_strict`` counts a predicted Grade 5 as wrong; ``acc_clipped`` maps it to 4 (the
    paper's DDR five-class accuracy is the clipped one).
    """
    yt = np.asarray(y_true, int)
    yp = np.asarray(y_pred, int)
    keep = yt != UNGRADABLE
    yt, yp = yt[keep], yp[keep]
    if len(yt) < 2:
        return None
    yp_clip = np.clip(yp, 0, MAX_SEV)
    return {"n": int(len(yt)),
            "acc_strict": float(np.mean(yt == yp)),
            "acc_clipped": float(np.mean(yt == yp_clip)),
            "macro_f1_strict": macro_f1(yt, yp, range(5)),
            "qwk": qwk(yt, yp_clip, range(5)),
            "pred5_on_gradable": int((yp == UNGRADABLE).sum())}


def six_class_recall(y_true, y_pred) -> dict:
    """Per-class recall over grades 0..5 plus overall accuracy (training Cell 14)."""
    yt = np.asarray(y_true, int)
    yp = np.asarray(y_pred, int)
    rec = recall_score(yt, yp, labels=list(range(6)), average=None, zero_division=0)
    return {"n": int(len(yt)), "per_class_recall": rec, "oa": float(np.mean(yt == yp)),
            "support": np.bincount(yt, minlength=6)}
