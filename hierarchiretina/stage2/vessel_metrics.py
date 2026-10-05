"""FOV-aware vessel metrics, post-processing and the held-out test-set evaluation.

All pixel counts are taken inside the FOV mask only. The test evaluation (`evaluate_test_set`)
reproduces the notebook cell used for the paper's Table "Stage II" (vessel row): flip TTA,
threshold from the best validation epoch, pixel-pooled Dice / IoU / sensitivity / specificity /
precision / ROC AUC / PR AUC, and per-image Dice mean +/- SD (population SD, ddof=0).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

EPS = 1e-7
# Youden-J threshold search window used during validation.
THR_MIN, THR_MAX = 0.05, 0.8
# Post-processing applied to deployed (real-world) masks.
MORPH_OPEN_RADIUS = 1
MIN_COMPONENT_AREA = 30
TTA_FLIPS = ([3], [2], [2, 3])  # horizontal, vertical, both (= 180 degree rotation)


def compute_metrics_fov(probs, tgts, fovs, thr) -> dict:
    """Pooled confusion-matrix metrics inside the FOV at threshold ``thr`` (pred = prob >= thr)."""
    inside = fovs > 0.5
    p = (probs[inside] >= thr).astype(np.uint8)
    t = (tgts[inside] >= 0.5).astype(np.uint8)
    tp = int((p & t).sum())
    fp = int((p & ~t.astype(bool)).sum())
    fn = int((~p.astype(bool) & t).sum())
    tn = int((~p.astype(bool) & ~t.astype(bool)).sum())
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    return {
        "dice": dice, "iou": tp / (tp + fp + fn + EPS), "f1": dice,
        "precision": tp / (tp + fp + EPS), "recall": tp / (tp + fn + EPS),
        "sensitivity": tp / (tp + fn + EPS), "specificity": tn / (tn + fp + EPS),
    }


def youden_threshold(probs, tgts, fovs, default: float = 0.5) -> float:
    """Threshold maximising Youden's J inside the FOV, clipped to [0.05, 0.8]."""
    inside = fovs > 0.5
    try:
        fpr, tpr, thrs = roc_curve(tgts[inside] >= 0.5, probs[inside])
        return float(np.clip(thrs[np.argmax(tpr - fpr)], THR_MIN, THR_MAX))
    except Exception:
        return default


def fov_auc(probs, tgts, fovs) -> float:
    """ROC AUC inside the FOV (NaN if undefined)."""
    inside = fovs > 0.5
    try:
        return float(roc_auc_score(tgts[inside] >= 0.5, probs[inside]))
    except Exception:
        return float("nan")


def postprocess_mask(prob, thr, morph_r: int = MORPH_OPEN_RADIUS,
                     min_area: int = MIN_COMPONENT_AREA, fov=None) -> np.ndarray:
    """Binarise (>= thr), restrict to FOV, open with an elliptical kernel, drop small components."""
    m = (prob >= thr).astype(np.uint8)
    if fov is not None:
        m = m * (fov > 0.5).astype(np.uint8)
    if morph_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_r + 1, 2 * morph_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if min_area > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        out = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[lab == i] = 1
        m = out
    return m


@torch.no_grad()
def predict_tta_batch(model, x: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
    """Sigmoid probabilities averaged over identity + horizontal, vertical and double flips."""
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    p = torch.sigmoid(out)
    if use_tta:
        for dims in TTA_FLIPS:
            o = model(torch.flip(x, dims=dims))
            if isinstance(o, tuple):
                o = o[0]
            p = p + torch.flip(torch.sigmoid(o), dims=dims)
        p = p / 4.0
    return p


def _confusion_metrics(tp, fp, fn, tn) -> dict:
    prec = (tp + EPS) / (tp + fp + EPS)
    rec = (tp + EPS) / (tp + fn + EPS)
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "sensitivity": rec,
        "specificity": (tn + EPS) / (tn + fp + EPS),
        "precision": prec,
        "f1": (2 * prec * rec) / (prec + rec + EPS),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + EPS),
    }


@torch.no_grad()
def evaluate_test_set(model, loader, device, thr: float, use_tta: bool = True):
    """FOV-aware test evaluation (no post-processing; prediction = prob > thr).

    Returns ``(per_image_df, pooled)`` where ``pooled`` holds the pixel-pooled metrics plus
    ``auc_roc`` and ``auc_pr`` (computed from the raw TTA probabilities inside the FOV).
    """
    model.eval()
    rows, probs_in, gts_in = [], [], []
    idx = 0
    for imgs, msks, fovs in loader:
        probs = predict_tta_batch(model, imgs.to(device, non_blocking=True), use_tta)
        probs = probs.cpu().float().numpy()
        gts, fmasks = msks.numpy(), fovs.numpy()
        for b in range(probs.shape[0]):
            f = fmasks[b, 0] > 0.5
            p_prob = probs[b, 0].astype(np.float32)
            g_bin = ((gts[b, 0] >= 0.5) & f).astype(np.uint8)
            p_bin = (((p_prob * f) > thr) & f).astype(np.uint8)
            fu = f.astype(np.uint8)
            tp = int((p_bin & g_bin).sum())
            fp = int((p_bin & (1 - g_bin) & fu).sum())
            fn = int(((1 - p_bin) & g_bin & fu).sum())
            tn = int(((1 - p_bin) & (1 - g_bin) & fu).sum())
            rows.append({"image_idx": idx, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         **_confusion_metrics(tp, fp, fn, tn)})
            probs_in.append(p_prob[f])
            gts_in.append((gts[b, 0] >= 0.5)[f].astype(np.uint8))
            idx += 1
    per_image = pd.DataFrame(rows)

    probs_in = np.concatenate(probs_in).astype(np.float32)
    gts_in = np.concatenate(gts_in).astype(np.uint8)
    pred_in = (probs_in > thr).astype(np.uint8)
    tp = int(((pred_in == 1) & (gts_in == 1)).sum())
    fp = int(((pred_in == 1) & (gts_in == 0)).sum())
    fn = int(((pred_in == 0) & (gts_in == 1)).sum())
    tn = int(((pred_in == 0) & (gts_in == 0)).sum())
    pooled = _confusion_metrics(tp, fp, fn, tn)
    try:
        pooled["auc_roc"] = float(roc_auc_score(gts_in, probs_in))
    except ValueError:
        pooled["auc_roc"] = float("nan")
    try:
        pooled["auc_pr"] = float(average_precision_score(gts_in, probs_in))
    except ValueError:
        pooled["auc_pr"] = float("nan")
    return per_image, pooled


def summarise_test_results(per_image: pd.DataFrame, pooled: dict, thr: float,
                           use_tta: bool = True) -> dict:
    """JSON-serialisable summary (image-averaged mean/SD with ddof=0, and pixel-pooled)."""
    metrics = ["dice", "iou", "sensitivity", "specificity", "precision", "f1", "accuracy"]
    return {
        "n_test_images": int(len(per_image)),
        "threshold": float(thr),
        "tta": bool(use_tta),
        "fov_aware": True,
        "image_avg": {m: [float(per_image[m].mean()), float(per_image[m].std(ddof=0))]
                      for m in metrics},
        "image_dice_range": [float(per_image["dice"].min()), float(per_image["dice"].max())],
        "n_image_dice_ge_0p70": int((per_image["dice"] >= 0.70).sum()),
        "pixel_pooled": {k: float(v) for k, v in pooled.items()},
    }


def paper_table_row(summary: dict) -> pd.DataFrame:
    """One-row table in the column order of the paper's Stage II table (vessel row)."""
    pp, ia = summary["pixel_pooled"], summary["image_avg"]
    return pd.DataFrame([{
        "Target": "Vessels", "n": summary["n_test_images"], "Thr.": round(summary["threshold"], 2),
        "Dice": pp["dice"], "IoU": pp["iou"], "Sens.": pp["sensitivity"],
        "Prec.": pp["precision"], "Spec.": pp["specificity"], "ROC AUC": pp["auc_roc"],
        "PR AUC": pp["auc_pr"],
        "Image Dice": f"{ia['dice'][0]:.3f} +/- {ia['dice'][1]:.3f}",
    }])
