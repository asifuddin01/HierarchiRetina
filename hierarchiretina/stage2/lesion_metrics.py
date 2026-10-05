"""Validation threshold sweep and test-set metrics for the HSMoE-AUNet lesion models.

``evaluate_test_set`` reproduces "Cell 8.5 - TEST-SET METRICS" of the notebooks, which
produced the Stage II table of the paper:

* pixel-pooled Dice / IoU / sensitivity / precision / specificity / F1 / accuracy at the
  deployed threshold on the *raw* probabilities, plus ROC-AUC and PR-AUC (average precision);
* per-image metrics (mean and population SD over images); for CWS they are computed on the
  post-processed (closed + small-component-filtered) masks;
* lesion-level sensitivity / precision / F1: 8-connected components; each reference lesion is
  greedily matched (in label order) to the unused predicted component with the highest
  IoU >= ``lesion_iou`` (0.2 for MA, 0.3 otherwise); CWS uses post-processed masks;
* image-level presence: an image counts as detected if any pixel exceeds the threshold
  (raw probabilities, also for CWS).
All test images contain the lesion (near-empty masks were filtered out), so image-level
specificity is undefined (NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from skimage import measure
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .hsmoe_aunet import get_lesion_config
from .lesion_inference import binarize, predict_probs

EPS = 1e-7


# =============================================================================
# Training-time metric (threshold 0.5, batch-pooled)
# =============================================================================
def batch_metrics(pred_prob: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> dict:
    """Dice / IoU / sensitivity / specificity / precision of one batch (pooled pixels).

    The validation Dice used for model selection and early stopping is the mean of this
    batch-level Dice over the validation batches.
    """
    p = (pred_prob > thresh).float().view(-1).cpu()
    t = target.view(-1).cpu()
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    sens = (tp + EPS) / (tp + fn + EPS)
    return dict(dice=dice, iou=(tp + EPS) / (tp + fp + fn + EPS), sensitivity=sens,
                specificity=(tn + EPS) / (tn + fp + EPS), precision=(tp + EPS) / (tp + fp + EPS),
                f1=dice, recall=sens)


# =============================================================================
# Validation threshold sweep
# =============================================================================
def sweep_thresholds_for(lesion: str) -> np.ndarray:
    """Threshold grid of the lesion notebook's validation sweep (Cell 8.0)."""
    start, stop, step = get_lesion_config(lesion).sweep_grid
    return np.arange(start, stop, step)


def wide_ma_thresholds() -> np.ndarray:
    """Grid of the wide MA validation sweep quoted in the paper (0.001 ... 0.5)."""
    return np.unique(np.round(np.concatenate([
        np.arange(0.001, 0.010, 0.001), np.arange(0.01, 0.10, 0.01),
        np.arange(0.10, 0.51, 0.05)]), 4))


@torch.no_grad()
def threshold_sweep(model, loader, thresholds, device, tta: bool = False, use_amp: bool = True,
                    fp32_sigmoid: bool = False, eps: float = EPS) -> pd.DataFrame:
    """Pixel-pooled Dice / IoU / recall / precision / specificity for each threshold.

    Confusion counts are accumulated batch by batch, which is identical to thresholding the
    concatenated validation probabilities as the notebooks did. ``eps`` is 1e-7 in the
    notebook sweeps and 1e-9 in the wide MA sweep script.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    tp = np.zeros(len(thresholds), dtype=np.int64)
    fp = np.zeros(len(thresholds), dtype=np.int64)
    n_pos = n_neg = 0
    model.eval()
    for imgs, masks in tqdm(loader, desc="threshold sweep"):
        probs = predict_probs(model, imgs.to(device), tta=tta, use_amp=use_amp,
                              fp32_sigmoid=fp32_sigmoid)
        p = probs.cpu().float().numpy().ravel()
        t = masks.numpy().ravel() > 0.5
        p_pos, p_neg = p[t], p[~t]
        n_pos += p_pos.size
        n_neg += p_neg.size
        for i, th in enumerate(thresholds):
            tp[i] += int((p_pos > th).sum())
            fp[i] += int((p_neg > th).sum())
    fn, tn = n_pos - tp, n_neg - fp
    tp, fp, fn, tn = (x.astype(np.float64) for x in (tp, fp, fn, tn))
    return pd.DataFrame({
        "threshold": thresholds,
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
    })


def best_threshold(sweep: pd.DataFrame) -> float:
    """Dice-maximising threshold (first one on ties, as ``idxmax``)."""
    return float(sweep.loc[sweep["dice"].idxmax(), "threshold"])


# =============================================================================
# Lesion-level matching
# =============================================================================
def match_lesions(pred_mask: np.ndarray, gt_mask: np.ndarray, iou_th: float):
    """Greedy one-to-one matching of 8-connected components; returns (tp, fp, fn).

    Equivalent to the notebook loop (for each reference component in label order, take the
    unused predicted component with the highest IoU >= ``iou_th``; ties -> lowest label) but
    computes all IoUs from one joint histogram instead of per-pair masks.
    """
    gt_lab = measure.label(gt_mask > 0)
    pr_lab = measure.label(pred_mask > 0)
    n_gt, n_pr = int(gt_lab.max()), int(pr_lab.max())
    if n_gt == 0 or n_pr == 0:
        return 0, n_pr, n_gt
    gt_area = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pr_area = np.bincount(pr_lab.ravel(), minlength=n_pr + 1)
    both = (gt_lab > 0) & (pr_lab > 0)
    pair = gt_lab[both].astype(np.int64) * (n_pr + 1) + pr_lab[both]
    codes, inter = np.unique(pair, return_counts=True)
    cand: dict[int, list[tuple[int, int]]] = {}
    for code, n in zip(codes, inter):
        g, p = divmod(int(code), n_pr + 1)
        cand.setdefault(g, []).append((p, int(n)))
    used, tp = set(), 0
    for g in range(1, n_gt + 1):
        best, best_p = 0.0, -1
        for p, n in cand.get(g, ()):            # sorted by predicted label
            if p in used:
                continue
            iou = n / int(gt_area[g] + pr_area[p] - n)
            if iou >= iou_th and iou > best:
                best, best_p = iou, p
        if best_p > 0:
            used.add(best_p)
            tp += 1
    return tp, n_pr - tp, n_gt - tp


# =============================================================================
# Test-set evaluation (Cell 8.5)
# =============================================================================
def _ratio_metrics(tp, fp, fn, tn):
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    rec = (tp + EPS) / (tp + fn + EPS)
    prec = (tp + EPS) / (tp + fp + EPS)
    spec = (tn + EPS) / (tn + fp + EPS)
    acc = (tp + tn) / (tp + fp + fn + tn + EPS)
    f1 = (2 * prec * rec) / (prec + rec + EPS)
    return dict(dice=dice, iou=iou, sensitivity=rec, specificity=spec, precision=prec, f1=f1,
                accuracy=acc)


@torch.no_grad()
def evaluate_test_set(model, loader, lesion: str, device, threshold: float | None = None,
                      use_amp: bool = True, lesion_iou: float | None = None):
    """Full test-set evaluation of one lesion model. Returns ``(summary, per_image_df)``."""
    cfg = get_lesion_config(lesion)
    thr = cfg.threshold if threshold is None else threshold
    iou_th = cfg.lesion_iou if lesion_iou is None else lesion_iou
    model.eval()

    all_probs, all_gts, per_image = [], [], []
    for imgs, masks in tqdm(loader, desc=f"{cfg.name} test"):
        probs = predict_probs(model, imgs.to(device), tta=cfg.tta, use_amp=use_amp)
        probs_np = probs.cpu().float().numpy()
        gts_np = masks.numpy().astype(np.uint8)
        for b in range(probs_np.shape[0]):
            p_prob = probs_np[b, 0].astype(np.float32)
            g_bin = (gts_np[b, 0] > 0).astype(np.uint8)
            p_bin = binarize(p_prob, thr, cfg)
            all_probs.append(p_prob)
            all_gts.append(g_bin)
            tp = int((p_bin & g_bin).sum())
            fp = int((p_bin & (1 - g_bin)).sum())
            fn = int(((1 - p_bin) & g_bin).sum())
            tn = int(((1 - p_bin) & (1 - g_bin)).sum())
            les_tp, les_fp, les_fn = match_lesions(p_bin, g_bin, iou_th)
            per_image.append({"image_idx": len(per_image),
                              **_ratio_metrics(tp, fp, fn, tn),
                              "lesion_tp": les_tp, "lesion_fp": les_fp, "lesion_fn": les_fn,
                              "has_gt": int(g_bin.sum() > 0),
                              "has_pred": int((p_prob > thr).sum() > 0)})
    df = pd.DataFrame(per_image)

    # pixel-pooled on raw probabilities
    flat_p = np.concatenate([p.ravel() for p in all_probs]).astype(np.float32)
    flat_g = np.concatenate([g.ravel() for g in all_gts]).astype(np.uint8)
    flat_b = flat_p > thr
    pos = flat_g == 1
    TP = int((flat_b & pos).sum())
    FP = int((flat_b & ~pos).sum())
    FN = int((~flat_b & pos).sum())
    TN = int((~flat_b & ~pos).sum())
    pooled = _ratio_metrics(TP, FP, FN, TN)
    pooled["auc_roc"] = float(roc_auc_score(flat_g, flat_p))
    pooled["auc_pr"] = float(average_precision_score(flat_g, flat_p))
    del flat_p, flat_g, flat_b

    les_tp, les_fp, les_fn = (int(df[c].sum()) for c in ("lesion_tp", "lesion_fp", "lesion_fn"))
    les_sens = les_tp / (les_tp + les_fn + EPS)
    les_prec = les_tp / (les_tp + les_fp + EPS)
    les_f1 = (2 * les_prec * les_sens) / (les_prec + les_sens + EPS)

    hg, hp = df["has_gt"].to_numpy(), df["has_pred"].to_numpy()
    d_tp = int(((hp == 1) & (hg == 1)).sum())
    d_fp = int(((hp == 1) & (hg == 0)).sum())
    d_fn = int(((hp == 0) & (hg == 1)).sum())
    d_tn = int(((hp == 0) & (hg == 0)).sum())

    metric_cols = ["dice", "iou", "sensitivity", "specificity", "precision", "f1", "accuracy"]
    summary = {
        "lesion": cfg.name, "n_test_images": len(df), "threshold": float(thr),
        "tta": cfg.tta, "postprocess": bool(cfg.postprocess), "lesion_iou_thr": float(iou_th),
        "image_avg": {c: [float(df[c].mean()), float(df[c].std(ddof=0))] for c in metric_cols},
        "pixel_pooled": {k: float(v) for k, v in pooled.items()},
        "lesion_level": {"tp": les_tp, "fp": les_fp, "fn": les_fn, "sensitivity": les_sens,
                         "precision": les_prec, "f1": les_f1},
        "image_level_presence": {
            "tp": d_tp, "fp": d_fp, "fn": d_fn, "tn": d_tn,
            "sensitivity": d_tp / (d_tp + d_fn + EPS) if d_tp + d_fn else float("nan"),
            "specificity": d_tn / (d_tn + d_fp + EPS) if d_tn + d_fp else float("nan")},
    }
    return summary, df


def paper_table_row(summary: dict) -> dict:
    """One row of the paper's Stage II table from an :func:`evaluate_test_set` summary."""
    pp, ll = summary["pixel_pooled"], summary["lesion_level"]
    dice_m, dice_sd = summary["image_avg"]["dice"]
    return {
        "Target": get_lesion_config(summary["lesion"]).full_name,
        "n": summary["n_test_images"], "Thr.": summary["threshold"],
        "Dice": pp["dice"], "IoU": pp["iou"], "Sens.": pp["sensitivity"],
        "Prec.": pp["precision"], "ROC AUC": pp["auc_roc"], "PR AUC": pp["auc_pr"],
        "Image Dice": f"{dice_m:.3f} +- {dice_sd:.3f}",
        "Lesion Sens.": ll["sensitivity"], "Lesion Prec.": ll["precision"],
        "Lesion F1": ll["f1"],
        "Image sens.": summary["image_level_presence"]["sensitivity"],
    }
