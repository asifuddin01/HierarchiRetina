"""Stage I evaluation: metrics, operating threshold, TTA inference, calibration, equal-load
model comparison and ensembles.

Everything here works on per-image probability vectors, so the paper's Stage I numbers can be
recomputed from the cached prediction CSVs without a GPU.
"""
from __future__ import annotations

import json
import os
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .config import Stage1Config

# --------------------------------------------------------------------------------------------
# Metrics and operating threshold
# --------------------------------------------------------------------------------------------


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """AUC plus confusion-matrix metrics at ``threshold`` (positive if prob >= threshold)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(labels, probs),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "threshold": threshold,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def find_optimal_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    rule: str = "2tpr-fpr",
    clip: tuple[float, float] | None = (0.20, 0.60),
) -> float:
    """ROC threshold maximising 2*TPR - FPR (``rule='2tpr-fpr'``) or TPR - FPR (``'youden'``),
    optionally clipped to ``clip``. The gate uses 2*TPR - FPR clipped to [0.20, 0.60]."""
    fpr, tpr, thr = roc_curve(labels, probs)
    score = 2 * tpr - fpr if rule == "2tpr-fpr" else tpr - fpr
    t = float(thr[np.argmax(score)])
    return float(np.clip(t, *clip)) if clip is not None else t


def threshold_rule(cfg: Stage1Config) -> Callable[[np.ndarray, np.ndarray], float]:
    """The per-epoch threshold function of a preset."""
    return lambda y, p: find_optimal_threshold(y, p, cfg.threshold_rule, cfg.threshold_clip)


def grade_accuracy(probs: np.ndarray, grades: np.ndarray, thr: float, grade: int = 1) -> float:
    """Fraction of images of one true grade predicted DR (Grade-1 'routing rate')."""
    m = np.asarray(grades) == grade
    if m.sum() == 0:
        return float("nan")
    return float((np.asarray(probs)[m] >= thr).mean())


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Reliability ECE on p(DR): sum_b (n_b / N) |mean(y_b) - mean(p_b)| over equal-width bins."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    idx = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI from image-level bootstrap resamples (paper: 500 resamples for AUC)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if labels[i].min() == labels[i].max():
            continue
        vals.append(metric(labels[i], probs[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_grade_routing(grades: np.ndarray, probs: np.ndarray, thr: float) -> pd.DataFrame:
    """Per true grade: count, images routed to Stage II, blocked, routing rate, mean probability.

    For Grade 0 the routing rate is the false-positive load; for Grades 1-5 it is sensitivity.
    """
    grades = np.asarray(grades)
    probs = np.asarray(probs)
    rows = []
    for g in sorted(np.unique(grades)):
        m = grades == g
        routed = int((probs[m] >= thr).sum())
        rows.append({"grade": int(g), "n": int(m.sum()), "routed": routed,
                     "blocked": int(m.sum()) - routed, "routed_rate": routed / m.sum(),
                     "mean_prob": float(probs[m].mean())})
    return pd.DataFrame(rows)


def screening_summary(
    y_true: np.ndarray, probs: np.ndarray, grades: np.ndarray, thr: float, n_bins: int = 15
) -> dict:
    """Every gate number quoted in the Stage I results paragraph."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    grades = np.asarray(grades)
    m = compute_metrics(y_true, probs, thr)
    pred = probs >= thr
    sub = lambda gs: pred[np.isin(grades, gs)].mean()  # noqa: E731
    return {
        **m,
        "ppv": m["tp"] / max(m["tp"] + m["fp"], 1),
        "npv": m["tn"] / max(m["tn"] + m["fn"], 1),
        "routed_to_stage2": int(pred.sum()),
        "stopped_grade0": int((~pred).sum()),
        "stopped_fraction": float((~pred).mean()),
        "sens_grades_2_4": float(sub([2, 3, 4])),
        "sens_grades_3_4": float(sub([3, 4])),
        "grade1_routed": float(sub([1])),
        "ungradable_passed": f"{int(pred[grades == 5].sum())}/{int((grades == 5).sum())}",
        "healthy_admitted": int(pred[grades == 0].sum()),
        f"ece_{n_bins}bins": expected_calibration_error(y_true, probs, n_bins),
        "brier": brier_score(y_true, probs),
    }


def threshold_sweep(
    y_true: np.ndarray, probs: np.ndarray, grades: np.ndarray, thresholds: Iterable[float]
) -> pd.DataFrame:
    """Reporting-only sweep (the operating threshold is never tuned on test data)."""
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, probs, float(t))
        rows.append({"threshold": float(t), "grade1_routed": grade_accuracy(probs, grades, t),
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                     "f1": m["f1"], "fn": m["fn"], "fp": m["fp"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# TTA inference
# --------------------------------------------------------------------------------------------
def _read_preprocessed(path, preprocess_fn, img_size) -> np.ndarray | None:
    try:
        im = cv2.imread(str(path))
        if im is None:
            return None
        return cv2.cvtColor(preprocess_fn(im, img_size), cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _prefetch(paths, preprocess_fn, img_size, n_threads: int = 6, ahead: int = 24):
    """Preprocess images in background threads and yield them in input order."""
    with ThreadPoolExecutor(n_threads) as ex:
        q: deque = deque()
        for p in paths:
            q.append(ex.submit(_read_preprocessed, p, preprocess_fn, img_size))
            if len(q) >= ahead:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()


@torch.no_grad()
def tta_predict_paths(
    model: torch.nn.Module,
    paths: Sequence,
    preprocess_fn: Callable[[np.ndarray, int], np.ndarray],
    img_size: int,
    tfms: Sequence,
    device: torch.device,
    amp: bool = True,
    channels_last: bool = True,
    img_batch: int = 4,
    n_threads: int = 6,
    progress: bool = False,
) -> tuple[np.ndarray, int]:
    """Mean sigmoid over the TTA views of every image. Unreadable images get 0.5.

    Views are generated in the main thread in input order, so seeding the RNGs before a call
    makes the random rotation view repeatable. Returns (probabilities, number of failures).
    """
    model.eval()
    v = len(tfms)
    probs = np.full(len(paths), 0.5, dtype=np.float64)
    n_fail = 0
    idx: list[int] = []
    views: list[torch.Tensor] = []
    use_amp = amp and device.type == "cuda"

    def flush() -> None:
        x = torch.stack(views).to(device, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            p = torch.sigmoid(model(x).float()).view(-1, v).mean(1)
        probs[idx] = p.cpu().numpy()
        idx.clear()
        views.clear()

    it = _prefetch(paths, preprocess_fn, img_size, n_threads)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=len(paths), desc="TTA", leave=False)
    for i, rgb in enumerate(it):
        if rgb is None:
            n_fail += 1
            continue
        idx.append(i)
        views.extend(t(image=rgb)["image"] for t in tfms)
        if len(idx) >= img_batch:
            flush()
    if idx:
        flush()
    return probs, n_fail


def seed_tta(tfms: Sequence, seed: int) -> None:
    """Seed python/numpy RNGs (albumentations 1.x) and each Compose (albumentations 2.x)."""
    random.seed(seed)
    np.random.seed(seed)
    for t in tfms:
        if hasattr(t, "set_random_seed"):
            t.set_random_seed(seed)


def checkpoint_signature(path: str | Path) -> str:
    st = Path(path).stat()
    return f"{st.st_size}|{int(st.st_mtime)}"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_chunked_tta(
    predict_chunk: Callable[[Sequence], tuple[np.ndarray, int]],
    tfms: Sequence,
    paths: Sequence,
    keys: Sequence[str],
    cache_csv: str | Path,
    chunk_size: int,
    seed: int = 42,
    signature: str = "",
) -> np.ndarray:
    """Resumable TTA over ``paths`` in fixed chunks; chunk ``i`` is seeded with ``seed + i``.

    Finished chunks are written to ``<cache>_parts/part_XXXXX.csv``; a re-run computes only the
    missing ones. The merged result is ``cache_csv`` with columns ``key, prob``. The original
    run used chunk 500 for validation and 1000 for the test pool.
    """
    cache_csv = Path(cache_csv)
    keys = [str(k) for k in keys]
    meta = cache_csv.with_name(cache_csv.stem + ".meta.json")
    sig = f"{signature}|{len(tfms)}|{chunk_size}"
    # A cache without a meta file (e.g. one copied from the original run) is trusted if it
    # covers every key; a cache whose meta signature differs is recomputed.
    if cache_csv.exists() and (not meta.exists()
                               or json.loads(meta.read_text()).get("sig") == sig):
        c = pd.read_csv(cache_csv, dtype={"key": str}).set_index("key")["prob"].reindex(keys)
        if c.notna().all():
            return c.values.astype(float)

    parts = cache_csv.with_name(cache_csv.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    sig_file = parts / "signature.txt"
    if sig_file.exists() and sig_file.read_text().strip() != sig:
        for f in parts.glob("part_*.csv"):
            f.unlink()
    sig_file.write_text(sig)

    out = []
    for i, s in enumerate(range(0, len(paths), chunk_size)):
        ck = keys[s:s + chunk_size]
        part = parts / f"part_{i:05d}.csv"
        if part.exists():
            c = pd.read_csv(part, dtype={"key": str})
            if c["key"].tolist() == ck and c["prob"].notna().all():
                out.append(c["prob"].values.astype(float))
                continue
        seed_tta(tfms, seed + i)
        pr, _ = predict_chunk(paths[s:s + chunk_size])
        _atomic_csv(pd.DataFrame({"key": ck, "prob": pr}), part)
        out.append(pr)

    probs = np.concatenate(out) if out else np.zeros(0)
    _atomic_csv(pd.DataFrame({"key": keys, "prob": probs}), cache_csv)
    meta.write_text(json.dumps({"sig": sig, "n": len(keys),
                                "created": datetime.now().isoformat(timespec="seconds")}))
    return probs


# --------------------------------------------------------------------------------------------
# Cached prediction files
# --------------------------------------------------------------------------------------------
def load_prediction_cache(path: str | Path, name: str = "prob") -> pd.DataFrame:
    """Read a cached prediction CSV -> DataFrame(stem, <name>).

    Accepts the 768-px cache (``key`` = relative path, ``prob``) and the baseline caches
    (``stem``, ``prob``).
    """
    df = pd.read_csv(path)
    stem_col = next((c for c in df.columns if c.lower() in
                     ("key", "stem", "image", "img", "filename", "id", "name")), df.columns[0])
    prob_col = next((c for c in df.columns if c.lower() in
                     ("prob", "probability", "pred", "p", "prob_768", "y_prob")),
                    df.select_dtypes("number").columns[-1])
    return pd.DataFrame({"stem": df[stem_col].apply(lambda x: Path(str(x)).stem),
                         name: df[prob_col].astype(float).values})


# --------------------------------------------------------------------------------------------
# Equal-load comparison and ensembles (Cells 5.C and 5.D)
# --------------------------------------------------------------------------------------------
def matched_threshold(probs: np.ndarray, n_pos: int) -> float:
    """Threshold at which ``probs >= t`` emits exactly ``n_pos`` positives (barring ties)."""
    return float(np.sort(np.asarray(probs))[::-1][n_pos - 1])


def score_at_load(
    y_true: np.ndarray, grades: np.ndarray, probs: np.ndarray, n_pos: int, label: str,
    thr: float | None = None,
) -> dict:
    """AUC and confusion metrics after thresholding to an equal routing load of ``n_pos``."""
    thr = matched_threshold(probs, n_pos) if thr is None else thr
    pred = (np.asarray(probs) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    g1 = np.asarray(grades) == 1
    return {"variant": label, "auc": roc_auc_score(y_true, probs), "thr": thr,
            "sens": tp / (tp + fn), "spec": tn / (tn + fp), "f1": 2 * tp / (2 * tp + fp + fn),
            "g1_acc": pred[g1].mean() if g1.sum() else np.nan, "fn": int(fn), "fp": int(fp)}


def rank_norm(v: np.ndarray) -> np.ndarray:
    """Probabilities -> percentile ranks in [0, 1] (calibration-free averaging)."""
    return pd.Series(v).rank(pct=True).values


def weighted_mean(
    frame: pd.DataFrame, cols: Sequence[str], weights: dict | None = None, mode: str = "prob"
) -> np.ndarray | None:
    """Weighted mean of model columns, of probabilities (``prob``) or of ranks (``rank``)."""
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return None
    w = np.array([1.0 if weights is None else weights.get(c, 1.0) for c in cols])
    m = np.column_stack([frame[c].values if mode == "prob" else rank_norm(frame[c].values)
                         for c in cols])
    return (m * w).sum(axis=1) / w.sum()


FOLD_COLS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
BASELINE_COLS = FOLD_COLS + ["swinv2", "single_cnx"]

#: Ensemble variants of Cell 5.D: (label, columns, weights).
ENSEMBLE_DEFS: list[tuple[str, list[str], dict | None]] = [
    ("768_alone", ["cnx768"], None),
    ("768 + single_cnx + swinv2", ["cnx768", "single_cnx", "swinv2"], None),
    ("768 + 5folds", ["cnx768"] + FOLD_COLS, None),
    ("768 + all 7", ["cnx768"] + BASELINE_COLS, None),
    ("768x3 + all 7", ["cnx768"] + BASELINE_COLS, {"cnx768": 3.0}),
    ("768x5 + all 7", ["cnx768"] + BASELINE_COLS, {"cnx768": 5.0}),
    ("old hybrid (7, no 768)", BASELINE_COLS, None),
]


def equal_load_comparison(
    merged: pd.DataFrame, n_pos: int, model_cols: Sequence[str], ref_col: str = "cnx768",
    min_auc_gain: float = 0.003,
) -> dict:
    """Cell 5.D on a frame with ``gt_binary``, ``true_grade`` and one probability column per
    model. Returns single-model rows, Spearman correlation with ``ref_col``, ensemble rows and
    the pre-registered verdict (keep an ensemble only if dAUC >= +0.003 and Grade-1 routing does
    not drop)."""
    y, g = merged["gt_binary"].values, merged["true_grade"].values
    singles = pd.DataFrame([score_at_load(y, g, merged[c].values, n_pos, c) for c in model_cols])
    corr = (merged[list(model_cols)].corr(method="spearman")[ref_col].drop(ref_col)
            .sort_values(ascending=False))
    rows = []
    for label, cols, w in ENSEMBLE_DEFS:
        for mode in ("prob", "rank"):
            if label == "768_alone" and mode == "rank":
                continue
            v = weighted_mean(merged, cols, w, mode)
            if v is None:
                continue
            name = label if label == "768_alone" else f"{label}  [{mode}]"
            rows.append(score_at_load(y, g, v, n_pos, name))
    ens = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    base = ens.loc[ens.variant == "768_alone"].iloc[0]
    best = ens.loc[ens.variant != "768_alone"].iloc[0]
    keep = (best.auc - base.auc >= min_auc_gain) and (best.g1_acc >= base.g1_acc)
    return {"singles": singles, "spearman": corr, "ensembles": ens,
            "verdict": {"baseline_auc": float(base.auc), "best_variant": best.variant,
                        "best_auc": float(best.auc), "gain": float(best.auc - base.auc),
                        "use_ensemble": bool(keep)}}


def hybrid_variants(merged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Cell 5.C: ensembles of the seven baseline models at the OOF-calibrated threshold.

    A: mean of the five folds; B: folds + SwinV2; C: all seven (the "hybrid"); D: (2 x fold
    mean + SwinV2) / 3; followed by each single model.
    """
    y = merged["gt_binary"].values
    folds = [c for c in FOLD_COLS if c in merged.columns]
    allc = [c for c in BASELINE_COLS if c in merged.columns]
    var: dict[str, np.ndarray] = {}
    if folds:
        var["A: 5-Fold ConvNeXt"] = merged[folds].mean(axis=1).values
    if folds and "swinv2" in merged:
        var["B: 5-Fold + SwinV2"] = merged[folds + ["swinv2"]].mean(axis=1).values
    if len(allc) > 1:
        var["C: Full Hybrid (all)"] = merged[allc].mean(axis=1).values
    if folds and "swinv2" in merged:
        var["D: Weighted (folds x2, swin x1)"] = (
            2 * merged[folds].mean(axis=1).values + merged["swinv2"].values) / 3
    for c in allc:
        var[f"[single] {c}"] = merged[c].values
    rows = []
    for name, p in var.items():
        m = compute_metrics(y, p, threshold)
        rows.append({"variant": name, "auc": m["auc"], "sens": m["sensitivity"],
                     "spec": m["specificity"], "f1": m["f1"], "acc": m["accuracy"],
                     "fn": m["fn"], "fp": m["fp"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Figure 2 (ROC with operating point; routing rate per true grade)
# --------------------------------------------------------------------------------------------
def plot_gate_figure(y_true, probs, grades, thr: float, out_path: str | Path | None = None):
    """Two-panel Stage I figure: (a) ROC with the validation-derived operating point,
    (b) fraction of each true grade routed to Stage II/III."""
    import matplotlib.pyplot as plt

    y_true, probs = np.asarray(y_true), np.asarray(probs)
    fpr, tpr, _ = roc_curve(y_true, probs)
    m = compute_metrics(y_true, probs, thr)
    tab = per_grade_routing(grades, probs, thr)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {m['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].scatter([1 - m["specificity"]], [m["sensitivity"]], color="k", zorder=5,
                  label=f"tau = {thr:.4f}")
    ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="(a) ROC")
    ax[0].legend(loc="lower right")
    ax[1].bar(tab["grade"].astype(str), tab["routed_rate"], color="0.4")
    for x, r in zip(tab["grade"].astype(str), tab["routed_rate"]):
        ax[1].text(x, r + 0.02, f"{r:.3f}", ha="center", fontsize=9)
    ax[1].set(ylim=(0, 1.1), xlabel="True grade", ylabel="Fraction routed to Stage II",
              title="(b) Routing by true grade")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig
