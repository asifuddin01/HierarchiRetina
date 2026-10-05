"""Stage I gate split: route the test pool into the Stage II input set.

Routing uses only the cached gate probabilities and the frozen validation-derived threshold;
ground truth is merged in for the manifest and the gate confusion matrix but never used to
route. Original (not preprocessed) images are hard-linked (same file system) or copied, because
Stage II has its own 1024-px front-end.

Images with ``prob >= GATE_THRESHOLD`` go to ``dr_from_test`` (Stage II/III); the rest go to
``nodr_from_test`` and receive the final cascade grade 0.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import GATE_THRESHOLD
from .data import SUPPORTED_EXTS, load_test_labels
from .evaluate import load_prediction_cache

DR_DIRNAME = "dr_from_test"
NODR_DIRNAME = "nodr_from_test"
MANIFEST_COLUMNS = ["filename", "stem", "prob", "pred_binary", "routed_to", "true_grade",
                    "gt_binary", "gate_outcome", "cascade_grade", "has_file"]


def index_images(folder: str | Path) -> tuple[dict[str, Path], list[str]]:
    """Map file stem -> path for every supported image (first occurrence wins)."""
    stem2path: dict[str, Path] = {}
    dups: list[str] = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXTS:
            if p.stem in stem2path:
                dups.append(p.stem)
            else:
                stem2path[p.stem] = p
    return stem2path, dups


def build_manifest(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    stem2path: dict[str, Path],
    threshold: float = GATE_THRESHOLD,
) -> pd.DataFrame:
    """Routing table, one row per predicted image, sorted by descending probability.

    ``cascade_grade`` is 0 for blocked images (their end-to-end prediction) and empty for images
    that continue to Stage II/III.
    """
    pred = pred.drop_duplicates("stem", keep="first")
    man = pred.merge(gt, on="stem", how="left")
    src = man["stem"].map(lambda s: stem2path.get(s))
    man["has_file"] = src.notna()
    man["filename"] = src.map(lambda p: p.name if p is not None else "")
    man["pred_binary"] = (man["prob"] >= threshold).astype(int)
    man["routed_to"] = np.where(man["pred_binary"] == 1, DR_DIRNAME, NODR_DIRNAME)
    man["cascade_grade"] = np.where(man["pred_binary"] == 0, 0, np.nan)
    man["gate_outcome"] = np.select(
        [(man.gt_binary == 1) & (man.pred_binary == 1),
         (man.gt_binary == 0) & (man.pred_binary == 0),
         (man.gt_binary == 0) & (man.pred_binary == 1),
         (man.gt_binary == 1) & (man.pred_binary == 0)],
        ["TP", "TN", "FP", "FN"], default="unknown")
    man["src_path"] = src
    return man.sort_values("prob", ascending=False).reset_index(drop=True)


def _same_filesystem(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def _transfer(src: Path, dst: Path, how: str) -> None:
    if dst.exists():
        dst.unlink()
    if how == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def route_test_pool(
    pred_cache_csv: str | Path,
    test_csv: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    manifest_csv: str | Path,
    threshold: float = GATE_THRESHOLD,
    link_mode: str = "auto",
    wipe_first: bool = True,
    progress: bool = True,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Write ``out_dir/dr_from_test``, ``out_dir/nodr_from_test`` and the manifest CSV.

    Args:
        pred_cache_csv: gate test predictions (``key``/``stem`` + ``prob``).
        link_mode: ``auto`` (hard link if on the same file system, else copy), ``hardlink``
            or ``copy``.
        wipe_first: empty both output folders before writing.
    Returns the manifest (without the internal source-path column).
    """
    out_dir = Path(out_dir)
    dr_dir, nodr_dir = out_dir / DR_DIRNAME, out_dir / NODR_DIRNAME
    pred = load_prediction_cache(pred_cache_csv, "prob")
    gt = load_test_labels(test_csv)
    stem2path, dups = index_images(image_dir)
    if dups:
        log(f"{len(dups)} duplicate stems on disk (different extensions); first kept")
    man = build_manifest(pred, gt, stem2path, threshold)
    missing = int((~man.has_file).sum())
    if missing:
        log(f"{missing} predicted stems have no file on disk")

    for d in (dr_dir, nodr_dir):
        if wipe_first and d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    mode = link_mode
    if mode == "auto":
        mode = "hardlink" if _same_filesystem(Path(image_dir), out_dir) else "copy"
    log(f"transfer mode: {mode}")

    rows = man[man.has_file]
    it = rows.itertuples(index=False)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=len(rows), desc="routing", unit="img")
    n_fail = 0
    for r in it:
        try:
            dst_dir = dr_dir if r.pred_binary == 1 else nodr_dir
            _transfer(r.src_path, dst_dir / r.src_path.name, mode)
        except Exception as e:  # keep going; report at the end
            n_fail += 1
            if n_fail <= 10:
                log(f"failed {r.stem}: {e}")
    log(f"transferred {len(rows) - n_fail:,} files, {n_fail} failed")

    out = man[MANIFEST_COLUMNS].copy()
    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(manifest_csv, index=False)
    return out


def verify_split(manifest: pd.DataFrame, out_dir: str | Path) -> dict:
    """Counts on disk vs manifest, the gate confusion matrix and sens/spec."""
    out_dir = Path(out_dir)
    on_disk = {d: sum(1 for p in (out_dir / d).glob("*") if p.suffix.lower() in SUPPORTED_EXTS)
               for d in (DR_DIRNAME, NODR_DIRNAME)}
    expected = {DR_DIRNAME: int((manifest.pred_binary == 1).sum()),
                NODR_DIRNAME: int((manifest.pred_binary == 0).sum())}
    cm = manifest["gate_outcome"].value_counts()
    tp, tn, fp, fn = (int(cm.get(k, 0)) for k in ("TP", "TN", "FP", "FN"))
    return {"expected": expected, "on_disk": on_disk, "match": expected == on_disk,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)}
