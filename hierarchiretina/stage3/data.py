"""Stage III datasets, fold assignment and loaders.

Development set: 18,570 DR-positive images (grades 1-5), each with five Stage II masks
(MA, HE, EX, CWS, vessel). Internally ``y = grade - 1`` (0..4); the gradability target is
``grade <= 4`` and the severity target ``y_sev = min(y, 3)`` (only used when gradable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocessing import build_aug, load_image_and_masks, resolve_file

LESIONS = ["ma", "he", "ex", "cws", "vessel"]          # channel order of x[:, 3:8]
MASK_SUFFIX = {"ma": "_mask", "he": "_he_mask", "ex": "_ex_mask",
               "cws": "_cws_mask", "vessel": "_mask"}  # stem + suffix + ext
IMG_EXTS = (".png", ".jpg", ".ppm", ".jpeg", ".tif", ".tiff")


def build_dataframe(csv_path: str | Path, img_dir: str | Path, mask_dirs: dict[str, str | Path],
                    mask_suffix: dict[str, str] = MASK_SUFFIX, id_col: str = "image",
                    label_col: str = "grade", exts: tuple[str, ...] = IMG_EXTS) -> pd.DataFrame:
    """Resolve every CSV row to its image and five mask files.

    Row order follows the CSV (after ``dropna``); fold assignment depends on it.
    """
    df = pd.read_csv(csv_path)
    df = df[[id_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(int)
    assert df[label_col].between(1, 5).all(), "expected grades 1..5"
    rows = []
    for stem, grade in zip(df[id_col].astype(str), df[label_col]):
        rows.append({
            "image_id": stem,
            "img_path": resolve_file(stem, img_dir, exts),
            **{f"mask_{l}": resolve_file(stem, mask_dirs[l], exts, mask_suffix[l])
               for l in LESIONS},
            "grade": int(grade), "y": int(grade) - 1,
        })
    return pd.DataFrame(rows)


def assign_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Stratified (on the 5 grades) K-fold split, as in the training notebook."""
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, vi) in enumerate(skf.split(df, df["y"])):
        df.loc[vi, "fold"] = f
    return df


def verify_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing image or mask path and print counts and the fold x grade table."""
    mask_cols = [f"mask_{l}" for l in LESIONS]
    n_img = int(df["img_path"].isna().sum())
    n_msk = int(df[mask_cols].isna().sum().sum())
    print(f"rows {len(df)} | missing images {n_img} | missing mask entries {n_msk}")
    if "fold" in df:
        print(pd.crosstab(df["fold"], df["grade"]))
    clean = df.dropna(subset=["img_path"] + mask_cols).reset_index(drop=True)
    print(f"usable rows: {len(clean)} / {len(df)}")
    return clean


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gradable"] = (df["grade"] <= 4).astype(int)
    df["y_sev"] = df["y"].clip(upper=3)
    return df


def prepare_development_set(csv_path, img_dir, mask_root, n_folds: int = 5, seed: int = 42,
                            mask_suffix: dict[str, str] = MASK_SUFFIX) -> pd.DataFrame:
    """CSV -> resolved paths -> stratified folds -> verification -> two-head targets.

    ``mask_root`` holds one sub-folder per lesion (``ma, he, ex, cws, vessel``).
    """
    mask_dirs = {l: Path(mask_root) / l for l in LESIONS}
    df = build_dataframe(csv_path, img_dir, mask_dirs, mask_suffix)
    return add_targets(verify_dataset(assign_folds(df, n_folds, seed)))


def grad_pos_weight(df: pd.DataFrame) -> float:
    """``pos_weight`` of the gradability BCE: n_ungradable / n_gradable over the whole
    development set (805 / 17,765 = 0.0453). The BCE positive class is *gradable*, so this
    down-weights the majority class."""
    n_grad = int((df["grade"] <= 4).sum())
    n_ungrad = len(df) - n_grad
    return max(n_ungrad, 1) / max(n_grad, 1)


class DRDatasetTwoHead(Dataset):
    """Returns ``(x[8,H,W], y_full, y_sev, gradable)``; x = 3 normalised RGB + 5 binary masks."""

    def __init__(self, df: pd.DataFrame, img_size: int = 512, train: bool = True,
                 use_clahe: bool = True):
        self.df = df.reset_index(drop=True)
        self.use_clahe = use_clahe
        self.tf = build_aug(img_size, len(LESIONS), train)
        self.mask_cols = [f"mask_{l}" for l in LESIONS]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img, masks = load_image_and_masks(r["img_path"], [r[c] for c in self.mask_cols],
                                          self.use_clahe)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {r['img_path']}")
        out =self.tf(image=img, **{f"mask{j}": m for j, m in enumerate(masks)})
        x = torch.cat([out["image"],
                       torch.stack([out[f"mask{j}"] for j in range(len(masks))], 0)], 0)
        y_full = int(r["y"])
        gradable = 1 if r["grade"] <= 4 else 0
        return x, torch.tensor(y_full), torch.tensor(min(y_full, 3)), torch.tensor(gradable)


def make_loaders_2h(df: pd.DataFrame, fold: int, img_size: int = 512, batch_size: int = 16,
                    num_workers: int = 0, num_classes: int = 5):
    """Training loader with an inverse-frequency sampler over the five grades (computed on the
    training folds, with replacement) and an ordered validation loader.

    Returns ``(train_loader, val_loader, val_df)``; ``val_df`` is in loader order.
    """
    tr = df[df.fold != fold].reset_index(drop=True)
    va = df[df.fold == fold].reset_index(drop=True)
    cw = 1.0 / np.maximum(np.bincount(tr["y"], minlength=num_classes), 1)
    sw = cw[tr["y"].values]
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), True)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    tl = DataLoader(DRDatasetTwoHead(tr, img_size, train=True), batch_size, sampler=sampler,
                    drop_last=True, **kw)
    vl = DataLoader(DRDatasetTwoHead(va, img_size, train=False), batch_size, shuffle=False, **kw)
    return tl, vl, va


# ----------------------------------------------------------------------------- test time
def build_infer_frame(img_dir: str | Path, mask_dirs: dict[str, str | Path],
                      mask_suffix: dict[str, str] = MASK_SUFFIX,
                      exts: tuple[str, ...] = IMG_EXTS + (".bmp", ".webp")) -> pd.DataFrame:
    """List every image in ``img_dir`` (the Stage I DR-routed folder) and resolve its masks.

    Prints the per-lesion mask availability; missing masks are zero-filled at load time, which
    degrades predictions silently, so check that every lesion is at 100%.
    """
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts)
    rows = [{"image": p.name, "stem": p.stem, "img_path": str(p),
             **{f"path_{l}": resolve_file(p.stem, mask_dirs[l], exts, mask_suffix.get(l, ""))
                for l in LESIONS}} for p in imgs]
    fdf = pd.DataFrame(rows)
    for l in LESIONS:
        n = int(fdf[f"path_{l}"].notna().sum()) if len(fdf) else 0
        print(f"  {l:<7} masks found {n:>7,} / {len(fdf):,}")
    return fdf


class InferDataset(Dataset):
    """Test-time dataset over ``build_infer_frame`` rows. Returns ``(x, stem, ok)``; an
    unreadable image yields a zero tensor and ``ok = 0``."""

    def __init__(self, fdf: pd.DataFrame, img_size: int = 512):
        self.df = fdf.reset_index(drop=True)
        self.img_size = img_size
        self.tf = build_aug(img_size, len(LESIONS), train=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img, masks = load_image_and_masks(r["img_path"], [r[f"path_{l}"] for l in LESIONS])
        if img is None:
            return torch.zeros(3 + len(LESIONS), self.img_size, self.img_size), r["stem"], 0
        out = self.tf(image=img, **{f"mask{j}": m for j, m in enumerate(masks)})
        x = torch.cat([out["image"],
                       torch.stack([out[f"mask{j}"] for j in range(len(masks))], 0)], 0)
        return x, r["stem"], 1
