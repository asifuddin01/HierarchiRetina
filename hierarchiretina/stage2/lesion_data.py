"""Data preparation, splits, augmentation and datasets for the HSMoE-AUNet lesion models.

Pipeline (identical for all four lesions, parameters from ``LESION_CONFIGS``):
    1. Pair every fundus image with its lesion mask by file stem (lesion-specific suffixes
       such as ``_MA`` or ``_Hard_Exudates`` are stripped from mask stems).
    2. Drop pairs whose mask has ``<= min_lesion_pixels`` positive pixels (>127), counted at the
       mask's *original* resolution.
    3. Crop image and mask to the bounding rectangle of the retinal disc (grey threshold 15,
       11x11 elliptical closing x3 and opening x2, 10-pixel margin) and binarise the mask.
    4. Save both as PNG and write a manifest CSV (sorted by stem).
    5. Split with ``train_test_split(test_size=0.30, random_state=42)`` and then
       ``train_test_split(temp, test_size=0.33, random_state=42)`` -> train / val / test.
Training images are resized (stretched) to 1024x1024 by the transforms below.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .hsmoe_aunet import LESION_CONFIGS, get_lesion_config

TARGET_SIZE = 1024
SPLIT_SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Retinal-area crop (training / test data; also used at deployment for EX and CWS)
# =============================================================================
def detect_retinal_area(img_bgr: np.ndarray, pad: int = 10) -> tuple[int, int, int, int]:
    """Bounding rectangle (x1, y1, x2, y2) of the largest bright region (the retinal disc)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(largest)
        return max(0, x - pad), max(0, y - pad), min(w, x + ww + pad), min(h, y + hh + pad)
    return 0, 0, w, h


def crop_retinal_area(img_bgr: np.ndarray, mask_gray: np.ndarray):
    """Crop image and mask to :func:`detect_retinal_area` of the image."""
    x1, y1, x2, y2 = detect_retinal_area(img_bgr)
    return img_bgr[y1:y2, x1:x2], mask_gray[y1:y2, x1:x2]


# =============================================================================
# Step 1-4: pairing, filtering, cropping, manifest
# =============================================================================
def pair_images_and_masks(img_dir: Path, mask_dir: Path,
                          strip_suffixes: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Return sorted (stem, image_path, mask_path) for stems present in both folders."""
    img_dict = {Path(p).stem: p for p in sorted(glob.glob(str(Path(img_dir) / "*")))}
    mask_dict = {}
    for p in sorted(glob.glob(str(Path(mask_dir) / "*"))):
        base = Path(p).stem
        for s in strip_suffixes:          # sequential str.replace, as in the notebooks
            base = base.replace(s, "")
        mask_dict[base] = p
    common = sorted(set(img_dict) & set(mask_dict))
    return [(s, img_dict[s], mask_dict[s]) for s in common]


def prepare_lesion_dataset(lesion: str, img_dir: Path, mask_dir: Path, out_img_dir: Path,
                           out_mask_dir: Path, manifest_path: Path) -> pd.DataFrame:
    """Filter near-empty masks, crop to the retinal area, save PNGs and the manifest CSV."""
    cfg = get_lesion_config(lesion)
    out_img_dir, out_mask_dir = Path(out_img_dir), Path(out_mask_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_images_and_masks(img_dir, mask_dir, cfg.mask_strip_suffixes)
    rows, skipped = [], 0
    for stem, img_path, mask_path in tqdm(pairs, desc=f"{cfg.name}: filter + crop"):
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            skipped += 1
            continue
        mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        if (mask_gray > 127).sum() <= cfg.min_lesion_pixels:
            skipped += 1
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1
            continue
        img_crop, mask_crop = crop_retinal_area(img_bgr, mask_gray)
        _, mask_crop = cv2.threshold(mask_crop, 127, 255, cv2.THRESH_BINARY)
        out_img, out_mask = out_img_dir / f"{stem}.png", out_mask_dir / f"{stem}.png"
        cv2.imwrite(str(out_img), img_crop)
        cv2.imwrite(str(out_mask), mask_crop)
        rows.append({"stem": stem, "image": str(out_img), "mask": str(out_mask),
                     "height": img_crop.shape[0], "width": img_crop.shape[1],
                     cfg.pixel_col: int((mask_crop > 127).sum())})
    manifest = pd.DataFrame(rows)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"{cfg.name}: pairs={len(pairs)} kept={len(rows)} skipped={skipped} -> {manifest_path}")
    return manifest


# =============================================================================
# Step 5: split
# =============================================================================
def split_manifest(manifest: pd.DataFrame, seed: int = SPLIT_SEED):
    """70 / ~20 / ~10 split exactly as in the notebooks; returns (train, val, test)."""
    train_df, temp_df = train_test_split(manifest, test_size=0.30, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=seed)
    return train_df, val_df, test_df


def load_splits(manifest_path: Path, lesion: str | None = None, check_sizes: bool = False):
    """Read a manifest CSV and split it. Optionally assert the paper split sizes."""
    manifest = pd.read_csv(manifest_path)
    tr, va, te = split_manifest(manifest)
    if check_sizes and lesion is not None:
        exp = get_lesion_config(lesion).expected_split_sizes
        got = (len(tr), len(va), len(te))
        assert got == exp, f"{lesion}: split {got} differs from the paper run {exp}"
    return tr, va, te


# =============================================================================
# Augmentation (arguments verbatim from each notebook)
# =============================================================================
# The notebooks were written for the Albumentations 1.x API. Under Albumentations >= 2.0 the
# arguments 'alpha_affine' (ElasticTransform) and 'max_holes/max_height/max_width/fill_value'
# (CoarseDropout) are ignored with a warning and the library defaults are used instead. The
# calls are kept verbatim so that a given Albumentations version reproduces the original run.
def _base_head():
    return [A.Resize(TARGET_SIZE, TARGET_SIZE),
            A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
            A.HorizontalFlip(p=0.5)]


def _tail():
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


def _train_ops_ma():
    return [
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                 A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)], p=0.7),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                 A.MedianBlur(blur_limit=3, p=1.0)], p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
    ]


def _train_ops_he():
    return [
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                 A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                 A.RandomGamma(gamma_limit=(80, 120), p=1.0)], p=0.8),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                 A.MedianBlur(blur_limit=3, p=1.0),
                 A.MotionBlur(blur_limit=3, p=1.0)], p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=12, p=0.5),
        A.RGBShift(r_shift_limit=12, g_shift_limit=8, b_shift_limit=8, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.2),
        A.CoarseDropout(max_holes=6, max_height=48, max_width=48, fill_value=0, p=0.15),
    ]


def _train_ops_ex():
    return [  # no vertical flip (optic-disc term uses the disc position)
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.12, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=1.0),
                 A.CLAHE(clip_limit=3.5, tile_grid_size=(4, 4), p=1.0),
                 A.RandomGamma(gamma_limit=(75, 130), p=1.0)], p=0.85),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                 A.MedianBlur(blur_limit=3, p=1.0)], p=0.25),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=6, g_shift_limit=10, b_shift_limit=6, p=0.3),
        A.ToGray(p=0.10),
        A.GridDistortion(num_steps=5, distort_limit=0.06, p=0.15),
        A.CoarseDropout(max_holes=4, max_height=40, max_width=40, fill_value=0, p=0.12),
    ]


def _train_ops_cws():
    return [  # no vertical flip (optic-disc term uses the disc position)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=1.0),
                 A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=1.0),
                 A.RandomGamma(gamma_limit=(85, 115), p=1.0)], p=0.75),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                 A.MedianBlur(blur_limit=5, p=1.0),
                 A.MotionBlur(blur_limit=5, p=1.0)], p=0.40),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=6, g_shift_limit=8, b_shift_limit=6, p=0.3),
        A.ElasticTransform(alpha=30, sigma=4, alpha_affine=4, p=0.20),
        A.GridDistortion(num_steps=5, distort_limit=0.08, p=0.20),
        A.CoarseDropout(max_holes=4, max_height=40, max_width=40, fill_value=0, p=0.12),
    ]


_TRAIN_OPS = {"MA": _train_ops_ma, "HE": _train_ops_he, "EX": _train_ops_ex,
              "CWS": _train_ops_cws}


def build_train_transform(lesion: str) -> A.Compose:
    """Lesion-specific training augmentation (resize 1024, flips, photometric, dropout)."""
    return A.Compose(_base_head() + _TRAIN_OPS[get_lesion_config(lesion).name]() + _tail())


def build_eval_transform() -> A.Compose:
    """Validation / test / deployment transform: resize to 1024, ImageNet normalisation."""
    return A.Compose([A.Resize(TARGET_SIZE, TARGET_SIZE)] + _tail())


# =============================================================================
# Dataset and loaders
# =============================================================================
class LesionDataset(Dataset):
    """(image, mask) pairs from a manifest; returns float tensors (3,H,W) and (1,H,W)."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return img.float(), mask.unsqueeze(0).float()


def build_dataloaders(lesion: str, train_df, val_df, test_df, batch_size: int | None = None,
                      num_workers: int = 0, pin_memory: bool | None = None):
    """Train (shuffled, drop_last) and val/test loaders with the lesion's batch size."""
    cfg = get_lesion_config(lesion)
    bs = batch_size or cfg.batch_size
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    ev = build_eval_transform()
    train_loader = DataLoader(LesionDataset(train_df, build_train_transform(lesion)),
                              batch_size=bs, shuffle=True, num_workers=num_workers,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(LesionDataset(val_df, ev), batch_size=bs, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(LesionDataset(test_df, ev), batch_size=bs, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


# =============================================================================
# Overlap of the Stage II splits with the grading test pool
# =============================================================================
_OVERLAP_SUFFIX = re.compile(r"(_ma|_he|_ex|_se|_cws|_mask)$")


def overlap_key(name) -> str:
    """Normalise a file name for matching: drop folder, extension, case and lesion suffix."""
    return _OVERLAP_SUFFIX.sub("", Path(str(name)).stem.lower())


def grading_source(name) -> str:
    """Source dataset of a grading image from its file name (sources shared with Stage II)."""
    n = str(name)
    if n.startswith("IDRiD"):
        return "IDRiD"
    if re.match(r"^\d{4}_\d(\.png)?$", n):
        return "FGADR"
    if re.match(r"^\d{8}_\d{5}_\d{4}_PP", n) or n.upper().startswith("IM"):
        return "Messidor-2"
    return "other"


def stage2_overlap(test_csv: Path, manifests: dict[str, Path],
                   vessel_img_dir: Path | None = None, image_col: str = "image"):
    """Count grading-test images that fall in each lesion split (and the vessel pool).

    Returns ``(table, distinct_train_hits, n_in_vessel_pool)`` where ``table`` has one row per
    (model, split, source) and ``distinct_train_hits`` lists the grading-test rows found in the
    training split of at least one lesion model.
    """
    test = pd.read_csv(test_csv)
    test["key"] = test[image_col].map(overlap_key)
    test["src"] = test[image_col].map(grading_source)
    test = test[test.src != "other"]

    rows, train_keys = [], set()
    for lesion, mpath in manifests.items():
        tr, va, te = split_manifest(pd.read_csv(mpath))
        exp = LESION_CONFIGS[lesion].expected_split_sizes
        assert (len(tr), len(va), len(te)) == exp, f"{lesion}: split did not reproduce"
        train_keys |= set(tr["stem"].map(overlap_key))
        for part, df in (("train", tr), ("val", va), ("test", te)):
            keys = set(df["stem"].map(overlap_key))
            for src, g in test.groupby("src"):
                rows.append({"model": lesion, "split": part, "grading_test_source": src,
                             "n_overlap": int(g.key.isin(keys).sum()),
                             "n_source_test": len(g)})
    n_vessel = None
    if vessel_img_dir is not None:
        vkeys = {overlap_key(p.name) for p in Path(vessel_img_dir).glob("*")}
        for src, g in test.groupby("src"):
            rows.append({"model": "Vessel", "split": "whole pool", "grading_test_source": src,
                         "n_overlap": int(g.key.isin(vkeys).sum()), "n_source_test": len(g)})
        n_vessel = int(test.key.isin(vkeys).sum())
    hits = test[test.key.isin(train_keys)]
    return pd.DataFrame(rows), hits, n_vessel
