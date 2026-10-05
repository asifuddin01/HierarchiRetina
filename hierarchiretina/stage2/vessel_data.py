"""Vessel data: image/mask pairing, dataset audit, FOV-aware preprocessing, splits and Dataset.

Pipeline per image (identical for training, validation, test and real-world inference):

1. FOV mask from the raw image: grey > 12, elliptical 25x25 closing and opening, largest
   external contour filled, 9x9 elliptical erosion (so the FOV rim itself is not scored).
2. The vessel ground truth is multiplied by the FOV mask.
3. LAB-lightness CLAHE (clip 2.0, 8x8 tiles), then green-channel histogram equalisation blended
   at strength 0.35.
4. Square crop to the FOV bounding box (1 % margin), applied jointly to image, mask and FOV.
5. Resize to 512 x 512 (bilinear for the image, nearest for mask and FOV).
6. Augmentation (training only), ImageNet normalisation.

Note that enhancement (step 3) is applied to the full image before cropping (step 4), as in the
original notebook.
"""
from __future__ import annotations

import csv
import math
import random
import re
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Split of the 288-image vessel pool (DRIVE 40 + STARE 20 + HRF 30 + MAPLES-DR 198) -> 230/28/30.
SPLIT_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# Fraction of vessel pixels inside the FOV reported in the paper (Stage II methods). The notebook
# estimated it on the first 50 (augmented) training samples; see `compute_fov_vessel_ratio`.
FOV_VESSEL_RATIO = 0.1445
MAX_POS_WEIGHT = 8.0


# --------------------------------------------------------------------------------------------
# Image / mask pairing
# --------------------------------------------------------------------------------------------
def norm_stem(stem: str) -> str:
    """Normalise a file stem so that an image and its mask map to the same key."""
    s = stem.lower().strip()
    suffixes = ["_mask", "_seg", "_label", "_gt", "_vessel", "_image", "_img",
                "mask", "image", "img", ".ah", ".bmp", ".ppm"]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    s = re.sub(r"^im(\d+)", r"im\1", s)
    s = s.replace("drive_test_image", "drive_test")
    s = s.replace("drive_train_image", "drive_train")
    s = s.replace("drive_test_mask", "drive_test")
    s = s.replace("drive_train_mask", "drive_train")
    s = re.sub(r"[_-](image|mask|seg|label|gt|vessel|ah|bmp|ppm)$", "", s)
    return s.strip("_- ")


def list_images(folder: str | Path) -> list[Path]:
    """Image files in ``folder`` in a deterministic order.

    The original notebook used ``Path.iterdir()`` on Windows/NTFS, which returns entries sorted by
    their upper-cased name; sorting by ``name.upper()`` reproduces that order on any OS (for ASCII
    names), which in turn is required to reproduce the random split.
    """
    files = [f for f in Path(folder).iterdir() if f.suffix.lower() in VALID_EXT]
    return sorted(files, key=lambda p: p.name.upper())


def load_dataset_files(img_dir: str | Path, mask_dir: str | Path):
    """Pair images with vessel masks by (normalised) stem.

    Returns ``(matched, missing_mask, missing_img)`` where ``matched`` maps image stem ->
    ``(image_path, mask_path)`` in directory order.
    """
    img_files = {f.stem: f for f in list_images(img_dir)}
    mask_files = {f.stem: f for f in list_images(mask_dir)}
    mask_norm = {norm_stem(k): v for k, v in mask_files.items()}
    matched, miss_mask, miss_img = {}, [], []
    for stem, ipath in img_files.items():
        ns = norm_stem(stem)
        if stem in mask_files:
            matched[stem] = (ipath, mask_files[stem])
        elif ns in mask_norm:
            matched[stem] = (ipath, mask_norm[ns])
        else:
            miss_mask.append(stem)
    img_norm_set = {norm_stem(s) for s in img_files}
    for stem in mask_files:
        if stem not in img_files and norm_stem(stem) not in img_norm_set:
            miss_img.append(stem)
    return matched, miss_mask, miss_img


def audit_dataset(matched: dict, n_max: int = 300) -> tuple[pd.DataFrame, dict]:
    """Per-pair statistics (resolution, vessel ratio, black-border ratio) and a summary.

    ``summary['crop_retina']`` reproduces the notebook rule: crop to the FOV when the mean
    black-border fraction exceeds 10 %.
    """
    rows, corrupted, mask_vals = [], [], set()
    for stem, (ipath, mpath) in list(matched.items())[:n_max]:
        img = cv2.imread(str(ipath))
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            corrupted.append(stem)
            continue
        h, w = img.shape[:2]
        mask_vals.update(np.unique(mask).tolist())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows.append({"stem": stem, "height": h, "width": w,
                     "vessel_ratio": float((mask > 127).mean()),
                     "black_ratio": float((gray < 10).mean())})
    df = pd.DataFrame(rows)
    summary = {
        "n_pairs": len(matched),
        "n_analysed": len(df),
        "n_corrupted": len(corrupted),
        "min_resolution": f"{df.height.min()}x{df.width.min()}" if len(df) else "",
        "max_resolution": f"{df.height.max()}x{df.width.max()}" if len(df) else "",
        "vessel_pct_mean": 100 * df.vessel_ratio.mean() if len(df) else float("nan"),
        "vessel_pct_std": 100 * df.vessel_ratio.std(ddof=0) if len(df) else float("nan"),
        "black_border_pct_mean": 100 * df.black_ratio.mean() if len(df) else float("nan"),
        "mask_unique_values": sorted(mask_vals)[:10],
    }
    summary["crop_retina"] = bool(df.black_ratio.mean() > 0.10) if len(df) else True
    return df, summary


# --------------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------------
def split_pairs(matched: dict, seed: int = SPLIT_SEED, train_ratio: float = TRAIN_RATIO,
                val_ratio: float = VAL_RATIO):
    """Random 80/10/10 split reproducing the original notebook's random-number sequence.

    In the notebook, ``random.seed(42)`` was followed by one ``random.shuffle`` of the stem list
    (to choose preview images) and then by ``random.shuffle`` of the pair list that defines the
    split. Both shuffles are replayed here with a private ``random.Random(seed)``, which yields the
    same sequence as the seeded module-level generator. With 288 pairs this gives 230/28/30.
    """
    rng = random.Random(seed)
    preview_stems = list(matched.keys())
    rng.shuffle(preview_stems)  # advances the RNG exactly as the notebook's preview shuffle did
    all_pairs = [(v[0], v[1]) for v in matched.values()]
    rng.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (all_pairs[:n_train], all_pairs[n_train:n_train + n_val],
            all_pairs[n_train + n_val:])


def save_split_csv(path: str | Path, train, val, test) -> None:
    """Write the split as ``split,image,mask`` (file names only) for exact reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "mask"])
        for name, pairs in (("train", train), ("val", val), ("test", test)):
            for ip, mp in pairs:
                w.writerow([name, Path(ip).name, Path(mp).name])


def load_split_csv(path: str | Path, img_dir: str | Path, mask_dir: str | Path):
    """Read a split CSV written by `save_split_csv` and return (train, val, test) path pairs."""
    out = {"train": [], "val": [], "test": []}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["split"]].append((Path(img_dir) / r["image"], Path(mask_dir) / r["mask"]))
    return out["train"], out["val"], out["test"]


# --------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------
def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the LAB lightness channel (clip 2.0, 8x8 tiles)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance_green(img_rgb: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Blend the green channel with its histogram-equalised version."""
    g2 = cv2.equalizeHist(img_rgb[:, :, 1].astype(np.uint8)).astype(np.float32)
    img = img_rgb.astype(np.float32)
    img[:, :, 1] = np.clip(img[:, :, 1] * (1 - strength) + g2 * strength, 0, 255)
    return img.astype(np.uint8)


def generate_fov_mask(img: np.ndarray, thresh: int = 12, is_rgb: bool = True) -> np.ndarray:
    """Binary (0/1, uint8) field-of-view mask: the circular retinal region, slightly eroded."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fov = np.zeros_like(gray)
    if cnts:
        cv2.drawContours(fov, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    else:
        fov[:] = 255
    fov = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return (fov > 127).astype(np.uint8)


def fov_crop_box(fov: np.ndarray, margin: float = 0.01) -> tuple[int, int, int, int]:
    """Square box ``(x1, y1, x2, y2)`` around the FOV bounding box (clipped to the image)."""
    ys, xs = np.where(fov > 0)
    h, w = fov.shape[:2]
    if len(xs) == 0:
        return 0, 0, w, h
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    mg = int(max(x_max - x_min, y_max - y_min) * margin)
    x1, y1 = max(0, x_min - mg), max(0, y_min - mg)
    x2, y2 = min(w, x_max + mg), min(h, y_max + mg)
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    return int(x1), int(y1), int(x2), int(y2)


def crop_to_fov(img_rgb, msk, fov, margin: float = 0.01):
    """Square crop of image, mask and FOV to the FOV bounding box."""
    if not (fov > 0).any():
        return img_rgb, msk, fov
    x1, y1, x2, y2 = fov_crop_box(fov, margin)
    return img_rgb[y1:y2, x1:x2], msk[y1:y2, x1:x2], fov[y1:y2, x1:x2]


def resize_square(img, msk, fov, size: int):
    """Resize image (bilinear), mask and FOV (nearest) to ``size x size``."""
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(msk.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    fov_r = cv2.resize(fov.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, msk_r, (fov_r > 0.5).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------------
def _gauss_noise() -> A.BasicTransform:
    """Gaussian noise with variance 8-30 (uint8 scale), across Albumentations versions.

    Albumentations >= 2 silently ignores ``var_limit`` and would fall back to a far stronger
    default; the equivalent standard-deviation range (normalised by 255) is used instead.
    """
    if int(A.__version__.split(".")[0]) >= 2:
        return A.GaussNoise(std_range=(math.sqrt(8) / 255.0, math.sqrt(30) / 255.0))
    return A.GaussNoise(var_limit=(8, 30))


def get_train_transform() -> A.Compose:
    """Training augmentation; the FOV mask is transformed together with the vessel mask."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.ElasticTransform(alpha=60, sigma=6, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.25),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=3.0),
        ], p=0.6),
        A.OneOf([A.Sharpen(alpha=(0.2, 0.4)), _gauss_noise()], p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"fov": "mask"})


def get_val_transform() -> A.Compose:
    """Validation / test transform: normalisation only."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"fov": "mask"})


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------
class RetinalVesselDataset(Dataset):
    """Returns ``(image [3,H,W], vessel mask [1,H,W], FOV mask [1,H,W])`` float tensors."""

    def __init__(self, pairs, img_size: int = 512, transform=None, crop_retina: bool = True):
        self.pairs = pairs
        self.img_size = img_size
        self.transform = transform
        self.crop_retina = crop_retina

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ipath, mpath = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(str(ipath)), cv2.COLOR_BGR2RGB)
        msk = (cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        fov = generate_fov_mask(img, is_rgb=True).astype(np.float32)
        msk = msk * fov  # ground truth is restricted to the FOV

        img = apply_clahe(img)
        img = enhance_green(img)
        if self.crop_retina:
            img, msk, fov = crop_to_fov(img, msk, fov, margin=0.01)
        img, msk, fov = resize_square(img, msk, fov, self.img_size)

        msk_u = (msk * 255).astype(np.uint8)
        fov_u = (fov * 255).astype(np.uint8)
        if self.transform:
            out = self.transform(image=img, mask=msk_u, fov=fov_u)
            img_t = out["image"]
            msk_t = (out["mask"].float() / 255.0).unsqueeze(0)
            fov_t = (out["fov"].float() / 255.0).unsqueeze(0)
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            msk_t = torch.from_numpy(msk).float().unsqueeze(0)
            fov_t = torch.from_numpy(fov).float().unsqueeze(0)
        return img_t, msk_t, fov_t


def compute_fov_vessel_ratio(ds: RetinalVesselDataset, n: int = 50) -> float:
    """Mean vessel fraction inside the FOV over the first ``n`` samples of ``ds``.

    As in the notebook this is evaluated on the (augmented) training set, so the value varies
    slightly between runs; the paper reports 14.45 %.
    """
    ratios = []
    for i in range(min(n, len(ds))):
        _, m, fv = ds[i]
        inside = fv.squeeze().numpy() > 0.5
        mm = m.squeeze().numpy()
        ratios.append(mm[inside].mean() if inside.sum() > 0 else mm.mean())
    return float(np.mean(ratios))


def pos_weight_from_ratio(vessel_ratio: float) -> float:
    """BCE positive weight = background/vessel ratio inside the FOV, capped at 8."""
    r = max(float(vessel_ratio), 0.01)
    return min((1 - r) / r, MAX_POS_WEIGHT)
