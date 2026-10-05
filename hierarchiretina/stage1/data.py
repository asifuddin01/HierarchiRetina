"""Stage I data: label tables, splits, sampler, dataset and augmentations.

Labels: Grade 0 -> No DR (0); Grades 1-5 -> DR (1). Grade 5 (ungradable) is DR-positive so that
ungradable images reach the gradability head of Stage III instead of being cleared as healthy.
"""
from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import IMAGENET_MEAN, IMAGENET_STD, Stage1Config
from .preprocessing import get_preprocess

IMAGE_COL_CANDIDATES = ("image", "img", "filename", "id", "name")
GRADE_COL_CANDIDATES = ("grade", "label", "diagnosis", "target", "level", "dr")
#: Extensions tried, in order, when resolving a CSV name to a file.
RESOLVE_EXTS = (".jpeg", ".jpg", ".png", ".JPG", ".JPEG", ".PNG", ".ppm", ".tif")
#: Extensions accepted when scanning an image folder.
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm"}


# --------------------------------------------------------------------------------------------
# Label tables
# --------------------------------------------------------------------------------------------
def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """(image column, grade column) by name, falling back to the first two columns."""
    img_col = next((c for c in df.columns if c.lower() in IMAGE_COL_CANDIDATES), df.columns[0])
    grade_col = next((c for c in df.columns if c.lower() in GRADE_COL_CANDIDATES), df.columns[1])
    return img_col, grade_col


def _ensure_ext(name: str) -> str:
    name = str(name).strip()
    return name if Path(name).suffix else name + ".jpeg"


def _resolve_image(name: str, search_dirs: list[Path]) -> str | None:
    stem = Path(name).stem
    for ext in RESOLVE_EXTS:
        for d in search_dirs:
            p = d / (stem + ext)
            if p.exists():
                return str(p)
    return None


def load_development_pool(csv_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    """Read ``train_grade.csv`` and resolve image paths.

    Returns columns ``image, grade, img_path, binary_label`` (rows without a file are dropped,
    duplicate names keep their first occurrence, row order is the CSV order).
    """
    df = pd.read_csv(csv_path)
    img_col, grade_col = detect_columns(df)
    df = df.rename(columns={img_col: "image", grade_col: "grade"})
    df["image"] = df["image"].astype(str).apply(_ensure_ext)
    image_dir = Path(image_dir)
    search_dirs = [image_dir, image_dir / "train", image_dir / "images"]
    df["img_path"] = df["image"].apply(lambda n: _resolve_image(n, search_dirs))
    df = df[df["img_path"].notna()].reset_index(drop=True)
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df = df.dropna(subset=["grade"]).reset_index(drop=True)
    df["grade"] = df["grade"].astype(int)
    df["binary_label"] = (df["grade"] > 0).astype(int)
    df = df.drop_duplicates(subset=["image"]).reset_index(drop=True)
    return df[["image", "grade", "img_path", "binary_label"]]


def load_test_labels(csv_path: str | Path) -> pd.DataFrame:
    """Read ``test_grade.csv`` -> columns ``stem, true_grade, gt_binary`` (CSV order)."""
    gt = pd.read_csv(csv_path)
    img_col, grade_col = detect_columns(gt)
    return pd.DataFrame({
        "stem": gt[img_col].apply(lambda x: Path(str(x)).stem),
        "true_grade": gt[grade_col].astype(int).values,
        "gt_binary": (gt[grade_col] > 0).astype(int).values,
    })


def list_images(folder: str | Path) -> list[Path]:
    """All supported images below ``folder``, sorted (the order used for test inference)."""
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


# --------------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------------
def split_development_pool(
    df: pd.DataFrame, cfg: Stage1Config
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/validation split exactly as in the preset's notebook.

    * ``grade_stratified`` (gate): 80/20, stratified on the six-level grade.
    * ``binary_stratified`` (512 single): 85/15, stratified on the binary label.
    * ``three_way`` (SwinV2): 10 % internal test, then 1/9 of the rest as validation, both
      stratified on the binary label; returns (train, val, internal_test).
    ``kfold`` presets use :func:`kfold_splits` instead.
    """
    s = cfg.seed
    if cfg.split == "grade_stratified":
        tr, va = train_test_split(df, test_size=cfg.val_split, random_state=s,
                                  stratify=df["grade"])
        return tr.reset_index(drop=True), va.reset_index(drop=True)
    if cfg.split == "binary_stratified":
        tr, va = train_test_split(df, test_size=cfg.val_split, random_state=s,
                                  stratify=df["binary_label"])
        return tr.reset_index(drop=True), va.reset_index(drop=True)
    if cfg.split == "three_way":
        trva, te = train_test_split(df, test_size=cfg.test_split, random_state=s,
                                    stratify=df["binary_label"])
        frac = cfg.val_split / (1.0 - cfg.test_split)
        tr, va = train_test_split(trva, test_size=frac, random_state=s,
                                  stratify=trva["binary_label"])
        return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
    raise ValueError(f"split={cfg.split!r} is not a single split; use kfold_splits()")


def kfold_splits(df: pd.DataFrame, cfg: Stage1Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold(n_folds, shuffle=True, seed) on the binary label (Cell 5.B)."""
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    return list(skf.split(df, df["binary_label"]))


# --------------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------------
def make_weighted_sampler(
    binary_labels: np.ndarray, grades: np.ndarray | None = None, grade1_weight: float = 1.0
) -> WeightedRandomSampler:
    """Class-balancing sampler (weight N / n_class), Grade-1 weights additionally x grade1_weight.

    Draws len(labels) samples with replacement per epoch.
    """
    binary = np.asarray(binary_labels)
    counts = Counter(binary.tolist())
    total = len(binary)
    w = np.array([total / counts[b] for b in binary], dtype=np.float64)
    if grades is not None and grade1_weight != 1.0:
        w[np.asarray(grades) == 1] *= grade1_weight
    return WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)


# --------------------------------------------------------------------------------------------
# Augmentations (albumentations 1.4.x and 2.x)
# --------------------------------------------------------------------------------------------
def _fill(cls: type, value: float = 0, legacy: str = "value") -> dict:
    """Constant border fill keyword for ``cls`` across albumentations versions."""
    params = inspect.signature(cls.__init__).parameters
    return {"fill": value} if "fill" in params else {legacy: value}


def _normalize() -> list:
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


def build_train_transforms(kind: str) -> A.Compose:
    """Training augmentation.

    ``gate``: flips, rotation <= 10 deg, mild photometric jitter, light blur / sharpen. Coarse
    dropout, elastic deformation and mixup are deliberately absent (they erase 2-5 px
    microaneurysms). ``baseline``: the 512/384-px recipe, which still contains them.
    """
    if kind == "gate":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4, **_fill(A.Rotate)),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=1.0),
                A.RandomGamma(gamma_limit=(88, 112), p=1.0),
            ], p=0.5),
            A.GaussianBlur(blur_limit=(3, 3), p=0.12),
            A.Sharpen(alpha=(0.05, 0.20), lightness=(0.95, 1.05), p=0.20),
            *_normalize(),
        ])
    if kind == "baseline":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.5, **_fill(A.Rotate)),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=1.0),
                A.RandomGamma(gamma_limit=(82, 118), p=1.0),
            ], p=0.55),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.Sharpen(alpha=(0.1, 0.35), lightness=(0.9, 1.1), p=1.0),
            ], p=0.35),
            A.CoarseDropout(num_holes_range=(2, 10), hole_height_range=(16, 42),
                            hole_width_range=(16, 42), p=0.35,
                            **_fill(A.CoarseDropout, legacy="fill_value")),
            A.ElasticTransform(alpha=10, sigma=5, p=0.2, border_mode=cv2.BORDER_CONSTANT,
                               **_fill(A.ElasticTransform)),
            *_normalize(),
        ])
    raise ValueError(f"unknown augmentation kind {kind!r}")


def build_eval_transforms() -> A.Compose:
    """Validation transform: ImageNet normalisation only."""
    return A.Compose(_normalize())


def build_tta_transforms(rotate_limit: int = 8, brightness: bool = False) -> list[A.Compose]:
    """Test-time views: identity, h-flip, v-flip, one random rotation in [-limit, +limit] deg.

    ``brightness=True`` adds a fifth view with RandomBrightnessContrast(0.1, 0.1); this 5-view
    set was used only when caching the baseline test predictions (Cell 5.C).
    """
    views = [
        A.Compose(_normalize()),
        A.Compose([A.HorizontalFlip(p=1.0), *_normalize()]),
        A.Compose([A.VerticalFlip(p=1.0), *_normalize()]),
        A.Compose([A.Rotate(limit=rotate_limit, p=1.0, border_mode=cv2.BORDER_CONSTANT,
                            **_fill(A.Rotate)), *_normalize()]),
    ]
    if brightness:
        views.append(A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            *_normalize(),
        ]))
    return views


# --------------------------------------------------------------------------------------------
# Dataset / loaders
# --------------------------------------------------------------------------------------------
class RetinalDataset(Dataset):
    """Reads a fundus image, applies the Stage I front-end, then an albumentations transform.

    Unreadable files become a black image (which the front-end still processes), as in the
    original notebooks.
    """

    def __init__(
        self,
        paths,
        labels,
        transform: A.Compose | None,
        preprocess_fn: Callable[[np.ndarray, int], np.ndarray],
        img_size: int,
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.paths[i]))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = self.preprocess_fn(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(float(self.labels[i]), dtype=torch.float32)


def build_loaders(
    cfg: Stage1Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Training loader (weighted sampler, drop_last) and validation loader (ordered)."""
    pre = get_preprocess(cfg.preprocess)
    sampler = make_weighted_sampler(
        train_df["binary_label"].values,
        train_df["grade"].values if "grade" in train_df else None,
        cfg.grade1_weight,
    )
    train_ds = RetinalDataset(train_df["img_path"], train_df["binary_label"],
                              build_train_transforms(cfg.train_aug), pre, cfg.img_size)
    val_ds = RetinalDataset(val_df["img_path"], val_df["binary_label"],
                            build_eval_transforms(), pre, cfg.img_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def iter_grade_table(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Iterator[dict]:
    """Rows of the per-grade train/validation count table."""
    for g in sorted(set(train_df["grade"]) | set(val_df["grade"])):
        yield {"grade": int(g), "train": int((train_df["grade"] == g).sum()),
               "val": int((val_df["grade"] == g).sum())}
