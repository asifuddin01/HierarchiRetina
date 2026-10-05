"""Stage III (LG-DRG) preprocessing.

Pipeline, identical at training and test time:

1. ``crop_fundus_circle``: crop the RGB image to the bounding box of pixels whose channel
   mean exceeds 7 (removes the black border). The same box is applied to every mask.
2. ``clahe_lab``: CLAHE (clip 2.5, 8x8 tiles) on the L channel of LAB.
3. ``build_aug``: ``LongestMaxSize(512)`` + zero ``PadIfNeeded`` to 512x512, ImageNet
   normalisation of the RGB channels. Training adds joint geometric augmentation (image and
   all five masks) and photometric augmentation (image only).

The original notebooks were written against albumentations 1.x. albumentations 2.x silently
ignores the 1.x ``CoarseDropout`` arguments, so ``build_aug`` maps them explicitly. The test
transform is identical in both versions.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
_ALB_MAJOR = int(A.__version__.split(".")[0])


def crop_fundus_circle(img: np.ndarray, tol: int = 7):
    """Crop the black border around the fundus.

    Returns ``(cropped, box)`` with ``box = (r0, r1, c0, c1)`` (inclusive), or ``(img, None)``
    when no pixel exceeds ``tol``.
    """
    gray = img.mean(2) if img.ndim == 3 else img
    mask = gray > tol
    if mask.sum() == 0:
        return img, None
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1], (r0, r1, c0, c1)


def clahe_lab(img: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    """CLAHE on the LAB lightness channel of an RGB uint8 image.

    Named ``clahe_green`` in the original notebooks, but it operates on LAB-L, not on green.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


clahe_green = clahe_lab  # name used in the original notebooks


def _coarse_dropout(img_size: int):
    """Six holes of exactly (img_size/16)^2 px on the image only, p=0.25 (albumentations 1.x
    semantics: ``min_holes``/``min_height`` default to the max values)."""
    h = img_size // 16
    if _ALB_MAJOR < 2:
        return A.CoarseDropout(max_holes=6, max_height=h, max_width=h, p=0.25)
    return A.CoarseDropout(num_holes_range=(6, 6), hole_height_range=(h, h),
                           hole_width_range=(h, h), fill=0, fill_mask=None, p=0.25)


def build_aug(img_size: int = 512, n_masks: int = 5, train: bool = True) -> A.Compose:
    """Albumentations pipeline for the RGB image plus ``n_masks`` binary masks.

    Masks are passed as ``mask0..mask{n-1}`` additional targets so that every geometric
    transform is applied jointly; photometric transforms and coarse dropout touch the image only.
    """
    extra = {f"mask{i}": "mask" for i in range(n_masks)}
    head = [A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)]
    tail = [A.Normalize(MEAN, STD), ToTensorV2()]
    if not train:
        return A.Compose(head + tail, additional_targets=extra)
    aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180,
                           border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.OneOf([A.GaussianBlur(3), A.MotionBlur(3)], p=0.2),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        A.HueSaturationValue(8, 12, 8, p=0.3),
        _coarse_dropout(img_size),
    ]
    return A.Compose(head + aug + tail, additional_targets=extra)


def resolve_file(stem: str, folder: str | Path, exts: tuple[str, ...], suffix: str = ""):
    """Return ``folder/{stem}{suffix}{ext}`` for the first existing extension, else None."""
    folder = Path(folder)
    name = f"{Path(str(stem)).stem}{suffix}"
    p = folder / name
    if p.exists():
        return str(p)
    for ext in exts:
        c = folder / f"{name}{ext}"
        if c.exists():
            return str(c)
    hits = list(folder.glob(f"{name}.*"))
    return str(hits[0]) if hits else None


def load_image_and_masks(img_path: str, mask_paths: list[str | None], use_clahe: bool = True):
    """Read one fundus image and its masks, apply crop (+CLAHE) and crop the masks with the
    same box. Missing or unreadable masks become all-zero maps of the cropped size.

    Returns ``(rgb_uint8, [mask_float32 in {0,1}, ...])``, or ``(None, None)`` if the image
    cannot be read.
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None, None
    im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im, box = crop_fundus_circle(im)
    if use_clahe:
        im = clahe_lab(im)
    h, w = im.shape[:2]
    masks = []
    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if isinstance(mp, str) else None
        if m is None:
            m = np.zeros((h, w), np.uint8)
        elif box is not None:
            r0, r1, c0, c1 = box
            m = m[r0:r1 + 1, c0:c1 + 1]
        masks.append((m > 127).astype(np.float32))
    return im, masks
