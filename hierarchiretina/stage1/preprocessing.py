"""Fundus preprocessing for Stage I.

Two front-ends exist:

* ``preprocess_gate`` (deployed 768-px gate): retina bounding box -> true square crop padded with
  black (never stretched) -> one resize -> mild green-channel CLAHE -> Ben Graham background
  subtraction with a resolution-scaled sigma -> LAB-lightness CLAHE -> mild unsharp masking.
* ``preprocess_letterbox`` (512/384-px baselines): tight square crop (falls back to the full
  frame when the disc covers < 75 % of it) -> aspect-preserving resize + letterbox -> Ben Graham
  (sigma 35) -> LAB CLAHE -> unsharp masking.

All functions take and return uint8 BGR images (OpenCV convention).
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def detect_retinal_bounds(
    img_bgr: np.ndarray,
    thresh_frac: float = 0.04,
    margin: int = 8,
    min_area_frac: float | None = 0.2,
) -> BBox | None:
    """Bounding box (x1, y1, x2, y2) of the retinal disc, or None.

    The disc is the largest connected component of a global intensity threshold after a 5x5
    morphological close + open. ``min_area_frac`` rejects boxes smaller than that fraction of the
    frame (used by the gate; the baselines pass None).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, mask = cv2.threshold(gray, int(thresh_frac * 255), 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x = stats[best, cv2.CC_STAT_LEFT]
    y = stats[best, cv2.CC_STAT_TOP]
    ww = stats[best, cv2.CC_STAT_WIDTH]
    hh = stats[best, cv2.CC_STAT_HEIGHT]
    if min_area_frac is not None and ww * hh < min_area_frac * w * h:
        return None
    return (max(0, x - margin), max(0, y - margin),
            min(w, x + ww + margin), min(h, y + hh + margin))


# --------------------------------------------------------------------------------------------
# Gate (768 px) front-end
# --------------------------------------------------------------------------------------------
def square_retinal_crop(img_bgr: np.ndarray) -> np.ndarray:
    """Square crop centred on the disc, side = max(box w, box h), black-padded past the edges.

    Falls back to the centred min(h, w) square when no disc is found.
    """
    h, w = img_bgr.shape[:2]
    b = detect_retinal_bounds(img_bgr)
    if b is None:
        side = min(h, w)
        cy, cx = h // 2, w // 2
    else:
        x1, y1, x2, y2 = b
        side = max(x2 - x1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

    half = side // 2
    nx1, ny1 = cx - half, cy - half
    nx2, ny2 = nx1 + side, ny1 + side
    pl, pt = max(0, -nx1), max(0, -ny1)
    pr, pb = max(0, nx2 - w), max(0, ny2 - h)

    crop = img_bgr[max(0, ny1):min(h, ny2), max(0, nx1):min(w, nx2)]
    if pl or pr or pt or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return crop


def green_channel_boost(img_bgr: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Blend a CLAHE-equalised green channel (clip 3.0, 8x8) into the green channel."""
    b, g, r = cv2.split(img_bgr)
    g_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g_new = cv2.addWeighted(g, 1 - strength, g_eq, strength, 0)
    return cv2.merge([b, g_new, r])


def ben_graham(
    img_bgr: np.ndarray, sigma: float | None = None, circle_frac: float = 0.49
) -> np.ndarray:
    """Ben Graham local-average subtraction, 4*(I - G_sigma*I) + 128, outside a circle set to 128.

    ``sigma=None`` scales with resolution: max(10, int(0.045 * height)), i.e. 34 at 768 px.
    """
    if sigma is None:
        sigma = max(10, int(img_bgr.shape[0] * 0.045))
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    out = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    h, w = out.shape[:2]
    mask = np.zeros_like(out)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * circle_frac), (1, 1, 1), -1)
    return (out * mask + 128 * (1 - mask)).astype(np.uint8)


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.2, grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, bb]), cv2.COLOR_LAB2BGR)


def mild_sharpen(
    img_bgr: np.ndarray,
    strength: float = 0.30,
    ksize: tuple[int, int] = (0, 0),
    sigma: float = 1.2,
) -> np.ndarray:
    """Unsharp mask: (1 + s) * I - s * G(I)."""
    blurred = cv2.GaussianBlur(img_bgr, ksize, sigma)
    return np.clip(
        cv2.addWeighted(img_bgr, 1 + strength, blurred, -strength, 0), 0, 255
    ).astype(np.uint8)


def preprocess_gate(img_bgr: np.ndarray, target_size: int = 768) -> np.ndarray:
    """Deployed Stage I front-end (square crop, single resize, lesion enhancement)."""
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    img = square_retinal_crop(img_bgr)
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > target_size else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (target_size, target_size), interpolation=interp)
    img = green_channel_boost(img, strength=0.25)
    img = ben_graham(img)
    img = apply_clahe(img, clip=2.2, grid=8)
    img = mild_sharpen(img, strength=0.30)
    return img


# --------------------------------------------------------------------------------------------
# Baseline (512 / 384 px) front-end
# --------------------------------------------------------------------------------------------
def tight_square_retinal_crop(img_bgr: np.ndarray, min_content_ratio: float = 0.75) -> np.ndarray:
    """Square crop clamped to the image; returns the full frame if the disc box is too small."""
    h, w = img_bgr.shape[:2]
    bounds = detect_retinal_bounds(img_bgr, min_area_frac=None)
    if bounds is None:
        return img_bgr
    x1, y1, x2, y2 = bounds
    if ((x2 - x1) * (y2 - y1)) / (h * w) < min_content_ratio:
        return img_bgr

    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1s = max(0, cx - side // 2)
    y1s = max(0, cy - side // 2)
    x2s = min(w, x1s + side)
    y2s = min(h, y1s + side)
    if x2s - x1s < side:
        x1s = max(0, x2s - side)
    if y2s - y1s < side:
        y1s = max(0, y2s - side)
    return img_bgr[y1s:y2s, x1s:x2s]


def preprocess_letterbox(img_bgr: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Baseline front-end used by the 512-px ConvNeXt and 384-px SwinV2 models."""
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    img = tight_square_retinal_crop(img_bgr)
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    ph, pw = target_size - nh, target_size - nw
    img = cv2.copyMakeBorder(img, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = ben_graham(img, sigma=35, circle_frac=0.48)
    img = apply_clahe(img, clip=2.2, grid=8)
    img = mild_sharpen(img, strength=0.28, ksize=(5, 5), sigma=1.0)
    return img


PREPROCESSORS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "square_crop": preprocess_gate,
    "letterbox": preprocess_letterbox,
}


def get_preprocess(name: str) -> Callable[[np.ndarray, int], np.ndarray]:
    """Preprocessing function ``f(img_bgr, target_size)`` by preset name."""
    return PREPROCESSORS[name]
