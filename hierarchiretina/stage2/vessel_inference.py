"""Real-world vessel-mask inference (FOV crop, flip TTA, FOV masking, post-processing).

For every input image: FOV mask -> CLAHE + green enhancement -> square FOV crop -> 512 x 512 ->
TTA probability -> FOV-masked at model resolution -> resized back into the crop box of a
full-resolution canvas -> binarised at the deployed threshold, restricted to the full-resolution
FOV, opened (radius 1), components < 30 px removed -> saved as a 0/255 PNG.

Output naming follows what LG-DRG (Stage III) expects for the vessel channel: the mask of image
``<stem>.<ext>`` is written as ``<stem>_mask.png`` into a dedicated vessel-mask folder
(LG-DRG resolves ``<mask_dir_vessel>/<stem>_mask.*``).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .vessel_data import (IMAGENET_MEAN, IMAGENET_STD, VALID_EXT, apply_clahe, enhance_green,
                          fov_crop_box, generate_fov_mask)
from .vessel_metrics import MIN_COMPONENT_AREA, MORPH_OPEN_RADIUS, postprocess_mask, \
    predict_tta_batch

VESSEL_MASK_SUFFIX = "_mask"   # LG-DRG: mask_suffix["vessel"] == "_mask"
DEPLOYED_THRESHOLD = 0.46      # Youden-J threshold of the best validation epoch (53)


def preprocess_for_inference(img_bgr: np.ndarray, img_size: int = 512):
    """Return the normalised input tensor [1,3,S,S] and the metadata needed to map back.

    Note: the original notebook computed the FOV here by passing the *BGR* image to
    ``generate_fov_mask`` with its default ``is_rgb=True`` (i.e. red and blue weights swapped in
    the grey conversion). This is kept unchanged so that the released masks are reproduced.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    fov_full = generate_fov_mask(img_bgr, is_rgb=True).astype(np.float32)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = enhance_green(apply_clahe(img_rgb))

    x1, y1, x2, y2 = fov_crop_box(fov_full, margin=0.01)
    crop, fov_crop = img_rgb[y1:y2, x1:x2], fov_full[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    img_r = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    fov_r = cv2.resize(fov_crop, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    arr = img_r.astype(np.float32) / 255.0
    for c in range(3):
        arr[..., c] = (arr[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    meta = dict(orig_h=orig_h, orig_w=orig_w, ch=ch, cw=cw, x1=x1, y1=y1, x2=x2, y2=y2,
                fov_r=(fov_r > 0.5).astype(np.float32))
    return t, meta


def to_original_resolution(prob: np.ndarray, meta: dict) -> np.ndarray:
    """FOV-mask the model-resolution probability map and paste it back at full resolution."""
    prob = prob * meta["fov_r"]
    pc = cv2.resize(prob, (meta["cw"], meta["ch"]), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((meta["orig_h"], meta["orig_w"]), dtype=np.float32)
    full[meta["y1"]:meta["y2"], meta["x1"]:meta["x2"]] = pc
    return full


@torch.no_grad()
def predict_vessel_mask(model, img_bgr: np.ndarray, thr: float, device, img_size: int = 512,
                        use_tta: bool = True):
    """Binary (0/1) vessel mask and the probability map, both at the input image's resolution."""
    t, meta = preprocess_for_inference(img_bgr, img_size)
    prob = predict_tta_batch(model, t.to(device), use_tta)[0, 0].float().cpu().numpy()
    full_prob = to_original_resolution(prob, meta)
    full_fov = generate_fov_mask(img_bgr, is_rgb=True).astype(np.float32)  # see note above
    mask = postprocess_mask(full_prob, thr, MORPH_OPEN_RADIUS, MIN_COMPONENT_AREA, fov=full_fov)
    return mask, full_prob


def generate_vessel_masks(model, input_dir, output_dir, thr: float, device,
                          img_size: int = 512, use_tta: bool = True,
                          suffix: str = VESSEL_MASK_SUFFIX, progress=None) -> dict:
    """Write ``<stem><suffix>.png`` (0/255) for every image in ``input_dir``.

    ``progress`` may be a wrapper such as ``tqdm``. Returns ``{"saved": n, "failed": [...]}``.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXT)
    it = progress(files) if progress else files
    saved, failed = 0, []
    for fp in it:
        img = cv2.imread(str(fp))
        if img is None:
            failed.append(fp.name)
            continue
        try:
            mask, _ = predict_vessel_mask(model, img, thr, device, img_size, use_tta)
            cv2.imwrite(str(output_dir / f"{fp.stem}{suffix}.png"), (mask * 255).astype(np.uint8))
            saved += 1
        except Exception as e:  # keep going over large folders; failures are reported
            failed.append(f"{fp.name} ({e})")
    return {"n_images": len(files), "saved": saved, "failed": failed}
