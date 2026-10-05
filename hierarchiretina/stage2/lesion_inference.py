"""Probability prediction (with optional flip TTA), CWS post-processing and deployment.

Deployment (``segment_folder``) reproduces "Cell 8 - Real-Life Inference Pipeline" of each
lesion notebook, which wrote the masks consumed by Stage III (LG-DRG):

    MA, HE : retinal circle crop (grey threshold 20, 15x15 ellipse close x3 / open x2,
             minimum enclosing circle) -> stretch to 1024x1024 -> single forward pass.
    EX, CWS: retinal rectangle crop (same as training data) -> aspect-preserving letterbox
             to 1024x1024 -> mean of identity / horizontal-flip / vertical-flip probabilities.
    All    : probabilities resized back to the crop (bilinear), thresholded at the lesion's
             deployed threshold and pasted into a zero canvas of the original image size.
    CWS    : 5x5 elliptical closing and removal of 8-connected components smaller than
             40 * max(1, crop_area / 1024^2) pixels, at original resolution.

Output files: ``<output_dir>/<image stem><mask_suffix>.png`` (uint8, 0/255), with suffixes
``_mask`` (MA), ``_he_mask`` (HE), ``_ex_mask`` (EX) and ``_cws_mask`` (CWS) -- the names that
LG-DRG's ``_resolve(stem, folder, exts, suffix)`` looks up.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from .hsmoe_aunet import LesionConfig, get_lesion_config
from .lesion_data import TARGET_SIZE, build_eval_transform, detect_retinal_area


def _autocast(device: torch.device | str, enabled: bool):
    dev = torch.device(device).type
    return torch.autocast(device_type=dev, enabled=enabled and dev == "cuda")


def _logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def predict_probs(model, imgs: torch.Tensor, tta: bool = False, use_amp: bool = True,
                  fp32_sigmoid: bool = False) -> torch.Tensor:
    """Sigmoid probabilities (B,1,H,W); ``tta`` averages identity, H-flip and V-flip.

    As in the notebooks, the sigmoid (and TTA mean) is taken in the dtype of the model output
    (FP16 under CUDA autocast). ``fp32_sigmoid=True`` casts the logits to FP32 first (used by
    the wide MA threshold sweep script).
    """
    def sig(o):
        return torch.sigmoid(o.float()) if fp32_sigmoid else o.sigmoid()

    with _autocast(imgs.device, use_amp):
        o1 = _logits(model, imgs)
        if not tta:
            return sig(o1)
        o2 = torch.flip(_logits(model, torch.flip(imgs, dims=[3])), dims=[3])
        o3 = torch.flip(_logits(model, torch.flip(imgs, dims=[2])), dims=[2])
        return (sig(o1) + sig(o2) + sig(o3)) / 3.0


def cws_postprocess(prob_map: np.ndarray, prob_thresh: float = 0.5, close_ksize: int = 5,
                    min_area: int = 40) -> np.ndarray:
    """Binarise, 5x5 elliptical closing, drop 8-connected components < ``min_area`` px."""
    bin_mask = (prob_map > prob_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    keep = np.zeros(n_labels, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def binarize(prob_map: np.ndarray, threshold: float, cfg: LesionConfig) -> np.ndarray:
    """Binary mask at model resolution (with CWS post-processing where configured)."""
    if cfg.postprocess:
        return cws_postprocess(prob_map, prob_thresh=threshold, **cfg.postprocess)
    return (prob_map > threshold).astype(np.uint8)


# =============================================================================
# Deployment pre/post-processing
# =============================================================================
def detect_retinal_circle(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Square around the minimum enclosing circle of the retina (MA/HE deployment crop)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), rad = cv2.minEnclosingCircle(largest)
        cx, cy, rad = int(cx), int(cy), int(rad)
        if rad > min(h, w) * 0.1:
            return max(0, cx - rad), max(0, cy - rad), min(w, cx + rad), min(h, cy + rad)
    return 0, 0, img_bgr.shape[1], img_bgr.shape[0]


_EVAL_TF = None


def _eval_tf():
    global _EVAL_TF
    if _EVAL_TF is None:
        _EVAL_TF = build_eval_transform()
    return _EVAL_TF


def preprocess_for_deployment(img_bgr: np.ndarray, cfg: LesionConfig):
    """Return the normalised (1,3,1024,1024) tensor and the geometry needed to paste back."""
    oh, ow = img_bgr.shape[:2]
    if cfg.inference_crop == "circle":
        x1, y1, x2, y2 = detect_retinal_circle(img_bgr)
        crop = img_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        meta = dict(box=(x1, y1, x2, y2), crop_hw=(ch, cw), orig_hw=(oh, ow), letterbox=None)
    else:
        x1, y1, x2, y2 = detect_retinal_area(img_bgr)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, ow, oh
        crop = img_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            crop, (ch, cw), (x1, y1, x2, y2) = img_bgr, (oh, ow), (0, 0, ow, oh)
        scale = TARGET_SIZE / max(ch, cw)
        nh, nw = int(ch * scale), int(cw * scale)
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=resized.dtype)
        y_off, x_off = (TARGET_SIZE - nh) // 2, (TARGET_SIZE - nw) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        meta = dict(box=(x1, y1, x2, y2), crop_hw=(ch, cw), orig_hw=(oh, ow),
                    letterbox=(x_off, y_off, nh, nw))
    tensor = _eval_tf()(image=rgb)["image"].unsqueeze(0).float()
    return tensor, meta


def postprocess_to_full_mask(prob: np.ndarray, meta: dict, cfg: LesionConfig,
                             threshold: float) -> np.ndarray:
    """Map a 1024x1024 probability map back to the original image as a 0/255 mask."""
    x1, y1, x2, y2 = meta["box"]
    ch, cw = meta["crop_hw"]
    oh, ow = meta["orig_hw"]
    if meta["letterbox"] is not None:
        x_off, y_off, nh, nw = meta["letterbox"]
        prob = prob[y_off:y_off + nh, x_off:x_off + nw]
    prob_crop = cv2.resize(prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    if cfg.postprocess:
        scale_factor = max(1.0, (ch * cw) / (TARGET_SIZE * TARGET_SIZE))
        pp = dict(cfg.postprocess)
        pp["min_area"] = int(pp["min_area"] * scale_factor)
        binary = cws_postprocess(prob_crop, prob_thresh=threshold, **pp)
    else:
        binary = (prob_crop > threshold).astype(np.uint8)
    full = np.zeros((oh, ow), dtype=np.uint8)
    full[y1:y2, x1:x2] = binary * 255
    return full


@torch.no_grad()
def segment_folder(model, lesion: str, input_dir: Path, output_dir: Path,
                   device: torch.device | str = "cuda", threshold: float | None = None,
                   batch_size: int = 4, use_amp: bool = True, overwrite: bool = True) -> dict:
    """Write one binary mask per image of ``input_dir`` (deployment pipeline of ``lesion``).

    Returns a summary dict with the number of saved masks and the files that failed.
    """
    cfg = get_lesion_config(lesion)
    thr = cfg.threshold if threshold is None else threshold
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in cfg.inference_exts)
    if not overwrite:
        files = [p for p in files if not (output_dir / f"{p.stem}{cfg.mask_suffix}.png").exists()]
    model.eval()
    saved, failed = 0, []
    for i in tqdm(range(0, len(files), batch_size), desc=f"{cfg.name} masks"):
        tensors, metas, paths = [], [], []
        for f in files[i:i + batch_size]:
            img = cv2.imread(str(f))
            if img is None:
                failed.append(f.name)
                continue
            t, m = preprocess_for_deployment(img, cfg)
            tensors.append(t), metas.append(m), paths.append(f)
        if not tensors:
            continue
        probs = predict_probs(model, torch.cat(tensors).to(device), tta=cfg.tta,
                              use_amp=use_amp)
        for j, (f, m) in enumerate(zip(paths, metas)):
            prob = probs[j, 0].cpu().float().numpy()
            mask = postprocess_to_full_mask(prob, m, cfg, thr)
            cv2.imwrite(str(output_dir / f"{f.stem}{cfg.mask_suffix}.png"), mask)
            saved += 1
    return {"lesion": cfg.name, "threshold": thr, "n_images": len(files), "saved": saved,
            "failed": failed, "output_dir": str(output_dir)}
