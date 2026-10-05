"""Stage I training: AMP, gradient accumulation, EMA, cosine warm-up schedule, early stopping on
validation AUC, per-epoch checkpoints with clean resume, and the 5-fold driver of the 512-px
baseline.

Checkpoint files keep the original names and keys (``best_auc_model.pth``/``latest_model.pth``
with ``model, optimizer, scheduler, scaler, ema_shadow, epoch, best_auc, patience_counter,
threshold`` ...), so checkpoints from the original notebooks can be evaluated or resumed.
The best checkpoint stores the EMA weights in ``model`` (EMA is applied before saving).
"""
from __future__ import annotations

import gc
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Stage1Config
from .evaluate import compute_metrics, grade_accuracy, threshold_rule

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch; set the cudnn flags used by the notebooks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMA:
    """Exponential moving average of trainable parameters (buffers are not averaged)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()
                       if p.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def cosine_warmup(
    optimizer: torch.optim.Optimizer, warmup: int, total: int, floor: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-epoch LR factor: linear (e+1)/warmup for e < warmup, then half-cosine to ``floor``."""
    def f(e: int) -> float:
        if e < warmup:
            return (e + 1) / max(1, warmup)
        prog = (e - warmup) / max(1, total - warmup)
        return max(floor, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Mixup on half of the batches (Beta(alpha, alpha)); identity when alpha <= 0."""
    if alpha <= 0 or np.random.rand() > 0.5:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# One epoch
# --------------------------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    cfg: Stage1Config,
    device: torch.device,
    epoch: int,
    progress: bool = True,
) -> float:
    """One pass over ``loader``; returns the mean (un-accumulated) training loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if cfg.reseed_each_epoch:
        # Deterministic sampler order per epoch (the gate notebook did this to support resume).
        seed_everything(cfg.seed + epoch, cfg.deterministic)
    use_amp = cfg.amp and device.type == "cuda"
    n_steps = len(loader)
    total = 0.0
    it = loader
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(loader, desc=f"train ep{epoch}", leave=False)
    for step, (x, y) in enumerate(it):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        if cfg.channels_last:
            x = x.to(memory_format=torch.channels_last)
        x, ya, yb, lam = mixup_batch(x, y, cfg.mixup_alpha)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            if lam == 1.0:
                loss = criterion(logits, ya)
            else:
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss = loss / cfg.grad_accum
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        if (step + 1) % cfg.grad_accum == 0 or (step + 1) == n_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                           error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
        total += loss.item() * cfg.grad_accum
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return total / n_steps


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: Stage1Config,
    device: torch.device,
    ema: EMA | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Validation pass (with EMA weights if ``ema`` is given). Returns loss, labels, probs."""
    if ema is not None:
        ema.apply_shadow(model)
    model.eval()
    use_amp = cfg.amp and device.type == "cuda"
    total, labels, probs = 0.0, [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        if cfg.channels_last:
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total += loss.item()
        probs.extend(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        labels.extend(y.cpu().numpy().ravel())
    if ema is not None:
        ema.restore(model)
    return total / len(loader), np.array(labels), np.array(probs)


# --------------------------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------------------------
def fit(
    cfg: Stage1Config,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    ckpt_dir: str | Path,
    device: torch.device | None = None,
    val_grades: np.ndarray | None = None,
    epochs: int | None = None,
    resume: bool = True,
    best_name: str = "best_auc_model.pth",
    latest_name: str = "latest_model.pth",
    history_name: str = "training_history.csv",
    extra_meta: dict | None = None,
    progress: bool = True,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Train with early stopping on validation AUC; resumable at epoch granularity.

    Per epoch: train -> validate with EMA -> scheduler step -> threshold from the preset rule on
    the validation probabilities -> save best (EMA weights, when AUC improves) and latest.
    ``val_grades`` enables the Grade-1 routing column. Returns the history table.
    """
    device = device or get_device()
    epochs = epochs or cfg.epochs
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = ckpt_dir / best_name, ckpt_dir / latest_name
    hist_path = ckpt_dir / history_name

    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.get_param_groups(cfg.lr, cfg.lr_backbone),
                                  weight_decay=cfg.weight_decay)
    scheduler = cosine_warmup(optimizer, cfg.warmup_epochs, epochs, cfg.lr_min / cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda",
                                  init_scale=1024, growth_interval=200)
    ema = EMA(model, cfg.ema_decay)
    pick_threshold = threshold_rule(cfg)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    start, best_auc, patience = 0, 0.0, 0
    history: dict[str, list] = defaultdict(list)
    if resume and latest_path.exists():
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if "ema_shadow" in ck:
            ema.shadow = {k: v.to(device).float() for k, v in ck["ema_shadow"].items()}
        start = ck["epoch"] + 1
        best_auc = ck.get("best_auc", 0.0)
        patience = ck.get("patience_counter", 0)
        del ck
        if hist_path.exists():
            for col, vals in pd.read_csv(hist_path).items():
                history[col] = vals.tolist()
        log(f"Resumed after epoch {start - 1} (best AUC {best_auc:.4f}, patience {patience})")
        if patience >= cfg.patience:
            log("Early-stopping criterion already met; nothing to do.")
            return pd.DataFrame(history)

    for epoch in range(start, epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, ema, cfg,
                                  device, epoch, progress)
        vl_loss, vl_lab, vl_pr = validate(model, val_loader, criterion, cfg, device, ema)
        scheduler.step()
        thr = pick_threshold(vl_lab, vl_pr)
        m = compute_metrics(vl_lab, vl_pr, thr if cfg.log_metrics_at_threshold else 0.5)
        g1 = grade_accuracy(vl_pr, val_grades, thr) if val_grades is not None else float("nan")
        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss, "val_auc": m["auc"],
               "val_f1": m["f1"], "val_sens": m["sensitivity"], "val_spec": m["specificity"],
               "val_acc": m["accuracy"], "grade1_acc": g1, "threshold": thr,
               "lr": optimizer.param_groups[1]["lr"], "minutes": (time.time() - t0) / 60}
        for k, v in row.items():
            history[k].append(v)
        log(f"ep{epoch:03d} | train {tr_loss:.4f} val {vl_loss:.4f} | AUC {m['auc']:.4f} "
            f"sens {m['sensitivity']:.4f} spec {m['specificity']:.4f} | G1 {g1:.3f} | "
            f"thr {thr:.4f} | {row['minutes']:.1f} min")

        state = {"epoch": epoch, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "threshold": thr, **(extra_meta or {})}
        if m["auc"] > best_auc:
            best_auc, patience = m["auc"], 0
            ema.apply_shadow(model)
            _atomic_save({**state, "model": model.state_dict(), "auc": m["auc"], "f1": m["f1"],
                          "grade1_acc": g1, "best_auc": best_auc, "patience_counter": patience,
                          "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()}}, best_path)
            ema.restore(model)
            log(f"   new best AUC {best_auc:.4f} (threshold {thr:.4f})")
        else:
            patience += 1
        _atomic_save({**state, "model": model.state_dict(), "best_auc": best_auc,
                      "patience_counter": patience,
                      "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()}}, latest_path)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        if patience >= cfg.patience:
            log(f"Early stop at epoch {epoch} (patience {cfg.patience})")
            break
    return pd.DataFrame(history)


# --------------------------------------------------------------------------------------------
# 5-fold driver (512-px baseline, Cell 5.B)
# --------------------------------------------------------------------------------------------
def fit_kfold(
    cfg: Stage1Config,
    df: pd.DataFrame,
    fold_dir: str | Path,
    device: torch.device | None = None,
    pretrained: bool = True,
    num_workers: int = 0,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Train ``cfg.n_folds`` models on StratifiedKFold splits; resumable per fold and epoch.

    After each fold the best (EMA) model predicts its held-out fold with 4-view TTA; the
    out-of-fold table ``oof_predictions.csv`` (img_path, true_label, oof_prob, fold) is the
    calibration set of the hybrid ensemble. Returns that table.
    """
    from .data import build_loaders, build_tta_transforms, kfold_splits
    from .evaluate import tta_predict_paths
    from .losses import build_criterion
    from .model import build_model, load_eval_weights
    from .preprocessing import get_preprocess

    device = device or get_device()
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    oof_path = fold_dir / "oof_predictions.csv"
    oof = (pd.read_csv(oof_path) if oof_path.exists()
           else pd.DataFrame(columns=["img_path", "true_label", "oof_prob", "fold"]))

    for k, (tr_idx, va_idx) in enumerate(kfold_splits(df, cfg), start=1):
        done = fold_dir / f"fold_{k}_DONE.flag"
        if done.exists():
            log(f"fold {k}: done, skipping")
            continue
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)
        train_loader, val_loader = build_loaders(cfg, tr_df, va_df, num_workers=num_workers)
        model = build_model(cfg, pretrained=pretrained)
        criterion = build_criterion(cfg, tr_df["binary_label"].values)
        fit(cfg, model, train_loader, val_loader, criterion, fold_dir, device,
            best_name=f"fold_{k}_best.pth", latest_name=f"fold_{k}_latest.pth",
            history_name=f"fold_{k}_history.csv", extra_meta={"fold": k}, log=log)

        load_eval_weights(model, fold_dir / f"fold_{k}_best.pth")
        model.to(device)
        probs, _ = tta_predict_paths(
            model, va_df["img_path"].tolist(), get_preprocess(cfg.preprocess), cfg.img_size,
            build_tta_transforms(cfg.tta_rotate_limit), device, cfg.amp, cfg.channels_last)
        oof = oof[oof["fold"] != k]
        oof = pd.concat([oof, pd.DataFrame({"img_path": va_df["img_path"].values,
                                            "true_label": va_df["binary_label"].values,
                                            "oof_prob": probs, "fold": k})], ignore_index=True)
        oof.to_csv(oof_path, index=False)
        done.touch()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return oof
