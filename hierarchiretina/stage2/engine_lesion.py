"""Training loop for the HSMoE-AUNet lesion models (one model per lesion).

Reproduces "Cell 5 - Training Loop" of the notebooks: AdamW, mixed precision (CUDA),
gradient accumulation, gradient-norm clipping, per-epoch LR scheduling, best-checkpoint
selection and early stopping on the validation Dice at threshold 0.5 (mean of batch-level
Dice), and per-epoch resume from ``last_checkpoint.pth``.

Checkpoint formats (unchanged from the notebooks, so released files are interchangeable):
    best_model.pth      {'model_state', 'epoch' (0-based), 'val_dice'}
    last_checkpoint.pth {'epoch', 'model_state', 'optim_state', 'scaler_state',
                         'sched_state', 'best_dice', 'history', 'no_improve'}
"""
from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .hsmoe_aunet import LesionConfig, get_lesion_config
from .lesion_metrics import batch_metrics

LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_f1", "val_recall",
               "val_sensitivity", "val_specificity", "val_precision", "lr"]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cuda_backends() -> None:
    """cuDNN benchmark and TF32 matmul/convolution, as enabled in the notebooks."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_optimizer(model: nn.Module, cfg: LesionConfig) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: optim.Optimizer, cfg: LesionConfig):
    """ReduceLROnPlateau on val Dice (MA), cosine warm restarts (HE, CWS) or cosine (EX)."""
    kw = dict(cfg.scheduler_kwargs)
    if cfg.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kw)
    if cfg.scheduler == "cosine_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kw)
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kw)
    raise ValueError(f"unknown scheduler {cfg.scheduler}")


def _autocast(device, enabled):
    return torch.autocast(device_type=torch.device(device).type, enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg: LesionConfig, device,
                    epoch: int, use_amp: bool) -> float:
    """One pass over the training loader; returns the mean (un-scaled) loss per batch."""
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    n = len(loader)
    pbar = tqdm(enumerate(loader), total=n, desc=f"ep {epoch + 1} train", leave=False)
    for step, (imgs, masks) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            out, aux3, aux2, moe_l = model(imgs)
            loss = criterion(out, aux3, aux2, masks, moe_l, img_norm=imgs, epoch=epoch)
            loss = loss / cfg.accum_steps
        if cfg.nan_guard and not torch.isfinite(loss):
            # CWS: drop the batch and the gradients accumulated so far
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        if (step + 1) % cfg.accum_steps == 0 or (step + 1) == n:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * cfg.accum_steps
        pbar.set_postfix(loss=f"{total / (step + 1):.4f}")
    return total / n


@torch.no_grad()
def validate(model, loader, criterion, cfg: LesionConfig, device, epoch: int,
             use_amp: bool) -> tuple[float, dict]:
    """Mean validation loss and mean batch metrics at threshold 0.5."""
    model.eval()
    loss_sum, n_loss = 0.0, 0
    agg = defaultdict(float)
    for imgs, masks in tqdm(loader, desc=f"ep {epoch + 1} val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        with _autocast(device, use_amp):
            out, aux3, aux2, moe_l = model(imgs)
            loss = criterion(out, aux3, aux2, masks, moe_l, img_norm=imgs, epoch=epoch)
        if not cfg.nan_guard or torch.isfinite(loss):
            loss_sum += loss.item()
            n_loss += 1
        for k, v in batch_metrics(out.sigmoid(), masks).items():
            agg[k] += v
    n = max(len(loader), 1)
    return loss_sum / max(n_loss, 1), {k: v / n for k, v in agg.items()}


def fit(model, criterion, train_loader, val_loader, lesion: str, checkpoint_dir: Path,
        device, log_csv: Path | None = None, max_epochs: int | None = None,
        resume: bool = True) -> dict:
    """Train ``model`` with the recipe of ``lesion`` and keep the best checkpoint.

    Returns ``{'best_dice', 'best_epoch', 'epochs_run'}`` (best_epoch is 1-based).
    """
    cfg = get_lesion_config(lesion)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last, ckpt_best = checkpoint_dir / "last_checkpoint.pth", checkpoint_dir / "best_model.pth"
    log_csv = Path(log_csv) if log_csv else checkpoint_dir / "training_log.csv"
    epochs = max_epochs or cfg.max_epochs
    use_amp = torch.device(device).type == "cuda"

    model.to(device)
    criterion.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_dice, best_epoch, no_improve = 0, 0.0, 0, 0
    history = defaultdict(list)
    if resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        scaler.load_state_dict(ck["scaler_state"])
        scheduler.load_state_dict(ck["sched_state"])
        start_epoch = ck["epoch"] + 1
        best_dice = ck.get("best_dice", 0.0)
        best_epoch = ck.get("best_epoch", 0)
        history = defaultdict(list, ck.get("history", {}))
        no_improve = ck.get("no_improve", 0)
        print(f"Resumed at epoch {start_epoch + 1} (best val Dice {best_dice:.4f})")
    if start_epoch == 0 or not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg,
                                  device, epoch, use_amp)
        va_loss, vm = validate(model, val_loader, criterion, cfg, device, epoch, use_amp)
        val_dice = vm["dice"]
        if cfg.scheduler == "plateau":
            scheduler.step(val_dice)
        else:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        improved = val_dice > best_dice
        if improved:
            best_dice, best_epoch, no_improve = val_dice, epoch + 1, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_dice": val_dice}, ckpt_best)
        else:
            no_improve += 1
        for k, v in vm.items():
            history[k].append(v)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                    "sched_state": scheduler.state_dict(), "best_dice": best_dice,
                    "best_epoch": best_epoch, "history": dict(history),
                    "no_improve": no_improve}, ckpt_last)
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, tr_loss, va_loss, vm["dice"], vm["iou"], vm["f1"],
                                    vm["recall"], vm["sensitivity"], vm["specificity"],
                                    vm["precision"], lr])
        print(f"ep {epoch + 1:03d}/{epochs} | loss {tr_loss:.4f}/{va_loss:.4f} | "
              f"Dice {val_dice:.4f} IoU {vm['iou']:.4f} Rec {vm['recall']:.4f} "
              f"Prec {vm['precision']:.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s"
              f"{' *' if improved else ''}")
        if no_improve >= cfg.patience:
            print(f"Early stopping: {cfg.patience} epochs without improvement.")
            break
    return {"best_dice": best_dice, "best_epoch": best_epoch, "epochs_run": epoch + 1}
