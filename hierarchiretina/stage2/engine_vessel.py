"""Training engine for SwinHRUNetPP (FOV-aware).

Final configuration (paper, Stage II training): AdamW, decoder LR 2e-4, encoder LR 0.1x, encoder
frozen for the first 3 epochs, weight decay 1e-4, 5 linear warm-up epochs then cosine decay to
1e-6, batch 4 with gradient accumulation 4, gradient clipping 1.0, mixed precision on CUDA, EMA
0.999 (updated once per epoch), up to 200 epochs. Validation uses the live model; the Youden-J
threshold of every epoch is stored in the history and the threshold of the best-Dice epoch is
the deployed threshold (0.46 for the released checkpoint, best epoch 53).
"""
from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .vessel_metrics import compute_metrics_fov, fov_auc, youden_threshold

TRAIN_CONFIG = {
    "epochs": 200,
    "batch_size": 4,
    "val_batch_size": 4,
    "grad_accum": 4,
    "lr": 2e-4,
    "encoder_lr_ratio": 0.1,
    "freeze_encoder_epochs": 3,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_epochs": 5,
    "ema_decay": 0.999,
    "default_threshold": 0.50,
    # Early-stopping patience as set in the original notebook (see README / report).
    "patience": 1,
}


class EMA:
    """Exponential moving average of model weights (float tensors averaged, others copied)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def strip_compile_prefix(sd: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix that ``torch.compile`` adds to state-dict keys."""
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def build_optimizer(model, cfg=TRAIN_CONFIG) -> torch.optim.Optimizer:
    """AdamW with a reduced learning rate for the Swin encoder."""
    enc = set(model.swin.parameters())
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p in enc],
         "lr": cfg["lr"] * cfg["encoder_lr_ratio"], "name": "encoder"},
        {"params": [p for p in model.parameters() if p not in enc],
         "lr": cfg["lr"], "name": "decoder"},
    ], weight_decay=cfg["weight_decay"])


def build_scheduler(opt, epochs, warmup, min_lr, base_lr):
    """Per-epoch linear warm-up followed by cosine decay with a floor of ``min_lr / base_lr``."""
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(epochs - warmup - 1, 1)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def set_encoder_grad(model, requires_grad: bool) -> None:
    for p in model.swin.parameters():
        p.requires_grad_(requires_grad)


def train_epoch(model, loader, optimizer, scaler, loss_fn, accum, device, grad_clip=1.0,
                use_amp=False) -> float:
    """One epoch with gradient accumulation; returns the mean (un-scaled) training loss.

    As in the original notebook, gradients of a trailing incomplete accumulation group are
    discarded by the ``zero_grad`` at the start of the next epoch.
    """
    model.train()
    total = 0.0
    optimizer.zero_grad()
    for i, (imgs, msks, fovs) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        fovs = fovs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            loss = loss_fn(out, msks, fovs) / accum
        scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total += loss.item() * accum
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, default_thr=0.5, use_amp=False) -> dict:
    """Validation loss and FOV-aware metrics at the Youden-J optimal threshold."""
    model.eval()
    vloss = 0.0
    all_p, all_t, all_f = [], [], []
    for imgs, msks, fovs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(imgs)
            main = out[0] if isinstance(out, tuple) else out
            vloss += loss_fn(main, msks.to(device), fovs.to(device)).item()
        all_p.append(torch.sigmoid(main).float().cpu().numpy())
        all_t.append(msks.numpy())
        all_f.append(fovs.numpy())
    probs = np.concatenate(all_p).ravel()
    tgts = np.concatenate(all_t).ravel()
    fovs = np.concatenate(all_f).ravel()
    thr = youden_threshold(probs, tgts, fovs, default=default_thr)
    m = compute_metrics_fov(probs, tgts, fovs, thr)
    m["auc"] = fov_auc(probs, tgts, fovs)
    m["val_loss"] = vloss / len(loader)
    m["opt_thr"] = thr
    return m


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, ema, best, history):
    """Checkpoint with the same keys as the original notebook (model/ema/opt/sched/...)."""
    state = dict(epoch=epoch, model=strip_compile_prefix(model.state_dict()),
                 opt=optimizer.state_dict(), sched=scheduler.state_dict(),
                 scaler=scaler.state_dict() if scaler else None,
                 history=dict(history), best=best)
    if ema is not None:
        state["ema"] = strip_compile_prefix(ema.model.state_dict())
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None):
    """Restore a training checkpoint; returns ``(epoch, best, history)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_compile_prefix(ckpt["model"]))
    if ema is not None and ckpt.get("ema"):
        ema.model.load_state_dict(strip_compile_prefix(ckpt["ema"]))
    if optimizer is not None and "opt" in ckpt:
        optimizer.load_state_dict(ckpt["opt"])
    if scheduler is not None and "sched" in ckpt:
        scheduler.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best", {"dice": 0.0, "epoch": -1}), ckpt.get("history")


def load_best_model_and_threshold(ckpt_path, model, fallback_thr: float = 0.5):
    """Load the deployed weights and threshold from ``best_dice.pth``.

    The EMA weights are used when present (as in every evaluation/inference cell of the original
    notebook), and the threshold is the Youden-J threshold stored for the best epoch.
    Returns ``(model, threshold, info)``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(strip_compile_prefix(ckpt[key]), strict=True)
    model.eval()
    best_epoch = ckpt.get("best", {}).get("epoch", -1)
    hist = ckpt.get("history", {}) or {}
    if best_epoch >= 0 and "opt_thr" in hist and best_epoch < len(hist["opt_thr"]):
        thr, src = float(hist["opt_thr"][best_epoch]), f"best epoch {best_epoch + 1}"
    elif "opt_thr" in hist and hist.get("dice"):
        idx = int(np.argmax(hist["dice"]))
        thr, src = float(hist["opt_thr"][idx]), f"argmax dice epoch {idx + 1}"
    else:
        thr, src = float(fallback_thr), "fallback"
    info = {"weights": key, "best_epoch": best_epoch + 1,
            "best_val_dice": ckpt.get("best", {}).get("dice", float("nan")),
            "threshold": thr, "threshold_source": src}
    return model, thr, info


def fit(model, train_loader, val_loader, loss_fn, device, ckpt_dir, cfg=TRAIN_CONFIG,
        resume: bool = True, log=print):
    """Full training loop. Writes ``best_dice.pth`` and ``latest.pth`` to ``ckpt_dir``.

    Resumption restarts at the epoch after the one stored in ``latest.pth``.
    Returns ``(history, best)``.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ck, latest_ck = ckpt_dir / "best_dice.pth", ckpt_dir / "latest.pth"
    use_amp = device.type == "cuda"

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg["epochs"], cfg["warmup_epochs"], cfg["min_lr"],
                                cfg["lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = EMA(model, cfg["ema_decay"])

    history, best, start_epoch = defaultdict(list), {"dice": 0.0, "epoch": -1}, 0
    if resume and latest_ck.exists():
        ep, best, h = load_checkpoint(latest_ck, model, optimizer, scheduler, scaler, ema)
        history = defaultdict(list, h or {})
        start_epoch = ep + 1
        log(f"Resumed at epoch {start_epoch + 1}, best Dice {best['dice']:.4f}")

    t0, patience_cnt = time.time(), 0
    for epoch in range(start_epoch, cfg["epochs"]):
        if epoch < cfg["freeze_encoder_epochs"]:
            set_encoder_grad(model, False)
        elif epoch == cfg["freeze_encoder_epochs"]:
            set_encoder_grad(model, True)

        tr_loss = train_epoch(model, train_loader, optimizer, scaler, loss_fn, cfg["grad_accum"],
                              device, cfg["grad_clip"], use_amp)
        ema.update(model)
        scheduler.step()
        val_m = validate(model, val_loader, loss_fn, device, cfg["default_threshold"], use_amp)

        history["train_loss"].append(tr_loss)
        for k, v in val_m.items():
            history[k].append(v)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_m["dice"] > best["dice"]:
            best = {"dice": val_m["dice"], "epoch": epoch}
            save_checkpoint(best_ck, epoch, model, optimizer, scheduler, scaler, ema, best,
                            history)
            patience_cnt = 0
        else:
            patience_cnt += 1
        save_checkpoint(latest_ck, epoch, model, optimizer, scheduler, scaler, ema, best, history)

        log(f"Ep {epoch + 1:03d}/{cfg['epochs']}  loss {tr_loss:.4f}  val {val_m['val_loss']:.4f}"
            f"  dice {val_m['dice']:.4f}  prec {val_m['precision']:.4f}"
            f"  rec {val_m['recall']:.4f}  auc {val_m['auc']:.4f}  thr {val_m['opt_thr']:.3f}"
            f"  [{(time.time() - t0) / 60:.0f} min]")
        if patience_cnt >= cfg["patience"]:
            log(f"Early stopping at epoch {epoch + 1} (patience {cfg['patience']})")
            break

    log(f"Best validation Dice {best['dice']:.4f} at epoch {best['epoch'] + 1}")
    return dict(history), best
