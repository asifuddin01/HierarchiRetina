"""LG-DRG training, evaluation, out-of-fold export and ensemble test inference.

Training recipe (training notebook, final configuration): AdamW with two parameter groups
(backbone lr 2e-5, everything else 2e-4), weight decay 0.05 on *all* parameters (biases, norms
and the gate scalar gamma included), linear warm-up + cosine, AMP, gradient accumulation,
clip-norm 5, EMA 0.9998 updated after every optimiser step, inverse-frequency sampling over the
five grades, model selection on the EMA model's five-class macro-AUC, at most 60 epochs with
patience 12. Checkpoints hold both raw and EMA weights; every reported number uses EMA.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .corn import lgdrg_predict
from .data import InferDataset, make_loaders_2h
from .losses import lgdrg_loss
from .metrics import macro_auc_5, qwk_sev_idx
from .model import LGDRG, load_lgdrg

GRADE_DESC = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative",
              5: "Ungradable - human review"}


@dataclass
class LGDRGConfig:
    """All Stage III hyper-parameters (values of the training notebook)."""
    backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    img_size: int = 512
    n_masks: int = 5
    num_classes: int = 5            # grades 1..5
    drop_path: float = 0.2
    drop_rate: float = 0.3          # head dropout
    epochs: int = 60
    batch_size: int = 16
    grad_accum: int = 8             # value in the training notebook's CFG (paper text says 4)
    lr: float = 2e-4                # heads, gate, mask encoder
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 3          # sized in batches, see build_scheduler
    clip_norm: float = 5.0
    use_amp: bool = True
    ema_decay: float = 0.9998
    num_workers: int = 0
    n_folds: int = 5
    early_stop_patience: int = 12
    seed: int = 42
    out_dir: str = "outputs/stage3/checkpoints"
    # gamma-fix experiment (fold 0 only); defaults reproduce the original training
    gamma_init: float | None = None
    gamma_no_decay: bool = False


def set_seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _autocast(device: torch.device, use_amp: bool):
    return torch.autocast(device.type, enabled=use_amp and device.type == "cuda")


class EMA:
    """Exponential moving average of the full ``state_dict`` (BN buffers included; integer
    buffers are copied)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def build_optimizer(model: LGDRG, cfg: LGDRGConfig) -> torch.optim.AdamW:
    """AdamW: backbone at lr x 0.1, everything else at lr; weight decay on all parameters.

    With ``cfg.gamma_no_decay`` (gamma-fix run) ``gate.gamma`` moves to a third group with the
    head lr and no weight decay; nothing else changes.
    """
    bb, hd, nd = [], [], []
    for n, p in model.named_parameters():
        if cfg.gamma_no_decay and n == "gate.gamma":
            nd.append(p)
        elif n.startswith("backbone"):
            bb.append(p)
        else:
            hd.append(p)
    if not cfg.gamma_no_decay:
        return torch.optim.AdamW([{"params": bb, "lr": cfg.lr * cfg.backbone_lr_mult},
                                  {"params": hd, "lr": cfg.lr}], weight_decay=cfg.weight_decay)
    assert len(nd) == 1, "gate.gamma not found"
    return torch.optim.AdamW([
        {"params": bb, "lr": cfg.lr * cfg.backbone_lr_mult, "weight_decay": cfg.weight_decay},
        {"params": hd, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": nd, "lr": cfg.lr, "weight_decay": 0.0},
    ])


def build_scheduler(opt, steps_per_epoch: int, cfg: LGDRGConfig):
    """Linear warm-up + cosine, preserved exactly as trained.

    Known quirk (kept on purpose, the released models were trained this way): the schedule
    length is counted in *batches* (``steps_per_epoch = len(train_loader)``) but
    ``sched.step()`` is called once per *optimiser update*, i.e. every ``grad_accum`` batches.
    Warm-up therefore lasts ``warmup_epochs * grad_accum`` epochs (24 with accumulation 8;
    12 with 4) and the cosine is far from complete when training stops at 60 epochs.
    """
    total = steps_per_epoch * cfg.epochs
    warm = steps_per_epoch * cfg.warmup_epochs

    def lr_lambda(s):
        if s < warm:
            return s / max(1, warm)
        prog = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, use_amp: bool = True, transform=None):
    """Score a two-head loader. Returns ``(metrics, Y, P, PF)`` with 0-indexed labels/preds
    (``P`` from ``lgdrg_predict`` at default thresholds) and 5-way probabilities ``PF``.

    ``transform(x)`` optionally modifies each input batch (mask ablation).
    """
    model.eval()
    P, Y, PF = [], [], []
    for x, y_full, _, _ in loader:
        if transform is not None:
            x = transform(x)
        x = x.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            s, g = model(x)
        pred, full = lgdrg_predict(s.float(), g.float())
        P.append(pred.cpu().numpy())
        PF.append(full.cpu().numpy())
        Y.append(y_full.numpy())
    P, Y, PF = np.concatenate(P), np.concatenate(Y), np.concatenate(PF)
    y_un = (Y == 4).astype(int)
    g5 = roc_auc_score(y_un, PF[:, 4]) if 0 < y_un.sum() < len(y_un) else float("nan")
    return {"auc": macro_auc_5(Y, PF), "g5_auc": g5, "qwk_sev": qwk_sev_idx(Y, P),
            "acc": accuracy_score(Y, P), "f1": f1_score(Y, P, average="macro")}, Y, P, PF


def save_ckpt(path, model, ema, opt, sched, scaler, epoch, best, patience, cfg) -> None:
    torch.save({"model": model.state_dict(), "ema": ema.shadow, "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                "best": best, "patience": patience, "cfg": asdict(cfg)}, path)


def train_fold(cfg: LGDRGConfig, df: pd.DataFrame, fold: int, pos_weight: float,
               device=None, seed_all: bool = True) -> tuple[float, Path]:
    """Train one fold; resumable per epoch from ``last_fold{k}.pt``.

    Writes ``best_fold{k}.pt`` (best EMA macro-AUC), ``last_fold{k}.pt``,
    ``history_fold{k}.json`` and ``done_fold{k}.flag`` into ``cfg.out_dir``.

    ``seed_all=False`` reproduces the gamma-fix cell, which seeded only torch and numpy.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_p, last_p = out / f"best_fold{fold}.pt", out / f"last_fold{fold}.pt"
    done_p, hist_p = out / f"done_fold{fold}.flag", out / f"history_fold{fold}.json"
    if done_p.exists():
        print(f"[skip] fold {fold} already complete")
        return float(json.loads(done_p.read_text()).get("best_auc", -1.0)), best_p

    if seed_all:
        set_seed(cfg.seed + fold)
    else:
        torch.manual_seed(cfg.seed + fold)
        np.random.seed(cfg.seed + fold)
    tl, vl, _ = make_loaders_2h(df, fold, cfg.img_size, cfg.batch_size, cfg.num_workers,
                                cfg.num_classes)
    model = LGDRG(cfg, pretrained=True).to(device)
    if cfg.gamma_init is not None:            # gamma-fix: open the gate before EMA is created
        with torch.no_grad():
            model.gate.gamma.fill_(cfg.gamma_init)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, len(tl), cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay)

    start, best, patience = 0, -1.0, 0
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else {}
    if last_p.exists():
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, patience = ck["epoch"] + 1, ck["best"], ck.get("patience", 0)
        print(f"[resume] fold {fold} from epoch {start}, best macro-AUC {best:.4f}")

    last_ep, stopped = start - 1, False
    for ep in range(start, cfg.epochs):
        last_ep = ep
        model.train()
        opt.zero_grad()
        sums = np.zeros(3)
        for i, (x, _, y_sev, grad) in enumerate(tqdm(tl, desc=f"E{ep:02d}/fold{fold}")):
            x = x.to(device, non_blocking=True)
            y_sev, grad = y_sev.to(device), grad.to(device)
            with _autocast(device, cfg.use_amp):
                s, g = model(x)
                loss, lg, ls = lgdrg_loss(s, g, y_sev, grad, pos_weight)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()          # once per update; see build_scheduler
                ema.update(model)
            sums += [loss.item() * cfg.grad_accum, float(lg), float(ls)]

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        m, *_ = evaluate(model, vl, device, cfg.use_amp)
        model.load_state_dict(backup)
        row = {"epoch": ep, "lr": float(sched.get_last_lr()[-1]),
               "train_loss": sums[0] / len(tl), "train_g_loss": sums[1] / len(tl),
               "train_s_loss": sums[2] / len(tl),
               "gamma": float(model.gate.gamma.detach().cpu()),
               "gamma_ema": float(ema.shadow["gate.gamma"].detach().cpu()),
               **{f"val_{k}": float(v) for k, v in m.items()}}
        for k, v in row.items():
            hist.setdefault(k, []).append(v)
        hist_p.write_text(json.dumps(hist, indent=2))
        print(f"  val macroAUC {m['auc']:.4f}  G5-AUC {m['g5_auc']:.4f}  "
              f"QWK {m['qwk_sev']:.4f}  acc {m['acc']:.4f}  gamma(EMA) {row['gamma_ema']:+.4f}")

        if m["auc"] > best:
            best, patience = m["auc"], 0
            save_ckpt(best_p, model, ema, opt, sched, scaler, ep, best, patience, cfg)
        else:
            patience += 1
        save_ckpt(last_p, model, ema, opt, sched, scaler, ep, best, patience, cfg)
        if patience >= cfg.early_stop_patience:
            stopped = True
            print(f"  early stop at epoch {ep}")
            break

    done_p.write_text(json.dumps({"fold": fold, "best_auc": float(best), "last_epoch": last_ep,
                                  "early_stopped": stopped}, indent=2))
    return best, best_p


def train_all_folds(cfg: LGDRGConfig, df: pd.DataFrame, pos_weight: float, device=None):
    rows = []
    for f in range(cfg.n_folds):
        best, path = train_fold(cfg, df, f, pos_weight, device)
        rows.append({"fold": f, "best_macro_auc": best, "ckpt": str(path)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- OOF
def oof_predictions(cfg: LGDRGConfig, df: pd.DataFrame, ckpt_dir, device=None,
                    template: str = "best_fold{k}.pt"):
    """Score every development image with the fold model that did not train on it.

    Returns ``(oof_df, per_fold_df)``. ``oof_df`` columns: image_id, fold, true_grade, y,
    pred_idx, pred_grade, p_grade_1..5 (``pred_idx`` from ``lgdrg_predict`` at 0.5).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    parts, per_fold = [], []
    for k in range(cfg.n_folds):
        model = load_lgdrg(Path(ckpt_dir) / template.format(k=k), cfg, device)
        _, vl, va = make_loaders_2h(df, k, cfg.img_size, cfg.batch_size, cfg.num_workers,
                                    cfg.num_classes)
        m, Y, P, PF = evaluate(model, vl, device, cfg.use_amp)
        assert np.array_equal(Y, va["y"].values), f"fold {k}: label order mismatch"
        per_fold.append({"fold": k, **m})
        parts.append(pd.DataFrame({"image_id": va["image_id"].values, "fold": k,
                                   "true_grade": va["grade"].values, "y": Y, "pred_idx": P,
                                   "pred_grade": P + 1,
                                   **{f"p_grade_{j + 1}": PF[:, j] for j in range(5)}}))
        print(f"  fold {k}: macroAUC {m['auc']:.4f}  G5-AUC {m['g5_auc']:.4f}  "
              f"QWK {m['qwk_sev']:.4f}  acc {m['acc']:.4f}  F1 {m['f1']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(parts, ignore_index=True), pd.DataFrame(per_fold)


# ----------------------------------------------------------------------------- test inference
def load_fold_models(cfg: LGDRGConfig, ckpt_dir, folds=(0, 1, 2, 3, 4), device=None,
                     template: str = "best_fold{k}.pt") -> dict[int, LGDRG]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return {k: load_lgdrg(Path(ckpt_dir) / template.format(k=k), cfg, device) for k in folds}


@torch.no_grad()
def ensemble_probs(models: dict, x: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
    """Mean of the members' 5-way probability vectors, ``(B, 5)``."""
    acc = None
    for m in models.values():
        with _autocast(x.device, use_amp):
            s, g = m(x)
        _, full = lgdrg_predict(s.float(), g.float())
        acc = full if acc is None else acc + full
    return acc / len(models)


def run_test_inference(models: dict, fdf: pd.DataFrame, cfg: LGDRGConfig, out_csv,
                       batch_size: int = 8, num_workers: int = 0, save_every: int = 50,
                       cache_csv=None) -> pd.DataFrame:
    """Batched, resumable 5-fold ensemble inference over the Stage I DR-routed images.

    Writes one row per image: stem, pred_grade (argmax of the averaged vector, 1..5),
    pred_label, confidence, p_grade_1..5, status, n_models. The reported decode is applied
    afterwards from ``p_grade_*`` (``corn.decode_full``).
    """
    device = next(next(iter(models.values())).parameters()).device
    cache_csv = Path(cache_csv or Path(out_csv).with_suffix(".cache.csv"))
    rows, done = [], set()
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv)
        rows, done = cached.to_dict("records"), set(cached["stem"].astype(str))
        print(f"resuming: {len(done):,} images cached")
    todo = fdf[~fdf["stem"].astype(str).isin(done)]
    dl = DataLoader(InferDataset(todo, cfg.img_size), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers)
    t0 = time.time()
    for bi, (x, stems, ok) in enumerate(tqdm(dl, desc="LG-DRG ensemble")):
        probs = ensemble_probs(models, x.to(device), cfg.use_amp).cpu().numpy()
        idx = probs.argmax(1)
        for j, stem in enumerate(stems):
            g = int(idx[j]) + 1
            rows.append({"stem": str(stem), "pred_grade": g, "pred_label": GRADE_DESC[g],
                         "confidence": float(probs[j, idx[j]]),
                         **{f"p_grade_{k + 1}": float(probs[j, k]) for k in range(5)},
                         "status": "OK" if int(ok[j]) == 1 else "FAILED_TO_READ",
                         "n_models": len(models)})
        if (bi + 1) % save_every == 0:
            pd.DataFrame(rows).to_csv(cache_csv, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(cache_csv, index=False)
    out.to_csv(out_csv, index=False)
    print(f"{len(todo):,} images in {(time.time() - t0) / 60:.1f} min -> {out_csv}")
    return out
