"""Does LG-DRG use the masks? Gate-scale read-out and test-time mask ablation.

1. ``read_gamma`` reads the learned gate scale from each fold's EMA weights.
2. ``run_mask_ablation`` re-scores fold 0's held-out images (single model, no retraining)
   under eight mask conditions and reports the change in severity QWK, accuracy, macro-F1,
   macro-AUC and G5-AUC (default 0.5 decode, ``lgdrg_predict``).
3. The gamma-fix retrain of fold 0 (gamma initialised to 0.1 and excluded from weight decay,
   everything else unchanged) is ``engine.train_fold`` with ``gamma_fix_config``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .engine import LGDRGConfig, evaluate

CH = {"ma": 0, "he": 1, "ex": 2, "cws": 3, "vessel": 4}   # index inside x[:, 3:8]
CONDITIONS = ["intact", "all_zero", "permuted",
              "zero_ma", "zero_he", "zero_ex", "zero_cws", "zero_vessel"]
GAMMA_FIX_INIT = 0.1


def read_gamma(ckpt_dir, n_folds: int = 5, template: str = "best_fold{k}.pt") -> pd.DataFrame:
    rows = []
    for k in range(n_folds):
        p = Path(ckpt_dir) / template.format(k=k)
        if not p.exists():
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["ema"]
        rows.append({"fold": k, "gamma": float(sd["gate.gamma"].flatten()[0])})
    return pd.DataFrame(rows)


def apply_condition(x: torch.Tensor, cond: str) -> torch.Tensor:
    """Return a modified copy of an 8-channel batch.

    ``permuted`` rolls the masks by one position in the batch, so every image receives another
    image's real masks (in-distribution but wrong); a batch of one falls back to zeros.
    """
    x = x.clone()
    if cond == "intact":
        return x
    if cond == "all_zero":
        x[:, 3:] = 0
    elif cond == "permuted":
        x[:, 3:] = x[:, 3:].roll(1, dims=0) if x.size(0) > 1 else 0
    elif cond.startswith("zero_"):
        x[:, 3 + CH[cond[5:]]] = 0
    else:
        raise ValueError(cond)
    return x


def run_mask_ablation(model, loader, device, use_amp: bool = True,
                      conditions=CONDITIONS) -> pd.DataFrame:
    """Score ``loader`` under each condition; ``d_qwk`` is relative to ``intact``.

    The validation loader is unshuffled, so ``permuted`` pairs are the same in every run.
    """
    rows = []
    for c in conditions:
        m, Y, P, PF = evaluate(model, loader, device, use_amp,
                               transform=lambda x, c=c: apply_condition(x, c))
        rows.append({"condition": c, "n": len(Y), "qwk_sev": m["qwk_sev"],
                     "acc": accuracy_score(Y, P), "macro_f1": f1_score(Y, P, average="macro"),
                     "macro_auc": m["auc"], "g5_auc": roc_auc_score((Y == 4).astype(int), PF[:, 4])})
        print(f"  {c:<12} QWK {rows[-1]['qwk_sev']:.4f}  acc {rows[-1]['acc']:.4f}  "
              f"macroAUC {rows[-1]['macro_auc']:.4f}")
    res = pd.DataFrame(rows)
    res["d_qwk"] = res.qwk_sev - res.loc[res.condition == "intact", "qwk_sev"].iloc[0]
    return res


def gamma_fix_config(cfg: LGDRGConfig, gamma_init: float = GAMMA_FIX_INIT) -> LGDRGConfig:
    """Fold-0 retrain config, as run: gamma initialised to ``gamma_init`` and excluded from
    weight decay, and gradient accumulation 8 (the value stored in the released gamma-fix
    checkpoint; the five fold models used 4). Checkpoints go to ``<out_dir>_gammafix``."""
    return dataclasses.replace(cfg, out_dir=str(cfg.out_dir).rstrip("\\/") + "_gammafix",
                               gamma_init=gamma_init, gamma_no_decay=True, grad_accum=8)


def ablation_table(original: pd.DataFrame, retrained: pd.DataFrame | None = None) -> pd.DataFrame:
    """Side-by-side QWK / delta-QWK table (paper Table "Mask reliance")."""
    t = original.set_index("condition")[["qwk_sev", "d_qwk"]].add_prefix("original_")
    if retrained is not None:
        t = t.join(retrained.set_index("condition")[["qwk_sev", "d_qwk"]].add_prefix("retrained_"))
    return t.loc[CONDITIONS].reset_index()


def summarise_gamma_fix(ckpt_path, ablation: pd.DataFrame) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {"gamma_init": GAMMA_FIX_INIT,
            "gamma_best_ema": float(ck["ema"]["gate.gamma"].flatten()[0]),
            "best_epoch": int(ck["epoch"]), "best_macro_auc": float(ck["best"]),
            "qwk_intact": float(ablation.loc[ablation.condition == "intact", "qwk_sev"].iloc[0]),
            "max_abs_dqwk": float(ablation.d_qwk.abs().max())}

