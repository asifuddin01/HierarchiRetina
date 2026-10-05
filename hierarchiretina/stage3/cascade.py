"""End-to-end cascade evaluation (Stage I gate -> Stage II masks -> Stage III LG-DRG).

Protocol (paper, Section "Cascade Evaluation Protocol"):

* **Stage B** (headline): every test image. Images blocked by the Stage I gate are scored as
  Grade 0, which is what the system outputs for them.
* **Stage A**: test images with true Grade 1-5 that reached Stage III.
* Severity QWK: true Grade 5 removed, predicted Grade 5 mapped to 4 (``metrics.qwk_sev``).
* 95% CIs: 1,000 image-level bootstrap resamples (500 for the gate AUC), one shared
  ``numpy`` generator seeded with 2026.

``run_final_evaluation`` is a faithful port of the single script that produced every reported
cascade number. The bootstrap calls are made in the original order so that the CIs are
reproduced exactly; do not reorder them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import calibrate_thresholds
from .corn import corn_from_full, decode_consecutive
from .metrics import auc, binary_metrics, bootstrap_ci, macro_f1, qwk, qwk_sev

GATE_THRESHOLD = 0.2391715943813324   # Stage I operating point (validation Youden-J, frozen)
BOOT_SEED = 2026
PC = [f"p_grade_{k}" for k in range(1, 6)]

# Source dataset from the image stem; order matters (most specific first).
SOURCE_PATTERNS = [
    ("idrid", r"^IDRiD_", True),
    ("messidor", r"^\d{8}_\d+_\d+_[A-Za-z]+$", False),
    ("messidor", r"^IM\d{4,}$", True),
    ("eyepacs", r"^\d+_(left|right)$", True),
    ("ddr", r"^\d+-\d+-\d+$", False),
    ("ddr", r"^\d{10,}$", False),
    ("aptos", r"^(?=.*[a-f])[a-f0-9]{8,}$", True),
    ("fgadr", r"^\d+_\d+$", False),
]
DATASETS = ["eyepacs", "ddr", "aptos", "fgadr", "messidor", "idrid"]


def detect_source(stem: str) -> str:
    s = Path(str(stem)).stem
    for name, pat, ic in SOURCE_PATTERNS:
        if re.match(pat, s, re.I if ic else 0):
            return name
    return "unknown"


# ----------------------------------------------------------------------------- inputs
def load_ground_truth(path, id_col: str = "image", label_col: str = "grade") -> pd.DataFrame:
    """Test labels (grades 0..5) for all 58,689 test images -> ``stem, true_grade``."""
    gt = pd.read_csv(path)
    return pd.DataFrame({"stem": gt[id_col].astype(str).map(lambda s: Path(s).stem),
                         "true_grade": gt[label_col].astype(int)})


def load_stage1_manifest(path) -> pd.DataFrame:
    """Stage I routing manifest: ``stem, prob, pred_binary`` (1 = passed the gate)."""
    man = pd.read_csv(path)
    man["stem"] = man["stem"].astype(str)
    return man


def load_lgdrg_predictions(path) -> pd.DataFrame:
    pr = pd.read_csv(path)
    if "status" in pr:
        pr = pr[pr.status == "OK"].copy()
    pr["stem"] = pr["stem"].astype(str)
    return pr


def build_cascade_frame(gt: pd.DataFrame, man: pd.DataFrame, pr: pd.DataFrame,
                        t, tg, expected_passed: int | None = 19154) -> pd.DataFrame:
    """One row per test image (ground-truth order): routing, LG-DRG grades (argmax and tuned
    decode) and the final cascade grade ``final`` (0 for blocked images)."""
    pr = pr.copy()
    pr["g_argmax"] = pr[PC].values.argmax(1) + 1
    pr["g_cal"] = decode_consecutive(*corn_from_full(pr[PC].values), t, tg)
    C = gt.merge(man[["stem", "prob", "pred_binary"]], on="stem", how="left").merge(
        pr[["stem", "g_argmax", "g_cal"] + PC], on="stem", how="left")
    C["passed"] = C.pred_binary.fillna(0).astype(int) == 1
    C["source"] = C.stem.map(detect_source)
    if expected_passed is not None:
        assert C.passed.sum() == expected_passed, C.passed.sum()
    assert (C.passed & C.g_cal.isna()).sum() == 0, "passed images without a prediction"
    for col in ("g_argmax", "g_cal"):
        C[f"final_{col}"] = np.where(C.passed, C[col].fillna(0), 0).astype(int)
    C["final"] = C.final_g_cal
    return C


# ----------------------------------------------------------------------------- blocks
def gate_summary(C, rng, gate_threshold=GATE_THRESHOLD):
    yt = C.true_grade.values
    yb = (yt > 0).astype(int)
    passed = C.passed.values
    n = len(C)
    tp, tn = int(((yb == 1) & passed).sum()), int(((yb == 0) & ~passed).sum())
    fp, fn = int(((yb == 0) & passed).sum()), int(((yb == 1) & ~passed).sum())
    probs = C.prob.values
    gate = {"threshold": gate_threshold, "auc": auc(yb, probs),
            "auc_ci": bootstrap_ci(lambda i: auc(yb[i], probs[i]), n, rng, 500),
            "sens": tp / (tp + fn), "spec": tn / (tn + fp), "ppv": tp / (tp + fp),
            "npv": tn / (tn + fn), "acc": (tp + tn) / n, "TP": tp, "TN": tn, "FP": fp,
            "FN": fn, "passed": int(passed.sum()), "blocked": int((~passed).sum())}
    per_grade = pd.DataFrame([{"grade": g, "n": int((yt == g).sum()),
                               "passed": int(((yt == g) & passed).sum()),
                               "pass_rate": float(passed[yt == g].mean()),
                               "mean_prob": float(probs[yt == g].mean())} for g in range(6)])
    gate["sens_referable_2to4"] = float(passed[(yt >= 2) & (yt <= 4)].mean())
    gate["sens_vision_threatening_3to4"] = float(passed[(yt >= 3) & (yt <= 4)].mean())
    return gate, per_grade


def headline(C, rng):
    """Stage A / Stage B QWK (with bootstrap CIs), accuracy, macro-F1, argmax comparison."""
    yt, yp, n = C.true_grade.values, C.final.values, len(C)
    A = C[C.passed & (C.true_grade > 0)]

    def qa(idx=None, col="final"):
        d = A if idx is None else A.iloc[idx]
        return qwk_sev(d.true_grade.values, d[col].values, include_healthy=False)[0]

    def qb(idx=None, col="final"):
        d = C if idx is None else C.iloc[idx]
        return qwk_sev(d.true_grade.values, d[col].values, include_healthy=True)[0]

    return {
        "stageA_n": int(len(A)), "stageA_qwk": qa(),
        "stageA_qwk_ci": bootstrap_ci(lambda i: qa(i), len(A), rng),
        "stageA_acc": float(np.mean(A.final == A.true_grade)),
        "stageB_n": n, "stageB_qwk": qb(), "stageB_qwk_ci": bootstrap_ci(lambda i: qb(i), n, rng),
        "stageB_acc": float(np.mean(yp == yt)), "stageB_macro_f1": macro_f1(yt, yp, range(6)),
        "stageB_qwk_argmax": qb(col="final_g_argmax"),
        "stageA_qwk_argmax": qa(col="final_g_argmax"),
        "stageB_qwk_as_published_6class": qwk(yt, yp, range(6)),
    }


def error_analysis(C):
    """Error attribution to stages, Stage III error distance, per-grade accuracy, confusion."""
    yt, yp = C.true_grade.values, C.final.values
    blk, cor = ~C.passed, yp == yt
    att = {"stageI_missed_DR": int((blk & (yt > 0)).sum()),
           "stageI_passed_healthy": int((C.passed & (yt == 0)).sum()),
           "stageIII_wrong_on_DR": int((C.passed & (yt > 0) & ~cor).sum())}
    att["total"] = sum(att.values())
    S = C[C.passed & (C.true_grade > 0)]
    sev = S[(S.true_grade <= 4) & (S.final <= 4)]
    dist = (sev.true_grade - sev.final).abs().value_counts().sort_index()
    per_grade = pd.DataFrame([{"grade": g, "n": int((S.true_grade == g).sum()),
                               "acc": float(np.mean(S.final[S.true_grade == g] == g)),
                               "mean_pred": float(S.final[S.true_grade == g].mean())}
                              for g in range(1, 6)])
    cm = np.zeros((6, 6), int)
    for a, b in zip(yt, yp):
        cm[a, b] += 1
    cm = pd.DataFrame(cm, index=[f"T{i}" for i in range(6)], columns=[f"P{i}" for i in range(6)])
    return att, {int(k): int(v) for k, v in dist.items()}, per_grade, cm


def per_dataset(C) -> pd.DataFrame:
    rows = []
    for s in DATASETS:
        D = C[C.source == s]
        Ad = D[D.passed & (D.true_grade > 0)]
        rows.append({
            "dataset": s, "n": len(D), "n_dr": int((D.true_grade > 0).sum()),
            "gate_sens": float(D.passed[D.true_grade > 0].mean()),
            "gate_spec": (float((~D.passed[D.true_grade == 0]).mean())
                          if (D.true_grade == 0).any() else np.nan),
            "dr_lost": int(((~D.passed) & (D.true_grade > 0)).sum()),
            "qwk_A": qwk_sev(Ad.true_grade, Ad.final, False)[0] if len(Ad) > 1 else np.nan,
            "acc_A": float(np.mean(Ad.final == Ad.true_grade)),
            "qwk_B": qwk_sev(D.true_grade, D.final, True)[0],
            "acc_B": float(np.mean(D.final == D.true_grade))})
    return pd.DataFrame(rows)


def ddr_protocols(C, rng) -> dict:
    """DDR official test split under the five-class (grades 0-4, gradable only) and six-class
    (all 4,105 images) protocols. A predicted Grade 5 on a gradable image is scored three ways:
    mapped to 4 (paper definition), dropped, or mapped to the farthest grade (worst case)."""
    D = C[C.source == "ddr"]
    G = D[D.true_grade <= 4]
    ytg, ypg = G.true_grade.values, G.final.values
    m5 = ypg == 5
    worst = ypg.copy()
    worst[m5] = np.where(ytg[m5] >= 2, 0, 4)
    return {
        "n_total": len(D), "n_gradable": len(G), "n_ungradable": int((D.true_grade == 5).sum()),
        "pred5_on_gradable": int(m5.sum()),
        "qwk5_pred5to4": qwk(ytg, np.clip(ypg, 0, 4), range(5)),
        "qwk5_pred5to4_ci": bootstrap_ci(lambda i: qwk(ytg[i], np.clip(ypg[i], 0, 4), range(5)),
                                         len(G), rng),
        "acc5_pred5to4": float(np.mean(np.clip(ypg, 0, 4) == ytg)),
        "qwk5_worstcase": qwk(ytg, worst, range(5)), "acc5_worstcase": float(np.mean(worst == ytg)),
        "qwk5_drop_pred5": qwk(ytg[~m5], ypg[~m5], range(5)), "n_drop_pred5": int((~m5).sum()),
        "acc6_all": float(np.mean(D.final == D.true_grade)),
        "acc6_ci": bootstrap_ci(lambda i: float(np.mean(D.final.values[i] ==
                                                        D.true_grade.values[i])), len(D), rng),
        "ungradable_recall": float(np.mean(D.final[D.true_grade == 5] == 5)),
        "ungradable_precision": float(np.mean(D.true_grade[D.final == 5] == 5)),
    }


def gate_sweep(C, gate_threshold=GATE_THRESHOLD) -> pd.DataFrame:
    """End-to-end QWK when the Stage I threshold is raised (upward only: images below the
    operating point were never segmented, so lower thresholds cannot be evaluated)."""
    yt = C.true_grade.values
    rows = []
    grid = np.unique(np.round(np.concatenate([[gate_threshold], np.arange(0.25, 0.91, 0.025)]), 6))
    for thr in grid:
        ps = ((C.prob >= thr) & C.passed).values
        ypp = np.where(ps, C.final, 0)
        rows.append({"threshold": thr, "passed": int(ps.sum()), "qwk": qwk_sev(yt, ypp, True)[0],
                     "acc": float(np.mean(ypp == yt)),
                     "dr_lost": int(((~ps) & (yt > 0)).sum()),
                     "healthy_in": int((ps & (yt == 0)).sum())})
    return pd.DataFrame(rows)


def referral(C, rng) -> dict:
    """Referral view: refer if output >= Grade 2 (ungradable included); vision-threatening DR
    = true Grades 3-4 output as 3-4."""
    yt, yp, n = C.true_grade.values, C.final.values, len(C)
    rt, rp = yt >= 2, yp >= 2
    return {
        "referable_ge2_incl_ungradable": binary_metrics(rt, rp),
        "referable_sens_ci": bootstrap_ci(lambda i: float((rt[i] & rp[i]).sum() / rt[i].sum()),
                                          n, rng),
        "referable_spec_ci": bootstrap_ci(lambda i: float((~rt[i] & ~rp[i]).sum() /
                                                          (~rt[i]).sum()), n, rng),
        "vision_threatening_3to4": binary_metrics((yt >= 3) & (yt <= 4), (yp >= 3) & (yp <= 4)),
    }


def gate_calibration(C) -> dict:
    yb = (C.true_grade.values > 0).astype(int)
    probs = C.prob.values
    bins = np.linspace(0, 1, 16)
    b = np.clip(np.digitize(probs, bins) - 1, 0, 14)
    ece = sum(abs(yb[b == k].mean() - probs[b == k].mean()) * (b == k).mean()
              for k in range(15) if (b == k).any())
    return {"ece_15bins": float(ece), "brier": float(np.mean((probs - yb) ** 2)),
            "fraction_stopped": float((~C.passed).mean())}


def leakage_sensitivity(C) -> list[dict]:
    """Headline recomputed without the sources that may overlap the Stage II training data."""
    out = []
    for label, srcs in [("eyepacs+ddr+aptos (no Stage II training overlap possible)",
                         ["eyepacs", "ddr", "aptos"]),
                        ("fgadr+messidor+idrid (possible Stage II overlap)",
                         ["fgadr", "messidor", "idrid"])]:
        D = C[C.source.isin(srcs)]
        A = D[D.passed & (D.true_grade > 0)]
        out.append({"subset": label, "n": len(D),
                    "stageB_qwk": qwk_sev(D.true_grade, D.final, True)[0],
                    "stageB_acc": float(np.mean(D.final == D.true_grade)),
                    "stageA_n": len(A), "stageA_qwk": qwk_sev(A.true_grade, A.final, False)[0]})
    return out


# ----------------------------------------------------------------------------- driver
def oof_calibration_summary(oof: pd.DataFrame, t, tg, best, start) -> tuple[dict, pd.DataFrame]:
    yo = oof.true_grade.values.astype(int)
    po = corn_from_full(oof[PC].values)
    am = oof[PC].values.argmax(1) + 1
    cal = decode_consecutive(*po, t, tg)
    summary = {
        "decode": "consecutive CORN rank, thresholds fitted on OOF (grades 1-5) only",
        "t": list(t), "tg": tg,
        "oof_qwk_argmax": qwk_sev(yo, am, False)[0], "oof_qwk_corn05": start,
        "oof_qwk_calibrated": best,
        "oof_acc_argmax": float(np.mean(am == yo)), "oof_acc_calibrated": float(np.mean(cal == yo)),
        "oof_g5_auc": auc((yo == 5).astype(int), oof.p_grade_5.values),
        "oof_macro_auc": float(np.mean([auc((yo == k).astype(int), oof[f"p_grade_{k}"].values)
                                        for k in range(1, 6)])),
        "n_oof": int(len(oof)),
    }
    per_grade = pd.DataFrame([{
        "grade": g, "n": int((yo == g).sum()),
        "acc_argmax": float(np.mean(am[yo == g] == g)), "acc_cal": float(np.mean(cal[yo == g] == g)),
        "meanpred_argmax": float(am[yo == g].mean()), "meanpred_cal": float(cal[yo == g].mean()),
        "auc": auc((yo == g).astype(int), oof[f"p_grade_{g}"].values)} for g in range(1, 6)])
    return summary, per_grade


def run_final_evaluation(oof_csv, pred_csv, gt_csv, manifest_csv, out_dir,
                         gt_id_col: str = "image", gt_label_col: str = "grade",
                         gate_threshold: float = GATE_THRESHOLD, seed: int = BOOT_SEED,
                         expected_passed: int | None = 19154, verbose: bool = False) -> dict:
    """Fit thresholds on OOF, decode the test ensemble, and compute every cascade number.

    Writes into ``out_dir``: corn_thresholds_consecutive.json, oof_per_grade.csv,
    cascade_final.csv, gate_per_grade.csv, stage3_test_per_grade.csv,
    confusion_end_to_end.csv, per_dataset.csv, gate_sweep.csv, leakage_sensitivity.json and
    final_numbers.json. Returns the ``final_numbers`` dict plus the tables.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 1. OOF calibration
    oof = pd.read_csv(oof_csv)
    t, tg, best, start = calibrate_thresholds(oof.true_grade.values, oof[PC].values,
                                              verbose=verbose)
    calib, oof_pg = oof_calibration_summary(oof, t, tg, best, start)
    json.dump(calib, open(out / "corn_thresholds_consecutive.json", "w"), indent=2)
    oof_pg.to_csv(out / "oof_per_grade.csv", index=False)

    # 2. cascade frame
    C = build_cascade_frame(load_ground_truth(gt_csv, gt_id_col, gt_label_col),
                            load_stage1_manifest(manifest_csv), load_lgdrg_predictions(pred_csv),
                            t, tg, expected_passed)
    C.to_csv(out / "cascade_final.csv", index=False)

    # 3-9. same order as the original script (the bootstrap generator is shared)
    gate, gate_pg = gate_summary(C, rng, gate_threshold)
    gate_pg.to_csv(out / "gate_per_grade.csv", index=False)
    head = headline(C, rng)
    att, dist, s3_pg, cm = error_analysis(C)
    s3_pg.to_csv(out / "stage3_test_per_grade.csv", index=False)
    cm.to_csv(out / "confusion_end_to_end.csv")
    pds = per_dataset(C)
    pds.to_csv(out / "per_dataset.csv", index=False)
    ddr = ddr_protocols(C, rng)
    sweep = gate_sweep(C, gate_threshold)
    sweep.to_csv(out / "gate_sweep.csv", index=False)
    ref = referral(C, rng)
    gate.update(gate_calibration(C))
    leak = leakage_sensitivity(C)
    json.dump(leak, open(out / "leakage_sensitivity.json", "w"), indent=2, default=float)

    res = {"calibration": calib, "gate": gate, "headline": head, "referral": ref,
           "attribution": att, "error_distance": dist, "ddr": ddr}
    json.dump(res, open(out / "final_numbers.json", "w"), indent=2, default=float)
    res["tables"] = {"cascade": C, "oof_per_grade": oof_pg, "gate_per_grade": gate_pg,
                     "stage3_test_per_grade": s3_pg, "confusion": cm, "per_dataset": pds,
                     "gate_sweep": sweep, "leakage": pd.DataFrame(leak)}
    return res
