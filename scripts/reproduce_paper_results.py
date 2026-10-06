#!/usr/bin/env python3
"""Reproduce the end-to-end results of the HierarchiRetina paper from the released predictions.

Inputs
  results/cascade_test_predictions.csv  model outputs for the 58,689 test images (no labels)
  results/stage3_oof_predictions.csv    LG-DRG out-of-fold probabilities for the 18,570
                                        development images (reference grade included, no IDs)
  --labels FILE                         reference grades of the test images, taken from the
                                        official dataset sources (0-4, 5 = ungradable); see
                                        results/README.md for how to build this file

What the script does
  1. Checks the SHA-256 of the prediction files against results/SHA256SUMS.
  2. Refits the four LG-DRG decision thresholds on the OOF file only
     (expected: t = (0.38, 0.60, 0.19), t_g = 0.10).
  3. Re-decodes the Stage III grade of every routed test image from the released
     probabilities and checks it against the released grade.
  4. Recomputes every end-to-end number in the paper: the proposed-gate row of Table II,
     Fig. 6, Tables VI to VIII, the examples of Fig. 7, the confusion matrix and error analysis
     of Fig. 8, and the numbers quoted in the text (Sections II-F, III and IV), together with
     the pooled out-of-fold figures of Stage III (Section III-C and Table IV).
  5. Prints every value next to the number printed in the paper.

The 95% bootstrap intervals use one numpy generator seeded with 2026 and called in the
original order, so they match the paper exactly. Only numpy and pandas are needed.

Usage
  python scripts/reproduce_paper_results.py --labels data/test/test_grade.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GATE_THRESHOLD = 0.2391715943813324      # Stage I operating point, fixed on validation data
SEED = 2026
PC = [f"p_grade_{k}" for k in range(1, 6)]
DATASETS = ["eyepacs", "ddr", "aptos", "fgadr", "messidor", "idrid"]
DATASET_NAMES = {"eyepacs": "EyePACS", "ddr": "DDR", "aptos": "APTOS 2019", "fgadr": "FGADR",
                 "messidor": "Messidor-2", "idrid": "IDRiD"}
RT = dict(float_precision="round_trip")  # read the released floats bit-exactly


# ----------------------------------------------------------------------------- metrics
def qwk(a, b, labels):
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    L = np.asarray(sorted(labels))
    k = len(L)
    ia, ib = np.searchsorted(L, a), np.searchsorted(L, b)
    O = np.bincount(ia * k + ib, minlength=k * k).reshape(k, k).astype(float)
    i, j = np.indices((k, k))
    W = (i - j) ** 2 / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def qwk_sev(yt, yp, include_healthy=True):
    """Severity QWK used throughout the paper: true Grade 5 removed, predicted Grade 5 -> 4."""
    yt = np.asarray(yt, int)
    yp = np.asarray(yp, int)
    keep = yt != 5
    if not include_healthy:
        keep &= yt > 0
    lo = 0 if include_healthy else 1
    yt, yp = yt[keep], np.clip(yp[keep], lo, 4)
    return qwk(yt, yp, range(lo, 5)), int(len(yt))


def auc(y, s):
    y = np.asarray(y, int)
    r = pd.Series(np.asarray(s, float)).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_f1(yt, yp, labels):
    f = []
    for c in labels:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot(rng, fn, n, B=1000):
    vals = []
    for _ in range(B):
        v = fn(rng.integers(0, n, n))
        if v == v:
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------- CORN decode
def corn_from_full(P5):
    """Recover the CORN conditional probabilities from the 5-way grade probabilities."""
    P5 = np.asarray(P5, float)
    pu = np.clip(P5[:, 4], 0, 1 - 1e-9)
    pg = np.clip(1 - pu, 1e-9, 1)
    ps = np.clip(P5[:, :4] / pg[:, None], 1e-12, None)
    ps /= ps.sum(1, keepdims=True)
    p1 = np.clip(1 - ps[:, 0], 1e-9, 1 - 1e-9)
    s23 = np.clip(ps[:, 2] + ps[:, 3], 1e-12, None)
    p2 = np.clip(s23 / p1, 1e-9, 1 - 1e-9)
    p3 = np.clip(ps[:, 3] / s23, 1e-9, 1 - 1e-9)
    return p1, p2, p3, pg


def decode(p1, p2, p3, pg, t, tg):
    """Consecutive CORN rank (stop at the first threshold not passed); Grade 5 if pg < tg."""
    a = p1 > t[0]
    b = a & (p2 > t[1])
    c = b & (p3 > t[2])
    rank = a.astype(int) + b.astype(int) + c.astype(int)
    return np.where(pg < tg, 5, rank + 1).astype(int)


def fit_thresholds(y, P5):
    """Coordinate ascent on OOF severity QWK (Grades 1-4), exactly as in the paper."""
    pp = corn_from_full(P5)

    def obj(t, tg):
        return qwk_sev(y, decode(*pp, t, tg), include_healthy=False)[0]

    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    tgrid = np.round(np.arange(0.10, 0.901, 0.02), 2)
    t, tg = [0.5, 0.5, 0.5], 0.5
    best = obj(tuple(t), tg)
    for _ in range(6):
        improved = False
        for i in range(3):
            bv = t[i]
            for v in grid:
                t[i] = float(v)
                s = obj(tuple(t), tg)
                if s > best + 1e-7:
                    best, bv, improved = s, float(v), True
            t[i] = bv
        for v in tgrid:
            s = obj(tuple(t), float(v))
            if s > best + 1e-7:
                best, tg, improved = s, float(v), True
        if not improved:
            break
    return tuple(t), tg, best


# ----------------------------------------------------------------------------- inputs
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_checksums(results):
    sums = results / "SHA256SUMS"
    if not sums.exists():
        print("  SHA256SUMS not found; skipping the checksum check")
        return
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        f = results / name.strip().lstrip("*")
        if f.exists():
            status = "OK" if sha256(f) == digest else "MISMATCH (file differs from the release)"
            print(f"  {name.strip():<36} {status}")


def load_labels(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    idc = next((cols[c] for c in ("image_id", "image", "img", "filename", "file", "id_code", "id",
                                  "name", "stem") if c in cols), None)
    lbc = next((cols[c] for c in ("grade", "true_grade", "label", "diagnosis", "level", "target",
                                  "dr") if c in cols), None)
    if idc is None or lbc is None:
        sys.exit(f"--labels: need an image-ID column and a grade column; found {list(df.columns)}")
    out = pd.DataFrame({"image_id": df[idc].astype(str).map(lambda s: Path(s).stem),
                        "true_grade": pd.to_numeric(df[lbc], errors="coerce")})
    out = out.dropna().drop_duplicates("image_id")
    out["true_grade"] = out.true_grade.astype(int)
    return out


# ----------------------------------------------------------------------------- report
ROWS = []


def fmt_like(value, paper):
    """Format a reproduced value the way the paper prints it."""
    if isinstance(value, tuple):
        lo, hi = value
        a, b = paper.split("-")
        return f"{fmt_like(lo, a)}-{fmt_like(hi, b)}"
    if isinstance(value, str):
        return value
    p = paper.replace(",", "")
    if "." in p:
        return f"{value:.{len(p.split('.')[1])}f}"
    v = int(round(value))
    return f"{v:,}" if "," in paper or v >= 10000 else str(v)


def check(section, name, value, paper):
    got = fmt_like(value, paper)
    ROWS.append((section, name, paper, got, "PASS" if got == paper else "DIFF"))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="CSV with an image-ID column and a grade column")
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"),
                    help="folder with the released prediction files (default: ./results)")
    ap.add_argument("--out", default="outputs/reproduced", help="where to write the recomputed tables")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1. Checksums")
    check_checksums(results)

    # 2. Thresholds, fitted on the OOF predictions only (no rng involved)
    print("2. Refitting the Stage III decision thresholds on the OOF predictions")
    oof = pd.read_csv(results / "stage3_oof_predictions.csv", **RT)
    yo = oof.true_grade.values.astype(int)
    t, tg, oof_best = fit_thresholds(yo, oof[PC].values)
    oof_cal = decode(*corn_from_full(oof[PC].values), t, tg)
    print(f"   t = ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}), t_g = {tg:.2f}")
    check("Section II-E", "Thresholds t1, t2, t3", f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})", "(0.38, 0.60, 0.19)")
    check("Section II-E", "Gradability threshold t_g", f"{tg:.2f}", "0.10")
    check("Section II-E", "OOF development images (Grades 1-5)", len(oof), "18,570")
    oof_argmax = oof[PC].values.argmax(1) + 1
    check("Section III-C", "Pooled OOF QWK, argmax decode", qwk_sev(yo, oof_argmax, False)[0], "0.771")
    check("Section III-C", "OOF AUC for ungradable images (Grade 5)", auc((yo == 5).astype(int), oof["p_grade_5"].values),
          "0.999")
    check("Table IV", "Pooled OOF QWK, tuned (n = 18,570)", oof_best, "0.781")
    check("Table IV", "Pooled OOF accuracy, tuned", float(np.mean(oof_cal == yo)), "0.774")

    # 3. Test frame, in the release order (keeps the bootstrap identical)
    pred = pd.read_csv(results / "cascade_test_predictions.csv", **RT)
    pred["image_id"] = pred.image_id.astype(str)
    passed = pred.stage1_passed.values == 1
    p5 = pred[[f"stage3_p_grade{k}" for k in range(1, 6)]].values
    g_cal = np.zeros(len(pred), int)
    g_cal[passed] = decode(*corn_from_full(p5[passed]), t, tg)
    same = np.array_equal(g_cal[passed], pred.stage3_grade.values[passed].astype(int))
    same_gate = np.array_equal(passed, pred.stage1_prob.values >= GATE_THRESHOLD)
    print(f"3. Re-decoded Stage III grades match the release: {same}; "
          f"gate decisions match the threshold: {same_gate}")
    if not (same and same_gate):
        sys.exit("The released predictions are not internally consistent; stopping.")

    lab = load_labels(args.labels)
    C = pred.merge(lab, on="image_id", how="left")
    missing = int(C.true_grade.isna().sum())
    if missing:
        ex = ", ".join(C.image_id[C.true_grade.isna()].head(5))
        sys.exit(f"{missing:,} test images have no label in {args.labels} (e.g. {ex}).")
    C = C.rename(columns={"image_id": "stem", "dataset": "source", "stage1_prob": "prob"})
    C["true_grade"] = C.true_grade.astype(int)
    C["passed"] = passed
    for k in range(1, 6):
        C[f"p_grade_{k}"] = C[f"stage3_p_grade{k}"]
    C["final"] = np.where(passed, g_cal, 0)
    C["final_g_argmax"] = C.final_grade_argmax.astype(int)
    print(f"   {len(C):,} test images labelled; {int(passed.sum()):,} passed the gate")

    rng = np.random.default_rng(SEED)
    yt, yp = C.true_grade.values, C.final.values
    yb = (yt > 0).astype(int)
    n = len(C)

    # 4a. Stage I gate (Table II, proposed model)
    tp = int(((yb == 1) & passed).sum())
    tn = int(((yb == 0) & ~passed).sum())
    fp = int(((yb == 0) & passed).sum())
    fn = int(((yb == 1) & ~passed).sum())
    probs = C.prob.values
    gate_auc = auc(yb, probs)
    gate_auc_ci = boot(rng, lambda i: auc(yb[i], probs[i]), n, 500)
    check("Table II", "Gate AUC", gate_auc, "0.929")
    check("Table II", "Gate sensitivity (Grade 0 vs 1-5)", tp / (tp + fn), "0.854")
    check("Table II", "Gate specificity", tn / (tn + fp), "0.887")
    check("Table II", "Fraction of Grade 1 images routed", float(passed[yt == 1].mean()), "0.645")
    check("Table II", "DR images blocked (FN)", fn, "2,459")
    check("Fig. 1", "Images passing the gate", int(passed.sum()), "19,154")
    check("Fig. 1", "Images stopped at the gate", int((~passed).sum()), "39,535")
    check("Section III-A", "Gate AUC, 95% CI", gate_auc_ci, "0.926-0.932")
    check("Section III-A", "Severe and proliferative DR routed (%)", 100 * float(passed[(yt == 3) | (yt == 4)].mean()),
          "97.8")
    check("Section III-A", "Ungradable images routed", f"{int(passed[yt == 5].sum())} of {int((yt == 5).sum())}",
          "345 of 346")
    check("Section III-A", "Test pool stopped at the gate (%)", 100 * float((~passed).mean()), "67.4")
    blocked_dr = ~passed & (yt > 0)
    check("Section III-A", "Blocked DR cases that are mild or moderate (%)",
          100 * float(np.isin(yt[blocked_dr], [1, 2]).mean()), "97")
    for g, paper in enumerate(("0.11", "0.64", "0.90", "0.99", "0.97", "1.00")):
        check("Fig. 6(b)", f"Fraction of Grade {g} routed", float(passed[yt == g].mean()), paper)

    # 4b. Cascade headline (Table VI); bootstrap order as in the paper
    A = C[C.passed & (C.true_grade > 0)]
    qA = qwk_sev(A.true_grade.values, A.final.values, False)[0]
    qA_ci = boot(rng, lambda i: qwk_sev(A.true_grade.values[i], A.final.values[i], False)[0], len(A))
    qB = qwk_sev(yt, yp, True)[0]
    qB_ci = boot(rng, lambda i: qwk_sev(yt[i], yp[i], True)[0], n)
    check("Table VI", "View B, tuned: n", n, "58,689")
    check("Table VI", "View B, tuned: QWK", qB, "0.804")
    check("Table VI", "View B, tuned: 95% CI", qB_ci, "0.798-0.809")
    check("Table VI", "View B, tuned: accuracy", float(np.mean(yp == yt)), "0.814")
    check("Table VI", "View B, tuned: macro-F1", macro_f1(yt, yp, range(6)), "0.659")
    check("Table VI", "View B, argmax: QWK", qwk_sev(yt, C.final_g_argmax.values, True)[0], "0.801")
    check("Section II-F", "View B QWK with Grade 5 on the ordinal axis", qwk(yt, yp, range(6)), "0.814")
    check("Table VI", "View A, tuned: n", len(A), "14,409")
    check("Table VI", "View A, tuned: QWK", qA, "0.732")
    check("Table VI", "View A, tuned: 95% CI", qA_ci, "0.722-0.743")
    check("Table VI", "View A, tuned: accuracy", float(np.mean(A.final == A.true_grade)), "0.743")
    check("Table VI", "View A, argmax: QWK",
          qwk_sev(A.true_grade.values, A.final_g_argmax.values, False)[0], "0.721")
    S3 = C[C.source.isin(["eyepacs", "ddr", "aptos"])]
    A3 = S3[S3.passed & (S3.true_grade > 0)]
    check("Table VI", "EyePACS+DDR+APTOS, View B: n", len(S3), "58,047")
    check("Table VI", "EyePACS+DDR+APTOS, View B: QWK", qwk_sev(S3.true_grade, S3.final, True)[0], "0.800")
    check("Table VI", "EyePACS+DDR+APTOS, View B: accuracy", float(np.mean(S3.final == S3.true_grade)), "0.815")
    check("Table VI", "EyePACS+DDR+APTOS, View A: n", len(A3), "13,992")
    check("Table VI", "EyePACS+DDR+APTOS, View A: QWK", qwk_sev(A3.true_grade, A3.final, False)[0], "0.727")

    # 4c. Error attribution and distance (Fig. 8)
    cor = yp == yt
    missed = int((~passed & (yt > 0)).sum())
    admitted = int((passed & (yt == 0)).sum())
    misgraded = int((passed & (yt > 0) & ~cor).sum())
    total = missed + admitted + misgraded
    check("Fig. 8(b)", "All end-to-end errors", total, "10,907")
    check("Fig. 8(b)", "Stage I: DR missed", missed, "2,459")
    check("Fig. 8(b)", "Stage I: DR missed (%)", 100 * missed / total, "22.5")
    check("Fig. 8(b)", "Stage I: healthy admitted", admitted, "4,745")
    check("Fig. 8(b)", "Stage I: healthy admitted (%)", 100 * admitted / total, "43.5")
    check("Fig. 8(b)", "Stage III: DR mis-graded", misgraded, "3,703")
    check("Fig. 8(b)", "Stage III: DR mis-graded (%)", 100 * misgraded / total, "34.0")
    check("Abstract", "Share of errors caused by screening (%)", 100 * (missed + admitted) / total, "66")
    S = C[C.passed & (C.true_grade > 0)]
    sev = S[(S.true_grade <= 4) & (S.final <= 4) & (S.true_grade != S.final)]
    d = (sev.true_grade - sev.final).abs()
    check("Fig. 8(c)", "Severity errors on Grades 1-4", len(sev), "3,543")
    check("Fig. 8(c)", "One grade from the reference", int((d == 1).sum()), "3,143")
    check("Fig. 8(c)", "One grade from the reference (%)", 100 * (d == 1).mean(), "88.7")
    check("Fig. 8(c)", "Two grades", int((d == 2).sum()), "379")
    check("Fig. 8(c)", "Three grades", int((d == 3).sum()), "21")
    cm = np.zeros((6, 6), int)
    for a, b in zip(yt, yp):
        cm[a, b] += 1
    pd.DataFrame(cm, index=[f"true {i}" for i in range(6)],
                 columns=[f"pred {i}" for i in range(6)]).to_csv(out / "confusion_end_to_end.csv")
    paper_cm = [["37,076", "3,035", "1,492", "12", "100", "106"],
                ["1,442", "1,890", "712", "3", "11", "2"],
                ["951", "1,119", "6,665", "567", "83", "97"],
                ["21", "9", "535", "789", "54", "9"],
                ["44", "10", "284", "156", "1,031", "38"],
                ["1", "2", "4", "0", "8", "331"]]
    for a in range(6):
        for b in range(6):
            check("Fig. 8(a)", f"True G{a}, output G{b}", int(cm[a, b]), paper_cm[a][b])

    # Fig. 7: one correctly graded test image per grade
    for stem, g, p_dr in (("IDRiD_029", 0, "0.20"), ("1232_left", 1, "0.55"), ("20170629112149103", 2, "1.00"),
                          ("13823_left", 3, "1.00"), ("25900_left", 4, "1.00"), ("007-8910-605", 5, "0.99")):
        r = C[C.stem == stem].iloc[0]
        check("Fig. 7", f"{stem}: p(DR)", float(r.prob), p_dr)
        check("Fig. 7", f"{stem}: reference grade", int(r.true_grade), str(g))
        check("Fig. 7", f"{stem}: output grade", int(r.final), str(g))

    # 4d. Per dataset (Table VIII)
    rows = []
    for s_ in DATASETS + ["all"]:
        D = C if s_ == "all" else C[C.source == s_]
        Ad = D[D.passed & (D.true_grade > 0)]
        h0 = D.true_grade == 0
        r = {"dataset": DATASET_NAMES.get(s_, "All"), "n": len(D), "dr": int((D.true_grade > 0).sum()),
             "grade0": int(h0.sum()), "gate_sens": float(D.passed[D.true_grade > 0].mean()),
             "gate_spec": float((~D.passed[h0]).mean()), "admitted_grade0": int((D.passed & h0).sum()),
             "dr_blocked": int((~D.passed & (D.true_grade > 0)).sum()),
             "qwk_A": qwk_sev(Ad.true_grade, Ad.final, False)[0],
             "qwk_B": qwk_sev(D.true_grade, D.final, True)[0],
             "acc_B": float(np.mean(D.final == D.true_grade))}
        rows.append(r)
    pds = pd.DataFrame(rows)
    pds.to_csv(out / "per_dataset.csv", index=False)
    paper8 = {  # n, DR, Grade 0, gate sens, gate spec, DR blocked, QWK A, QWK B, Acc B
        "EyePACS": ("53,576", "14,043", "39,533", "0.837", "0.883", "2,288", "0.721", "0.785", "0.812"),
        "DDR": ("4,105", "2,225", "1,880", "0.931", "0.969", "154", "0.750", "0.878", "0.856"),
        "APTOS 2019": ("366", "167", "199", "0.994", "0.945", "1", "0.745", "0.925", "0.847"),
        "FGADR": ("277", "256", "21", "0.996", "0/21", "1", "0.872", "0.810", "0.776"),
        "Messidor-2": ("262", "108", "154", "0.889", "0.818", "12", "0.782", "0.843", "0.752"),
        "IDRiD": ("103", "69", "34", "0.957", "0.647", "3", "0.747", "0.830", "0.660"),
        "All": ("58,689", "16,868", "41,821", "0.854", "0.887", "2,459", "0.732", "0.804", "0.814"),
    }
    check("Section III-F", "EyePACS share of the test pool (%)", 100 * float((C.source == "eyepacs").mean()), "91")
    for r in rows:
        p = paper8[r["dataset"]]
        k = f"Table VIII, {r['dataset']}"
        check(k, "n", r["n"], p[0])
        check(k, "DR images", r["dr"], p[1])
        check(k, "Grade 0 images", r["grade0"], p[2])
        check(k, "Gate sensitivity", r["gate_sens"], p[3])
        if p[4].startswith("0/"):   # FGADR: no Grade 0 image was stopped
            spec = f"{r['grade0'] - r['admitted_grade0']}/{r['grade0']}"
            check(k, "Gate specificity", spec, p[4])
        else:
            check(k, "Gate specificity", r["gate_spec"], p[4])
        check(k, "DR blocked", r["dr_blocked"], p[5])
        check(k, "QWK, View A", r["qwk_A"], p[6])
        check(k, "QWK, View B", r["qwk_B"], p[7])
        check(k, "Accuracy, View B", r["acc_B"], p[8])

    # 4e. DDR official test split (Table VII); bootstrap order as in the paper
    D = C[C.source == "ddr"]
    G = D[D.true_grade <= 4]
    ytg, ypg = G.true_grade.values, G.final.values
    m5 = ypg == 5
    worst = ypg.copy()
    worst[m5] = np.where(ytg[m5] >= 2, 0, 4)
    q5 = qwk(ytg, np.clip(ypg, 0, 4), range(5))
    q5_ci = boot(rng, lambda i: qwk(ytg[i], np.clip(ypg[i], 0, 4), range(5)), len(G))
    acc6 = float(np.mean(D.final == D.true_grade))
    acc6_ci = boot(rng, lambda i: float(np.mean(D.final.values[i] == D.true_grade.values[i])), len(D))
    check("Table VII", "Six classes: n", len(D), "4,105")
    check("Table VII", "Six classes: accuracy (%)", 100 * acc6, "85.63")
    check("Table VII", "Six classes: accuracy 95% CI (%)", (100 * acc6_ci[0], 100 * acc6_ci[1]), "84.6-86.7")
    check("Table VII", "Six classes: QWK, ungradable as a sixth level",
          qwk(D.true_grade, D.final, range(6)), "0.913")
    for g, paper in ((3, "52.1"), (4, "82.2")):
        check("Table VII", f"Six classes: Grade {g} recall (%)",
              100 * float(np.mean(D.final[D.true_grade == g] == g)), paper)
    check("Table VII", "Five classes: gradable images", len(G), "3,759")
    check("Table VII", "Five classes: ungradable calls on gradable images", int(m5.sum()), "81")
    check("Table VII", "Five classes, ungradable -> Grade 4: accuracy (%)",
          100 * float(np.mean(np.clip(ypg, 0, 4) == ytg)), "85.10")
    check("Table VII", "Five classes, ungradable -> Grade 4: QWK", q5, "0.878")
    check("Table VII", "Five classes, ungradable dropped: n", int((~m5).sum()), "3,678")
    check("Table VII", "Five classes, ungradable dropped: QWK", qwk(ytg[~m5], ypg[~m5], range(5)), "0.892")
    check("Table VII", "Five classes, worst case: accuracy (%)", 100 * float(np.mean(worst == ytg)), "84.70")
    check("Table VII", "Five classes, worst case: QWK", qwk(ytg, worst, range(5)), "0.851")

    # 4f. Gate sweep (Fig. 6(c)-(d)); no rng
    sw = []
    for thr in np.unique(np.round(np.concatenate([[GATE_THRESHOLD], np.arange(0.25, 0.91, 0.025)]), 6)):
        ps_ = (C.prob >= thr) & C.passed
        ypp = np.where(ps_, C.final, 0)
        sw.append({"threshold": thr, "passed": int(ps_.sum()), "qwk": qwk_sev(yt, ypp, True)[0],
                   "dr_blocked": int(((~ps_) & (yt > 0)).sum()), "healthy_admitted": int((ps_ & (yt == 0)).sum())})
    sweep = pd.DataFrame(sw)
    sweep.to_csv(out / "gate_sweep.csv", index=False)
    top = sweep.loc[sweep.qwk.idxmax()]
    check("Fig. 6(c)", "Peak end-to-end QWK of the sweep", top.qwk, "0.825")
    check("Fig. 6(c)", "Threshold at the peak", top.threshold, "0.675")
    check("Fig. 6(d)", "DR images blocked at that threshold", top.dr_blocked, "3,779")
    ps_top = ((C.prob >= top.threshold) & C.passed).values
    yp_top = np.where(ps_top, C.final.values, 0)
    extra = passed & ~ps_top & (yt > 0)
    check("Section III-E", "Extra DR cases blocked at that threshold", int(extra.sum()), "1,320")
    check("Section III-E", "Extra blocked cases that are mild or moderate (%)",
          100 * float(np.isin(yt[extra], [1, 2]).mean()), "94")
    admitted_h = passed & (yt == 0)
    check("Section III-E", "Admitted healthy eyes off by two or more grades (%)",
          100 * float((yp[admitted_h] >= 2).mean()), "36")
    check("Section III-E", "Referral sensitivity at that threshold",
          float(((yt >= 2) & (yp_top >= 2)).sum() / (yt >= 2).sum()), "0.795")
    check("Section III-E", "Referral specificity at that threshold",
          float(((yt < 2) & (yp_top < 2)).sum() / (yt < 2).sum()), "0.969")

    # 4g. Referral (text); bootstrap order as in the paper
    ref_t, ref_p = yt >= 2, yp >= 2
    sens = float((ref_t & ref_p).sum() / ref_t.sum())
    spec = float((~ref_t & ~ref_p).sum() / (~ref_t).sum())
    sens_ci = boot(rng, lambda i: float((ref_t[i] & ref_p[i]).sum() / ref_t[i].sum()), n)
    spec_ci = boot(rng, lambda i: float((~ref_t[i] & ~ref_p[i]).sum() / (~ref_t[i]).sum()), n)
    vt_t, vt_p = (yt >= 3) & (yt <= 4), (yp >= 3) & (yp <= 4)
    check("Section III-D", "Referral sensitivity (Grade >= 2 or ungradable)", sens, "0.832")
    check("Section III-D", "Referral specificity", spec, "0.947")
    check("Section III-D", "Sensitivity for vision-threatening DR (Grades 3-4)",
          float((vt_t & vt_p).sum() / vt_t.sum()), "0.681")
    vt_miss = vt_t & ~vt_p
    check("Section III-D", "Missed vision-threatening DR", int(vt_miss.sum()), "950")
    check("Section III-D", "Missed vision-threatening DR graded moderate", int((vt_miss & (yp == 2)).sum()), "819")

    # 4h. Gradability head on the test pool (Section IV)
    check("Section IV", "Ungradable test images", int((yt == 5).sum()), "346")
    check("Section IV", "Ungradable images output as Grade 5", int(((yt == 5) & (yp == 5)).sum()), "331")
    check("Section IV", "Gradable images output as Grade 5", int(((yt < 5) & (yp == 5)).sum()), "252")

    # ------------------------------------------------------------------ report and files
    numbers = {
        "thresholds": {"t": list(t), "tg": tg, "oof_qwk_tuned": oof_best},
        "gate": {"auc": gate_auc, "auc_ci": gate_auc_ci, "TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "view_A": {"n": len(A), "qwk": qA, "qwk_ci": qA_ci},
        "view_B": {"n": n, "qwk": qB, "qwk_ci": qB_ci, "acc": float(np.mean(yp == yt))},
        "attribution": {"stageI_missed_DR": missed, "stageI_admitted_healthy": admitted,
                        "stageIII_misgraded_DR": misgraded, "total": total},
        "ddr": {"acc6": acc6, "acc6_ci": acc6_ci, "qwk5_ungradable_to_4": q5, "qwk5_ci": q5_ci},
        "referral": {"sens": sens, "sens_ci": sens_ci, "spec": spec, "spec_ci": spec_ci},
    }
    json.dump(numbers, open(out / "reproduced_numbers.json", "w"), indent=2, default=float)
    report = pd.DataFrame(ROWS, columns=["where", "quantity", "paper", "reproduced", "status"])
    report.to_csv(out / "comparison_with_paper.csv", index=False)

    print("\n4. Comparison with the numbers printed in the paper")
    w = max(len(f"{a} | {b}") for a, b, *_ in ROWS)
    for a, b, p, g, s in ROWS:
        print(f"   {(a + ' | ' + b):<{w}}  paper {p:>12}  reproduced {g:>12}  {s}")
    ok = int((report.status == "PASS").sum())
    print(f"\n{ok} of {len(report)} values match the paper. Tables written to {out}/")
    return 0 if ok == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
