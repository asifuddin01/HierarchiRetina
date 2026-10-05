"""OOF-only fitting of the four LG-DRG decision thresholds.

Coordinate ascent maximising Stage A severity QWK (grades 1-4, true 5 dropped, predicted 5 ->
4) of the consecutive CORN decode on the pooled out-of-fold predictions. No test image
influences the result. On the released OOF file this returns t=(0.38, 0.60, 0.19), t_g=0.10.
"""
from __future__ import annotations

import numpy as np

from .corn import corn_from_full, decode_consecutive
from .metrics import qwk_sev

T_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)   # each CORN threshold
TG_GRID = np.round(np.arange(0.10, 0.901, 0.02), 2)  # gradability threshold


def calibrate_thresholds(y_true, P5, grid=T_GRID, tg_grid=TG_GRID, rounds: int = 6,
                         tol: float = 1e-7, decode=decode_consecutive, verbose: bool = True):
    """Fit ``(t1, t2, t3), t_g`` on OOF data.

    ``y_true``: true grades 1..5; ``P5``: OOF 5-way probabilities (columns p_grade_1..5).
    Each round sweeps t1, t2, t3 in turn over ``grid`` and then t_g over ``tg_grid``,
    starting from (0.5, 0.5, 0.5), 0.5; stops after a round without improvement.

    Returns ``(t, tg, best_qwk, start_qwk)``.
    """
    y = np.asarray(y_true, int)
    pp = corn_from_full(P5)

    def obj(t, tg):
        return qwk_sev(y, decode(*pp, t, tg), include_healthy=False)[0]

    t = [0.5, 0.5, 0.5]
    tg = 0.5
    best = start = obj(tuple(t), tg)
    for r in range(rounds):
        improved = False
        for i in range(3):
            bv = t[i]
            for v in grid:
                t[i] = float(v)
                s = obj(tuple(t), tg)
                if s > best + tol:
                    best, bv, improved = s, float(v), True
            t[i] = bv
        for v in tg_grid:
            s = obj(tuple(t), float(v))
            if s > best + tol:
                best, tg, improved = s, float(v), True
        if verbose:
            print(f"  round {r + 1}: t=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) tg={tg:.2f} "
                  f"QWK={best:.4f}")
        if not improved:
            break
    return tuple(t), tg, best, start
