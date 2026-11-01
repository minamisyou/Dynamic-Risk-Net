"""Metrics of the paper's section IV: RCA, weighted F1, per-level recall / precision (Table 5), minADE / minFDE / MR
(K = 6, best-of-K, 2 m miss threshold), Table 7 (predicted level by scene CV-TTC range) and Table 8 (mean min CV-TTC
by ground-truth level)."""
from typing import Dict, List

import numpy as np

from .data import LEVELS

T7_EDGES = (5.0, 3.0, 1.5)      # rows: > 5, 3-5, 1.5-3, <= 1.5 s


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    m = np.zeros((LEVELS, LEVELS), np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    m = confusion(y_true, y_pred)
    tp = np.diag(m).astype(float); support = m.sum(1); predicted = m.sum(0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
    precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), np.nan)
    f1 = np.where(np.nan_to_num(recall) + np.nan_to_num(precision) > 0, 2 * np.nan_to_num(recall) * np.nan_to_num(precision) /
                  np.maximum(np.nan_to_num(recall) + np.nan_to_num(precision), 1e-9), 0.0)
    wf1 = float((f1 * support).sum() / max(support.sum(), 1))
    return {"n": int(len(y_true)), "rca": float(tp.sum() / max(len(y_true), 1)), "weighted_f1": wf1,
            "recall": [None if np.isnan(r) else round(float(r), 4) for r in recall],
            "precision": [None if np.isnan(p) else round(float(p), 4) for p in precision],
            "support": support.tolist(), "confusion": m.tolist(),
            "within_one": float((np.abs(y_true - y_pred) <= 1).mean()) if len(y_true) else None}


def trajectory_metrics(reg: np.ndarray, cls: np.ndarray, gt: np.ndarray, has: np.ndarray, miss_m: float = 2.0) -> Dict:
    """reg (A, K, F, 2), gt (A, F, 2), has (A, F)."""
    valid = has[:, -1]
    if valid.sum() == 0:
        return {"n_actors": 0, "minADE": None, "minFDE": None, "MR": None}
    reg, gt, has = reg[valid], gt[valid], has[valid]
    d = np.linalg.norm(reg - gt[:, None], axis=-1)                                      # (A, K, F)
    ade = (d * has[:, None]).sum(-1) / np.maximum(has.sum(-1)[:, None], 1)              # (A, K)
    fde = d[:, :, -1]
    best = fde.argmin(1)
    i = np.arange(len(gt))
    return {"n_actors": int(len(gt)), "minADE": float(ade.min(1).mean()), "minFDE": float(fde[i, best].mean()),
            "MR": float((fde[i, best] > miss_m).mean() * 100.0), "minADE_at_minFDE": float(ade[i, best].mean())}


def _row(t: float) -> int:
    return 0 if t > T7_EDGES[0] else 1 if t > T7_EDGES[1] else 2 if t > T7_EDGES[2] else 3


def table7(frame_of_unit: np.ndarray, time: np.ndarray, pred: np.ndarray, cap: float, mode: str = "scene") -> Dict:
    """Predicted level (grouped L0-2 / L3 / L4, %) by CV-TTC range.

    mode "scene": each keyframe is placed in the row of its minimum CV-TTC and the lane units holding that minimum are
    counted ("relevant lane segments"). mode "lane": every lane unit with a CV-TTC below the cap is counted in its own row."""
    has_t = time < cap - 1e-9
    M = np.zeros((4, LEVELS), np.int64)
    if mode == "scene":
        for f in np.unique(frame_of_unit):
            sel = (frame_of_unit == f) & has_t
            if not sel.any():
                continue
            tmin = time[sel].min()
            hold = sel & (np.abs(time - tmin) <= 0.05)
            np.add.at(M, (np.full(int(hold.sum()), _row(tmin), np.int64), pred[hold].astype(np.int64)), 1)
    else:
        np.add.at(M, (np.array([_row(t) for t in time[has_t]], np.int64), pred[has_t].astype(np.int64)), 1)
    tot = M.sum(1, keepdims=True).astype(float)
    pct = np.where(tot > 0, 100.0 * M / np.maximum(tot, 1), np.nan)
    rows = [[float(np.nansum(r[:3])) if tot[k, 0] > 0 else None, float(r[3]) if tot[k, 0] > 0 else None, float(r[4]) if tot[k, 0] > 0 else None]
            for k, r in enumerate(pct)]
    return {"rows": [">5 s", "3-5 s", "1.5-3 s", "<=1.5 s"], "columns": ["L0-2 %", "L3 %", "L4 %"], "values": rows, "n": tot[:, 0].astype(int).tolist(), "mode": mode}


def table8(level: np.ndarray, time: np.ndarray, cap: float) -> Dict:
    t = np.minimum(time, cap)
    return {"mean_min_cv_ttc_by_level": [float(t[level == k].mean()) if (level == k).any() else None for k in range(LEVELS)],
            "n": [int((level == k).sum()) for k in range(LEVELS)], "cap_s": cap}
