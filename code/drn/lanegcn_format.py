"""Scene record -> LaneGCN-style training sample (numpy only).

Coordinates are in the ego frame at prediction time. The lane graph has the base keys
LaneGCN's data.py builds before `preprocess()` (dilated neighbours and node-level
left/right are added by LaneGCN's own preprocess code).
"""
from typing import Any, Dict, Optional

import numpy as np

from .lanes import lane_nodes, to_ego_frame

IGNORE = -1  # risk target for lane units without a human label
ACTOR_CLASSES = ("ego", "vehicle", "pedestrian", "bicycle", "motorcycle")
KIN_FIELDS = ("vx", "vy", "ax", "ay", "cos_dyaw", "sin_dyaw", "width", "length", "height", "is_large")
LANE_ATTR_FIELDS = ("is_connector", "in_junction", "merge")


def _rot(vx: float, vy: float, yaw: float):
    c, s = np.cos(-yaw), np.sin(-yaw)
    return c * vx - s * vy, s * vx + c * vy


def past_motion(pos: np.ndarray, mask: np.ndarray, hz: float = 10.0, span_s: float = 0.5):
    
    k = max(int(round(span_s * hz)), 1)
    A, T = mask.shape
    vel = np.zeros((A, 2), np.float32); acc = np.zeros((A, 2), np.float32)
    if T > k:
        ok = mask[:, -1] & mask[:, -1 - k]
        vel[ok] = (pos[ok, -1] - pos[ok, -1 - k]) / (k / hz)
    if T > 2 * k:
        ok2 = mask[:, -1] & mask[:, -1 - k] & mask[:, -1 - 2 * k]
        vprev = (pos[ok2, -1 - k] - pos[ok2, -1 - 2 * k]) / (k / hz)
        acc[ok2] = (vel[ok2] - vprev) / (k / hz)
    return vel, acc


def actor_state(agents, ego, vel=None, acc=None):
    """Current state per agent in the ego frame (paper Eq. 1: heading psi_j, box d_j, class c_j; Eq. 6: |v|, |a|)."""
    kin = np.zeros((len(agents), len(KIN_FIELDS)), np.float32)
    cls = np.zeros(len(agents), np.int64)
    for i, a in enumerate(agents):
        if vel is not None:
            (vx, vy), (ax, ay) = vel[i], acc[i]
        else:
            vx, vy = (a.get("vx", 0.0), a.get("vy", 0.0)) if a.get("vel_valid", True) else (0.0, 0.0)
            ax, ay = (a.get("ax", 0.0), a.get("ay", 0.0)) if a.get("acc_valid", False) else (0.0, 0.0)
            vx, vy = _rot(vx, vy, ego["yaw"]); ax, ay = _rot(ax, ay, ego["yaw"])
        dy = a.get("yaw", ego["yaw"]) - ego["yaw"]
        w, l, h = (list(a.get("size", [0, 0, 0])) + [0, 0, 0])[:3]
        kin[i] = [vx, vy, ax, ay, np.cos(dy), np.sin(dy), w, l, h, 1.0 if a.get("is_large") else 0.0]
        cls[i] = 0 if i == 0 else ACTOR_CLASSES.index(a.get("cls", "vehicle")) if a.get("cls") in ACTOR_CLASSES else 1
    return kin, cls


def lane_attr(units, lane_idcs):
    """Per lane node: lane attributes from the map (paper III-B: lane attributes such as intersection membership)."""
    per_lane = np.array([[1.0 if u.get("kind") == "lane_connector" else 0.0, 1.0 if u.get("in_junction") else 0.0,
                          1.0 if u.get("n_incoming", 0) > 1 else 0.0] for u in units],
                        np.float32).reshape(-1, len(LANE_ATTR_FIELDS))
    return per_lane[np.asarray(lane_idcs, np.int64)] if len(per_lane) else np.zeros((0, len(LANE_ATTR_FIELDS)), np.float32)


def lanegcn_sample(rec: Dict[str, Any], labels: Optional[Dict[str, int]], node_spacing_m: float, hz: float = 10.0,
                   causal_kin: bool = True) -> Dict[str, Any]:
    """labels maps unit_id -> available level for this keyframe; unlabelled units remain IGNORE."""
    ego = rec["ego"]
    units = [dict(l, pts=np.asarray(l["pts"], dtype=float)) for l in rec["lanes"]]
    graph = lane_nodes(units, ego, node_spacing_m)
    agents = [ego] + list(rec["actors"])
    n_hist, n_fut = len(ego["hist"]), len(ego["fut"])
    feats = np.zeros((len(agents), n_hist, 3), np.float32)
    ctrs = np.zeros((len(agents), 2), np.float32)
    gt = np.zeros((len(agents), n_fut, 2), np.float32)
    has = np.zeros((len(agents), n_fut), bool)
    for i, a in enumerate(agents):
        hist = to_ego_frame(np.asarray(a["hist"], dtype=float), ego)
        mask = np.asarray(a["hist_mask"], dtype=bool)
        step = np.zeros_like(hist)
        step[1:] = hist[1:] - hist[:-1]
        step[1:][~(mask[1:] & mask[:-1])] = 0.0
        feats[i, :, :2] = step
        feats[i, :, 2] = mask
        ctrs[i] = hist[-1]
        gt[i] = to_ego_frame(np.asarray(a["fut"], dtype=float), ego)
        has[i] = np.asarray(a["fut_mask"], dtype=bool)
    pos = np.zeros((len(agents), n_hist, 2), np.float32)
    hmask = np.zeros((len(agents), n_hist), bool)
    for i, a in enumerate(agents):
        m = np.asarray(a["hist_mask"], dtype=bool)
        hmask[i] = m
        pos[i][m] = to_ego_frame(np.asarray(a["hist"], dtype=float), ego)[m]
    if causal_kin:
        vel, acc = past_motion(pos, hmask, hz)
        kin, cls = actor_state(agents, ego, vel, acc)
    else:
        kin, cls = actor_state(agents, ego)
    graph["attr"] = lane_attr(units, graph["lane_idcs"])
    lane_risk = np.full(len(units), IGNORE, np.int64)
    if labels:
        for li, u in enumerate(units):
            if u.get("unit_id") in labels:
                lane_risk[li] = int(labels[u["unit_id"]])
    c, s = np.cos(-ego["yaw"]), np.sin(-ego["yaw"])
    return {
        "sample_token": rec["sample_token"],
        "feats": feats, "ctrs": ctrs, "gt_preds": gt, "has_preds": has,
        "hist_pos": pos,                                      # (A, T, 2) past positions, ego frame (paper Eq. 1: T_j positions)
        "kin": kin, "cls": cls,
        "kin_source": "past" if causal_kin else "record",
        "orig": np.array([ego["x"], ego["y"]], np.float32),
        "rot": np.array([[c, -s], [s, c]], np.float32),
        "graph": graph,
        "risk_targets_lane": lane_risk,                       # one per lane unit
        "risk_targets": lane_risk[graph["lane_idcs"]],        # one per lane node
    }
