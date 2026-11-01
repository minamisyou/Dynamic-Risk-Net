"""Time to conflict per lane unit.

Two computed quantities, both constant-velocity:
  min_ttc_s    strict CV-TTC: the first time two road users (discs) actually touch.
  loose_ttc_s  distance / closing rate of any approaching pair, whether or not their courses meet.
The annotated part (estimates where neither describes the conflict) is combined with them in `hybrid_ttc`.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .lanes import point_polyline_dist


def actor_radius(size: List[float]) -> float:
    """Collision radius of an actor approximated by a disc of half its width (size = w, l, h)."""
    return 0.5 * float(size[0])


def pair_ttc(p1: np.ndarray, v1: np.ndarray, r1: float,
             p2: np.ndarray, v2: np.ndarray, r2: float,
             margin: float = 0.0) -> Optional[Tuple[float, np.ndarray]]:
    """First time two constant-velocity discs touch, and the midpoint where they do.

    Returns None when they never touch. Already-overlapping discs give TTC 0.
    """
    p = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    v = np.asarray(v2, dtype=float) - np.asarray(v1, dtype=float)
    radius = r1 + r2 + margin
    c = float(p @ p) - radius * radius
    if c <= 0.0:
        # Already inside the safety margin. Only physically overlapping discs count as TTC 0;
        # otherwise fall back to the bare radii so side-by-side agents are not "colliding".
        radius = r1 + r2
        c = float(p @ p) - radius * radius
        if c <= 0.0:
            # Annotated boxes that touch (a pedestrian beside a parked car, a truck and its trailer)
            # are only a collision event when the two are actually closing on each other.
            if float(p @ v) < 0.0 and float(v @ v) > 0.25:
                return 0.0, (np.asarray(p1, dtype=float) + np.asarray(p2, dtype=float)) / 2.0
            return None
    a = float(v @ v)
    if a < 1e-9:
        return None
    b = 2.0 * float(p @ v)
    disc = b * b - 4.0 * a * c
    if disc < 0.0 or b >= 0.0:  # no contact, or moving apart
        return None
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    if t < 0.0:
        return None
    q1 = np.asarray(p1, dtype=float) + np.asarray(v1, dtype=float) * t
    q2 = np.asarray(p2, dtype=float) + np.asarray(v2, dtype=float) * t
    return float(t), (q1 + q2) / 2.0


def lane_min_ttc(units: List[Dict[str, Any]], ego: Dict[str, Any], actors: List[Dict[str, Any]],
                 cfg: Dict[str, float]) -> Dict[str, Any]:
    """Minimum pairwise CV-TTC per lane unit.

    A pair's TTC is attributed to every lane unit whose centreline passes within
    `lane_dist_m` of the predicted contact point. Lanes with no predicted contact
    inside the horizon get `horizon_s` (i.e. "no collision within the horizon").
    Pedestrian-pedestrian pairs are skipped.
    """
    horizon = float(cfg["horizon_s"])
    agents = [{"id": "ego", "p": np.array([ego["x"], ego["y"]]), "v": np.array([ego["vx"], ego["vy"]]),
               "r": actor_radius(ego["size"]), "cls": "vehicle"}]
    for a in actors:
        agents.append({"id": a["instance_token"], "p": np.array([a["x"], a["y"]]),
                       "v": np.array([a["vx"], a["vy"]]), "r": actor_radius(a["size"]), "cls": a.get("cls")})
    events = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if agents[i]["cls"] == "pedestrian" and agents[j]["cls"] == "pedestrian":
                continue  # pedestrian-pedestrian contact is not a driving risk
            res = pair_ttc(agents[i]["p"], agents[i]["v"], agents[i]["r"],
                           agents[j]["p"], agents[j]["v"], agents[j]["r"], cfg["margin_m"])
            if res is not None and res[0] <= horizon:
                events.append((res[0], res[1], agents[i]["id"], agents[j]["id"]))
    events.sort(key=lambda e: e[0])
    per_lane = {}
    d = float(cfg["lane_dist_m"])
    for u in units:
        best, pair = horizon, None
        x0, y0 = u["pts"][:, 0].min() - d, u["pts"][:, 1].min() - d
        x1, y1 = u["pts"][:, 0].max() + d, u["pts"][:, 1].max() + d
        for t, pt, a_id, b_id in events:  # earliest first: stop at the first event on this lane
            if x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1 and point_polyline_dist(pt, u["pts"]) <= d:
                best, pair = t, [a_id, b_id]
                break
        per_lane[u["lane_token"]] = {"min_ttc_s": float(best), "pair": pair}
    scene_min = min([e[0] for e in events], default=horizon)
    return {"per_lane": per_lane, "scene_min_ttc_s": float(scene_min), "n_events": len(events)}


def lane_loose_ttc(units: List[Dict[str, Any]], ego: Dict[str, Any], actors: List[Dict[str, Any]],
                   cfg: Dict[str, Any]) -> Dict[str, float]:
    """Distance / closing rate per lane unit ("how soon is the nearest approaching pair here").

    For every pair whose separation is shrinking and is at most `loose.max_pair_m`, the time is the centre
    distance divided by the rate at which it shrinks. No collision-course test is made, so two vehicles about to
    pass each other in neighbouring lanes get a short time as well. A pair's time is attributed to every lane unit
    within `loose.lane_dist_m` of either road user or of their midpoint at that time. Lanes with no such pair get
    `horizon_s`. Pedestrian-pedestrian pairs are skipped.
    """
    horizon = float(cfg["horizon_s"])
    lc = cfg.get("loose") or {}
    max_pair, d = float(lc.get("max_pair_m", 30.0)), float(lc.get("lane_dist_m", 3.0))
    agents = [(np.array([ego["x"], ego["y"]], dtype=float), np.array([ego["vx"], ego["vy"]], dtype=float), "vehicle")]
    agents += [(np.array([a["x"], a["y"]], dtype=float), np.array([a["vx"], a["vy"]], dtype=float), a.get("cls")) for a in actors]
    events = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if agents[i][2] == "pedestrian" and agents[j][2] == "pedestrian":
                continue
            p, v = agents[j][0] - agents[i][0], agents[j][1] - agents[i][1]
            rng, pv = float(np.hypot(p[0], p[1])), float(p @ v)
            if pv >= -1e-6 or rng > max_pair or rng < 1e-6:
                continue
            t = rng * rng / -pv
            if t > horizon:
                continue
            tt = min(t, horizon)
            mid = 0.5 * ((agents[i][0] + agents[i][1] * tt) + (agents[j][0] + agents[j][1] * tt))
            events.append((t, (agents[i][0], agents[j][0], mid)))
    events.sort(key=lambda e: e[0])
    out = {}
    for u in units:
        best = horizon
        x0, y0 = u["pts"][:, 0].min() - d, u["pts"][:, 1].min() - d
        x1, y1 = u["pts"][:, 0].max() + d, u["pts"][:, 1].max() + d
        for t, pts in events:
            if any(x0 <= q[0] <= x1 and y0 <= q[1] <= y1 and point_polyline_dist(q, u["pts"]) <= d for q in pts):
                best = t
                break
        out[u["lane_token"]] = float(best)
    return out


def consensus_time(values: List[Optional[float]], bases: List[str]) -> Tuple[Optional[float], str]:
    """Annotators' time to conflict for one unit: the median when at least two gave a value, else none.

    The basis is "calc" when more of those who gave a value accepted the computed one than estimated their own.
    """
    given = sorted(float(v) for v in values if v is not None)
    if len(given) < 2 or len(given) * 2 <= len(values):
        return None, "none"
    mid = len(given) // 2
    val = given[mid] if len(given) % 2 else 0.5 * (given[mid - 1] + given[mid])
    n_calc, n_est = sum(1 for b in bases if b == "calc"), sum(1 for b in bases if b == "est")
    return float(val), ("calc" if n_calc > n_est else "est")


def hybrid_ttc(computed: Optional[float], judged: Optional[float], basis: str, horizon: float) -> float:
    """Computed where the computation applies, annotated where it does not.

    Where the annotators estimated a time themselves (basis "est": a waiting party, a curved path, a road user the
    computation does not know) that estimate is used; everywhere else the computed value stands.
    """
    if basis == "est" and judged is not None:
        return float(min(judged, horizon))
    return float(horizon if computed is None else min(computed, horizon))
