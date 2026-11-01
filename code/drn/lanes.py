"""Lane units and lane graph from the nuScenes map expansion.

A *lane unit* is one nuScenes `lane` or `lane_connector` record, clipped to the ROI
around the ego vehicle. A *label unit* is what annotators label: one or more lane units
that carry the same traffic movement (see `assign_label_units`). A public label is
(sample_token, unit_id, risk_level) and expands to every member lane_token. The finer
LaneGCN-style node graph used for training is derived from the same polylines by `lane_nodes()`.
"""
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .compat import patch_matplotlib


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def to_ego_frame(xy: np.ndarray, ego: Dict[str, Any]) -> np.ndarray:
    """Global (N,2) -> ego frame with x forward, y left."""
    c, s = math.cos(-ego["yaw"]), math.sin(-ego["yaw"])
    d = np.asarray(xy, dtype=float).reshape(-1, 2) - np.array([ego["x"], ego["y"]])
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def point_polyline_dist(p: np.ndarray, pts: np.ndarray) -> float:
    """Distance from point p to a polyline (segment-exact)."""
    if len(pts) == 1:
        return float(np.linalg.norm(pts[0] - p))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    denom = (ab ** 2).sum(axis=1)
    denom[denom == 0] = 1e-12
    t = np.clip(((p - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return float(np.linalg.norm(proj - p, axis=1).min())


class MapIndex:
    """Pre-discretised lanes and the polygons the scenario rules need, for one map."""

    def __init__(self, dataroot: str, location: str, resolution_m: float = 1.0,
                 cache_dir: Optional[str] = None):
        self.location = location
        cache = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache = os.path.join(cache_dir, "mapindex_v2_%s_r%.2f.pkl" % (location, resolution_m))
        if cache and os.path.exists(cache):
            with open(cache, "rb") as fh:
                self.__dict__.update(pickle.load(fh))
            return
        patch_matplotlib()
        from nuscenes.map_expansion import arcline_path_utils
        from nuscenes.map_expansion.map_api import NuScenesMap
        nmap = NuScenesMap(dataroot=dataroot, map_name=location)

        self.tokens = []  # type: List[str]
        self.kind = {}  # type: Dict[str, str]
        self.poly = {}  # type: Dict[str, np.ndarray]
        self.incoming = {}  # type: Dict[str, List[str]]
        self.outgoing = {}  # type: Dict[str, List[str]]
        for layer in ("lane", "lane_connector"):
            for rec in getattr(nmap, layer):
                tok = rec["token"]
                path = nmap.arcline_path_3.get(tok)
                if not path:
                    continue
                poses = arcline_path_utils.discretize_lane(path, resolution_meters=resolution_m)
                pts = np.asarray(poses, dtype=float)[:, :2]
                # discretize_lane repeats the joint between consecutive arcline segments.
                keep = np.ones(len(pts), dtype=bool)
                keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6
                pts = pts[keep]
                if len(pts) < 2:
                    continue
                self.tokens.append(tok)
                self.kind[tok] = layer
                self.poly[tok] = pts
                conn = nmap.connectivity.get(tok, {})
                self.incoming[tok] = list(conn.get("incoming", []))
                self.outgoing[tok] = list(conn.get("outgoing", []))
        self.bbox = np.asarray([[self.poly[t][:, 0].min(), self.poly[t][:, 1].min(),
                                 self.poly[t][:, 0].max(), self.poly[t][:, 1].max()] for t in self.tokens])

        def polys(records):
            out = []
            for rec in records:
                try:
                    pg = nmap.extract_polygon(rec["polygon_token"])
                except Exception:
                    continue
                if pg.is_valid and not pg.is_empty:
                    out.append((rec["token"], np.asarray(pg.exterior.coords)[:, :2]))
            return out

        self.intersections = polys([r for r in nmap.road_segment if r.get("is_intersection")])
        self.ped_crossings = polys(nmap.ped_crossing)
        # nuScenes maps give traffic light positions only; there is no signal state anywhere in nuScenes.
        self.traffic_lights = np.asarray(
            [[r["pose"]["tx"], r["pose"]["ty"], r["pose"].get("tz", 0.0)] for r in nmap.traffic_light if "pose" in r],
            dtype=float).reshape(-1, 3)
        if cache:
            with open(cache, "wb") as fh:
                pickle.dump(self.__dict__, fh)

    # ------------------------------------------------------------- polygons
    @staticmethod
    def _near(polys: List[Tuple[str, np.ndarray]], x: float, y: float, radius: float) -> List[Tuple[str, Polygon]]:
        out = []
        for tok, coords in polys:
            if (coords[:, 0].min() - radius <= x <= coords[:, 0].max() + radius and
                    coords[:, 1].min() - radius <= y <= coords[:, 1].max() + radius):
                out.append((tok, Polygon(coords)))
        return out

    def intersections_near(self, x: float, y: float, radius: float) -> List[Tuple[str, Polygon]]:
        return self._near(self.intersections, x, y, radius)

    def ped_crossings_near(self, x: float, y: float, radius: float) -> List[Tuple[str, Polygon]]:
        return self._near(self.ped_crossings, x, y, radius)

    def map_context(self, ego: Dict[str, Any], radius: float) -> Dict[str, Any]:
        """Intersection outlines, pedestrian crossings and traffic light positions near the ego (global frame)."""
        ex, ey = ego["x"], ego["y"]

        def outlines(polys):
            out = []
            for tok, pg in polys:
                if pg.distance(Point(ex, ey)) <= radius:
                    simple = pg.simplify(0.3)
                    out.append({"token": tok, "pts": np.asarray(simple.exterior.coords)[:, :2]})
            return out

        tl = self.traffic_lights
        near = tl[np.hypot(tl[:, 0] - ex, tl[:, 1] - ey) <= radius] if len(tl) else tl
        return {"intersections": outlines(self.intersections_near(ex, ey, radius)),
                "ped_crossings": outlines(self.ped_crossings_near(ex, ey, radius)),
                "traffic_lights": near}

    # ---------------------------------------------------------------- lanes
    def lane_units(self, ego: Dict[str, Any], roi_radius_m: float, label_roi: Dict[str, float],
                   lane_cfg: Dict[str, float], unit_cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ex, ey = ego["x"], ego["y"]
        r = roi_radius_m
        cand = np.where((self.bbox[:, 0] <= ex + r) & (self.bbox[:, 2] >= ex - r) &
                        (self.bbox[:, 1] <= ey + r) & (self.bbox[:, 3] >= ey - r))[0]
        inter = self.intersections_near(ex, ey, r)
        units = []
        for ci in cand:
            tok = self.tokens[ci]
            pts = self.poly[tok]
            inside = np.linalg.norm(pts - np.array([ex, ey]), axis=1) <= r
            run = _longest_run(inside)
            if run is None or run[1] - run[0] < 2:
                continue
            clip = pts[run[0]:run[1]]
            mid = clip[len(clip) // 2]
            in_junction = any(pg.contains(Point(mid[0], mid[1])) for _, pg in inter)
            in_int = self.kind[tok] == "lane_connector" or in_junction
            d = clip[-1] - clip[0]
            ego_pts = to_ego_frame(clip, ego)
            box = ((ego_pts[:, 0] >= label_roi["x_min"]) & (ego_pts[:, 0] <= label_roi["x_max"]) &
                   (np.abs(ego_pts[:, 1]) <= label_roi["y_abs"]))
            brun = _longest_run(box)
            len_in_box = polyline_length(ego_pts[brun[0]:brun[1]]) if brun else 0.0
            units.append({
                "lane_token": tok, "kind": self.kind[tok],
                "pts": clip, "length_m": polyline_length(clip),
                "heading": float(math.atan2(d[1], d[0])),
                "in_intersection": bool(in_int), "in_junction": bool(in_junction),
                "n_incoming": len(self.incoming.get(tok, [])),
                "label_target": bool(len_in_box >= label_roi["min_length_m"]),
                "starts_in_roi": bool(run[0] == 0), "ends_in_roi": bool(run[1] == len(pts)),
            })
        # The lane the ego is driving on is always a label target, however short it is.
        best, best_d = None, 2.5
        for u in units:
            if abs(wrap_angle(u["heading"] - ego["yaw"])) > math.radians(45.0):
                continue
            d = point_polyline_dist(np.array([ex, ey]), u["pts"])
            if d < best_d:
                best, best_d = u, d
        for u in units:
            u["is_ego_lane"] = u is best
        if best is not None:
            best["label_target"] = True
        present = {u["lane_token"] for u in units}
        for u in units:
            u["succ"] = [t for t in self.outgoing.get(u["lane_token"], []) if t in present]
            u["pred"] = [t for t in self.incoming.get(u["lane_token"], []) if t in present]
        _assign_neighbors(units, lane_cfg["neighbor_max_dist_m"], math.radians(lane_cfg["neighbor_max_angle_deg"]))
        assign_label_units(units, self, ego, unit_cfg or {})
        return units


def _longest_run(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Longest contiguous True run as (start, end_exclusive)."""
    best, start = None, None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _assign_neighbors(units: List[Dict[str, Any]], max_dist: float, max_angle: float) -> None:
    """Left/right neighbours: roughly parallel lane units running side by side."""
    for u in units:
        u["left"], u["right"] = [], []
    boxes = [(u["pts"][:, 0].min(), u["pts"][:, 1].min(), u["pts"][:, 0].max(), u["pts"][:, 1].max()) for u in units]
    lines = {}  # type: Dict[int, LineString]
    for i, u in enumerate(units):
        if u["kind"] != "lane":
            continue
        samples = u["pts"][:: max(1, len(u["pts"]) // 8)]
        for j, v in enumerate(units):
            if i == j or v["kind"] != "lane":
                continue
            if abs(wrap_angle(u["heading"] - v["heading"])) > max_angle:
                continue
            a, b = boxes[i], boxes[j]
            if a[0] > b[2] + max_dist or b[0] > a[2] + max_dist or a[1] > b[3] + max_dist or b[1] > a[3] + max_dist:
                continue
            if v["lane_token"] in u["succ"] or v["lane_token"] in u["pred"]:
                continue
            # Sample u and measure the lateral offset to v; require overlap along most of the sample.
            dists = np.array([point_polyline_dist(p, v["pts"]) for p in samples])
            close = dists <= max_dist
            if close.mean() < 0.5 or dists[close].mean() < 1.0:
                continue
            if j not in lines:
                lines[j] = LineString(v["pts"])
            p = samples[int(np.argmin(dists))]
            q = np.asarray(lines[j].interpolate(lines[j].project(Point(p[0], p[1]))).coords[0])
            cross = math.cos(u["heading"]) * (q[1] - p[1]) - math.sin(u["heading"]) * (q[0] - p[0])
            (u["left"] if cross > 0 else u["right"]).append(v["lane_token"])


def _ends(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Start point, end point, start heading and end heading of a polyline."""
    k = min(3, len(pts) - 1)
    d0, d1 = pts[k] - pts[0], pts[-1] - pts[-1 - k]
    return pts[0], pts[-1], math.atan2(d0[1], d0[0]), math.atan2(d1[1], d1[0])


def assign_label_units(units: List[Dict[str, Any]], mapidx: "MapIndex", ego: Dict[str, Any],
                       cfg: Dict[str, Any]) -> None:
    """Group label-target lane units into label units and write `unit_id` on every lane unit.

    Four merges, each switchable in config (`label_units`):
      chain     a lane and its only continuation (one outgoing, one incoming in the whole map),
                outside junction areas
      ego_path  the ego's lane and the only way onward from it (up to two steps: a connector and the
                lane after it), so the ego's unit is "the ego's path ahead"
      parallel  lanes running side by side in the same direction (the left/right neighbours)
      movement  connectors of the same turning movement: start points and end points both within
                `movement_end_dist_m`, start and end headings within `movement_heading_deg`
                (measured on the full, unclipped map polylines)
    The ego's path only takes part in `chain` and `ego_path`, so the ego's own unit is never widened sideways.
    `unit_id` is the smallest member lane_token; lane units that are not label targets get None.
    Units are ordered ego unit first, then by distance from the ego (`unit_rank`).
    """
    targets = [u for u in units if u["label_target"]]
    by_tok = {u["lane_token"]: u for u in targets}
    parent = {t: t for t in by_tok}

    def find(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if cfg.get("chain", True):
        for u in targets:
            out = mapidx.outgoing.get(u["lane_token"], [])
            if len(out) != 1 or out[0] not in by_tok:
                continue
            v = by_tok[out[0]]
            if mapidx.incoming.get(v["lane_token"], []) != [u["lane_token"]]:
                continue
            if u["in_junction"] or v["in_junction"]:
                continue
            union(u["lane_token"], v["lane_token"])
    ego_path = {u["lane_token"] for u in targets if u["is_ego_lane"]}
    if cfg.get("ego_path", True):
        cur = next((u for u in targets if u["is_ego_lane"]), None)
        for _ in range(2):
            out = mapidx.outgoing.get(cur["lane_token"], []) if cur is not None else []
            if len(out) != 1 or out[0] not in by_tok:
                break
            union(cur["lane_token"], out[0])
            ego_path.add(out[0])
            cur = by_tok[out[0]]
    if cfg.get("parallel", True):
        for u in targets:
            if u["lane_token"] in ego_path or u["kind"] != "lane":
                continue
            for t in u["left"] + u["right"]:
                if t in by_tok and t not in ego_path:
                    union(u["lane_token"], t)
    if cfg.get("movement", True):
        dist = float(cfg.get("movement_end_dist_m", 5.0))
        ang = math.radians(float(cfg.get("movement_heading_deg", 30.0)))
        conns = [(u, _ends(mapidx.poly[u["lane_token"]])) for u in targets
                 if u["kind"] == "lane_connector" and u["lane_token"] not in ego_path]
        for i in range(len(conns)):
            for j in range(i + 1, len(conns)):
                a, b = conns[i][1], conns[j][1]
                if (np.linalg.norm(a[0] - b[0]) <= dist and np.linalg.norm(a[1] - b[1]) <= dist and
                        abs(wrap_angle(a[2] - b[2])) <= ang and abs(wrap_angle(a[3] - b[3])) <= ang):
                    union(conns[i][0]["lane_token"], conns[j][0]["lane_token"])
    for u in units:
        u["unit_id"] = find(u["lane_token"]) if u["label_target"] else None
    here = np.array([ego["x"], ego["y"]])
    rank = {}  # type: Dict[str, Tuple[int, float]]
    for u in targets:
        d = point_polyline_dist(here, u["pts"])
        ego_first = 0 if u["is_ego_lane"] else 1
        cur = rank.get(u["unit_id"])
        rank[u["unit_id"]] = (ego_first, d) if cur is None else (min(cur[0], ego_first), min(cur[1], d))
    order = sorted(rank, key=lambda t: (rank[t][0], rank[t][1], t))
    for u in units:
        u["unit_rank"] = order.index(u["unit_id"]) if u["unit_id"] is not None else None


def label_units(lanes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Label units of a frame, in annotation order, from the `unit_id` written on its lane units."""
    groups = {}  # type: Dict[str, List[Dict[str, Any]]]
    for l in lanes:
        if l.get("unit_id") is not None:
            groups.setdefault(l["unit_id"], []).append(l)
    out = []
    for uid, members in sorted(groups.items(), key=lambda kv: kv[1][0]["unit_rank"]):
        kinds = {m["kind"] for m in members}
        ttc = [m["min_ttc_s"] for m in members if "min_ttc_s" in m]
        best = min(members, key=lambda m: m.get("min_ttc_s", float("inf")))
        out.append({
            "unit_id": uid, "lane_tokens": sorted(m["lane_token"] for m in members),
            "kind": kinds.pop() if len(kinds) == 1 else "mixed",
            "in_intersection": any(m["in_intersection"] for m in members),
            "is_ego_lane": any(m["is_ego_lane"] for m in members),
            "length_m": float(sum(m["length_m"] for m in members)),
            "min_ttc_s": float(min(ttc)) if ttc else None, "min_ttc_pair": best.get("min_ttc_pair"),
            "loose_ttc_s": min((float(m["loose_ttc_s"]) for m in members if m.get("loose_ttc_s") is not None), default=None),
        })
    return out


def lane_nodes(units: List[Dict[str, Any]], ego: Dict[str, Any], spacing_m: float) -> Dict[str, Any]:
    """LaneGCN-style node graph (ego frame) built from lane units.

    Returns the keys LaneGCN's data.py produces: ctrs, feats, lane_idcs, intersect,
    pre_pairs/suc_pairs/left_pairs/right_pairs (lane level) and pre/suc (node level).
    """
    ctrs, feats, lane_idcs, intersect = [], [], [], []
    node_ranges = []
    for li, u in enumerate(units):
        pts = to_ego_frame(u["pts"], ego)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(round(s[-1] / spacing_m)), 1)
        grid = np.linspace(0.0, s[-1], n + 1)
        res = np.stack([np.interp(grid, s, pts[:, 0]), np.interp(grid, s, pts[:, 1])], axis=1)
        start = len(ctrs)
        for k in range(n):
            ctrs.append((res[k] + res[k + 1]) / 2.0)
            feats.append(res[k + 1] - res[k])
            lane_idcs.append(li)
            intersect.append(1.0 if u["in_intersection"] else 0.0)
        node_ranges.append((start, len(ctrs)))
    idx = {u["lane_token"]: i for i, u in enumerate(units)}
    pre_u, pre_v, suc_u, suc_v = [], [], [], []
    for li, (a, b) in enumerate(node_ranges):
        for k in range(a, b):
            if k > a:
                pre_u.append(k)
                pre_v.append(k - 1)
            if k < b - 1:
                suc_u.append(k)
                suc_v.append(k + 1)
        for t in units[li]["pred"]:
            if units[li]["starts_in_roi"] and units[idx[t]]["ends_in_roi"]:
                pre_u.append(a)
                pre_v.append(node_ranges[idx[t]][1] - 1)
        for t in units[li]["succ"]:
            if units[li]["ends_in_roi"] and units[idx[t]]["starts_in_roi"]:
                suc_u.append(b - 1)
                suc_v.append(node_ranges[idx[t]][0])

    def pairs(key):
        return np.asarray([[i, idx[t]] for i, u in enumerate(units) for t in u[key]], dtype=np.int64).reshape(-1, 2)

    return {
        "ctrs": np.asarray(ctrs, dtype=np.float32).reshape(-1, 2),
        "feats": np.asarray(feats, dtype=np.float32).reshape(-1, 2),
        "lane_idcs": np.asarray(lane_idcs, dtype=np.int64),
        "intersect": np.asarray(intersect, dtype=np.float32),
        "num_nodes": len(ctrs),
        "pre": {"u": np.asarray(pre_u, dtype=np.int64), "v": np.asarray(pre_v, dtype=np.int64)},
        "suc": {"u": np.asarray(suc_u, dtype=np.int64), "v": np.asarray(suc_v, dtype=np.int64)},
        "pre_pairs": pairs("pred"), "suc_pairs": pairs("succ"),
        "left_pairs": pairs("left"), "right_pairs": pairs("right"),
        "lane_tokens": [u["lane_token"] for u in units],
    }
