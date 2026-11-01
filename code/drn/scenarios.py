"""Rule-based mesoscopic scenario templates (paper Sec. IV-A1a).

Each rule returns {"hit": bool, ...evidence}. Rules only *select* scenes for human
annotation; they never produce risk labels. All thresholds come from config.yaml.
"""
import math
from typing import Any, Dict, List, Optional

import numpy as np
from shapely.geometry import Point, Polygon

from .lanes import MapIndex, point_polyline_dist, to_ego_frame, wrap_angle

TEMPLATES = ("unprotected_intersection", "merge_weave", "pedestrian_curb", "occlusion")
WHEELED = ("vehicle", "motorcycle", "bicycle")


def _ego_xy(actor: Dict[str, Any], ego: Dict[str, Any]) -> np.ndarray:
    return to_ego_frame(np.array([[actor["x"], actor["y"]]]), ego)[0]


def _motion_heading(a: Dict[str, Any]) -> float:
    return math.atan2(a["vy"], a["vx"])


def _corridor_global(ego: Dict[str, Any], ahead: float, half_width: float) -> Polygon:
    c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
    corners = [(0.0, -half_width), (ahead, -half_width), (ahead, half_width), (0.0, half_width)]
    return Polygon([(ego["x"] + c * x - s * y, ego["y"] + s * x + c * y) for x, y in corners])


def intersection_rule(ego, actors, mapidx: MapIndex, cfg, moving_speed: float) -> Dict[str, Any]:
    """Ego is about to enter an intersection while a moving vehicle on a conflicting course is at it."""
    corridor = _corridor_global(ego, cfg["ahead_m"], cfg["corridor_half_width_m"])
    best = {"hit": False, "n_conflicting": 0, "intersection_token": None, "signalized": None}
    for tok, pg in mapidx.intersections_near(ego["x"], ego["y"], cfg["ahead_m"] + 20.0):
        if not pg.intersects(corridor):
            continue
        signalized = False
        if len(mapidx.traffic_lights):
            d = [pg.distance(Point(x, y)) for x, y in mapidx.traffic_lights[:, :2]
                 if abs(x - ego["x"]) < 150 and abs(y - ego["y"]) < 150]
            signalized = bool(d and min(d) <= cfg["signal_radius_m"])
        if cfg.get("exclude_signalized") and signalized:
            continue
        n = 0
        for a in actors:
            if a["cls"] not in WHEELED or a["speed"] < moving_speed:
                continue
            if pg.distance(Point(a["x"], a["y"])) > cfg["near_m"]:
                continue
            if abs(wrap_angle(_motion_heading(a) - ego["yaw"])) >= math.radians(cfg["min_heading_diff_deg"]):
                n += 1
        if n > best["n_conflicting"]:
            best = {"hit": True, "n_conflicting": n, "intersection_token": tok, "signalized": signalized}
    return best


def _ego_lane(ego, units) -> Optional[Dict[str, Any]]:
    best, best_d = None, 3.0
    p = np.array([ego["x"], ego["y"]])
    for u in units:
        if abs(wrap_angle(u["heading"] - ego["yaw"])) > math.radians(45.0):
            continue
        d = point_polyline_dist(p, u["pts"])
        if d < best_d:
            best, best_d = u, d
    return best


def merge_rule(ego, actors, units, cfg, moving_speed: float) -> Dict[str, Any]:
    """A merge point lies ahead on the ego route with traffic on another incoming lane,
    or several vehicles travel alongside the ego in adjacent lanes."""
    by_tok = {u["lane_token"]: u for u in units}
    out = {"hit": False, "merge_ahead": False, "n_branch_vehicles": 0, "n_adjacent": 0}
    start = _ego_lane(ego, units)
    if start is not None:
        seen, frontier, route = {start["lane_token"]}, [start], []
        while frontier:  # lanes reachable ahead within ahead_m
            u = frontier.pop()
            route.append(u)
            for t in u["succ"]:
                v = by_tok.get(t)
                if v is None or t in seen:
                    continue
                if to_ego_frame(v["pts"][:1], ego)[0][0] <= cfg["ahead_m"]:
                    seen.add(t)
                    frontier.append(v)
        for u in route:
            if u["n_incoming"] < 2:
                continue
            mp = u["pts"][0]  # merge point = where the incoming lanes join
            mx = to_ego_frame(mp[None], ego)[0][0]
            if not (0.0 < mx <= cfg["ahead_m"]):
                continue
            branches = [by_tok[t] for t in u["pred"] if t in by_tok and t not in seen]
            n = 0
            for a in actors:
                if a["cls"] not in WHEELED or a["speed"] < moving_speed:
                    continue
                pa = np.array([a["x"], a["y"]])
                if np.linalg.norm(pa - mp) > cfg["branch_m"]:
                    continue
                if any(point_polyline_dist(pa, b["pts"]) <= 2.0 for b in branches):
                    n += 1
            if n > 0:
                out["merge_ahead"] = True
                out["n_branch_vehicles"] = max(out["n_branch_vehicles"], n)
    n_adj = 0
    for a in actors:
        if a["cls"] not in WHEELED or a["speed"] < moving_speed:
            continue
        x, y = _ego_xy(a, ego)
        if not (cfg["adjacent_lon_min_m"] <= x <= cfg["adjacent_lon_max_m"]):
            continue
        if not (cfg["adjacent_lat_min_m"] <= abs(y) <= cfg["adjacent_lat_max_m"]):
            continue
        if abs(wrap_angle(_motion_heading(a) - ego["yaw"])) <= math.radians(cfg["adjacent_heading_deg"]):
            n_adj += 1
    out["n_adjacent"] = n_adj
    out["hit"] = bool(out["merge_ahead"] or n_adj >= cfg["min_adjacent"])
    return out


def pedestrian_rule(ego, actors, units, mapidx: MapIndex, cfg) -> Dict[str, Any]:
    """Pedestrians ahead of the ego that are close to a lane centreline."""
    crossings = mapidx.ped_crossings_near(ego["x"], ego["y"], cfg["ahead_m"] + 20.0)
    n, n_cross, dmin = 0, 0, None
    for a in actors:
        if a["cls"] != "pedestrian":
            continue
        x, y = _ego_xy(a, ego)
        if not (0.0 <= x <= cfg["ahead_m"] and abs(y) <= cfg["lateral_m"]):
            continue
        p = np.array([a["x"], a["y"]])
        d = min([point_polyline_dist(p, u["pts"]) for u in units], default=float("inf"))
        if d <= cfg["max_lane_dist_m"]:
            n += 1
            dmin = d if dmin is None else min(dmin, d)
            if any(pg.contains(Point(a["x"], a["y"])) for _, pg in crossings):
                n_cross += 1
    return {"hit": bool(n >= cfg["min_count"]), "n_pedestrians": n, "n_on_crossing": n_cross,
            "min_lane_dist_m": None if dmin is None else float(dmin)}


def occlusion_rule(ego, actors, cfg, intersection_ahead: bool, moving_speed: float) -> Dict[str, Any]:
    """Several poorly visible road users ahead that could enter the ego's path (moving agents or
    pedestrians; parked occluded cars do not count), or a large vehicle blocking the view toward an intersection."""
    low = set(str(t) for t in cfg["low_visibility_tokens"])
    n_low, large = 0, False
    for a in actors:
        x, y = _ego_xy(a, ego)
        if x <= 0.0:
            continue
        relevant = a["cls"] == "pedestrian" or a["speed"] >= moving_speed
        if relevant and math.hypot(x, y) <= cfg["radius_m"] and a["visibility"] in low:
            n_low += 1
        if a.get("is_large") and x <= cfg["large_ahead_m"] and abs(y) <= cfg["large_lateral_m"]:
            large = True
    blocked = bool(large and intersection_ahead)
    return {"hit": bool(n_low >= cfg["min_occluded"] or blocked), "n_low_visibility": n_low,
            "large_vehicle_blocking": blocked}


def evaluate(ego: Dict[str, Any], actors: List[Dict[str, Any]], units: List[Dict[str, Any]],
             mapidx: MapIndex, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ms = float(cfg["moving_speed_mps"])
    ic = cfg["intersection"]
    corridor = _corridor_global(ego, ic["ahead_m"], ic["corridor_half_width_m"])
    int_ahead = any(pg.intersects(corridor)
                    for _, pg in mapidx.intersections_near(ego["x"], ego["y"], ic["ahead_m"] + 20.0))
    return {
        "unprotected_intersection": intersection_rule(ego, actors, mapidx, ic, ms),
        "merge_weave": merge_rule(ego, actors, units, cfg["merge"], ms),
        "pedestrian_curb": pedestrian_rule(ego, actors, units, mapidx, cfg["pedestrian"]),
        "occlusion": occlusion_rule(ego, actors, cfg["occlusion"], int_ahead, ms),
    }
