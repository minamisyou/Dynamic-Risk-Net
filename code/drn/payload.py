"""Compact per-keyframe export (ego frame, decimetres) for offline human review.

Per frame: "L" lane polylines [flag (0 context / 1 label target / 2 target and ego lane), is_connector,
in_intersection, min CV-TTC, flat polyline, label-unit index or -1], "U" label units
[unit_id, is_ego, kind (l/c/m), in_intersection, min CV-TTC, member lane tokens], "A" actors, map context.
Adds what a viewer needs to overlay lanes on the CAM_FRONT image: camera intrinsics and
the 3x4 transform from the keyframe ego frame to the camera frame.
"""
import base64
import json
import lzma
import os
from typing import Any, Dict, List

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import LineString, Polygon

from .build import curated, load_record
from .lanes import to_ego_frame
from .source import NuscSource

CLS = {"vehicle": "v", "pedestrian": "p", "bicycle": "b", "motorcycle": "m"}
# nuScenes attribute -> one letter, appended to each actor row (empty string when the dataset gives none)
ATTR = {"vehicle.moving": "m", "vehicle.stopped": "s", "vehicle.parked": "k", "cycle.with_rider": "r", "cycle.without_rider": "n",
        "pedestrian.moving": "m", "pedestrian.standing": "s", "pedestrian.sitting_lying_down": "l"}


def _dm(a: np.ndarray) -> List[int]:
    return np.round(np.asarray(a, dtype=float).reshape(-1) * 10.0).astype(int).tolist()


def _simplify_ring(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 4:
        return pts
    return np.asarray(Polygon(pts).simplify(tol).exterior.coords)


def cam_from_ego(src: NuscSource, sample: Dict[str, Any]) -> Dict[str, Any]:
    nusc = src.nusc
    sd_l = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    sd_c = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    pl = nusc.get("ego_pose", sd_l["ego_pose_token"])
    pc = nusc.get("ego_pose", sd_c["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sd_c["calibrated_sensor_token"])
    r_l = Quaternion(pl["rotation"]).rotation_matrix
    r_c = Quaternion(pc["rotation"]).rotation_matrix
    r_s = Quaternion(cs["rotation"]).rotation_matrix
    rot = r_s.T @ r_c.T @ r_l
    trans = r_s.T @ (r_c.T @ (np.asarray(pl["translation"]) - np.asarray(pc["translation"])) - np.asarray(cs["translation"]))
    k = np.asarray(cs["camera_intrinsic"], dtype=float)
    return {"K": [round(float(k[0, 0]), 1), round(float(k[1, 1]), 1), round(float(k[0, 2]), 1), round(float(k[1, 2]), 1)],
            "T": np.round(np.concatenate([rot, trans[:, None]], axis=1), 4).reshape(-1).tolist(),
            "wh": [int(sd_c["width"]), int(sd_c["height"])]}


def frame_payload(rec: Dict[str, Any], cam: Dict[str, Any], view: Dict[str, float]) -> Dict[str, Any]:
    ego = rec["ego"]
    c, s = np.cos(-ego["yaw"]), np.sin(-ego["yaw"])

    def in_view(pts: np.ndarray) -> bool:
        return bool(((pts[:, 0] >= view["x_min"]) & (pts[:, 0] <= view["x_max"]) & (np.abs(pts[:, 1]) <= view["y_abs"])).any())

    unit_idx = {u["unit_id"]: i for i, u in enumerate(rec["label_units"])}
    units = [[u["unit_id"], 1 if u["is_ego_lane"] else 0, {"lane": "l", "lane_connector": "c"}.get(u["kind"], "m"),
              1 if u["in_intersection"] else 0, round(float(u["min_ttc_s"]), 1), u["lane_tokens"]]
             for u in rec["label_units"]]
    lanes = []
    for l in rec["lanes"]:
        pts = to_ego_frame(np.asarray(l["pts"], dtype=float), ego)
        if not l["label_target"] and not in_view(pts):
            continue
        simple = np.asarray(LineString(pts).simplify(0.25 if l["label_target"] else 0.5).coords)
        flag = 2 if (l["label_target"] and l.get("is_ego_lane")) else (1 if l["label_target"] else 0)
        row = [flag, 1 if l["kind"] == "lane_connector" else 0,
               1 if l["in_intersection"] else 0, round(float(l["min_ttc_s"]), 1), _dm(simple)]
        row.append(unit_idx[l["unit_id"]] if l["label_target"] else -1)
        lanes.append(row)
    actors = []
    for a in rec["actors"]:
        x, y = to_ego_frame(np.array([[a["x"], a["y"]]]), ego)[0]
        if not (view["x_min"] - 5 <= x <= view["x_max"] + 5 and abs(y) <= view["y_abs"] + 5):
            continue
        vx, vy = c * a["vx"] - s * a["vy"], s * a["vx"] + c * a["vy"]
        actors.append([CLS.get(a["cls"], "?"), int(round(x * 10)), int(round(y * 10)),
                       int(round(np.degrees(a["yaw"] - ego["yaw"]))), int(round(vx * 10)), int(round(vy * 10)),
                       int(round(a["size"][0] * 10)), int(round(a["size"][1] * 10)), a["visibility"],
                       1 if a.get("is_large") else 0, "".join(ATTR.get(n, "") for n in a.get("attributes", []))])
    mc = rec.get("map_context", {})
    tl = np.asarray(mc.get("traffic_lights", []), dtype=float).reshape(-1, 3)
    tl_e = []
    if len(tl):
        xy = to_ego_frame(tl[:, :2], ego)
        tl_e = [[int(round(p[0] * 10)), int(round(p[1] * 10)), int(round(z * 10))] for p, z in zip(xy, tl[:, 2])
                if -10 <= p[0] <= 90 and abs(p[1]) <= 40]
    t = rec["templates"]
    return {
        "t": rec["sample_token"], "s": rec["scene_name"], "ts": rec["timestamp"], "loc": rec["location"],
        "img": os.path.basename(rec["cam_front"]),
        "tpl": {"I": [int(t["unprotected_intersection"]["hit"]), t["unprotected_intersection"]["n_conflicting"],
                      int(bool(t["unprotected_intersection"]["signalized"]))],
                "M": [int(t["merge_weave"]["hit"]), int(t["merge_weave"]["merge_ahead"]), t["merge_weave"]["n_adjacent"]],
                "P": [int(t["pedestrian_curb"]["hit"]), t["pedestrian_curb"]["n_pedestrians"], t["pedestrian_curb"]["n_on_crossing"]],
                "O": [int(t["occlusion"]["hit"]), t["occlusion"]["n_low_visibility"], int(t["occlusion"]["large_vehicle_blocking"])]},
        "ego": [round(float(ego["speed"]), 1)], "cam": cam, "L": lanes, "U": units, "A": actors, "TL": tl_e,
        "X": [_dm(_simplify_ring(to_ego_frame(np.asarray(p["pts"], dtype=float), ego), 0.5)) for p in mc.get("ped_crossings", [])],
        "I": [_dm(_simplify_ring(to_ego_frame(np.asarray(p["pts"], dtype=float), ego), 1.0)) for p in mc.get("intersections", [])],
        "ttc": round(float(rec["ttc"]["scene_min_ttc_s"]), 1),
    }


def run_payload(cfg: Dict[str, Any], out_file: str, view: Dict[str, float] = None) -> Dict[str, Any]:
    view = view or {"x_min": -25.0, "x_max": 65.0, "y_abs": 35.0}
    src = NuscSource(cfg["nuscenes"]["dataroot"], cfg["nuscenes"]["version"])
    frames = []
    for row in curated(cfg):
        rec = load_record(cfg, row)
        sample = src.nusc.get("sample", rec["sample_token"])
        frames.append(frame_payload(rec, cam_from_ego(src, sample), view))
    raw = json.dumps(frames, separators=(",", ":")).encode("utf-8")
    with open(out_file, "wb") as fh:
        fh.write(base64.b64encode(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)))
    return {"frames": len(frames), "raw_bytes": len(raw), "b64_bytes": os.path.getsize(out_file)}
