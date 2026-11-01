"""nuScenes access layer: ego state, actor states, and 10 Hz history/future tracks.

Everything that touches nuscenes-devkit tables lives here, so the rest of the
pipeline works on plain dicts and numpy arrays.
"""
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from .compat import patch_matplotlib

CLASS_RULES = [
    ("human.pedestrian", "pedestrian"),
    ("vehicle.bicycle", "bicycle"),
    ("vehicle.motorcycle", "motorcycle"),
    ("vehicle.", "vehicle"),
]
LARGE_VEHICLE_PREFIXES = ("vehicle.truck", "vehicle.bus", "vehicle.trailer", "vehicle.construction")


def coarse_class(category_name: str) -> Optional[str]:
    """Map a nuScenes category to pedestrian / bicycle / motorcycle / vehicle (None = ignored)."""
    for prefix, cls in CLASS_RULES:
        if category_name.startswith(prefix):
            return cls
    return None


def quat_yaw(rotation: List[float]) -> float:
    return float(Quaternion(rotation).yaw_pitch_roll[0])


def resample_track(ts: np.ndarray, xy: np.ndarray, t0: float, hz: float,
                   n_hist: int, n_fut: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate a keyframe track onto a fixed-rate grid around t0.

    History has n_hist steps ending at t0 (inclusive); future has n_fut steps starting
    one step after t0. Steps outside the observed time span are masked, never extrapolated.
    """
    k = np.arange(-(n_hist - 1), n_fut + 1)
    grid = t0 + k / float(hz)
    valid = (grid >= ts[0] - 1e-6) & (grid <= ts[-1] + 1e-6)
    out = np.stack([np.interp(grid, ts, xy[:, 0]), np.interp(grid, ts, xy[:, 1])], axis=1)
    out[~valid] = 0.0
    return out[:n_hist], valid[:n_hist], out[n_hist:], valid[n_hist:]


class NuscSource:
    def __init__(self, dataroot: str, version: str, verbose: bool = False):
        patch_matplotlib()
        from nuscenes.nuscenes import NuScenes
        self.dataroot = dataroot
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self._track_cache_scene = None  # type: Optional[str]
        self._tracks = {}  # type: Dict[str, Dict[str, Any]]
        self._ego_track = None  # type: Optional[Dict[str, np.ndarray]]

    # ------------------------------------------------------------------ scenes
    def scenes(self) -> List[Dict[str, Any]]:
        return list(self.nusc.scene)

    def location(self, scene: Dict[str, Any]) -> str:
        return self.nusc.get("log", scene["log_token"])["location"]

    def samples(self, scene: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            yield sample
            token = sample["next"]

    def cam_front_path(self, sample: Dict[str, Any]) -> str:
        sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        return os.path.join(self.dataroot, sd["filename"])

    def cam_front_relpath(self, sample: Dict[str, Any]) -> str:
        return self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])["filename"]

    # ------------------------------------------------------------------ tracks
    def _ego_pose(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        return self.nusc.get("ego_pose", sd["ego_pose_token"])

    def _prepare_scene(self, scene: Dict[str, Any]) -> None:
        if self._track_cache_scene == scene["token"]:
            return
        ts, xy, yaw = [], [], []
        tracks = {}  # type: Dict[str, Dict[str, list]]
        for sample in self.samples(scene):
            t = sample["timestamp"] * 1e-6
            pose = self._ego_pose(sample)
            ts.append(t)
            xy.append(pose["translation"][:2])
            yaw.append(quat_yaw(pose["rotation"]))
            for ann_token in sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)
                tr = tracks.setdefault(ann["instance_token"], {"t": [], "xy": [], "ann": []})
                tr["t"].append(t)
                tr["xy"].append(ann["translation"][:2])
                tr["ann"].append(ann_token)
        self._ego_track = {"t": np.asarray(ts), "xy": np.asarray(xy, dtype=float), "yaw": np.asarray(yaw)}
        self._tracks = {k: {"t": np.asarray(v["t"]), "xy": np.asarray(v["xy"], dtype=float), "ann": v["ann"]}
                        for k, v in tracks.items()}
        self._track_cache_scene = scene["token"]

    @staticmethod
    def _fd_velocity(ts: np.ndarray, xy: np.ndarray, i: int) -> Tuple[float, float, bool]:
        """Centred finite-difference velocity at index i (one-sided at the ends)."""
        lo, hi = max(i - 1, 0), min(i + 1, len(ts) - 1)
        dt = ts[hi] - ts[lo]
        if hi == lo or dt <= 0 or dt > 1.5:
            return 0.0, 0.0, False
        v = (xy[hi] - xy[lo]) / dt
        return float(v[0]), float(v[1]), True

    # ------------------------------------------------------------------ states
    def ego_state(self, scene: Dict[str, Any], sample: Dict[str, Any], traj_cfg: Dict[str, Any]) -> Dict[str, Any]:
        self._prepare_scene(scene)
        et = self._ego_track
        t0 = sample["timestamp"] * 1e-6
        i = int(np.argmin(np.abs(et["t"] - t0)))
        vx, vy, vel_ok = self._fd_velocity(et["t"], et["xy"], i)
        n_hist = int(round(traj_cfg["history_s"] * traj_cfg["hz"]))
        n_fut = int(round(traj_cfg["future_s"] * traj_cfg["hz"]))
        hist, hmask, fut, fmask = resample_track(et["t"], et["xy"], t0, traj_cfg["hz"], n_hist, n_fut)
        pose = self._ego_pose(sample)
        return {
            "x": float(pose["translation"][0]), "y": float(pose["translation"][1]),
            "yaw": quat_yaw(pose["rotation"]),
            "vx": vx, "vy": vy, "speed": float(math.hypot(vx, vy)), "vel_valid": vel_ok,
            "size": [1.73, 4.08, 1.56],  # nuScenes Renault Zoe (w, l, h)
            "hist": hist.tolist(), "hist_mask": hmask.astype(int).tolist(),
            "fut": fut.tolist(), "fut_mask": fmask.astype(int).tolist(),
        }

    def actors(self, scene: Dict[str, Any], sample: Dict[str, Any], ego: Dict[str, Any],
               radius_m: float, traj_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._prepare_scene(scene)
        t0 = sample["timestamp"] * 1e-6
        n_hist = int(round(traj_cfg["history_s"] * traj_cfg["hz"]))
        n_fut = int(round(traj_cfg["future_s"] * traj_cfg["hz"]))
        out = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cls = coarse_class(ann["category_name"])
            if cls is None:
                continue
            x, y = float(ann["translation"][0]), float(ann["translation"][1])
            if math.hypot(x - ego["x"], y - ego["y"]) > radius_m:
                continue
            tr = self._tracks[ann["instance_token"]]
            i = tr["ann"].index(ann_token)
            vel = self.nusc.box_velocity(ann_token)[:2]
            if np.any(np.isnan(vel)):
                vx, vy, vel_ok = self._fd_velocity(tr["t"], tr["xy"], i)
            else:
                vx, vy, vel_ok = float(vel[0]), float(vel[1]), True
            ax = ay = 0.0
            acc_ok = False
            if i > 0:
                # Same finite-difference estimator at both steps, so the difference is consistent.
                cvx, cvy, c_ok = self._fd_velocity(tr["t"], tr["xy"], i)
                pvx, pvy, p_ok = self._fd_velocity(tr["t"], tr["xy"], i - 1)
                dt = tr["t"][i] - tr["t"][i - 1]
                if c_ok and p_ok and 0 < dt <= 1.5:
                    ax, ay, acc_ok = (cvx - pvx) / dt, (cvy - pvy) / dt, True
            hist, hmask, fut, fmask = resample_track(tr["t"], tr["xy"], t0, traj_cfg["hz"], n_hist, n_fut)
            out.append({
                "instance_token": ann["instance_token"], "ann_token": ann_token,
                "category": ann["category_name"], "cls": cls,
                "is_large": ann["category_name"].startswith(LARGE_VEHICLE_PREFIXES),
                "x": x, "y": y, "yaw": quat_yaw(ann["rotation"]),
                "vx": vx, "vy": vy, "speed": float(math.hypot(vx, vy)), "vel_valid": vel_ok,
                "ax": float(ax), "ay": float(ay), "acc_valid": acc_ok,
                "size": [float(s) for s in ann["size"]],  # (w, l, h)
                "visibility": str(ann["visibility_token"]),
                # nuScenes attributes, e.g. vehicle.moving / vehicle.stopped / vehicle.parked, cycle.with_rider
                "attributes": [self.nusc.get("attribute", t)["name"] for t in ann.get("attribute_tokens", [])],
                "num_lidar_pts": int(ann["num_lidar_pts"]),
                "hist": hist.tolist(), "hist_mask": hmask.astype(int).tolist(),
                "fut": fut.tolist(), "fut_mask": fmask.astype(int).tolist(),
            })
        return out
