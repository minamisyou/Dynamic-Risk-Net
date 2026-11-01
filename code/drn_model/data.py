"""Samples, caching, batching and the class-aware sampler.

A sample is the dict `drn.lanegcn_format.lanegcn_sample()` returns, plus the fields the evaluation needs:
  unit_of_lane   lane -> label-unit index (-1 for lanes outside the labelled set)
  unit_level     per label unit, the ground-truth level (0-4)
  unit_time      per label unit, the CV-TTC used by Tables 7 and 8 (seconds; cap when there is none)
  split          train / val / test
"""
import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

IGNORE = -1
LEVELS = 5


# ----------------------------------------------------------------------------- labels
def read_labels(path: str) -> Dict[str, Dict[str, Any]]:
    """labels csv -> {sample_token: {"units": {unit_id: level}, "time": {unit_id: seconds or None}, "split": ...}}.

    Accepts the lane-row annotation layout in `labels.csv`. Blank labels represent units without a strict majority
    and remain IGNORE; their time and split fields are still retained."""
    out = {}  # type: Dict[str, Dict[str, Any]]
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            d = out.setdefault(r["sample_token"], {"units": {}, "time": {}, "split": r.get("split", ""), "row": r})
            if r.get("split", "") != d["split"]:
                raise ValueError("Conflicting split assignments within sample %s" % r["sample_token"])
            label = (r.get("label") or "").strip()
            if r["unit_id"] in d.get("rows", {}):
                previous = (d["rows"][r["unit_id"]].get("label") or "").strip()
                if label != previous:
                    raise ValueError("Conflicting lane-row labels for sample/unit %s/%s" % (r["sample_token"], r["unit_id"]))
            if label:
                value = int(label)
                if value not in range(LEVELS):
                    raise ValueError("Risk label must be in 0..4: %r" % label)
                d["units"][r["unit_id"]] = value
            d["row"] = r
            d.setdefault("rows", {})[r["unit_id"]] = r
    return out


def unit_time(row: Dict[str, str], column: str, cap: float) -> float:
    v = row.get(column, "")
    try:
        t = float(v)
    except (TypeError, ValueError):
        return cap
    return min(t, cap)


# ----------------------------------------------------------------------------- prepare
def prepare(out_dir: str, labels_path: str, cache_path: str, node_spacing_m: float = 2.0, time_column: Optional[str] = None,
            time_cap: Optional[float] = None, splits_path: Optional[str] = None) -> Dict[str, Any]:
    """Scene records + labels -> a cache; blank labels stay ignored by risk training and evaluation."""
    with open(labels_path, newline="", encoding="utf-8") as fh:
        columns = csv.DictReader(fh).fieldnames or []
    if time_column is None:
        time_column = next((c for c in ("cv_ttc_s",) if c in columns), None)
    if not time_column or time_column not in columns:
        raise ValueError("labels.csv must contain cv_ttc_s, or specify time_column")
    if time_cap is None:
        time_cap = 10.0
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from drn.lanegcn_format import lanegcn_sample

    labels = read_labels(labels_path)
    split_override = json.load(open(splits_path, encoding="utf-8")) if splits_path else None
    samples = []
    scene_splits = {}
    paths = sorted(glob.glob(os.path.join(out_dir, "scenes", "*", "*.json")))
    for p in paths:
        tok = os.path.basename(p)[:-5]
        if tok not in labels:
            continue
        rec = json.load(open(p, encoding="utf-8"))
        lab = labels[tok]
        s = lanegcn_sample(rec, lab["units"], node_spacing_m)
        unit_ids = [u["unit_id"] for u in rec["label_units"]]
        uidx = {u: i for i, u in enumerate(unit_ids)}
        s["unit_of_lane"] = np.array([uidx.get(l.get("unit_id"), -1) if l.get("label_target") else -1 for l in rec["lanes"]], np.int64)
        s["unit_level"] = np.array([lab["units"].get(u, IGNORE) for u in unit_ids], np.int64)
        s["unit_time"] = np.array([unit_time(lab["rows"][u], time_column, time_cap) if u in lab["rows"] else time_cap for u in unit_ids], np.float32)
        s["unit_ids"] = unit_ids
        s["scene_name"] = rec["scene_name"]
        scene_key = rec.get("scene_token") or rec["scene_name"]
        chosen = lab["split"]
        if split_override is not None:
            matches = [split_override[k] for k in (rec.get("scene_token"), rec["scene_name"]) if k in split_override]
            if not matches:
                raise ValueError("--splits: no split for scene %s" % scene_key)
            if len(set(matches)) != 1:
                raise ValueError("--splits: conflicting token/name assignments for %s" % scene_key)
            chosen = matches[0]
        if chosen not in ("train", "val", "test"):
            raise ValueError("Invalid or missing split for scene %s: %r" % (scene_key, chosen))
        if scene_key in scene_splits and scene_splits[scene_key] != chosen:
            raise ValueError("Scene assigned to multiple splits: %s" % scene_key)
        scene_splits[scene_key] = chosen
        s["split"] = chosen
        s["lane_tokens"] = s["graph"]["lane_tokens"]
        samples.append(s)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save({"samples": samples, "node_spacing_m": node_spacing_m, "time_column": time_column, "time_cap": time_cap,
                "labels_path": os.path.abspath(labels_path), "out_dir": os.path.abspath(out_dir)}, cache_path)
    counts = {k: sum(1 for s in samples if s["split"] == k) for k in ("train", "val", "test")}
    return {"samples": len(samples), "records_seen": len(paths), "splits": counts, "cache": cache_path}


def relabel(cache_path: str, labels_path: str, out_path: str, time_column: str = "cv_ttc_s") -> Dict[str, Any]:
    """Copy of a cache with the unit levels / times (and split) taken from another labels csv of the same keyframes and
    label units). Keyframes or units missing from the CSV are an error."""
    data = load_cache(cache_path)
    with open(labels_path, newline="", encoding="utf-8") as fh:
        if time_column not in (csv.DictReader(fh).fieldnames or []):
            raise ValueError("Missing TTC column: %s" % time_column)
    labels = read_labels(labels_path)
    cap = float(data.get("time_cap", 10.0))
    changed = 0
    scene_splits = {}
    for s in data["samples"]:
        lab = labels[s["sample_token"]]
        chosen = lab["split"] or s["split"]
        scene_key = s.get("scene_token") or s["scene_name"]
        if chosen not in ("train", "val", "test"):
            raise ValueError("Invalid split for scene %s: %r" % (scene_key, chosen))
        if scene_key in scene_splits and scene_splits[scene_key] != chosen:
            raise ValueError("Scene assigned to multiple splits: %s" % scene_key)
        scene_splits[scene_key] = chosen
        new = np.array([lab["units"].get(u, IGNORE) for u in s["unit_ids"]], np.int64)
        changed += int((new != s["unit_level"]).sum())
        s["unit_level"] = new
        s["unit_time"] = np.array([unit_time(lab["rows"][u], time_column, cap) for u in s["unit_ids"]], np.float32)
        s["split"] = chosen
    data["labels_path"] = os.path.abspath(labels_path)
    data["time_column"] = time_column
    torch.save(data, out_path)
    return {"samples": len(data["samples"]), "units_changed": changed, "cache": out_path}


SIGNAL_GROUPS = ("red", "yellow", "green", "light_unseen", "no_light")
SIGNAL_OF_STATE = {"red": "red", "red_and_green_arrow": "red", "yellow": "yellow", "green": "green", "green_arrow": "green",
                   "not_visible": "light_unseen", "unclear": "light_unseen", "off_or_flashing": "light_unseen"}


def load_cache(path: str) -> Dict[str, Any]:
    try:
        data = torch.load(path, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        data = torch.load(path)

    if "samples" in data and any(x.get("kin_source") != "past" for x in data["samples"]):
        raise ValueError("Cache uses outdated actor kinematics; create a new cache with prepare or convert with causal-kin")
    return data


def causal_kin(cache_in: str, cache_out: str, hz: float = 10.0) -> Dict[str, Any]:
    
    from drn.lanegcn_format import past_motion
    try:
        data = torch.load(cache_in, weights_only=False)
    except TypeError:
        data = torch.load(cache_in)
    dv = []
    for s in data["samples"]:
        pos = np.asarray(s["hist_pos"], np.float32); mask = np.asarray(s["feats"])[:, :, 2] > 0.5
        vel, acc = past_motion(pos, mask, hz)
        kin = np.array(s["kin"], np.float32, copy=True)
        dv.append(np.linalg.norm(kin[:, 0:2] - vel, axis=1))
        kin[:, 0:2] = vel; kin[:, 2:4] = acc
        s["kin"] = kin; s["kin_source"] = "past"
    data["kin_source"] = "past"
    torch.save(data, cache_out)
    dv = np.concatenate(dv) if dv else np.zeros(0)
    return {"samples": len(data["samples"]), "actors": int(len(dv)), "speed_change_mean_mps": float(dv.mean()) if len(dv) else None,
            "speed_change_p90_mps": float(np.percentile(dv, 90)) if len(dv) else None, "cache": cache_out}


# ----------------------------------------------------------------------------- dataset / batching
class RiskDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, Any]], split: Optional[str] = None):
        self.samples = [s for s in samples if split is None or s["split"] == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def class_counts(self) -> np.ndarray:
        c = np.zeros(LEVELS, np.int64)
        for s in self.samples:
            lv = s["unit_level"]
            for k in lv[lv >= 0]:
                c[k] += 1
        return c

    def frame_weights(self) -> np.ndarray:
        """Class-aware oversampling: a keyframe is drawn in proportion to the rarest level it contains."""
        c = np.maximum(self.class_counts(), 1).astype(float)
        inv = c.sum() / c
        w = np.zeros(len(self.samples))
        for i, s in enumerate(self.samples):
            lv = s["unit_level"]; lv = lv[lv >= 0]
            w[i] = inv[lv].max() if len(lv) else inv[0]
        return w / w.sum()


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate the actors and lane nodes of a batch; `*_batch` tensors say which sample each row belongs to."""
    B = len(samples)
    actor_feats, actor_ctrs, actor_batch, gt, has = [], [], [], [], []
    actor_kin, actor_cls, actor_pos, node_attr = [], [], [], []
    node_ctrs, node_feats, node_batch, node_lane, node_inter = [], [], [], [], []
    pre_u, pre_v, suc_u, suc_v, left_u, left_v, right_u, right_v = [], [], [], [], [], [], [], []
    lane_batch, lane_unit, unit_level, unit_time, unit_batch = [], [], [], [], []
    n_nodes = n_lanes = n_units = 0
    for b, s in enumerate(samples):
        g = s["graph"]
        A = s["feats"].shape[0]
        actor_feats.append(_t(s["feats"])); actor_ctrs.append(_t(s["ctrs"])); actor_batch.append(torch.full((A,), b, dtype=torch.long))
        gt.append(_t(s["gt_preds"])); has.append(_t(s["has_preds"], torch.bool))
        if "kin" in s:
            actor_kin.append(_t(s["kin"])); actor_cls.append(_t(s["cls"], torch.long)); actor_pos.append(_t(s["hist_pos"]))
            node_attr.append(_t(s["graph"]["attr"]))
        N = int(g["num_nodes"]); L = len(g["lane_tokens"])
        node_ctrs.append(_t(g["ctrs"])); node_feats.append(_t(g["feats"])); node_batch.append(torch.full((N,), b, dtype=torch.long))
        node_lane.append(_t(g["lane_idcs"], torch.long) + n_lanes); node_inter.append(_t(g["intersect"]))
        pre_u.append(_t(g["pre"]["u"], torch.long) + n_nodes); pre_v.append(_t(g["pre"]["v"], torch.long) + n_nodes)
        suc_u.append(_t(g["suc"]["u"], torch.long) + n_nodes); suc_v.append(_t(g["suc"]["v"], torch.long) + n_nodes)
        lp = np.asarray(g["left_pairs"]).reshape(-1, 2); rp = np.asarray(g["right_pairs"]).reshape(-1, 2)
        left_u.append(_t(lp[:, 0], torch.long) + n_lanes); left_v.append(_t(lp[:, 1], torch.long) + n_lanes)
        right_u.append(_t(rp[:, 0], torch.long) + n_lanes); right_v.append(_t(rp[:, 1], torch.long) + n_lanes)
        lane_batch.append(torch.full((L,), b, dtype=torch.long))
        uol = np.asarray(s["unit_of_lane"], np.int64); lane_unit.append(_t(np.where(uol >= 0, uol + n_units, -1), torch.long))
        U = len(s["unit_level"])
        unit_level.append(_t(s["unit_level"], torch.long)); unit_time.append(_t(s["unit_time"])); unit_batch.append(torch.full((U,), b, dtype=torch.long))
        n_nodes += N; n_lanes += L; n_units += U
    cat = torch.cat
    return {"B": B, "actor_feats": cat(actor_feats), "actor_ctrs": cat(actor_ctrs), "actor_batch": cat(actor_batch),
            "gt_preds": cat(gt), "has_preds": cat(has),
            "node_ctrs": cat(node_ctrs), "node_feats": cat(node_feats), "node_batch": cat(node_batch), "node_lane": cat(node_lane),
            "node_inter": cat(node_inter), "pre_u": cat(pre_u), "pre_v": cat(pre_v), "suc_u": cat(suc_u), "suc_v": cat(suc_v),
            "left_u": cat(left_u), "left_v": cat(left_v), "right_u": cat(right_u), "right_v": cat(right_v),
            "lane_batch": cat(lane_batch), "lane_unit": cat(lane_unit), "n_lanes": n_lanes, "n_units": n_units,
            "unit_level": cat(unit_level), "unit_time": cat(unit_time), "unit_batch": cat(unit_batch),
            "sample_tokens": [s["sample_token"] for s in samples],
            **({"actor_kin": cat(actor_kin), "actor_cls": cat(actor_cls), "actor_pos": cat(actor_pos), "node_attr": cat(node_attr)}
               if len(actor_kin) == B else {})}


def to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(ds: RiskDataset, batch_size: int, shuffle: bool, oversample: bool, seed: int, num_workers: int = 0):
    g = torch.Generator(); g.manual_seed(seed)
    if oversample and len(ds):
        w = torch.as_tensor(ds.frame_weights(), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collate, num_workers=num_workers)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=g, num_workers=num_workers)
