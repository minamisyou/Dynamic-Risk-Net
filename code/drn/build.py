"""Stage 1: walk nuScenes keyframes, build scene records, run the scenario rules, keep the hits."""
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from . import scenarios
from .compat import out_path
from .lanes import MapIndex, label_units
from .source import NuscSource
from .ttc import lane_loose_ttc, lane_min_ttc

DROP_VALUES = ("0", "n", "no", "drop", "x")


def _round(obj: Any, nd: int = 3) -> Any:
    if isinstance(obj, np.ndarray):
        return np.round(obj, nd).tolist()
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


def build_record(src: NuscSource, mapidx: MapIndex, scene: Dict[str, Any], sample: Dict[str, Any],
                 cfg: Dict[str, Any]) -> Dict[str, Any]:
    ego = src.ego_state(scene, sample, cfg["trajectory"])
    actors = src.actors(scene, sample, ego, cfg["roi"]["radius_m"], cfg["trajectory"])
    units = mapidx.lane_units(ego, cfg["roi"]["radius_m"], cfg["label_roi"], cfg["lane"], cfg.get("label_units"))
    templates = scenarios.evaluate(ego, actors, units, mapidx, cfg["templates"])
    ttc = lane_min_ttc(units, ego, actors, cfg["ttc"])
    for u in units:
        u["min_ttc_s"] = ttc["per_lane"][u["lane_token"]]["min_ttc_s"]
        u["min_ttc_pair"] = ttc["per_lane"][u["lane_token"]]["pair"]
    loose = lane_loose_ttc(units, ego, actors, cfg["ttc"])
    for u in units:
        u["loose_ttc_s"] = loose[u["lane_token"]]
    return {
        "sample_token": sample["token"], "scene_token": scene["token"], "scene_name": scene["name"],
        "location": mapidx.location, "timestamp": sample["timestamp"],
        "cam_front": src.cam_front_relpath(sample),
        "ego": ego, "actors": actors, "lanes": units, "label_units": label_units(units), "templates": templates,
        "map_context": mapidx.map_context(ego, cfg["roi"]["radius_m"]),
        "ttc": {"scene_min_ttc_s": ttc["scene_min_ttc_s"], "n_events": ttc["n_events"]},
    }


def read_token_list(path: str) -> set:
    """Sample tokens from a file: one per line, or the first column of a CSV (a `sample_token` header and lines
    starting with # are skipped). Reads scenes.csv, labels files and selection_compact.csv alike."""
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            tok = line.split(",")[0].strip()
            if tok and not tok.startswith("#") and tok != "sample_token":
                out.add(tok)
    return out


def run_build(cfg: Dict[str, Any], limit_scenes: Optional[int] = None, keep_all: bool = False,
              src: Optional[NuscSource] = None, only_tokens: Optional[set] = None) -> Dict[str, Any]:
    """`only_tokens`: build just these keyframes (each is written whatever the extraction filters say). Used to
    rebuild a published list of scenes without walking the whole dataset through the filters."""
    src = src or NuscSource(cfg["nuscenes"]["dataroot"], cfg["nuscenes"]["version"])
    ext = cfg["extraction"]
    maps = {}  # type: Dict[str, MapIndex]
    index_path = out_path(cfg, "index.jsonl")
    n_frames = n_retained = 0
    hit_counts = {t: 0 for t in scenarios.TEMPLATES}
    rows = []
    with open(index_path, "w", encoding="utf-8") as index_fh:
        for si, scene in enumerate(src.scenes()):
            if limit_scenes is not None and si >= limit_scenes:
                break
            samples = list(src.samples(scene))
            if only_tokens is not None:
                samples = [s for s in samples if s["token"] in only_tokens]
                if not samples:
                    continue
            loc = src.location(scene)
            if loc not in maps:
                maps[loc] = MapIndex(cfg["nuscenes"]["dataroot"], loc, cfg["lane"]["resolution_m"],
                                     cache_dir=out_path(cfg, "cache"))
            last_kept_t = None
            for sample in samples:
                n_frames += 1
                rec = build_record(src, maps[loc], scene, sample, cfg)
                hits = [t for t in scenarios.TEMPLATES if rec["templates"][t]["hit"]]
                n_label = len(rec["label_units"])
                t = sample["timestamp"] * 1e-6
                reason = ""
                if not hits and ext.get("require_template", True):
                    reason = "no_template"
                elif n_label < ext["min_label_units"]:
                    reason = "few_label_units"
                elif last_kept_t is not None and t - last_kept_t < ext["min_gap_s"] - 1e-6:
                    reason = "min_gap"
                if only_tokens is not None:
                    reason = ""
                retained = reason == ""
                if retained:
                    last_kept_t = t
                    n_retained += 1
                    for h in hits:
                        hit_counts[h] += 1
                    rows.append([rec["sample_token"], rec["scene_name"], rec["location"],
                                 "|".join(hits), n_label, "", ""])
                if retained or keep_all:
                    p = out_path(cfg, "scenes", rec["scene_name"], rec["sample_token"] + ".json")
                    with open(p, "w", encoding="utf-8") as fh:
                        json.dump(_round(rec), fh)
                index_fh.write(json.dumps({
                    "sample_token": rec["sample_token"], "scene_token": rec["scene_token"],
                    "scene_name": rec["scene_name"], "location": rec["location"],
                    "timestamp": rec["timestamp"], "n_actors": len(rec["actors"]),
                    "n_lanes": len(rec["lanes"]), "n_label_units": n_label,
                    "n_label_lanes": sum(1 for u in rec["lanes"] if u["label_target"]),
                    "templates": hits, "retained": retained, "reason": reason,
                }) + "\n")
    review_path = out_path(cfg, "review.csv")
    previous = {}
    if os.path.exists(review_path):
        with open(review_path, newline="", encoding="utf-8") as fh:
            previous = {r["sample_token"]: (r.get("keep", ""), r.get("note", "")) for r in csv.DictReader(fh)}
    for row in rows:
        row[5], row[6] = previous.get(row[0], ("", ""))
    with open(review_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_token", "scene_name", "location", "templates", "n_label_units", "keep", "note"])
        w.writerows(rows)
    summary = {"frames": n_frames, "retained": n_retained, "template_hits": hit_counts,
               "index": index_path, "review": review_path}
    if only_tokens is not None:
        summary["listed"] = len(only_tokens)
        summary["listed_not_found"] = len(only_tokens) - n_frames
    with open(out_path(cfg, "build_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def curated(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keyframes the later stages work on.

    Retained keyframes after the manual review column (blank keep = keep); when `candidates.restrict_to` names
    candidate rounds, only the keyframes of those rounds; when `sampling.apply` is true, only the keyframes marked
    selected in selection.csv (see drn/candidates.py)."""
    dropped = set()
    review_path = os.path.join(cfg["output_dir"], "review.csv")
    if os.path.exists(review_path):
        with open(review_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("keep") or "").strip().lower() in DROP_VALUES:
                    dropped.add(row["sample_token"])
    out = []
    with open(os.path.join(cfg["output_dir"], "index.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["retained"] and row["sample_token"] not in dropped:
                out.append(row)
    return out


def load_record(cfg: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    p = os.path.join(cfg["output_dir"], "scenes", row["scene_name"], row["sample_token"] + ".json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def set_review(cfg: Dict[str, Any], sample_token: str, keep: bool, note: str = "") -> None:
    path = os.path.join(cfg["output_dir"], "review.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["sample_token"] == sample_token:
            row["keep"] = "1" if keep else "0"
            row["note"] = note
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_token", "scene_name", "location", "templates",
                                           "n_label_units", "keep", "note"])
        w.writeheader()
        w.writerows(rows)
