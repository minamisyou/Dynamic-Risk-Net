"""Stage 4: scene-level train/val/test split and the public release files.

The release contains no nuScenes data: only sample tokens, lane tokens, labels and
split assignments. Anyone with nuScenes can regenerate the scenes with this package.
"""
import csv
import json
import os
import random
from typing import Any, Dict, List

from .consensus import annotators_for

RELEASE_README = """# Dynamic-Risk Net lane-segment risk labels (derived from nuScenes)

Files
- labels.csv   : one row per lane of every labelled unit
    sample_token  nuScenes keyframe token
    lane_token    nuScenes map-expansion `lane` / `lane_connector` token
    unit_id       label unit this lane belongs to in this keyframe (lanes merged into one unit share a label;
                  the id is the smallest member lane_token). Count labels by distinct (sample_token, unit_id).
    label         consensus risk level 0-4 (Normal, Take Care, Slow Down, Ready to Brake, Stop)
    source        `majority` (strict majority of annotators) or `adjudicated`
    raw_*         each annotator's independent label
    cv_ttc_s      computed constant-velocity TTC: first time two road users (discs) touch near this unit; the horizon
                  value (10 s by default) means none
    loose_ttc_s   computed distance / closing rate of the nearest approaching pair near this unit (no collision-course
                  test); the horizon value means none
    ttc_judged_s  annotators' time to the conflict that decided the level (median); empty when they gave none or the
                  guide version used had no time question
    ttc_basis     `calc` (annotators accepted the computed value), `est` (they estimated it), `none`
    ttc_hybrid_s  the computed value (see config.yaml ttc.hybrid_base), replaced by ttc_judged_s where ttc_basis is `est`.
                  Where it is estimated it comes from the same annotators as the label: say so wherever label and time
                  are compared.
    split         train / val / test (split by nuScenes scene)
- splits.json  : scene_token -> split
- scenes.jsonl : per keyframe, the scenario templates that selected it and the annotated signal state
- stats.json / stats.md : label distribution and inter-annotator agreement, computed from the raw labels

This release contains no nuScenes sensor data, annotations or maps. Obtain nuScenes
(https://www.nuscenes.org, CC BY-NC-SA 4.0) and run `python -m drn build` with the
config.yaml shipped next to this file to create the scene records.
"""


def assign_splits(scene_tokens: List[str], fractions: Dict[str, float], seed: int) -> Dict[str, str]:
    tokens = sorted(set(scene_tokens))
    random.Random(seed).shuffle(tokens)
    n = len(tokens)
    n_train = int(round(n * fractions["train"]))
    n_val = int(round(n * fractions["val"]))
    out = {}
    for i, t in enumerate(tokens):
        out[t] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def _num(v: Any) -> str:
    return "" if v is None else "%.2f" % float(v)


def run_export(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cons_dir = os.path.join(cfg["output_dir"], "consensus")
    with open(os.path.join(cons_dir, "labels.jsonl"), encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    if not items:
        raise SystemExit("No consensus labels yet. Run `python -m drn consensus` after annotation.")
    annotators = annotators_for(cfg)
    splits = assign_splits([i["scene_token"] for i in items], cfg["split"], cfg["seed"])
    rel = os.path.join(cfg["output_dir"], "release")
    os.makedirs(rel, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    with open(os.path.join(rel, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_token", "lane_token", "unit_id", "label", "source"] + ["raw_" + a for a in annotators] +
                   ["cv_ttc_s", "loose_ttc_s", "ttc_judged_s", "ttc_basis", "ttc_hybrid_s", "split"])
        for i in items:
            sp = splits[i["scene_token"]]
            counts[sp] += 1
            for lane_token in i["lane_tokens"]:
                w.writerow([i["sample_token"], lane_token, i["unit_id"], i["label"], i["source"]] +
                           [i["raw"][a] for a in annotators] +
                           [_num(i.get("min_ttc_s")), _num(i.get("loose_ttc_s")), _num(i.get("ttc_judged_s")), i.get("ttc_basis", ""),
                            _num(i.get("ttc_hybrid_s")), sp])
    with open(os.path.join(rel, "splits.json"), "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=1)
    seen = {}
    for i in items:
        seen[i["sample_token"]] = {"sample_token": i["sample_token"], "scene_token": i["scene_token"],
                                   "templates": i["templates"], "split": splits[i["scene_token"]]}
    sig_path = os.path.join(cons_dir, "signals.jsonl")
    if os.path.exists(sig_path):
        with open(sig_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec["sample_token"] in seen:
                        seen[rec["sample_token"]]["signal"] = rec["signal"]
    with open(os.path.join(rel, "scenes.jsonl"), "w", encoding="utf-8") as fh:
        for v in seen.values():
            fh.write(json.dumps(v) + "\n")
    for name in ("stats.json", "stats.md"):
        src = os.path.join(cons_dir, name)
        if os.path.exists(src):
            with open(src, encoding="utf-8") as a, open(os.path.join(rel, name), "w", encoding="utf-8") as b:
                b.write(a.read())
    with open(os.path.join(rel, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(RELEASE_README)
    return {"release_dir": rel, "labels_per_split": counts, "scenes": len(splits), "keyframes": len(seen)}
