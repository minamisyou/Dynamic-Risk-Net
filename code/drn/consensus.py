"""Stage 3: merge independent annotator labels, majority vote, agreement statistics.

Nothing here invents a label: an item without a strict majority stays unresolved until an
adjudication entry exists for it. Adjudication never overrides an existing majority; where an
adjudicator reviewed a large disagreement and disagreed with the majority, that is reported
in majority_challenged.jsonl for a person to look at.
"""
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import scenarios
from .build import curated, load_record
from .ttc import consensus_time, hybrid_ttc, lane_loose_ttc

N_LEVELS = 5
LEVEL_NAMES = ["Normal", "Take Care", "Slow Down", "Ready to Brake", "Stop"]
def annotators_for(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["annotation"]["annotators"])


def _entries(cfg: Dict[str, Any], annotator: str) -> List[Dict[str, Any]]:
    path = os.path.join(cfg["output_dir"], "labels", annotator + ".jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_labels(cfg: Dict[str, Any], annotator: str) -> Dict[Tuple[str, str], int]:
    """Latest label per (sample_token, unit_id) for one annotator."""
    out = {}  # type: Dict[Tuple[str, str], int]
    for rec in _entries(cfg, annotator):
        for unit_id, level in rec["labels"].items():
            out[(rec["sample_token"], unit_id)] = int(level)
    return out


def read_times(cfg: Dict[str, Any], annotator: str) -> Dict[Tuple[str, str], Tuple[Optional[float], str]]:
    """Latest (time to conflict, basis) per (sample_token, unit_id) for one annotator; entries without times are skipped."""
    out = {}  # type: Dict[Tuple[str, str], Tuple[Optional[float], str]]
    for rec in _entries(cfg, annotator):
        if "ttc" in rec and "ttc_basis" in rec:
            for unit_id, basis in rec["ttc_basis"].items():
                out[(rec["sample_token"], unit_id)] = (rec["ttc"].get(unit_id), basis)
    return out


def ensure_loose(rec: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Records built before loose_ttc_s existed: compute it from the stored lanes, ego and actors."""
    if all(u.get("loose_ttc_s") is not None for u in rec["label_units"]):
        return
    import numpy as np
    lanes = [dict(l, pts=np.asarray(l["pts"], dtype=float)) for l in rec["lanes"]]
    loose = lane_loose_ttc(lanes, rec["ego"], rec["actors"], cfg["ttc"])
    for u in rec["label_units"]:
        u["loose_ttc_s"] = min(loose[t] for t in u["lane_tokens"])


TTC_BANDS = ((">5", 5.0, float("inf")), ("3-5", 3.0, 5.0), ("1.5-3", 1.5, 3.0), ("<=1.5", -1.0, 1.5))


def ttc_tables(items: List[Dict[str, Any]], horizon: float) -> Dict[str, Any]:
    """Level against time to conflict, for each definition of the time (strict, loose, judged, hybrid).

    by_level: mean time per level, lanes without a value counted at the horizon (the paper's Table 8 layout).
    by_range: per time range, how the lanes that have a value in it split into levels 0-2 / 3 / 4
    (the paper's Table 7 layout, with the consensus label in place of a model prediction).
    """
    out = {}
    for name in ("min_ttc_s", "loose_ttc_s", "ttc_judged_s", "ttc_hybrid_s"):
        vals = [(i["label"], i.get(name)) for i in items]
        if all(v is None for _, v in vals):
            continue
        capped = [(l, horizon if v is None else min(float(v), horizon)) for l, v in vals]
        by_level = {}
        for lvl in range(N_LEVELS):
            xs = [v for l, v in capped if l == lvl]
            by_level[str(lvl)] = (sum(xs) / len(xs)) if xs else None
        by_range = {}
        for band, lo, hi in TTC_BANDS:
            xs = [l for l, v in capped if lo < v <= hi and v < horizon]
            by_range[band] = {"n": len(xs), "level_0_2": sum(1 for l in xs if l <= 2),
                              "level_3": sum(1 for l in xs if l == 3), "level_4": sum(1 for l in xs if l == 4)}
        out[name] = {"by_level": by_level, "by_range": by_range,
                     "lanes_with_value": sum(1 for _, v in capped if v < horizon)}
    return out


def read_signals(cfg: Dict[str, Any], annotator: str) -> Dict[str, str]:
    """Latest signal state per sample_token for one annotator (entries without one are skipped)."""
    return {rec["sample_token"]: rec["signal"] for rec in _entries(cfg, annotator) if rec.get("signal")}


def read_adjudication(cfg: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    path = os.path.join(cfg["output_dir"], "adjudication.jsonl")
    out = {}  # type: Dict[Tuple[str, str], Dict[str, Any]]
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    out[(rec["sample_token"], rec["unit_id"])] = rec
    return out


def fleiss_kappa(rows: List[List[int]], n_categories: int = N_LEVELS) -> Optional[float]:
    """Fleiss' kappa for items each rated by the same number of raters.

    rows[i] is the list of ratings for item i. Returns None when undefined.
    """
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(r) != n for r in rows):
        return None
    totals = [0] * n_categories
    p_sum = 0.0
    for r in rows:
        counts = Counter(r)
        for c, k in counts.items():
            totals[c] += k
        p_sum += (sum(k * k for k in counts.values()) - n) / float(n * (n - 1))
    p_bar = p_sum / len(rows)
    p_e = sum((t / float(len(rows) * n)) ** 2 for t in totals)
    if abs(1.0 - p_e) < 1e-12:
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def majority(ratings: List[Any]) -> Optional[Any]:
    """Strict-majority value, or None when no value has more than half the votes."""
    value, k = Counter(ratings).most_common(1)[0]
    return value if k * 2 > len(ratings) else None


def run_consensus(cfg: Dict[str, Any]) -> Dict[str, Any]:
    annotators = annotators_for(cfg)
    if len(annotators) < 2:
        raise SystemExit("Need at least two annotators in config.yaml.")
    per_ann = {a: read_labels(cfg, a) for a in annotators}
    per_sig = {a: read_signals(cfg, a) for a in annotators}
    per_time = {a: read_times(cfg, a) for a in annotators}
    horizon = float(cfg["ttc"]["horizon_s"])
    base_key = "min_ttc_s" if cfg["ttc"].get("hybrid_base", "loose") == "strict" else "loose_ttc_s"
    time_spreads = []
    adjud = read_adjudication(cfg)
    rows = curated(cfg)
    items, pending, challenged, raw_rows, signals = [], [], [], [], []
    incomplete = scenes_done = sig_complete = sig_agree = 0
    template_counts = {t: 0 for t in scenarios.TEMPLATES}
    for row in rows:
        rec = load_record(cfg, row)
        ensure_loose(rec, cfg)
        scene_complete = True
        for lane in rec["label_units"]:
            key = (rec["sample_token"], lane["unit_id"])
            ratings = [per_ann[a][key] for a in annotators if key in per_ann[a]]
            if len(ratings) < len(annotators):
                incomplete += 1
                scene_complete = False
                continue
            raw_rows.append(ratings)
            label, how = majority(ratings), "majority"
            adj = adjud.get(key)
            if label is None:
                if adj is None:
                    pending.append({"sample_token": key[0], "unit_id": key[1], "ratings": ratings})
                    scene_complete = False
                    continue
                label, how = int(adj["label"]), "adjudicated"
            elif adj is not None and int(adj["label"]) != label:
                challenged.append({"sample_token": key[0], "unit_id": key[1], "ratings": ratings, "majority": int(label),
                                   "adjudicator": int(adj["label"]), "rule": adj.get("rule", ""), "reason": adj.get("reason", "")})
            item_time = {}
            if all(key in per_time[a] for a in annotators):       # every annotator answered the time question
                raw_t = {a: per_time[a][key] for a in annotators}
                judged, basis = consensus_time([v for v, _ in raw_t.values()], [b for _, b in raw_t.values()])
                given = [v for v, _ in raw_t.values() if v is not None]
                if len(given) == len(annotators):
                    time_spreads.append(max(given) - min(given))
                item_time = {"ttc_judged_s": judged, "ttc_basis": basis, "ttc_raw": {a: list(raw_t[a]) for a in annotators},
                             "ttc_hybrid_s": hybrid_ttc(lane.get(base_key), judged, basis, horizon), "ttc_hybrid_base": base_key}
            items.append({
                "sample_token": key[0], "unit_id": key[1], "lane_tokens": lane["lane_tokens"], "scene_token": rec["scene_token"],
                "scene_name": rec["scene_name"], "label": int(label), "source": how,
                "raw": {a: per_ann[a][key] for a in annotators},
                "kind": lane["kind"], "in_intersection": lane["in_intersection"], "is_ego_lane": lane["is_ego_lane"],
                "min_ttc_s": lane["min_ttc_s"], "loose_ttc_s": lane.get("loose_ttc_s"), **item_time,
                "templates": [t for t in scenarios.TEMPLATES if rec["templates"][t]["hit"]],
            })
        sig = [per_sig[a][rec["sample_token"]] for a in annotators if rec["sample_token"] in per_sig[a]]
        if len(sig) == len(annotators):
            sig_complete += 1
            sig_agree += int(len(set(sig)) == 1)
            signals.append({"sample_token": rec["sample_token"], "signal": majority(sig) or "unresolved",
                            "raw": dict(zip(annotators, sig))})
        if scene_complete:
            scenes_done += 1
            for t in scenarios.TEMPLATES:
                if rec["templates"][t]["hit"]:
                    template_counts[t] += 1

    disagreements = [r for r in raw_rows if len(set(r)) > 1]
    dist = Counter(i["label"] for i in items)
    ttc_by_level = {}
    for lvl in range(N_LEVELS):
        vals = [i["min_ttc_s"] for i in items if i["label"] == lvl]
        ttc_by_level[str(lvl)] = (sum(vals) / len(vals)) if vals else None
    stats = {
        "annotators": annotators,
        "curated_scenes": len(rows), "scenes_fully_labelled": scenes_done,
        "labels_final": len(items), "labels_incomplete": incomplete, "labels_pending_adjudication": len(pending),
        "labels_adjudicated": sum(1 for i in items if i["source"] == "adjudicated"),
        "majority_challenged_by_adjudicator": len(challenged),
        "label_distribution": {str(l): dist.get(l, 0) for l in range(N_LEVELS)},
        "template_prevalence": template_counts,
        "fleiss_kappa": fleiss_kappa(raw_rows),
        "exact_agreement": (sum(1 for r in raw_rows if len(set(r)) == 1) / len(raw_rows)) if raw_rows else None,
        "disagreement_within_one_level": (sum(1 for r in disagreements if max(r) - min(r) <= 1) / len(disagreements))
        if disagreements else None,
        "disagreement_share_l2_l3": (sum(1 for r in disagreements if set(r) == {2, 3}) / len(disagreements))
        if disagreements else None,
        "mean_min_ttc_by_level_s": ttc_by_level, "ttc_horizon_s": cfg["ttc"]["horizon_s"],
        "ttc_tables": ttc_tables(items, horizon),
        "ttc_basis_by_level": {str(l): dict(Counter(i["ttc_basis"] for i in items if i["label"] == l and "ttc_basis" in i))
                               for l in range(N_LEVELS)},
        "ttc_annotated_units": sum(1 for i in items if "ttc_basis" in i),
        "ttc_all_gave_value": len(time_spreads),
        "ttc_spread_median_s": (sorted(time_spreads)[len(time_spreads) // 2] if time_spreads else None),
        "ttc_spread_within_1s": (sum(1 for x in time_spreads if x <= 1.0) / len(time_spreads)) if time_spreads else None,
        "signal_frames": sig_complete, "signal_exact_agreement": (sig_agree / sig_complete) if sig_complete else None,
        "signal_distribution": dict(Counter(s["signal"] for s in signals)),
    }
    out_dir = os.path.join(cfg["output_dir"], "consensus")
    os.makedirs(out_dir, exist_ok=True)
    for name, rows_out in (("labels.jsonl", items), ("pending_adjudication.jsonl", pending),
                           ("majority_challenged.jsonl", challenged), ("signals.jsonl", signals)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for r in rows_out:
                fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    with open(os.path.join(out_dir, "stats.md"), "w", encoding="utf-8") as fh:
        fh.write(stats_markdown(stats))
    return stats


def _fmt(v: Optional[float], pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return "%.1f%%" % (100.0 * v) if pct else "%.3f" % v


def stats_markdown(s: Dict[str, Any]) -> str:
    total = max(s["labels_final"], 1)
    lines = ["# Dataset statistics (computed from annotator files)", "",
             "Annotators: " + ", ".join(s["annotators"]), "",
             "Curated scenes: %d (fully labelled: %d)" % (s["curated_scenes"], s["scenes_fully_labelled"]),
             "Final label units: %d (adjudicated: %d) | incomplete: %d | pending adjudication: %d"
             % (s["labels_final"], s["labels_adjudicated"], s["labels_incomplete"], s["labels_pending_adjudication"]),
             "Majority labels an adjudicator disagreed with (kept as majority, listed in majority_challenged.jsonl): %d"
             % s["majority_challenged_by_adjudicator"], "",
             "## Label distribution (after majority vote)", "", "| Level | Count | Fraction |", "|---|---|---|"]
    for lvl in range(N_LEVELS):
        c = s["label_distribution"][str(lvl)]
        lines.append("| %d (%s) | %d | %.1f%% |" % (lvl, LEVEL_NAMES[lvl], c, 100.0 * c / total))
    lines += ["", "## Scenario template prevalence (fully labelled scenes)", "", "| Template | Scenes |", "|---|---|"]
    for t, c in s["template_prevalence"].items():
        lines.append("| %s | %d |" % (t, c))
    lines += ["", "## Agreement between annotators (raw labels)", "", "| Statistic | Value |", "|---|---|",
              "| Fleiss' kappa (5-class) | %s |" % _fmt(s["fleiss_kappa"]),
              "| Exact agreement, all annotators | %s |" % _fmt(s["exact_agreement"], True),
              "| Disagreements within +-1 level | %s |" % _fmt(s["disagreement_within_one_level"], True),
              "| Disagreements that are L2 vs L3 only | %s |" % _fmt(s["disagreement_share_l2_l3"], True),
              "| Signal state: frames rated by all | %d |" % s["signal_frames"],
              "| Signal state: exact agreement | %s |" % _fmt(s["signal_exact_agreement"], True),
              "", "Signal state distribution: " + (", ".join("%s=%d" % kv for kv in sorted(s["signal_distribution"].items())) or "n/a"),
              "", "## Mean minimum CV-TTC by label level (horizon %.1f s)" % s["ttc_horizon_s"], "",
              "| Level | Mean min CV-TTC (s) |", "|---|---|"]
    for lvl in range(N_LEVELS):
        lines.append("| %d (%s) | %s |" % (lvl, LEVEL_NAMES[lvl], _fmt(s["mean_min_ttc_by_level_s"][str(lvl)])))
    names = {"min_ttc_s": "strict CV-TTC (computed)", "loose_ttc_s": "distance / closing rate (computed)",
             "ttc_judged_s": "annotators' time (accepted computed value or own estimate)",
             "ttc_hybrid_s": "hybrid: computed, replaced by the estimate where annotators estimated"}
    tables = s.get("ttc_tables") or {}
    if tables:
        lines += ["", "## Mean time to conflict by label level, per definition (lanes without a value count as %.0f s)" % s["ttc_horizon_s"], "",
                  "| Definition | L0 | L1 | L2 | L3 | L4 | lanes with a value |", "|---|---|---|---|---|---|---|"]
        for key, t in tables.items():
            lines.append("| %s | %s | %d |" % (names[key], " | ".join(_fmt(t["by_level"][str(l)]) for l in range(N_LEVELS)), t["lanes_with_value"]))
        lines += ["", "## Label level by time-to-conflict range (lanes that have a value; L0-2 / L3 / L4)", "",
                  "| Definition | > 5 s | 3-5 s | 1.5-3 s | <= 1.5 s |", "|---|---|---|---|---|"]
        for key, t in tables.items():
            cells = []
            for band in (">5", "3-5", "1.5-3", "<=1.5"):
                r = t["by_range"][band]
                cells.append("n=%d: %s" % (r["n"], " / ".join("%.0f%%" % (100.0 * r[k] / r["n"]) for k in ("level_0_2", "level_3", "level_4")))
                             if r["n"] else "n=0")
            lines.append("| %s | %s |" % (names[key], " | ".join(cells)))
    if s.get("ttc_annotated_units"):
        lines += ["", "## Where the annotated time comes from (units, by label level)", "",
                  "| Level | computed value accepted | estimated | none |", "|---|---|---|---|"]
        for lvl in range(N_LEVELS):
            b = s["ttc_basis_by_level"][str(lvl)]
            lines.append("| %d (%s) | %d | %d | %d |" % (lvl, LEVEL_NAMES[lvl], b.get("calc", 0), b.get("est", 0), b.get("none", 0)))
        lines += ["", "Units where every annotator gave a time: %d; spread between them (max - min): median %s s, within 1 s on %s."
                  % (s["ttc_all_gave_value"], _fmt(s["ttc_spread_median_s"]), _fmt(s["ttc_spread_within_1s"], True)),
                  "", "The annotated times come from the same annotators who gave the levels, so the level/time tables above are",
                  "partly built in wherever a time is estimated. Report the computed-only rows next to the hybrid ones."]
    return "\n".join(lines) + "\n"
