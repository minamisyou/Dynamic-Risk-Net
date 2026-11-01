"""End-to-end run on the synthetic fixture: build -> annotate (via the web API) -> consensus -> export."""
import csv
import json
import os

import numpy as np
import pytest

from drn.annotate import create_app
from drn.build import curated, load_record, run_build
from drn.consensus import run_consensus
from drn.export import run_export
from drn.lanegcn_format import IGNORE, lanegcn_sample


@pytest.fixture(scope="module")
def built(fake_cfg):
    return run_build(fake_cfg)


def _record(cfg, token):
    row = [r for r in curated(cfg) if r["sample_token"] == token][0]
    return load_record(cfg, row)


def test_build_selects_and_describes_scenes(fake_cfg, built):
    assert built["frames"] == 6
    assert built["retained"] == 3                       # min_gap_s = 1.0 keeps every other keyframe
    rows = curated(fake_cfg)
    assert [r["sample_token"] for r in rows] == ["sample_0", "sample_2", "sample_4"]
    rec = _record(fake_cfg, "sample_4")
    assert all(rec["templates"][t]["hit"] for t in rec["templates"])
    assert rec["templates"]["unprotected_intersection"]["n_conflicting"] == 1
    assert rec["templates"]["pedestrian_curb"]["n_on_crossing"] == 1
    assert rec["templates"]["occlusion"]["n_low_visibility"] == 2
    assert {a["cls"] for a in rec["actors"]} == {"vehicle", "pedestrian"}      # barrier is ignored
    lanes = {l["lane_token"]: l for l in rec["lanes"]}
    assert set(lanes) == {"L_E_in", "L_E_in2", "L_E_out", "L_N_in", "L_N_out", "C_E", "C_E2", "C_N"}
    assert lanes["C_E"]["in_intersection"] and not lanes["L_E_in"]["in_intersection"]
    assert lanes["L_E_in"]["succ"] == ["C_E"] and sorted(lanes["L_E_out"]["pred"]) == ["C_E", "C_E2"]
    assert lanes["L_E_in"]["right"] == ["L_E_in2"] and lanes["L_E_in2"]["left"] == ["L_E_in"]


def test_label_units_merge_same_movement(fake_cfg, built):
    rec = _record(fake_cfg, "sample_4")
    units = rec["label_units"]
    by_id = {u["unit_id"]: u for u in units}
    # The two connectors that leave side-by-side lanes and end on the same exit lane are one movement.
    assert by_id["C_E"]["lane_tokens"] == ["C_E", "C_E2"] and by_id["C_E"]["in_intersection"]
    # The ego's own lane comes first and is not widened by its parallel neighbour.
    assert units[0]["unit_id"] == "L_E_in" and units[0]["is_ego_lane"] and units[0]["lane_tokens"] == ["L_E_in"]
    assert by_id["L_E_in2"]["lane_tokens"] == ["L_E_in2"]
    # A lane does not chain into a connector inside the junction.
    assert by_id["L_N_in"]["lane_tokens"] == ["L_N_in"] and by_id["C_N"]["lane_tokens"] == ["C_N"]
    assert len(units) == 7 and sum(len(u["lane_tokens"]) for u in units) == 8
    lanes = {l["lane_token"]: l for l in rec["lanes"]}
    assert lanes["C_E2"]["unit_id"] == "C_E" and by_id["C_E"]["min_ttc_s"] == min(lanes["C_E"]["min_ttc_s"], lanes["C_E2"]["min_ttc_s"])


def test_ego_path_unit_follows_the_only_way_onward(fake_cfg, built):
    from drn.lanes import assign_label_units, label_units
    from fake_nusc import CONNECTIVITY

    class Stub:
        outgoing = {k: v["outgoing"] for k, v in CONNECTIVITY.items()}
        incoming = {k: v["incoming"] for k, v in CONNECTIVITY.items()}

    rec = _record(fake_cfg, "sample_4")
    lanes = [dict(l, pts=np.asarray(l["pts"], dtype=float)) for l in rec["lanes"]]
    Stub.poly = {l["lane_token"]: l["pts"] for l in lanes}
    assign_label_units(lanes, Stub, rec["ego"], dict(fake_cfg["label_units"], ego_path=True))
    units = label_units(lanes)
    # L_E_in has one way onward (C_E, then L_E_out): the ego's unit is that whole path, listed first.
    assert units[0]["is_ego_lane"] and units[0]["lane_tokens"] == ["C_E", "L_E_in", "L_E_out"] and units[0]["kind"] == "mixed"
    assert {u["unit_id"]: u["lane_tokens"] for u in units}["C_E2"] == ["C_E2"]      # not pulled into the ego's path
    assert len(units) == 6


def test_ttc_matches_hand_calculation(fake_cfg, built):
    rec = _record(fake_cfg, "sample_4")                 # t = 2.0 s
    # Ego (x=978, 6 m/s east) and the crossing car (y=982.667, 3.833 m/s north) both reach (1002, 998) at t=4.0 s.
    # Contact radius = 1.73/2 + 1.9/2 + 0.5 margin = 2.315 m; closing speed = hypot(6, 3.833) = 7.12 m/s.
    expected = 4.0 - 2.315 / np.hypot(6.0, 23.0 / 6.0)
    lanes = {l["lane_token"]: l for l in rec["lanes"]}
    assert abs(lanes["C_E"]["min_ttc_s"] - expected) < 0.01
    assert lanes["C_E"]["min_ttc_pair"] == ["ego", "inst_crossing_car"]
    assert lanes["L_E_in"]["min_ttc_s"] == fake_cfg["ttc"]["horizon_s"]
    # Loose value for the same pair: both are 24 m from the meeting point (range 24*sqrt(2)), 6 s and 6.26 s away;
    # distance / closing rate = range^2 / -(p.v) with p = (24, -24), v = (-6, 23/6).
    loose = (2 * 24.0 ** 2) / (24.0 * 6.0 + 24.0 * 23.0 / 6.0)
    by_id = {u["unit_id"]: u for u in rec["label_units"]}
    assert lanes["C_E"]["loose_ttc_s"] <= loose + 0.01 and by_id["C_E"]["loose_ttc_s"] == min(lanes["C_E"]["loose_ttc_s"], lanes["C_E2"]["loose_ttc_s"])


def test_annotation_tool_consensus_and_export(fake_cfg, built):
    client = create_app(fake_cfg, mode="annotate").test_client()
    assert client.get("/").status_code == 200
    items = client.get("/api/list?annotator=A1").get_json()["items"]
    assert len(items) == 3 and not any(i["done"] for i in items)
    assert client.get("/img/sample_0").status_code == 200
    assert client.get("/api/list?annotator=nobody").status_code == 400

    # Scripted ratings, for exercising the code path only (these are not labels of anything).
    def rating(annotator, sample, k):
        base = k % 5
        if sample == "sample_2" and k == 0:
            return {"A1": 2, "A2": 3, "A3": 3}[annotator]          # L2 vs L3 disagreement -> majority 3
        if sample == "sample_4" and k == 1:
            return {"A1": 0, "A2": 2, "A3": 4}[annotator]          # no majority -> adjudication
        return base

    for ann in ("A1", "A2", "A3"):
        for item in items:
            tok = item["sample_token"]
            scene = client.get("/api/scene/%s?annotator=%s" % (tok, ann)).get_json()
            assert scene["labels"] == {}                            # blind: nothing from other annotators
            targets = [u["unit_id"] for u in scene["units"]]
            labels = {t: rating(ann, tok, k) for k, t in enumerate(targets)}
            partial = dict(list(labels.items())[:-1])
            bad = client.post("/api/label", json={"annotator": ann, "sample_token": tok, "labels": partial})
            assert bad.status_code == 400                           # every target lane must be labelled
            no_signal = client.post("/api/label", json={"annotator": ann, "sample_token": tok, "labels": labels})
            assert no_signal.status_code == 400                     # the signal state is required
            signal = "red" if (tok == "sample_2" and ann != "A3") else "none"
            ok = client.post("/api/label", json={"annotator": ann, "sample_token": tok, "labels": labels,
                                                 "signal": signal, "elapsed_s": 12.5})
            assert ok.status_code == 200
    again = client.get("/api/scene/sample_0?annotator=A2").get_json()
    n_units = {t: len(_record(fake_cfg, t)["label_units"]) for t in ("sample_0", "sample_2", "sample_4")}
    assert len(again["labels"]) == n_units["sample_0"] and all(i["done"] for i in client.get("/api/list?annotator=A2").get_json()["items"])

    stats = run_consensus(fake_cfg)
    n_targets = sum(n_units.values())
    assert n_targets == 6 + 7 + 7
    assert stats["labels_final"] == n_targets - 1 and stats["labels_pending_adjudication"] == 1
    assert stats["labels_incomplete"] == 0 and stats["scenes_fully_labelled"] == 2
    assert abs(stats["exact_agreement"] - (n_targets - 2) / n_targets) < 1e-9
    assert stats["disagreement_within_one_level"] == 0.5 and stats["disagreement_share_l2_l3"] == 0.5
    assert 0.8 < stats["fleiss_kappa"] < 1.0
    pending = [json.loads(l) for l in open(os.path.join(fake_cfg["output_dir"], "consensus", "pending_adjudication.jsonl"))]
    assert pending[0]["sample_token"] == "sample_4" and pending[0]["ratings"] == [0, 2, 4]

    with open(os.path.join(fake_cfg["output_dir"], "adjudication.jsonl"), "w") as fh:
        fh.write(json.dumps({"sample_token": "sample_4", "unit_id": pending[0]["unit_id"], "label": 2}) + "\n")
    stats = run_consensus(fake_cfg)
    assert stats["labels_final"] == n_targets and stats["labels_pending_adjudication"] == 0
    assert sum(stats["label_distribution"].values()) == n_targets
    assert stats["template_prevalence"]["pedestrian_curb"] == 3
    assert stats["labels_adjudicated"] == 1
    assert stats["signal_frames"] == 3 and abs(stats["signal_exact_agreement"] - 2 / 3) < 1e-9
    assert stats["signal_distribution"] == {"none": 2, "red": 1}      # 2-of-3 red wins on sample_2

    out = run_export(fake_cfg)
    with open(os.path.join(out["release_dir"], "labels.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len({(r["sample_token"], r["unit_id"]) for r in rows}) == n_targets       # one label per unit ...
    assert len(rows) == sum(sum(1 for l in _record(fake_cfg, t)["lanes"] if l["label_target"]) for t in n_units)  # ... one row per lane
    merged = [r for r in rows if r["sample_token"] == "sample_4" and r["unit_id"] == "C_E"]
    assert sorted(r["lane_token"] for r in merged) == ["C_E", "C_E2"] and len({r["label"] for r in merged}) == 1
    assert {r["source"] for r in rows} == {"majority", "adjudicated"}
    assert set(rows[0]) == {"sample_token", "lane_token", "unit_id", "label", "source",
                            "raw_A1", "raw_A2", "raw_A3", "cv_ttc_s", "loose_ttc_s", "ttc_judged_s", "ttc_basis",
                            "ttc_hybrid_s", "split"}
    assert {r["ttc_basis"] for r in rows} == {""} and all(r["loose_ttc_s"] for r in rows)   # no time question was answered
    disputed = [r for r in rows if r["sample_token"] == "sample_2" and r["raw_A1"] == "2" and r["raw_A2"] == "3"]
    assert disputed and disputed[0]["label"] == "3"
    assert os.path.exists(os.path.join(out["release_dir"], "stats.md"))


def test_review_mode_drops_scene(fake_cfg, built):
    client = create_app(fake_cfg, mode="review").test_client()
    assert client.post("/api/review", json={"sample_token": "sample_0", "keep": False, "note": "false positive"}).status_code == 200
    assert [r["sample_token"] for r in curated(fake_cfg)] == ["sample_2", "sample_4"]
    assert client.post("/api/review", json={"sample_token": "sample_0", "keep": True}).status_code == 200
    assert len(curated(fake_cfg)) == 3
    run_build(fake_cfg)
    with open(os.path.join(fake_cfg["output_dir"], "review.csv"), newline="") as fh:
        assert [r["keep"] for r in csv.DictReader(fh)][0] == "1"


def test_lanegcn_sample_shapes(fake_cfg, built):
    rec = _record(fake_cfg, "sample_4")
    labels = {"C_E": 3, "L_E_in": 1}
    s = lanegcn_sample(rec, labels, fake_cfg["lane"]["node_spacing_m"])
    n_agents = 1 + len(rec["actors"])
    assert s["feats"].shape == (n_agents, 20, 3) and s["gt_preds"].shape == (n_agents, 30, 2)
    assert np.allclose(s["ctrs"][0], 0.0)                            # ego sits at the origin of its own frame
    assert np.allclose(s["feats"][0, -1, :2], [0.6, 0.0], atol=1e-3) # 6 m/s east = 0.6 m per 0.1 s step, forward
    g = s["graph"]
    assert g["num_nodes"] == len(g["ctrs"]) == len(s["risk_targets"]) and g["num_nodes"] > 100
    idx = g["lane_tokens"].index("C_E")
    assert set(s["risk_targets"][g["lane_idcs"] == idx]) == {3}
    assert set(s["risk_targets"][g["lane_idcs"] == g["lane_tokens"].index("C_E2")]) == {3}   # same unit as C_E
    assert (s["risk_targets_lane"] == IGNORE).sum() == len(rec["lanes"]) - 3
    assert len(g["suc"]["u"]) == len(g["suc"]["v"]) > 0
    # successor edges must connect the end of L_E_in to the start of C_E
    li, ci = g["lane_tokens"].index("L_E_in"), g["lane_tokens"].index("C_E")
    last_in = np.where(g["lane_idcs"] == li)[0][-1]
    first_c = np.where(g["lane_idcs"] == ci)[0][0]
    assert (last_in, first_c) in set(zip(g["suc"]["u"].tolist(), g["suc"]["v"].tolist()))
