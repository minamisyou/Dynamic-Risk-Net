"""drn_model on the synthetic fixture: prepare -> train (2 epochs, CPU) -> evaluate; batching and loss sanity."""
import csv
import glob
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drn_model import data as D  # noqa: E402
from drn_model.loss import class_weights, total_loss  # noqa: E402
from drn_model.metrics import classification_metrics, table7, table8, trajectory_metrics  # noqa: E402
from drn_model.model import build_model  # noqa: E402
from drn_model.train import evaluate, train  # noqa: E402


@pytest.fixture(scope="module")
def built(fake_cfg):
    from drn.build import run_build
    return run_build(fake_cfg)


@pytest.fixture(scope="module")
def labels_csv(fake_cfg, built, tmp_path_factory):
    """Synthetic fixture using the human annotation CSV layout."""
    path = str(tmp_path_factory.mktemp("labels") / "labels.csv")
    paths = sorted(glob.glob(os.path.join(fake_cfg["output_dir"], "scenes", "*", "*.json")))
    split_of = {os.path.basename(p)[:-5]: ("train", "val", "test", "train")[i % 4] for i, p in enumerate(paths)}
    headers = ["sample_token", "lane_token", "unit_id", "label", "source", "raw_A1", "raw_A2", "raw_A3",
               "cv_ttc_s", "loose_ttc_s", "ttc_judged_s", "ttc_basis", "ttc_hybrid_s", "split"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for p in paths:
            rec = json.load(open(p, encoding="utf-8"))
            tok = os.path.basename(p)[:-5]
            rec["scene_token"] = "fixture_scene_" + tok
            rec["scene_name"] = "fixture_scene_" + tok
            with open(p, "w", encoding="utf-8") as record_file:
                json.dump(rec, record_file)
            for idx, unit in enumerate(rec["label_units"]):
                level = (idx + int(tok[-1], 16)) % 5
                ttc = min(9.5, 0.75 + level * 1.5)
                for lane in unit["lane_tokens"]:
                    w.writerow([tok, lane, unit["unit_id"], level, "majority", level, level, level,
                                ttc, ttc, "", "fixture", ttc, split_of[tok]])
    return path

@pytest.fixture(scope="module")
def cache(fake_cfg, labels_csv, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("cache") / "fixture.pt")
    info = D.prepare(fake_cfg["output_dir"], labels_csv, path, node_spacing_m=2.0)
    assert info["samples"] > 0 and info["splits"]["train"] > 0
    return path


def test_prepare_and_collate(cache):
    data = D.load_cache(cache)
    s = data["samples"][0]
    assert s["unit_level"].shape[0] == len(s["unit_ids"]) == s["unit_time"].shape[0]
    assert (s["unit_of_lane"] < len(s["unit_ids"])).all()
    labelled = s["unit_of_lane"] >= 0
    assert labelled.sum() >= len(s["unit_ids"])                              # every unit has at least one lane
    b = D.collate(data["samples"][:3])
    assert b["B"] == 3 and b["n_units"] == sum(len(x["unit_ids"]) for x in data["samples"][:3])
    assert b["node_lane"].max() < b["n_lanes"] and b["lane_unit"].max() < b["n_units"]
    assert b["pre_u"].max() < b["node_ctrs"].shape[0] if len(b["pre_u"]) else True
    ds = D.RiskDataset(data["samples"], "train")
    w = ds.frame_weights(); assert abs(w.sum() - 1) < 1e-9 and (w > 0).all()


@pytest.mark.parametrize("pool,risk_on", [("max", "unit"), ("mean", "unit"), ("mean", "node")])
def test_forward_and_loss(cache, pool, risk_on):
    data = D.load_cache(cache)
    b = D.collate(data["samples"][:4])
    model = build_model("drn", pool)
    from drn_model.train import input_stats
    model.set_input_stats(input_stats(data["samples"]))
    out = model(b)
    assert out["unit_logits"].shape == (b["n_units"], 5)
    assert torch.isfinite(out["unit_logits"]).all()
    assert out["traj_reg"].shape == (b["actor_feats"].shape[0], 6, b["gt_preds"].shape[1], 2)
    w = class_weights(np.array([100, 20, 10, 5, 2]))
    assert abs(float(w.mean()) - 1) < 1e-6 and w[4] > w[0]
    terms = total_loss(out, b, w, 0.5, risk_on)
    assert torch.isfinite(terms["total"]) and terms["total"].requires_grad
    terms["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_metrics_by_hand():
    y = np.array([0, 0, 1, 2, 3, 4, 4]); p = np.array([0, 1, 1, 2, 2, 4, 4])
    m = classification_metrics(y, p)
    assert abs(m["rca"] - 5 / 7) < 1e-9 and m["recall"][0] == 0.5 and m["recall"][4] == 1.0 and m["within_one"] == 1.0
    t = table8(np.array([0, 4, 4]), np.array([9.0, 1.0, 2.0]), 8.0)
    assert t["mean_min_cv_ttc_by_level"][0] == 8.0 and t["mean_min_cv_ttc_by_level"][4] == 1.5
    t7 = table7(np.array([0, 0, 1]), np.array([1.0, 6.0, 4.0]), np.array([4, 0, 3]), 8.0, "scene")
    assert t7["values"][3] == [0.0, 0.0, 100.0] and t7["values"][1] == [0.0, 100.0, 0.0] and t7["n"] == [0, 1, 0, 1]
    reg = np.zeros((2, 6, 30, 2)); reg[0, 2] = 1.0; gt = np.ones((2, 30, 2)); has = np.ones((2, 30), bool)
    tr = trajectory_metrics(reg, np.zeros((2, 6)), gt, has)
    assert tr["minADE"] == pytest.approx(np.hypot(1, 1) / 2) and tr["MR"] == 0.0


def test_train_evaluate_smoke(cache, tmp_path):
    run = str(tmp_path / "run_drn")
    s = train(cache, "drn", run, seed=0, epochs=2, batch_size=4, augment="mirror,rot90", amp=False, device_name="cpu", deterministic=True)
    assert s["epochs"] == 2 and os.path.exists(os.path.join(run, "best.pt")) and os.path.exists(os.path.join(run, "history.jsonl"))
    for split in ("val", "test"):
        res = evaluate(cache, run, split, device_name="cpu")
        assert 0.0 <= res["classification"]["rca"] <= 1.0 and "trajectory" in res and len(res["table7_scene"]["values"]) == 4
    # same seed, same machine -> same weights
    run2 = str(tmp_path / "run_drn_again")
    train(cache, "drn", run2, seed=0, epochs=2, batch_size=4, augment="mirror,rot90", amp=False, device_name="cpu", deterministic=True)
    a = D.load_cache(os.path.join(run, "best.pt"))["model"]; b = D.load_cache(os.path.join(run2, "best.pt"))["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_paper_features_and_risk_score(cache):
    from drn_model.model import checkpoint_model_name, RiskInfluenceFusion, pairs
    data = D.load_cache(cache)
    s = data["samples"][0]
    assert s["kin"].shape == (s["feats"].shape[0], 10) and s["cls"][0] == 0 and s["hist_pos"].shape[:2] == s["feats"].shape[:2]
    assert s["graph"]["attr"].shape == (s["graph"]["num_nodes"], 3)
    assert checkpoint_model_name({"model": "drn", "arch": "drn"}) == "drn"
    b = D.collate(data["samples"][:2])
    f = RiskInfluenceFusion()
    eq, ek = pairs(b["node_ctrs"], b["node_batch"], b["actor_ctrs"], b["actor_batch"], 30.0)
    if len(eq):
        C = f.score(b, eq, ek)
        assert (C >= 0).all() and torch.isfinite(C).all()


def test_resume_continues_to_the_same_epoch_count(cache, tmp_path):
    run = str(tmp_path / "r")
    train(cache, "drn", run, seed=0, epochs=1, batch_size=4, device_name="cpu")
    s = train(cache, "drn", run, seed=0, epochs=2, batch_size=4, device_name="cpu", resume=True)
    hist = [json.loads(l) for l in open(os.path.join(run, "history.jsonl"))]
    assert [h["epoch"] for h in hist] == [1, 2] and s["config"]["resumed_at_epoch"] == 1


def test_max_pool_head(cache):
    data = D.load_cache(cache)
    b = D.collate(data["samples"][:4])
    from drn_model.train import input_stats
    m = build_model("drn", pool="max"); m.set_input_stats(input_stats(data["samples"]))
    out = m(b)
    assert out["unit_logits"].shape == (b["n_units"], 5) and torch.isfinite(out["unit_logits"]).all()
    total_loss(out, b, class_weights(np.ones(5)), 0.5, "unit")["total"].backward()


def test_mirror_and_rotation(cache):
    from drn_model.model import mirror_batch, rotate_batch
    from drn_model.train import input_stats
    data = D.load_cache(cache)
    b = D.collate(data["samples"][:4])
    m = build_model("drn", "max"); m.set_input_stats(input_stats(data["samples"]))
    m.eval()
    assert torch.isfinite(m(b)["unit_logits"]).all()
    bm = mirror_batch(b)
    assert torch.equal(mirror_batch(bm)["node_ctrs"], b["node_ctrs"]) and torch.equal(bm["left_u"], b["right_u"])
    assert torch.isfinite(m(bm)["unit_logits"]).all()
    br = rotate_batch(rotate_batch(b, 1.2), -1.2)
    assert torch.allclose(br["node_ctrs"], b["node_ctrs"], atol=1e-4) and torch.allclose(br["actor_kin"], b["actor_kin"], atol=1e-4)


def test_table4_ablation_terms(cache):
    from drn_model.train import input_stats
    data = D.load_cache(cache)
    b = D.collate(data["samples"][:3])
    for drop in (["prox"], ["dir"], ["threat"]):
        m = build_model("drn", "max", {"drop": drop}); m.set_input_stats(input_stats(data["samples"]))
        assert m.layers[0].drop == tuple(drop) and torch.isfinite(m(b)["unit_logits"]).all()


def test_only_drn():
    with pytest.raises(ValueError):
        build_model("drn_plus")


@pytest.mark.parametrize("rc", ["lane", "node"])
def test_risk_conditioning(cache, rc):
    from drn_model.train import input_stats
    data = D.load_cache(cache)
    b = D.collate(data["samples"][:4])
    m = build_model("drn", "max", {"risk_cond": rc}); m.set_input_stats(input_stats(data["samples"]))
    out = m(b)
    assert m.risk_cond == rc and torch.isfinite(out["traj_reg"]).all()


def test_ris_details(cache):
    from drn_model.train import input_stats
    data = D.load_cache(cache)
    b = D.collate(data["samples"][:4])
    m = build_model("drn", "max")
    L = m.layers[0]
    assert L.radius == 15.0 and L.dir_gate and L.threat_fix and float(L.threat.bias) == 0.5
    m2 = build_model("drn", "max", {"radius": 30.0, "dir_gate": False, "threat_fix": False})
    assert m2.layers[0].radius == 30.0 and not m2.layers[0].dir_gate
    for mm in (m, m2):
        mm.set_input_stats(input_stats(data["samples"]))
        assert torch.isfinite(mm(b)["unit_logits"]).all()


def test_kinematics_use_past_only(fake_cfg, built):
    
    from drn.lanegcn_format import lanegcn_sample, past_motion
    p = glob.glob(os.path.join(fake_cfg["output_dir"], "scenes", "*", "sample_4.json"))[0]
    rec = json.load(open(p))
    s = lanegcn_sample(rec, None, 2.0)
    assert s["kin_source"] == "past"
    assert np.allclose(s["kin"][0, :2], [6.0, 0.0], atol=1e-3)
    rec2 = json.loads(json.dumps(rec))
    for a in [rec2["ego"]] + rec2["actors"]:
        a["vx"], a["vy"], a["ax"], a["ay"] = 99.0, -99.0, 9.0, 9.0
        a["fut"] = [[x + 50.0, y - 50.0] for x, y in a["fut"]]
    s2 = lanegcn_sample(rec2, None, 2.0)
    assert np.array_equal(s["kin"], s2["kin"])
    pos = np.zeros((1, 20, 2), np.float32); pos[0, :, 0] = (np.arange(20) - 19) * 0.1 * 2.0
    v, a = past_motion(pos, np.ones((1, 20), bool), 10.0)
    assert np.allclose(v, [[2.0, 0.0]]) and np.allclose(a, 0.0)
    v, a = past_motion(pos, np.r_[np.zeros(16, bool), np.ones(4, bool)][None], 10.0)
    assert np.allclose(v, 0.0) and np.allclose(a, 0.0)


def test_causal_kin_converts_old_cache(cache, tmp_path):
    out = str(tmp_path / "c.pt")
    info = D.causal_kin(cache, out)
    assert info["samples"] > 0
    a, b = D.load_cache(cache), D.load_cache(out)
    for x, y in zip(a["samples"], b["samples"]):
        assert np.allclose(x["kin"], y["kin"], atol=1e-5) and y["kin_source"] == "past"


def test_min_ade_is_paper_definition():
    gt = np.zeros((1, 2, 2), np.float32); has = np.ones((1, 2), bool)
    reg = np.zeros((1, 2, 2, 2), np.float32)
    reg[0, 0] = [[3.0, 0.0], [0.5, 0.0]]
    reg[0, 1] = [[1.0, 0.0], [1.0, 0.0]]
    t = trajectory_metrics(reg, np.zeros((1, 2)), gt, has)
    assert abs(t["minADE"] - 1.0) < 1e-6 and abs(t["minFDE"] - 0.5) < 1e-6 and abs(t["minADE_at_minFDE"] - 1.75) < 1e-6


def test_split_override_keys(fake_cfg, labels_csv, tmp_path):
    paths = sorted(glob.glob(os.path.join(fake_cfg["output_dir"], "scenes", "*", "*.json")))
    recs = [json.load(open(p)) for p in paths]
    by_token = {r["scene_token"]: "test" for r in recs}
    sp = tmp_path / "splits.json"; sp.write_text(json.dumps(by_token))
    D.prepare(fake_cfg["output_dir"], labels_csv, str(tmp_path / "a.pt"), splits_path=str(sp))
    assert {s["split"] for s in D.load_cache(str(tmp_path / "a.pt"))["samples"]} == {"test"}
    sp.write_text(json.dumps({r["scene_name"]: "val" for r in recs}))
    D.prepare(fake_cfg["output_dir"], labels_csv, str(tmp_path / "b.pt"), splits_path=str(sp))
    assert {s["split"] for s in D.load_cache(str(tmp_path / "b.pt"))["samples"]} == {"val"}
    sp.write_text(json.dumps({"no-such-scene": "val"}))
    with pytest.raises(ValueError):
        D.prepare(fake_cfg["output_dir"], labels_csv, str(tmp_path / "c.pt"), splits_path=str(sp))


def test_eval_json_has_no_paper_constants(cache, tmp_path):
    run = str(tmp_path / "run")
    train(cache, "drn", run, epochs=1, max_train=8, device_name="cpu")
    res = evaluate(cache, run, "val", device_name="cpu")
    assert "paper" not in res


def test_prepare_defaults_and_unresolved_labels(fake_cfg, labels_csv, tmp_path):
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    unit = (rows[0]["sample_token"], rows[0]["unit_id"])
    for row in rows:
        row["cv_ttc_s"] = "2.0"
        row["ttc_hybrid_s"] = "7.0"
        if (row["sample_token"], row["unit_id"]) == unit:
            row["label"] = ""
    path = tmp_path / "human.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cache = str(tmp_path / "human.pt")
    D.prepare(fake_cfg["output_dir"], str(path), cache)
    data = D.load_cache(cache)
    assert data["time_column"] == "cv_ttc_s" and data["time_cap"] == 10.0
    assert all(np.allclose(x["unit_time"], 2.0) for x in data["samples"])
    sample = next(x for x in data["samples"] if x["sample_token"] == unit[0])
    assert sample["unit_level"][sample["unit_ids"].index(unit[1])] == D.IGNORE


def test_split_override_rejects_partial_conflicting_and_invalid(fake_cfg, labels_csv, tmp_path):
    records = [json.load(open(p)) for p in glob.glob(os.path.join(fake_cfg["output_dir"], "scenes", "*", "*.json"))]
    valid = {x["scene_token"]: "test" for x in records}
    cases = [dict(valid), dict(valid), dict(valid)]
    del cases[0][records[0]["scene_token"]]
    cases[1][records[0]["scene_name"]] = "val"

    record_path = glob.glob(os.path.join(fake_cfg["output_dir"], "scenes", "*", "*.json"))[0]
    rec = json.load(open(record_path)); rec["scene_name"] = "distinct_alias"
    with open(record_path, "w") as fh: json.dump(rec, fh)
    cases[1] = dict(valid, distinct_alias="val")
    cases[2][records[0]["scene_token"]] = "invalid"
    for i, mapping in enumerate(cases):
        path = tmp_path / ("splits%d.json" % i); path.write_text(json.dumps(mapping))
        with pytest.raises(ValueError):
            D.prepare(fake_cfg["output_dir"], labels_csv, str(tmp_path / "cache.pt"), splits_path=str(path))
    rec["scene_name"] = records[0]["scene_name"]
    with open(record_path, "w") as fh: json.dump(rec, fh)


def test_read_labels_rejects_conflicting_lane_rows(tmp_path):
    fields = ["sample_token", "unit_id", "label", "split"]
    original = dict(sample_token="s", unit_id="u", label="1", split="train")
    for replacement in (dict(original, split="test"), dict(original, label="2"), dict(original, label="")):
        path = tmp_path / "conflict.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows([original, replacement])
        with pytest.raises(ValueError):
            D.read_labels(str(path))


def test_relabel_updates_ttc_metadata_and_validates_column(cache, labels_csv, tmp_path):
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    for row in rows:
        row["ttc_hybrid_s"] = "7.0"
    path = tmp_path / "other_time.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out = str(tmp_path / "relabeled.pt")
    D.relabel(cache, str(path), out, time_column="ttc_hybrid_s")
    data = D.load_cache(out)
    assert data["time_column"] == "ttc_hybrid_s"
    assert all(np.allclose(s["unit_time"], 7.0) for s in data["samples"])
    with pytest.raises(ValueError):
        D.relabel(cache, str(path), str(tmp_path / "bad.pt"), time_column="missing_column")
