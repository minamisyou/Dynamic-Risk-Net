import argparse
import json


def main():
    ap = argparse.ArgumentParser(prog="drn_model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="scene records + labels -> cache file")
    p.add_argument("--out", required=True, help="output_dir of `python -m drn build`")
    p.add_argument("--labels", required=True, help="lane-row human annotation CSV (labels.csv)")
    p.add_argument("--cache", required=True)
    p.add_argument("--node-spacing", type=float, default=2.0)
    p.add_argument("--time-column", default="cv_ttc_s", help="column with the CV-TTC for Tables 7/8")
    p.add_argument("--time-cap", type=float, default=10.0)
    p.add_argument("--splits", help="optional JSON {scene_token or scene_name: split} overriding the split column (e.g. the released splits.json)")
    c = sub.add_parser("causal-kin", help="rewrite actor velocity / acceleration of an old cache from past positions only")
    c.add_argument("--cache", required=True); c.add_argument("--out", required=True); c.add_argument("--hz", type=float, default=10.0)
    t = sub.add_parser("train", help="train DRN (defaults = the released setting)")
    t.add_argument("--cache", required=True); t.add_argument("--model", default="drn", choices=["drn"])
    t.add_argument("--run", required=True); t.add_argument("--seed", type=int, default=0); t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=32); t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--lambda-traj", type=float, default=0.5, help="lambda_motion (paper: 0.5)")
    t.add_argument("--risk-on", choices=["node", "unit"], default="unit", help="risk loss per labelled lane segment (unit) or per lane node (node, paper Eq. 12 as written)")
    t.add_argument("--pool", choices=["mean", "max"], default="max", help="pooling of node embeddings per labelled lane segment")
    t.add_argument("--risk-cond", choices=["lane", "node"], default="lane", help="risk distribution fed to the trajectory head (Eq. 9): the actor's lane or its nearest 2 m node")
    t.add_argument("--ris-radius", type=float, default=15.0, help="A(i): actors within this many metres of a lane node (default 15)")
    t.add_argument("--no-dir-gate", action="store_true", help="keep DirectionalRisk for actors slower than 0.5 m/s")
    t.add_argument("--no-threat-fix", action="store_true", help="raw |v|, |a| into Threat and default bias")
    t.add_argument("--ablate", action="append", choices=["prox", "dir", "threat"], help="Table 4: drop a term of the Risk Influence Score")
    t.add_argument("--augment", default="mirror,rot90", help="comma list: mirror, rot (±45 deg), rot90, rot180; 'none' for no augmentation")
    t.add_argument("--weight-decay", type=float, default=1e-2); t.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights used for val selection and saving (0 = off)")
    t.add_argument("--cw-power", type=float, default=0.25, help="class weight = inverse frequency ** power")
    t.add_argument("--no-oversample", action="store_true"); t.add_argument("--amp", action="store_true", help="mixed precision (off by default)"); t.add_argument("--device")
    t.add_argument("--resume", action="store_true", help="continue from last.pt of --run (after a crash)"); t.add_argument("--no-deterministic", action="store_true"); t.add_argument("--max-train", type=int); t.add_argument("--workers", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--cache", required=True); e.add_argument("--run", required=True); e.add_argument("--split", default="test")
    e.add_argument("--checkpoint", default="best.pt"); e.add_argument("--device"); e.add_argument("--out-name")
    a = ap.parse_args()
    if a.cmd == "prepare":
        from .data import prepare
        print(json.dumps(prepare(a.out, a.labels, a.cache, a.node_spacing, a.time_column, a.time_cap, a.splits), indent=1))
    elif a.cmd == "causal-kin":
        from .data import causal_kin
        print(json.dumps(causal_kin(a.cache, a.out, a.hz), indent=1))
    elif a.cmd == "train":
        from .train import train
        aug = None if a.augment in (None, "", "none") else a.augment
        s = train(a.cache, a.model, a.run, a.seed, a.epochs, a.batch_size, a.lr, weight_decay=a.weight_decay, lambda_traj=a.lambda_traj,
                  risk_on=a.risk_on, cw_power=a.cw_power, pool=a.pool, model_opts=dict({"risk_cond": a.risk_cond, "radius": a.ris_radius, "dir_gate": not a.no_dir_gate, "threat_fix": not a.no_threat_fix},
                                                                                         **({"drop": a.ablate} if a.ablate else {})), augment=aug,
                  oversample=not a.no_oversample, amp=a.amp, device_name=a.device, deterministic=not a.no_deterministic,
                  num_workers=a.workers, max_train=a.max_train, resume=a.resume, ema=a.ema)
        print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1))
    elif a.cmd == "evaluate":
        from .train import evaluate
        res = evaluate(a.cache, a.run, a.split, a.checkpoint, device_name=a.device, out_name=a.out_name)
        c = res["classification"]
        print(json.dumps({"split": a.split, "n_units": c["n"], "rca": round(c["rca"], 4), "weighted_f1": round(c["weighted_f1"], 4),
                          "recall": c["recall"], "trajectory": res.get("trajectory"), "table7_scene": res["table7_scene"]["values"],
                          "table8_gt": res["table8_gt"]["mean_min_cv_ttc_by_level"]}, indent=1))


if __name__ == "__main__":
    main()
