"""Assemble a release with the human-labelled sample, code, DRN weights and configurations."""
import argparse
import json
import os
import shutil
import sys

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.join(RELEASE_ROOT, "code")
RUNS = ["checkpoints/drn_s%d" % seed for seed in range(3)]
CODE = ["drn", "drn_model", "tests", "tools/release/make_release.py", "config.yaml", "requirements.txt"]
SAMPLE = os.path.join(RELEASE_ROOT, "sample200")
SAMPLE_FILES = ["README.md", "labels.csv", "scenes.csv", "scenes.jsonl", "splits.json"]
DOC_FILES = ["README.md", "DATA_CARD.md", "ENVIRONMENT.txt", "LICENSE", "THIRD_PARTY_NOTICES.md"]
RUN_CONFIG_FIELDS = ("model", "seed", "epochs", "batch_size", "lr", "weight_decay", "lambda_traj", "risk_on", "cw_power", "pool", "model_opts", "augment", "ema", "arch", "oversample", "amp", "deterministic")
MODEL_CONFIG_FIELDS = ("model", "seed", "pool", "model_opts")
PRIVATE = ["/home/", "I:\\", "I:/", "minamisyou", "drn_work", "C:\\Users", "drn_plus", "drn_geo"]


def copy(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"), dirs_exist_ok=True)
    elif os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    else:
        print("missing (skipped):", src)


def clean_config(config, fields):
    return {key: config[key] for key in fields if key in config}


def sanitize_checkpoint(src, dst):
    import torch

    if hasattr(torch.serialization, "add_safe_globals") and hasattr(torch, "torch_version"):
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    clean = {"model": checkpoint["model"], "config": clean_config(checkpoint["config"], MODEL_CONFIG_FIELDS)}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    temp = dst + ".tmp"
    torch.save(clean, temp)
    verify = torch.load(temp, map_location="cpu", weights_only=True)
    if set(verify) != {"model", "config"} or set(verify["model"]) != set(clean["model"]):
        os.remove(temp)
        raise RuntimeError("checkpoint verification failed")
    if any(not torch.equal(clean["model"][key], verify["model"][key]) for key in clean["model"]):
        os.remove(temp)
        raise RuntimeError("checkpoint weights changed during metadata cleanup")
    os.replace(temp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="empty destination directory for the release")
    ap.add_argument("--docs", default=RELEASE_ROOT)
    ap.add_argument("--runs-root", default=RELEASE_ROOT, help="folder containing checkpoints/drn_s0, drn_s1 and drn_s2")
    ap.add_argument("--sample-dir", default=SAMPLE, help="directory containing the selected sample files")
    ap.add_argument("--include-attachment", action="append", default=[],
                    help="explicitly include a reviewed PDF/XLSX from --docs; repeat for multiple files")
    args = ap.parse_args()
    docs = os.path.realpath(args.docs)
    for item in DOC_FILES:
        if not os.path.isfile(os.path.join(docs, item)):
            sys.exit("required release document is missing: " + item)
    attachments = []
    for item in args.include_attachment:
        path = os.path.realpath(os.path.join(docs, item))
        if os.path.dirname(path) != docs or not path.lower().endswith((".pdf", ".xlsx")) or not os.path.isfile(path):
            sys.exit("attachment must be an existing PDF/XLSX directly inside --docs: " + item)
        attachments.append(os.path.basename(path))
    dest = os.path.realpath(args.dest)
    if os.path.exists(dest) and os.listdir(dest):
        sys.exit("destination must be empty: " + dest)
    os.makedirs(dest, exist_ok=True)

    for item in CODE:
        copy(os.path.join(CODE_ROOT, item), os.path.join(dest, "code", item))
    for item in SAMPLE_FILES:
        copy(os.path.join(args.sample_dir, item), os.path.join(dest, "sample200", item))

    for seed, run in enumerate(RUNS):
        run_dir = os.path.join(args.runs_root, run)
        source_checkpoint = os.path.join(run_dir, "best.pt")
        if not os.path.exists(source_checkpoint):
            continue
        checkpoint_dir = os.path.join(dest, "checkpoints", "drn_s%d" % seed)
        os.makedirs(checkpoint_dir, exist_ok=True)
        sanitize_checkpoint(source_checkpoint, os.path.join(checkpoint_dir, "best.pt"))
        config_path = os.path.join(checkpoint_dir, "config.json")
        source_config = os.path.join(run_dir, "config.json")
        if os.path.exists(source_config):
            with open(source_config, encoding="utf-8") as f:
                config = json.load(f)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(clean_config(config, RUN_CONFIG_FIELDS), f, indent=1)

    for item in DOC_FILES + attachments:
        copy(os.path.join(docs, item), os.path.join(dest, item))

    leaks = []
    for root, _, files in os.walk(dest):
        for name in files:
            path = os.path.join(root, name)
            if name.endswith((".py", ".md", ".json", ".jsonl", ".csv", ".yaml", ".txt", ".sh", ".cmd")) and name not in ("make_release.py", "test_model.py"):
                with open(path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                leaks.extend("%s: %s" % (os.path.relpath(path, dest), token) for token in PRIVATE if token in content)
    if leaks:
        sys.exit("private paths in the release:\n" + "\n".join(leaks))

    print("release written to %s" % dest)


if __name__ == "__main__":
    main()
