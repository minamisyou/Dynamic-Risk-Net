"""Training and evaluation loops."""
import json
import math
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from .data import IGNORE, RiskDataset, load_cache, make_loader, to_device
from .loss import class_weights, total_loss
from .metrics import classification_metrics, table7, table8, trajectory_metrics
from .model import ARCH, build_model, checkpoint_model_name, mirror_batch, rotate_batch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def device_of(name: Optional[str] = None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_eval(model, loader, device, cap: float, amp: bool = False) -> Dict[str, Any]:
    model.eval()
    lv, pr, tm, fr, tok = [], [], [], [], []
    regs, clss, gts, hass = [], [], [], []
    offset = 0
    for batch in loader:
        b = to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(b)
        pred = out["unit_logits"].float().argmax(1).cpu().numpy()
        level = b["unit_level"].cpu().numpy()
        keep = level != IGNORE
        lv.append(level[keep]); pr.append(pred[keep]); tm.append(b["unit_time"].cpu().numpy()[keep])
        fr.append(b["unit_batch"].cpu().numpy()[keep] + offset)
        offset += b["B"]
        if "traj_reg" in out:
            regs.append(out["traj_reg"].float().cpu().numpy()); clss.append(out["traj_cls"].float().cpu().numpy())
            gts.append(b["gt_preds"].cpu().numpy()); hass.append(b["has_preds"].cpu().numpy())
    lv = np.concatenate(lv) if lv else np.zeros(0, np.int64); pr = np.concatenate(pr) if pr else np.zeros(0, np.int64)
    tm = np.concatenate(tm) if tm else np.zeros(0); fr = np.concatenate(fr) if fr else np.zeros(0, np.int64)
    res = {"classification": classification_metrics(lv, pr), "table7_scene": table7(fr, tm, pr, cap, "scene"),
           "table7_lane": table7(fr, tm, pr, cap, "lane"), "table7_gt_scene": table7(fr, tm, lv, cap, "scene"),
           "table8_gt": table8(lv, tm, cap), "table8_pred": table8(pr, tm, cap), "n_scenes": int(offset)}
    if regs:
        res["trajectory"] = trajectory_metrics(np.concatenate(regs), np.concatenate(clss), np.concatenate(gts), np.concatenate(hass))
    return res


def input_stats(samples) -> Dict[str, Any]:
    """Mean / std per input dimension over the training samples (paper III-B normalisation)."""
    node = np.concatenate([np.concatenate([s["graph"]["ctrs"], s["graph"]["feats"], np.asarray(s["graph"]["intersect"]).reshape(-1, 1),
                                           s["graph"]["attr"]], 1) for s in samples])
    hist = []
    for s in samples:
        m = s["feats"][:, :, 2] > 0
        hist.append(np.concatenate([s["hist_pos"], s["feats"][:, :, :2]], -1)[m])
    hist = np.concatenate(hist); kin = np.concatenate([s["kin"] for s in samples])
    f = lambda x: (x.mean(0).tolist(), x.std(0).tolist())
    return {"node": f(node), "hist": f(hist), "kin": f(kin)}


def train(cache: str, model_name: str, run_dir: str, seed: int = 0, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
          weight_decay: float = 1e-2, lambda_traj: float = 0.5, risk_on: str = "unit", cw_power: float = 0.25, pool: str = "max", model_opts: Optional[Dict[str, Any]] = None, augment: Optional[str] = None, oversample: bool = True, amp: bool = False, device_name: Optional[str] = None,
          deterministic: bool = True, num_workers: int = 0, max_train: Optional[int] = None, log_every: int = 20,
          resume: bool = False, ema: float = 0.0) -> Dict[str, Any]:
    seed_everything(seed, deterministic)
    device = device_of(device_name)
    data = load_cache(cache)
    cap = float(data.get("time_cap", 8.0))
    train_ds = RiskDataset(data["samples"], "train"); val_ds = RiskDataset(data["samples"], "val"); test_ds = RiskDataset(data["samples"], "test")
    if max_train:
        train_ds.samples = train_ds.samples[:max_train]
    counts = train_ds.class_counts()
    weights = class_weights(counts, cw_power)
    model_opts = model_opts or {}
    model = build_model(model_name, pool, model_opts)
    aug_rng = random.Random(seed * 7919 + 17)
    stats = None
    if hasattr(model, "set_input_stats"):
        stats = input_stats(train_ds.samples)
        model.set_input_stats(stats)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ema_model = None
    if ema:
        import copy
        ema_model = copy.deepcopy(model)
        for q in ema_model.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    os.makedirs(run_dir, exist_ok=True)
    start_ep, best, resumed = 0, (-1.0, -1), None
    last_path = os.path.join(run_dir, "last.pt")
    if resume and os.path.exists(last_path):
        try:
            ck = torch.load(last_path, map_location=device, weights_only=False)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            ck = torch.load(last_path, map_location=device)
        if "opt" in ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start_ep, best, resumed = int(ck["epoch"]), tuple(ck.get("best", (-1.0, -1))), int(ck["epoch"])
            hist_path = os.path.join(run_dir, "history.jsonl")
            if os.path.exists(hist_path):                                     # keep exactly the epochs the checkpoint has seen
                lines = [l for l in open(hist_path) if l.strip() and json.loads(l)["epoch"] <= start_ep]
                open(hist_path, "w").writelines(lines)
    elif os.path.exists(os.path.join(run_dir, "history.jsonl")):
        os.remove(os.path.join(run_dir, "history.jsonl"))                    # a fresh run starts a fresh history
    cfg = {"cache": os.path.abspath(cache), "model": model_name, "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
           "weight_decay": weight_decay, "lambda_traj": lambda_traj, "risk_on": risk_on, "cw_power": cw_power, "pool": pool, "model_opts": model_opts, "augment": augment, "ema": ema,
           "arch": ARCH, "oversample": oversample, "amp": bool(scaler.is_enabled()),
           "device": str(device), "deterministic": deterministic, "n_params": n_params, "class_counts_train": counts.tolist(),
           "class_weights": [round(float(w), 4) for w in weights], "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
           "resumed_at_epoch": resumed, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    val_loader = make_loader(val_ds, batch_size, False, False, seed)
    history = []
    t0 = time.time()
    for ep in range(start_ep, epochs):
        model.train()
        loader = make_loader(train_ds, batch_size, True, oversample, seed * 1000 + ep, num_workers)
        sums = {}; n = 0
        for it, batch in enumerate(loader):
            b = to_device(batch, device)
            aug = (augment or "").split(",")
            if "mirror" in aug and aug_rng.random() < 0.5:
                b = mirror_batch(b)
            rot_deg = 180.0 if "rot180" in aug else 90.0 if "rot90" in aug else 45.0 if "rot" in aug else 0.0
            if rot_deg:
                r = math.radians(rot_deg)
                b = rotate_batch(b, aug_rng.uniform(-r, r))
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                out = model(b)
                terms = total_loss(out, b, weights, lambda_traj, risk_on)
            opt.zero_grad(set_to_none=True)
            scaler.scale(terms["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.mul_(ema).add_(pm.detach(), alpha=1.0 - ema)
                        else:
                            pe.copy_(pm)
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
        sched.step()
        eval_model = ema_model if ema_model is not None else model
        ev = run_eval(eval_model, val_loader, device, cap, scaler.is_enabled())
        rca = ev["classification"]["rca"]
        rec = {"epoch": ep + 1, "loss": {k: round(v / max(n, 1), 4) for k, v in sums.items()}, "val_rca": round(rca, 4),
               "val_wf1": round(ev["classification"]["weighted_f1"], 4), "val_minADE": (ev.get("trajectory") or {}).get("minADE"),
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        with open(os.path.join(run_dir, "history.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if rca > best[0]:
            best = (rca, ep + 1)
            torch.save({"model": eval_model.state_dict(), "epoch": ep + 1, "config": cfg, "val_rca": rca}, os.path.join(run_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "config": cfg, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": list(best)}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
    summary = {"best_val_rca": best[0], "best_epoch": best[1], "epochs": epochs, "seconds": round(time.time() - t0, 1), "config": cfg}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    return summary


def evaluate(cache: str, run_dir: str, split: str = "test", checkpoint: str = "best.pt", batch_size: int = 32, device_name: Optional[str] = None,
             out_name: Optional[str] = None) -> Dict[str, Any]:
    device = device_of(device_name)
    data = load_cache(cache)
    cap = float(data.get("time_cap", 8.0))
    try:
        ck = torch.load(os.path.join(run_dir, checkpoint), map_location=device, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        ck = torch.load(os.path.join(run_dir, checkpoint), map_location=device)
    name = checkpoint_model_name(ck["config"])
    model = build_model(name, ck["config"].get("pool", "max"), ck["config"].get("model_opts")).to(device)
    model.load_state_dict(ck["model"])
    ds = RiskDataset(data["samples"], split)
    res = run_eval(model, make_loader(ds, batch_size, False, False, 0), device, cap, amp=False)
    res.update({"split": split, "checkpoint": checkpoint, "epoch": ck.get("epoch"), "model": ck["config"]["model"], "seed": ck["config"]["seed"],
                "n_scenes": len(ds), "time_column": data.get("time_column"), "time_cap": cap})
    path = os.path.join(run_dir, out_name or ("eval_%s.json" % split))
    json.dump(res, open(path, "w"), indent=1)
    return res
