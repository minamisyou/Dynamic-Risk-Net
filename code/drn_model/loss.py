"""Composite loss (paper III-F): class-weighted cross-entropy on lane-unit risk + trajectory classification and
best-of-K Smooth-L1 regression. The risk term is averaged over lane units so that scenes with many lanes do not dominate.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE, LEVELS


def class_weights(counts: np.ndarray, power: float = 1.0) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (the "weighted cross-entropy from our previous work")."""
    c = np.maximum(np.asarray(counts, float), 1.0)
    w = (c.sum() / c) ** power
    return torch.as_tensor(w / w.mean(), dtype=torch.float32)


def risk_loss(unit_logits: torch.Tensor, unit_level: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    keep = unit_level != IGNORE
    if keep.sum() == 0:
        return unit_logits.sum() * 0.0
    return F.cross_entropy(unit_logits[keep], unit_level[keep], weight=weights.to(unit_logits.device))


def node_targets(batch) -> torch.Tensor:
    """Ground-truth level of every lane node: the level of the labelled unit its lane segment belongs to (IGNORE otherwise)."""
    lu = batch["lane_unit"][batch["node_lane"]]
    tgt = torch.full_like(lu, IGNORE)
    keep = lu >= 0
    tgt[keep] = batch["unit_level"][lu[keep]]
    return tgt


def risk_loss_nodes(node_logits: torch.Tensor, batch, weights: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 12: L_risk = -1/|V_L| sum_i sum_c w_c y_ic log p_ic, averaged over the labelled lane nodes."""
    tgt = node_targets(batch)
    keep = tgt != IGNORE
    if keep.sum() == 0:
        return node_logits.sum() * 0.0
    ce = F.cross_entropy(node_logits[keep].float(), tgt[keep], reduction="none")
    return (weights.to(ce.device)[tgt[keep]] * ce).mean()


def trajectory_loss(reg: torch.Tensor, cls: torch.Tensor, gt: torch.Tensor, has: torch.Tensor) -> Dict[str, torch.Tensor]:
    """reg (A, K, F, 2), cls (A, K), gt (A, F, 2), has (A, F). Best mode = smallest final displacement among valid steps."""
    valid = has.any(1)
    if valid.sum() == 0:
        z = reg.sum() * 0.0
        return {"reg": z, "cls": z}
    reg, cls, gt, has = reg[valid], cls[valid], gt[valid], has[valid]
    last = has.float().cumsum(1).argmax(1)                                           # index of the last valid step
    gt_last = gt[torch.arange(len(gt)), last]                                          # (A, 2)
    reg_last = reg[torch.arange(len(gt)), :, last]                                     # (A, K, 2)
    dist = (reg_last - gt_last.unsqueeze(1)).norm(dim=-1)                              # (A, K)
    best = dist.argmin(1)
    pred = reg[torch.arange(len(gt)), best]                                            # (A, F, 2)
    l_reg = F.smooth_l1_loss(pred[has], gt[has], reduction="mean")
    l_cls = F.cross_entropy(cls, best)
    return {"reg": l_reg, "cls": l_cls}


def total_loss(out: Dict[str, torch.Tensor], batch: Dict, weights: torch.Tensor, lambda_traj: float = 0.5,
               risk_on: str = "unit") -> Dict[str, torch.Tensor]:
    """L_total = lambda_risk L_risk + lambda_motion L_motion (Eq. 11, lambda_risk = 1). risk_on = "node" is Eq. 12 (paper);
    "unit" averages over labelled lane units (the v0 runs)."""
    l_risk = risk_loss_nodes(out["node_logits"], batch, weights) if risk_on == "node" else risk_loss(out["unit_logits"], batch["unit_level"], weights)
    terms = {"risk": l_risk}
    total = l_risk
    if "traj_reg" in out:
        tl = trajectory_loss(out["traj_reg"], out["traj_cls"], batch["gt_preds"], batch["has_preds"])
        terms.update({"traj_reg": tl["reg"], "traj_cls": tl["cls"]})
        total = total + lambda_traj * (tl["reg"] + tl["cls"])
    terms["total"] = total
    return terms
