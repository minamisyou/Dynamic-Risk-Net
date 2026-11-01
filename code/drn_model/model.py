"""Dynamic-Risk Net (DRN) with the paper's equations.

Inputs of paper Eq. 1 (position and velocity history, kinematic state, heading, box, class), the lane attributes of III-B,
the Risk Influence Score fusion of Eq. 2-7 (T = 3 layers), the risk head of Eq. 8 (d_L -> 128 -> C) and risk-aware
forecasting of Eq. 9-10 (K = 6 modes). Adaptation to the nuScenes lane-segment dataset: a labelled lane segment covers many 2 m
lane nodes, so node embeddings are max-pooled per segment before the head (RiskHeadP, pool="max"); the risk
distribution of Eq. 9 is taken from the actor's lane, pooled the same way.
"""
import math
import warnings
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

D_LANE, D_ACTOR, N_LEVELS, K_MODES = 64, 128, 5, 6
warnings.filterwarnings("ignore", category=UserWarning, module="torch")       # k-hop lane adjacency uses torch.sparse; its beta warnings are noise here

def add_at(target: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """target[index] += src, out of place. index_put with accumulate has a deterministic CUDA path (index_add_ does not),
    and the source is cast to the target's dtype so mixed precision cannot mix types."""
    return target.index_put((index,), src.to(target.dtype), accumulate=True)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = add_at(src.new_zeros((n,) + src.shape[1:]), index, src)
    cnt = add_at(torch.zeros(n, device=src.device, dtype=src.dtype), index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))


def scatter_max(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = src.new_full((n,) + src.shape[1:], torch.finfo(src.dtype).min)
    return out.scatter_reduce(0, index.view(-1, 1).expand_as(src), src, reduce="amax", include_self=True)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=None):
        super().__init__()
        h = d_hidden or d_out
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- encoders

class MapNet(nn.Module):
    """Lane-node encoder with dilated predecessor / successor propagation (LaneGCN MapNet, simplified) -> d_L."""

    def __init__(self, d=D_LANE, hops=(1, 2, 4), layers=2):
        super().__init__()
        self.inp = MLP(5, d)                                 # ctr (2), segment vector (2), intersection flag (1)
        self.hops = hops
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({"self": nn.Linear(d, d), "pre": nn.ModuleList([nn.Linear(d, d) for _ in hops]),
                                              "suc": nn.ModuleList([nn.Linear(d, d) for _ in hops]), "norm": nn.LayerNorm(d)}))

    @staticmethod
    def _hop_edges(u, v, n, k):
        """k-hop edges by composing the adjacency k times (small graphs; dense bool matrices per batch are avoided by sparse matmul)."""
        if k == 1 or len(u) == 0:
            return u, v
        idx = torch.stack([u, v]); val = torch.ones(len(u), device=u.device)
        A = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        P = A
        for _ in range(k - 1):
            P = torch.sparse.mm(P, A).coalesce()
        return P.indices()[0], P.indices()[1]

    def forward(self, ctrs, feats, inter, pre_u, pre_v, suc_u, suc_v):
        n = ctrs.shape[0]
        x = self.inp(torch.cat([ctrs / 50.0, feats / 5.0, inter.view(-1, 1)], 1))
        edges = {}
        for k in self.hops:
            edges[("pre", k)] = self._hop_edges(pre_u, pre_v, n, k)
            edges[("suc", k)] = self._hop_edges(suc_u, suc_v, n, k)
        for layer in self.layers:
            agg = layer["self"](x)
            for name in ("pre", "suc"):
                for j, k in enumerate(self.hops):
                    u, v = edges[(name, k)]
                    if len(u):
                        msg = layer[name][j](x[v])
                        agg = add_at(agg, u, msg)
            x = F.relu(layer["norm"](agg)) + x
        return x


def pairs(cq, bq, ck, bk, radius):
    """(query, key) index pairs of the same sample within `radius` metres, built one sample at a time so memory stays
    linear in the number of pairs rather than in (all queries x all keys of the batch)."""
    qi, ki = [], []
    for b in torch.unique(bq):
        iq = torch.nonzero(bq == b).view(-1); ik = torch.nonzero(bk == b).view(-1)
        if len(iq) == 0 or len(ik) == 0:
            continue
        d = torch.cdist(cq[iq].float(), ck[ik].float())
        a, c = torch.nonzero(d <= radius, as_tuple=True)
        qi.append(iq[a]); ki.append(ik[c])
    if not qi:
        e = torch.zeros(0, dtype=torch.long, device=cq.device)
        return e, e
    return torch.cat(qi), torch.cat(ki)


class TrajHead(nn.Module):
    def __init__(self, d=D_ACTOR, n_fut=30, k=K_MODES):
        super().__init__()
        self.k, self.n_fut = k, n_fut
        self.reg = nn.ModuleList([MLP(d, 2 * n_fut, 2 * d) for _ in range(k)])
        self.cls = MLP(d, k, d)

    def forward(self, actor, ctrs):
        reg = torch.stack([m(actor).view(-1, self.n_fut, 2) for m in self.reg], 1)       # (A, K, F, 2)
        reg = reg + ctrs.view(-1, 1, 1, 2)
        return reg, self.cls(actor)


N_CLASSES_ACTOR = 5          # ego, vehicle, pedestrian, bicycle, motorcycle (drn.lanegcn_format.ACTOR_CLASSES)
D_RISK = D_ACTOR // 4        # paper: d_risk = d_A / 4 = 32
ARCH = "drn"


class InputNorm(nn.Module):
    """Per-dimension standardisation with training-set statistics (paper III-B: 'per-dimension standardized using
    training-set statistics'). The statistics are buffers, so they travel with the checkpoint."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim)); self.register_buffer("std", torch.ones(dim))

    def set(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32)); self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-3))

    def forward(self, x):
        return (x - self.mean) / self.std


class ActorNetP(nn.Module):
    """Paper Eq. 1, h_A = [T_j, psi_j, d_j, c_j]: 1-D convolutions over the history of position and velocity (step),
    plus the current kinematic state (velocity, acceleration, heading, box) and a learnable class embedding -> d_A."""

    def __init__(self, d=D_ACTOR, n_kin=10):
        super().__init__()
        self.hist_norm = InputNorm(4); self.kin_norm = InputNorm(n_kin)
        self.conv = nn.Sequential(nn.Conv1d(5, 32, 3, padding=1), nn.GroupNorm(1, 32), nn.ReLU(inplace=True),
                                  nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(1, 64), nn.ReLU(inplace=True),
                                  nn.Conv1d(64, d, 3, stride=2, padding=1), nn.GroupNorm(1, d), nn.ReLU(inplace=True))
        self.kin = MLP(n_kin, d)
        self.cls = nn.Embedding(N_CLASSES_ACTOR, d)
        self.out = MLP(d, d)

    def forward(self, b):
        mask = b["actor_feats"][:, :, 2:3]
        h = torch.cat([self.hist_norm(torch.cat([b["actor_pos"], b["actor_feats"][:, :, :2]], -1)) * mask, mask], -1)
        h = self.conv(h.transpose(1, 2))[:, :, -1]
        return F.relu(self.out(h + self.kin(self.kin_norm(b["actor_kin"])) + self.cls(b["actor_cls"])))


class MapNetP(MapNet):
    """MapNet with the lane attributes of paper III-B (centre, direction, intersection, connector, junction, merge)."""

    def __init__(self, d=D_LANE, n_attr=3):
        super().__init__(d)
        self.inp = MLP(5 + n_attr, d)
        self.norm_in = InputNorm(5 + n_attr)

    def forward_b(self, b):
        x0 = torch.cat([b["node_ctrs"], b["node_feats"], b["node_inter"].view(-1, 1), b["node_attr"]], 1)
        return self._run(self.inp(self.norm_in(x0)), b["pre_u"], b["pre_v"], b["suc_u"], b["suc_v"])

    def _run(self, x, pre_u, pre_v, suc_u, suc_v):
        n = x.shape[0]
        edges = {}
        for k in self.hops:
            edges[("pre", k)] = self._hop_edges(pre_u, pre_v, n, k)
            edges[("suc", k)] = self._hop_edges(suc_u, suc_v, n, k)
        for layer in self.layers:
            agg = layer["self"](x)
            for name in ("pre", "suc"):
                for j, k in enumerate(self.hops):
                    u, v = edges[(name, k)]
                    if len(u):
                        agg = add_at(agg, u, layer[name][j](x[v]))
            x = F.relu(layer["norm"](agg)) + x
        return x

def _sp_inv(y):
    return math.log(math.expm1(y))


class RiskInfluenceFusion(nn.Module):
    """One layer of the Dynamic Risk Fusion Module (paper Eq. 2-7).

    C_ij = w1 Proximity + w2 DirectionalRisk + w3 Threat, with w = softplus(.) >= 0;
      Proximity = exp(-d(p_j, l_i)^2 / (2 sigma_p^2)), d = distance from the actor to lane node i's centreline segment,
      DirectionalRisk = (1 - cos(v_j, u_i)) / 2, Threat = ReLU(W_T [|v_j|, |a_j|] + b_T).
    h_i <- sigma(W_self h_i + sum_{k in N(i)} W_lane h_k + sum_{j in A(i)} W_risk C_ij h_j)    (Eq. 7)
    N(i): predecessor / successor nodes and the left / right neighbour lanes; A(i): actors within `radius` metres.
    A residual connection and LayerNorm wrap the update (implementation detail for depth T = 3).
    Implementation details not fixed by the paper (released setting): radius 15 m; DirectionalRisk = 0 for actors slower than
    0.5 m/s, whose direction of motion is undefined (dir_gate); Threat input [|v| / 10, |a| / 3] with the bias starting at 0.5 so
    the ReLU does not start dead (threat_fix).
    """

    def __init__(self, d_l=D_LANE, d_a=D_ACTOR, radius=15.0, drop=(), dir_gate=True, threat_fix=True):
        super().__init__()
        self.radius, self.drop = radius, tuple(drop)            # drop: Table 4 ablation ("prox", "dir", "threat")
        self.dir_gate, self.threat_fix = dir_gate, threat_fix   # see the class docstring
        self.w = nn.Parameter(torch.zeros(3))                                    # softplus(0) = 0.69 each
        self.sigma_raw = nn.Parameter(torch.tensor(_sp_inv(3.0)))               # sigma_p starts at 3 m
        self.threat = nn.Linear(2, 1)
        if threat_fix:
            with torch.no_grad():
                self.threat.bias.fill_(0.5)
        self.w_self = nn.Linear(d_l, d_l); self.w_pre = nn.Linear(d_l, d_l); self.w_suc = nn.Linear(d_l, d_l); self.w_side = nn.Linear(d_l, d_l)
        self.w_risk = nn.Linear(d_a, d_l)
        self.norm = nn.LayerNorm(d_l)

    def score(self, b, eq, ek):
        c = b["node_ctrs"][eq]; u = b["node_feats"][eq]; p = b["actor_ctrs"][ek]
        kin = b["actor_kin"][ek]; v = kin[:, 0:2]; a = kin[:, 2:4]
        L2 = (u * u).sum(-1).clamp(min=1e-6)
        t = (((p - (c - u / 2)) * u).sum(-1) / L2).clamp(0.0, 1.0)
        d = (p - (c - u / 2) - t.unsqueeze(-1) * u).norm(dim=-1)              # actor -> centreline segment
        sigma = F.softplus(self.sigma_raw) + 0.1
        prox = torch.exp(-d ** 2 / (2 * sigma ** 2))
        alpha = (v * u).sum(-1) / (v.norm(dim=-1) * u.norm(dim=-1) + 1e-3)
        dirr = (1.0 - alpha) / 2.0
        if self.dir_gate:
            dirr = dirr * (v.norm(dim=-1) >= 0.5).to(dirr.dtype)
        sv, sa = v.norm(dim=-1), a.norm(dim=-1)
        if self.threat_fix:
            sv, sa = sv / 10.0, sa / 3.0
        thr = F.relu(self.threat(torch.stack([sv, sa], -1))).squeeze(-1)
        w = F.softplus(self.w)
        C = (0 if "prox" in self.drop else w[0] * prox) + (0 if "dir" in self.drop else w[1] * dirr) + (0 if "threat" in self.drop else w[2] * thr)
        return C

    def forward(self, x, actor, b, pairs):
        eq, ek = pairs
        agg = self.w_self(x)
        if len(b["pre_u"]):
            agg = add_at(agg, b["pre_u"], self.w_pre(x[b["pre_v"]]))
        if len(b["suc_u"]):
            agg = add_at(agg, b["suc_u"], self.w_suc(x[b["suc_v"]]))
        lane_x = scatter_mean(x, b["node_lane"], b["n_lanes"])
        side = torch.zeros_like(lane_x)
        for u, v in ((b["left_u"], b["left_v"]), (b["right_u"], b["right_v"])):
            if len(u):
                side = add_at(side, u, self.w_side(lane_x[v]))
        agg = agg + side[b["node_lane"]]
        if len(eq):
            C = self.score(b, eq, ek)
            msg = self.w_risk(actor)[ek]
            agg = add_at(agg, eq, C.unsqueeze(-1) * msg)
        return self.norm(x + F.relu(agg))


def mirror_batch(b):
    """Left-right mirror of a collated batch (y -> -y in the ego frame). Risk levels are unchanged by mirroring; the
    left / right lane neighbours swap. Used as training augmentation (--augment mirror)."""
    b = dict(b)
    def neg(t, idx):
        t = t.clone(); t[idx] = -t[idx]; return t
    b["actor_feats"] = neg(b["actor_feats"], (slice(None), slice(None), 1))
    b["actor_ctrs"] = neg(b["actor_ctrs"], (slice(None), 1))
    b["gt_preds"] = neg(b["gt_preds"], (slice(None), slice(None), 1))
    b["node_ctrs"] = neg(b["node_ctrs"], (slice(None), 1)); b["node_feats"] = neg(b["node_feats"], (slice(None), 1))
    if "actor_kin" in b:
        k = b["actor_kin"].clone(); k[:, [1, 3, 5]] = -k[:, [1, 3, 5]]; b["actor_kin"] = k
        b["actor_pos"] = neg(b["actor_pos"], (slice(None), slice(None), 1))
    b["left_u"], b["right_u"] = b["right_u"], b["left_u"]; b["left_v"], b["right_v"] = b["right_v"], b["left_v"]
    return b


def rotate_batch(b, angle):
    """Rotate every ego-frame position, displacement, velocity, acceleration and heading of a batch by `angle` (rad).
    Risk levels are rotation invariant. Training augmentation (--augment rot)."""
    b = dict(b)
    c, s_ = math.cos(angle), math.sin(angle)
    def rot(t, i0):
        t = t.clone(); x, y = t[..., i0].clone(), t[..., i0 + 1].clone()
        t[..., i0] = c * x - s_ * y; t[..., i0 + 1] = s_ * x + c * y; return t
    b["actor_feats"] = rot(b["actor_feats"], 0); b["actor_ctrs"] = rot(b["actor_ctrs"], 0); b["gt_preds"] = rot(b["gt_preds"], 0)
    b["node_ctrs"] = rot(b["node_ctrs"], 0); b["node_feats"] = rot(b["node_feats"], 0)
    if "actor_kin" in b:
        k = rot(rot(rot(b["actor_kin"], 0), 2), 4)                       # velocity, acceleration, (cos, sin) of heading
        b["actor_kin"] = k; b["actor_pos"] = rot(b["actor_pos"], 0)
    return b


class RiskHeadP(nn.Module):
    """Paper Eq. 8: shared MLP d_L -> 128 -> C with ReLU after the first layer.

    In the paper a lane node *is* a labelled lane segment. Here a labelled unit (one or more nuScenes lanes) is covered by
    many 2 m nodes, so the segment representation has to be pooled from its nodes:
      pool="mean"  logits of every node, averaged per lane and per unit (first p1 runs);
      pool="max"   node embeddings max-pooled per lane and per unit, then the MLP (the unit takes the level of its most
                   critical stretch, as the annotation guide defines it)."""

    def __init__(self, d=D_LANE, pool="mean"):
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True), nn.Linear(128, N_LEVELS))

    def forward(self, lane_nodes, b):
        node_logits = self.net(lane_nodes)
        keep = b["lane_unit"] >= 0
        if self.pool == "max":
            lane_emb = scatter_max(lane_nodes, b["node_lane"], b["n_lanes"])
            unit_emb = scatter_max(lane_emb[keep], b["lane_unit"][keep], b["n_units"])
            return node_logits, self.net(lane_emb), self.net(unit_emb)
        lane_logits = scatter_mean(node_logits, b["node_lane"], b["n_lanes"])
        unit_logits = scatter_mean(lane_logits[keep], b["lane_unit"][keep], b["n_units"])
        return node_logits, lane_logits, unit_logits


def nearest_node(b):
    """For every actor, the index of the closest lane node of its own sample (paper III-E: the lane segment the actor
    occupies or is spatially closest to); -1 when the sample has no lane node."""
    out = torch.full((b["actor_ctrs"].shape[0],), -1, dtype=torch.long, device=b["actor_ctrs"].device)
    for s in torch.unique(b["actor_batch"]):
        ia = torch.nonzero(b["actor_batch"] == s).view(-1); il = torch.nonzero(b["node_batch"] == s).view(-1)
        if len(il):
            out[ia] = il[torch.cdist(b["actor_ctrs"][ia].float(), b["node_ctrs"][il].float()).argmin(1)]
    return out


class DynamicRiskNetP(nn.Module):
    """Map encoder -> T layers of Risk Influence Score fusion -> risk head; actor encoder -> risk-aware trajectory head."""

    def __init__(self, T=3, pool="max", drop=(), risk_cond="lane", radius=15.0, dir_gate=True, threat_fix=True):
        super().__init__()
        self.risk_cond = risk_cond      # Eq. 9 risk distribution: "lane" (the actor's lane, pooled like the segments) or "node" (nearest 2 m node)
        self.map = MapNetP()
        self.actor = ActorNetP()
        self.layers = nn.ModuleList([RiskInfluenceFusion(radius=radius, drop=drop, dir_gate=dir_gate, threat_fix=threat_fix) for _ in range(T)])
        self.risk = RiskHeadP(pool=pool)
        self.risk_emb = nn.Linear(N_LEVELS, D_RISK, bias=False)                     # E in R^{C x d_risk}, Eq. 9
        self.traj = TrajHead(d=D_ACTOR + D_RISK)

    def set_input_stats(self, st):
        self.map.norm_in.set(*st["node"])
        self.actor.hist_norm.set(*st["hist"]); self.actor.kin_norm.set(*st["kin"])

    def forward(self, b):
        if "actor_kin" not in b:
            raise ValueError("this cache has no paper features (actor_kin / node_attr); re-run `drn_model prepare`")
        lane = self.map.forward_b(b)
        actor = self.actor(b)
        pr = pairs(b["node_ctrs"], b["node_batch"], b["actor_ctrs"], b["actor_batch"], self.layers[0].radius)
        for layer in self.layers:
            lane = layer(lane, actor, b, pr)
        out = {}
        out["node_logits"], out["lane_logits"], out["unit_logits"] = self.risk(lane, b)
        nn_i = nearest_node(b)
        if self.risk_cond == "lane":
            src = out["lane_logits"][b["node_lane"][nn_i.clamp(min=0)]]
        else:
            src = out["node_logits"][nn_i.clamp(min=0)]
        y = torch.softmax(src.float(), -1) * (nn_i >= 0).unsqueeze(-1)
        h = torch.cat([actor, self.risk_emb(y.to(actor.dtype))], -1)                       # Eq. 9-10
        out["traj_reg"], out["traj_cls"] = self.traj(h, b["actor_ctrs"])
        return out


def build_model(name: str = "drn", pool: str = "max", opts=None) -> nn.Module:
    """name: "drn". opts: {"drop": ["prox" | "dir" | "threat"]} removes that term of C_ij (paper Table 4 ablation);
    {"risk_cond": "lane" | "node"} chooses the risk distribution of Eq. 9; {"radius": metres, "dir_gate": bool, "threat_fix": bool}
    are the Risk Influence Score details (see RiskInfluenceFusion). Missing keys take the released setting."""
    if name != "drn":
        raise ValueError("unknown model %r (this release contains drn only)" % name)
    opts = opts or {}
    return DynamicRiskNetP(T=3, pool=pool, drop=tuple(opts.get("drop", ())), risk_cond=opts.get("risk_cond", "lane"),
                           radius=float(opts.get("radius", 15.0)), dir_gate=bool(opts.get("dir_gate", True)), threat_fix=bool(opts.get("threat_fix", True)))


MODEL_NAMES = ("drn",)


def checkpoint_model_name(config) -> str:
    return config["model"]
