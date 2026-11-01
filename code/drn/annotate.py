"""Stage 2: local web tool for independent lane-segment risk annotation (and scene review).

Each annotator's labels go to labels/<annotator>.jsonl (append-only; the latest entry
per lane wins). The API only ever returns the requesting annotator's own labels, so
annotators work blind to each other.
"""
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from .build import curated, load_record, set_review
from .annotation_fields import SIGNALS, parse_times
from .consensus import LEVEL_NAMES, read_labels, read_signals, read_times
from .lanes import to_ego_frame

PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>DRN lane risk annotation</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}
 header{display:flex;gap:16px;align-items:center;padding:8px 16px;background:#fff;border-bottom:1px solid #d9dce1}
 header b{font-size:15px} select,button,input{font-size:14px;padding:4px 8px}
 main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:12px;padding:12px}
 .card{background:#fff;border:1px solid #d9dce1;border-radius:6px;padding:8px}
 img{width:100%;display:block;border-radius:4px}
 svg{width:100%;height:auto;background:#fafbfc;border-radius:4px}
 .lane{fill:none;stroke-linecap:round;cursor:pointer}
 .lane.ctx{stroke:#c9cdd3;stroke-width:.5;cursor:default}
 table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:3px 6px;border-bottom:1px solid #eceef1;text-align:left}
 tr.sel{background:#fff6d6} .lv{display:inline-block;min-width:22px;text-align:center;border-radius:3px;color:#fff;padding:1px 4px;cursor:pointer;margin-right:2px;opacity:.35}
 .lv.on{opacity:1} .hint{font-size:12px;color:#5b616b;line-height:1.5} .warn{color:#b3261e;font-weight:600}
 .legend span{margin-right:10px;font-size:12px}
</style></head><body>
<header>
 <b>Dynamic-Risk Net · lane risk annotation</b>
 <label id="annbox">Annotator <select id="ann"></select></label>
 <label id="sigbox">ego 신호 <select id="sig"><option value="">선택</option><option>red</option><option>yellow</option><option>green</option><option>none</option><option>unclear</option></select></label>
 <span id="prog"></span>
 <button id="prev">◀ 이전</button><button id="next">다음 ▶</button>
 <button id="save">저장 후 다음 (Enter)</button>
 <span id="msg" class="warn"></span>
</header>
<main>
 <div class="card"><img id="cam" alt="CAM_FRONT">
  <p class="hint">기준: 각 라벨 단위(같은 색·번호의 lane 묶음)에서 <b>실제 상충이 얼마나 가까운지</b>를 5단계로 판단합니다.
  0 Normal · 1 Take Care · 2 Slow Down · 3 Ready to Brake · 4 Stop.<br>
  조작: 단위 클릭 또는 ↑/↓ 로 선택 → 숫자키 0–4 로 라벨. 굵은 선만 라벨 대상이고, 여러 lane으로 된 단위는 가장 위험한 부분 기준입니다. 화살표는 actor 속도(1초 이동량).<br>
  노란 영역 = 교차로, 회청색 = 횡단보도, 흰 사각형 = 신호등 위치.<br>
  ego 신호: ego가 지금 있거나 곧 진입할 교차로의 신호 상태(카메라에서 보이는 대로). 자세한 기준은 어노테이션 가이드.</p>
  <div id="review" style="display:none">
   <button id="keep">유지 (K)</button> <button id="drop">제외 – 규칙 오탐 (D)</button>
   <input id="note" placeholder="메모(선택)" size="40">
  </div>
  <div id="meta" class="hint"></div>
 </div>
 <div class="card"><svg id="bev" viewBox="-32 -58 64 80"></svg>
  <div class="legend" id="legend"></div>
  <table id="tbl"><thead><tr><th>#</th><th>종류</th><th>라벨</th><th>상충까지 시간(초)</th></tr></thead><tbody></tbody></table>
  <p class="hint">시간: 이 단위의 레벨을 정한 상충까지의 시간입니다. <b>계산</b> = 표시된 등속 TTC가 그 상충의 시간일 때,
  <b>추정</b> = 계산으로 잡히지 않아 직접 어림한 값(서 있는 차, 굽은 경로, 목록에 없는 객체 등), <b>없음</b> = 잴 상충이 없음.
  자세한 기준은 어노테이션 가이드의 "Time to conflict".</p>
 </div>
</main>
<script>
const MODE = "__MODE__", ANNS = __ANNS__, NAMES = __NAMES__;
const COL = ["#6aa84f","#2e86de","#f39c12","#e74c3c","#6c3483"];
let list = [], idx = 0, scene = null, labels = {}, times = {}, sel = 0, t0 = 0;
const $ = id => document.getElementById(id);
const ann = () => MODE === "review" ? "REVIEW" : $("ann").value;
function sx(p){ return -p[1]; } function sy(p){ return -p[0]; }
async function loadList(){
  const r = await fetch("/api/list?annotator=" + ann()); const j = await r.json();
  list = j.items; const first = list.findIndex(x => !x.done); idx = first < 0 ? 0 : first; await loadScene();
}
async function loadScene(){
  if(!list.length){ $("msg").textContent = "큐레이션된 scene이 없습니다."; return; }
  const tok = list[idx].sample_token;
  const r = await fetch("/api/scene/" + tok + "?annotator=" + ann()); scene = await r.json();
  labels = Object.assign({}, scene.labels); sel = 0; t0 = Date.now();
  times = {}; for(const u of scene.units){ const t = (scene.ttc || {})[u.unit_id], b = (scene.ttc_basis || {})[u.unit_id]; times[u.unit_id] = {b: b || "none", t: t === undefined ? null : t}; }
  $("sig").value = scene.signal || "";
  $("msg").textContent = scene.prefilled ? "초안이 채워져 있습니다. 확인·수정한 뒤 저장하세요." : "";
  $("cam").src = "/img/" + tok;
  $("meta").textContent = scene.scene_name + " · " + tok + " · templates: " + scene.templates.join(", ");
  $("prog").textContent = (idx + 1) + " / " + list.length + " (완료 " + list.filter(x => x.done).length + ")";
  if(MODE === "review"){ $("note").value = ""; }
  draw();
}
function targets(){ return scene.units; }
function draw(){
  const svg = $("bev"); let h = "";
  const poly = (pts, style) => `<polygon points="${pts.map(p => sx(p)+","+sy(p)).join(" ")}" ${style}/>`;
  for(const p of scene.context.intersections) h += poly(p, 'fill="#ffecb3" fill-opacity=".6"');
  for(const p of scene.context.crossings) h += poly(p, 'fill="#78909c" fill-opacity=".35" stroke="#546e7a" stroke-width=".15"');
  for(const l of scene.lanes){ if(l.label_target) continue;
    h += `<polyline class="lane ctx" points="${l.pts.map(p => sx(p)+","+sy(p)).join(" ")}"/>`; }
  const T = targets();
  T.forEach((l, i) => {
    const lv = labels[l.unit_id]; const c = lv === undefined ? "#7f8790" : COL[lv];
    for(const part of l.parts){
      const P = part.map(p => sx(p)+","+sy(p)).join(" ");
      if(i === sel) h += `<polyline class="lane" stroke="#111" stroke-width="2.2" points="${P}"/>`;
      h += `<polyline class="lane" data-i="${i}" stroke="${c}" stroke-width="${i===sel?1.4:1.0}" points="${P}"/>`;
    }
    const big = l.parts.reduce((a, b) => b.length > a.length ? b : a);
    const m = big[Math.floor(big.length/2)];
    h += `<circle cx="${sx(m)}" cy="${sy(m)}" r="1.5" fill="#fff" stroke="${c}" stroke-width=".3"/><text x="${sx(m)}" y="${sy(m)+.7}" font-size="2" text-anchor="middle">${i+1}</text>`;
  });
  for(const a of scene.actors){
    const col = a.cls === "pedestrian" ? "#a23b72" : (a.cls === "vehicle" ? "#2e6fab" : "#00897b");
    const w = a.size[0], len = a.size[1], deg = -a.yaw * 180 / Math.PI;  // screen: forward = up
    if(a.cls === "pedestrian") h += `<circle cx="${sx([a.x,a.y])}" cy="${sy([a.x,a.y])}" r=".8" fill="${col}" stroke="#fff" stroke-width=".2"/>`;
    else h += `<rect x="${-w/2}" y="${-len/2}" width="${w}" height="${len}" fill="${col}" fill-opacity=".6" stroke="#222" stroke-width=".1" transform="translate(${sx([a.x,a.y])},${sy([a.x,a.y])}) rotate(${deg})"/>`;
    if(a.speed > 0.5) h += `<line x1="${sx([a.x,a.y])}" y1="${sy([a.x,a.y])}" x2="${sx([a.x+a.vx,a.y+a.vy])}" y2="${sy([a.x+a.vx,a.y+a.vy])}" stroke="${col}" stroke-width=".25"/>`;
  }
  for(const t of scene.context.signals) h += `<rect x="${sx(t)-.6}" y="${sy(t)-.6}" width="1.2" height="1.2" fill="#fff" stroke="#000" stroke-width=".25"/>`;
  h += `<rect x="-0.87" y="-2.04" width="1.73" height="4.08" fill="#178a2f" stroke="#0b4d19" stroke-width=".1"/>`;
  h += `<line x1="0" y1="0" x2="${sx([scene.ego.vx,scene.ego.vy])}" y2="${sy([scene.ego.vx,scene.ego.vy])}" stroke="#178a2f" stroke-width=".3"/>`;
  svg.innerHTML = h;
  svg.querySelectorAll("polyline[data-i]").forEach(el => el.onclick = () => { sel = +el.dataset.i; draw(); });
  const tb = $("tbl").querySelector("tbody"); tb.innerHTML = "";
  T.forEach((l, i) => {
    const tr = document.createElement("tr"); if(i === sel) tr.className = "sel";
    let b = ""; for(let k = 0; k < 5; k++) b += `<span class="lv ${labels[l.unit_id]===k?"on":""}" style="background:${COL[k]}" data-i="${i}" data-k="${k}">${k}</span>`;
    tr.innerHTML = `<td>${i+1}</td><td>${l.kind === "lane" ? "lane" : (l.kind === "mixed" ? "lane+connector" : "connector")}${l.parts.length > 1 ? " ×" + l.parts.length : ""}${l.in_intersection ? " · 교차로" : ""}${l.is_ego_lane ? " · EGO" : ""}</td><td>${MODE === "review" ? "" : b}</td><td>${MODE === "review" ? "" : timeCell(l, i)}</td>`;
    tr.onclick = () => { sel = i; draw(); }; tb.appendChild(tr);
  });
  tb.querySelectorAll(".lv").forEach(el => el.onclick = ev => { ev.stopPropagation(); setLabel(+el.dataset.i, +el.dataset.k); });
  tb.querySelectorAll(".tb").forEach(el => { el.onclick = ev => ev.stopPropagation(); el.onchange = () => {
    const u = T[+el.dataset.i], cur = times[u.unit_id]; cur.b = el.value;
    cur.t = el.value === "none" ? null : (el.value === "calc" && u.cv_ttc !== null && u.cv_ttc < 9.95 ? u.cv_ttc : cur.t); draw(); }; });
  tb.querySelectorAll(".tv").forEach(el => { el.onclick = ev => ev.stopPropagation(); el.onkeydown = ev => ev.stopPropagation();
    el.onchange = () => { const v = parseFloat(el.value); times[T[+el.dataset.i].unit_id].t = isNaN(v) ? null : Math.min(Math.max(v, 0), 10); }; });
  $("legend").innerHTML = NAMES.map((n, k) => `<span><b style="color:${COL[k]}">■</b> ${k} ${n}</span>`).join("");
}
function timeCell(l, i){
  const cur = times[l.unit_id] || {b: "none", t: null}; const cv = (l.cv_ttc === null || l.cv_ttc === undefined || l.cv_ttc >= 9.95) ? "없음" : l.cv_ttc.toFixed(1) + "s";
  const opt = [["none", "없음"], ["calc", "계산"], ["est", "추정"]].map(o => `<option value="${o[0]}" ${cur.b === o[0] ? "selected" : ""}>${o[1]}</option>`).join("");
  return `<select class="tb" data-i="${i}">${opt}</select> <input class="tv" data-i="${i}" size="3" value="${cur.t === null ? "" : cur.t}" ${cur.b === "none" ? "disabled" : ""}> <span class="hint">계산 ${cv}</span>`;
}
function setLabel(i, k){
  const T = targets(); labels[T[i].unit_id] = k;
  const nxt = T.findIndex((l, j) => j > i && labels[l.unit_id] === undefined);
  sel = nxt >= 0 ? nxt : i; draw();
}
async function save(){
  const T = targets(); const missing = T.filter(l => labels[l.unit_id] === undefined).length;
  if(missing){ $("msg").textContent = "라벨이 없는 단위 " + missing + "개"; return; }
  if(!$("sig").value){ $("msg").textContent = "ego 신호 상태를 선택하세요 (신호가 없으면 none)"; return; }
  const bad = T.filter(l => times[l.unit_id].b !== "none" && times[l.unit_id].t === null).length;
  if(bad){ $("msg").textContent = "시간 값이 비어 있는 단위 " + bad + "개 (계산/추정을 골랐으면 초를 넣으세요)"; return; }
  const ttc = {}, basis = {}; for(const l of T){ ttc[l.unit_id] = times[l.unit_id].b === "none" ? null : times[l.unit_id].t; basis[l.unit_id] = times[l.unit_id].b; }
  const body = {annotator: ann(), sample_token: scene.sample_token, labels: labels, signal: $("sig").value, ttc: ttc, ttc_basis: basis,
                elapsed_s: (Date.now() - t0) / 1000};
  const r = await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if(!r.ok){ $("msg").textContent = "저장 실패: " + (await r.text()); return; }
  list[idx].done = true; step(1);
}
async function review(keep){
  const body = {sample_token: scene.sample_token, keep: keep, note: $("note").value};
  const r = await fetch("/api/review", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if(!r.ok){ $("msg").textContent = "저장 실패"; return; } list[idx].done = true; step(1);
}
function step(d){ idx = Math.min(Math.max(idx + d, 0), list.length - 1); loadScene(); }
document.addEventListener("keydown", e => {
  if(e.target.tagName === "INPUT" || !scene) return;
  const T = targets();
  if(MODE === "review"){ if(e.key === "k") review(true); if(e.key === "d") review(false); }
  else if(e.key >= "0" && e.key <= "4"){ setLabel(sel, +e.key); }
  else if(e.key === "Enter"){ save(); }
  if(e.key === "ArrowDown"){ sel = Math.min(sel + 1, T.length - 1); draw(); e.preventDefault(); }
  if(e.key === "ArrowUp"){ sel = Math.max(sel - 1, 0); draw(); e.preventDefault(); }
});
$("prev").onclick = () => step(-1); $("next").onclick = () => step(1); $("save").onclick = save;
$("keep").onclick = () => review(true); $("drop").onclick = () => review(false);
if(MODE === "review"){ $("annbox").style.display = "none"; $("sigbox").style.display = "none"; $("save").style.display = "none"; $("review").style.display = "block"; }
else { $("ann").innerHTML = ANNS.map(a => `<option>${a}</option>`).join(""); $("ann").onchange = loadList; }
loadList();
</script></body></html>
"""


def scene_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Scene record in the ego frame (x forward, y left), trimmed to what the UI draws."""
    ego = rec["ego"]
    c, s = np.cos(-ego["yaw"]), np.sin(-ego["yaw"])

    def rot(vx: float, vy: float) -> List[float]:
        return [float(c * vx - s * vy), float(s * vx + c * vy)]

    lanes = [{"lane_token": l["lane_token"], "kind": l["kind"], "in_intersection": l["in_intersection"],
              "label_target": l["label_target"],
              "pts": np.round(to_ego_frame(np.asarray(l["pts"]), ego), 2).tolist()} for l in rec["lanes"]]
    by_tok = {l["lane_token"]: l["pts"] for l in lanes}
    units = [{"unit_id": u["unit_id"], "kind": u["kind"], "in_intersection": u["in_intersection"],
              "is_ego_lane": u["is_ego_lane"], "cv_ttc": u.get("min_ttc_s"),
              "parts": [by_tok[t] for t in u["lane_tokens"]]} for u in rec["label_units"]]
    actors = []
    for a in rec["actors"]:
        x, y = to_ego_frame(np.array([[a["x"], a["y"]]]), ego)[0]
        vx, vy = rot(a["vx"], a["vy"])
        actors.append({"cls": a["cls"], "x": float(x), "y": float(y), "yaw": float(a["yaw"] - ego["yaw"]),
                       "vx": vx, "vy": vy, "speed": a["speed"], "size": a["size"]})
    evx, evy = rot(ego["vx"], ego["vy"])
    mc = rec.get("map_context", {})

    def ring(p):
        return np.round(to_ego_frame(np.asarray(p["pts"], dtype=float), ego), 2).tolist()

    tl = np.asarray(mc.get("traffic_lights", []), dtype=float).reshape(-1, 3)
    context = {"intersections": [ring(p) for p in mc.get("intersections", [])],
               "crossings": [ring(p) for p in mc.get("ped_crossings", [])],
               "signals": np.round(to_ego_frame(tl[:, :2], ego), 2).tolist() if len(tl) else []}
    return {"sample_token": rec["sample_token"], "scene_name": rec["scene_name"], "context": context,
            "templates": [t for t, v in rec["templates"].items() if v["hit"]],
            "ego": {"vx": evx, "vy": evy, "speed": ego["speed"]}, "lanes": lanes, "units": units, "actors": actors}


def _load_prefill(cfg: Dict[str, Any], name: str) -> Dict[str, Dict[str, Any]]:
    """Load an existing label set as a starting point for annotation."""
    base = os.path.join(cfg["output_dir"], name)
    out = {}  # type: Dict[str, Dict[str, Any]]
    with open(os.path.join(base, "labels.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out.setdefault(rec["sample_token"], {"labels": {}, "signal": ""})["labels"][rec["unit_id"]] = int(rec["label"])
    sig = os.path.join(base, "signals.jsonl")
    if os.path.exists(sig):
        with open(sig, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec["sample_token"] in out and rec["signal"] in SIGNALS:
                        out[rec["sample_token"]]["signal"] = rec["signal"]
    return out


def create_app(cfg: Dict[str, Any], mode: str = "annotate", prefill: str = ""):
    from flask import Flask, abort, jsonify, request, send_file
    prefill_data = _load_prefill(cfg, prefill) if (prefill and mode == "annotate") else {}

    app = Flask(__name__)
    rows = curated(cfg) if mode == "annotate" else _all_retained(cfg)
    by_token = {r["sample_token"]: r for r in rows}
    annotators = list(cfg["annotation"]["annotators"])
    labels_dir = os.path.join(cfg["output_dir"], "labels")
    os.makedirs(labels_dir, exist_ok=True)

    def targets_of(rec: Dict[str, Any]) -> List[str]:
        return [u["unit_id"] for u in rec["label_units"]]

    @app.route("/")
    def index():
        return (PAGE.replace("__MODE__", mode).replace("__ANNS__", json.dumps(annotators))
                .replace("__NAMES__", json.dumps(LEVEL_NAMES)))

    @app.route("/api/list")
    def api_list():
        annotator = request.args.get("annotator", "")
        if mode == "review":
            reviewed = _review_state(cfg)
            return jsonify({"items": [{"sample_token": r["sample_token"], "scene_name": r["scene_name"],
                                       "done": reviewed.get(r["sample_token"], "") != ""} for r in rows]})
        if annotator not in annotators:
            abort(400, "unknown annotator")
        mine = read_labels(cfg, annotator)
        done_samples = {k[0] for k in mine}
        return jsonify({"items": [{"sample_token": r["sample_token"], "scene_name": r["scene_name"],
                                   "done": r["sample_token"] in done_samples} for r in rows]})

    @app.route("/api/scene/<token>")
    def api_scene(token: str):
        if token not in by_token:
            abort(404)
        rec = load_record(cfg, by_token[token])
        payload = scene_payload(rec)
        annotator = request.args.get("annotator", "")
        payload["labels"], payload["signal"], payload["prefilled"] = {}, "", False
        if mode == "annotate" and annotator in annotators:
            mine = read_labels(cfg, annotator)
            payload["labels"] = {t: mine[(token, t)] for t in targets_of(rec) if (token, t) in mine}
            payload["signal"] = read_signals(cfg, annotator).get(token, "")
            mine_t = read_times(cfg, annotator)
            payload["ttc"] = {t: mine_t[(token, t)][0] for t in targets_of(rec) if (token, t) in mine_t}
            payload["ttc_basis"] = {t: mine_t[(token, t)][1] for t in targets_of(rec) if (token, t) in mine_t}
            if not payload["labels"] and token in prefill_data:
                pre = prefill_data[token]
                payload["labels"] = {t: pre["labels"][t] for t in targets_of(rec) if t in pre["labels"]}
                payload["signal"], payload["prefilled"] = pre["signal"], True
        return jsonify(payload)

    @app.route("/img/<token>")
    def img(token: str):
        if token not in by_token:
            abort(404)
        rec = load_record(cfg, by_token[token])
        path = os.path.join(cfg["nuscenes"]["dataroot"], rec["cam_front"])
        if not os.path.exists(path):
            abort(404)
        return send_file(path)

    @app.route("/api/label", methods=["POST"])
    def api_label():
        if mode != "annotate":
            abort(400)
        body = request.get_json(force=True)
        annotator, token = body.get("annotator"), body.get("sample_token")
        if annotator not in annotators or token not in by_token:
            abort(400, "unknown annotator or scene")
        rec = load_record(cfg, by_token[token])
        want = set(targets_of(rec))
        got = {k: int(v) for k, v in (body.get("labels") or {}).items() if k in want}
        if set(got) != want or any(v not in range(5) for v in got.values()):
            abort(400, "every label unit needs a label in 0..4")
        if body.get("signal") not in SIGNALS:
            abort(400, "signal must be one of " + ", ".join(SIGNALS))
        try:
            times = parse_times(body.get("ttc"), body.get("ttc_basis"), sorted(want), "label")
        except SystemExit as exc:
            abort(400, str(exc))
        entry = {"sample_token": token, "annotator": annotator, "source": "human", "labels": got, "signal": body["signal"],
                 "elapsed_s": float(body.get("elapsed_s") or 0.0), "saved_at": time.time()}
        entry.update(times)
        if token in prefill_data:   # this annotator started from another label set: keep that on record
            pre = prefill_data[token]["labels"]
            entry["prefill"] = prefill
            entry["prefill_changed"] = sum(1 for t, v in got.items() if pre.get(t) != v)
        with open(os.path.join(labels_dir, annotator + ".jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return jsonify({"ok": True})

    @app.route("/api/review", methods=["POST"])
    def api_review():
        if mode != "review":
            abort(400)
        body = request.get_json(force=True)
        if body.get("sample_token") not in by_token:
            abort(400)
        set_review(cfg, body["sample_token"], bool(body.get("keep")), str(body.get("note") or ""))
        return jsonify({"ok": True})

    return app


def _all_retained(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    with open(os.path.join(cfg["output_dir"], "index.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["retained"]:
                out.append(row)
    return out


def _review_state(cfg: Dict[str, Any]) -> Dict[str, str]:
    import csv
    path = os.path.join(cfg["output_dir"], "review.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["sample_token"]: (r.get("keep") or "").strip() for r in csv.DictReader(fh)}
