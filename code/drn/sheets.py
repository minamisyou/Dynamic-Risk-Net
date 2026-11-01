"""Review sheets for annotation outside the web tool.

For every curated keyframe this writes, under <output_dir>/review_sheets/:
  <base>.jpg     CAM_FRONT with the label units projected and numbered + a bird's-eye view
  <base>.txt     text digest: each numbered unit in words, who is on / beside / approaching it, CV-TTC
  <base>_tl.jpg  full-resolution crops around mapped traffic-signal positions (when any project into the image)
  index.json     frame list with the unit_id behind every unit number
"""
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .build import curated, load_record
from .payload import cam_from_ego, frame_payload
from .source import NuscSource

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0b0", "#f032e6", "#7a9a01",
           "#c2185b", "#008080", "#9a6324", "#800000", "#5d6d00", "#000075", "#d35400", "#1b7f5c"]
CAM_W, CAM_H = 1280, 720
BEV_W, BEV_H = 720, 1000
X_MIN, X_MAX, Y_ABS = -14.0, 50.0, 23.0          # BEV extent in the ego frame (x forward, y left)
VIEW = {"x_min": -25.0, "x_max": 65.0, "y_abs": 35.0}
CLS = {"v": "vehicle", "p": "pedestrian", "b": "bicycle", "m": "motorcycle"}
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]


def _font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pts_of(flat: List[int]) -> np.ndarray:
    return np.asarray(flat, dtype=float).reshape(-1, 2) / 10.0


def densify(p: np.ndarray, step: float = 1.0) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return p
    g = np.linspace(0.0, s[-1], max(int(s[-1] / step) + 1, 2))
    return np.stack([np.interp(g, s, p[:, 0]), np.interp(g, s, p[:, 1])], axis=1)


def project(p_ego: np.ndarray, cam: Dict[str, Any], z: float = 0.0):
    """Ego-frame points at height z -> pixel coordinates in the full camera image, and depth."""
    T = np.asarray(cam["T"], dtype=float).reshape(3, 4)
    fx, fy, cx, cy = cam["K"]
    P = np.concatenate([p_ego, np.full((len(p_ego), 1), z)], axis=1)
    c = P @ T[:, :3].T + T[:, 3]
    depth = c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * c[:, 0] / depth + cx
        v = fy * c[:, 1] / depth + cy
    return np.stack([u, v], axis=1), depth


def bev_xy(p: np.ndarray) -> np.ndarray:
    sx = BEV_W / (2 * Y_ABS)
    sy = BEV_H / (X_MAX - X_MIN)
    return np.stack([(Y_ABS - p[:, 1]) * sx, (X_MAX - p[:, 0]) * sy], axis=1)


def _place(tags: list, x: float, y: float, r: float = 17):
    for k in range(40):
        ang = k * 2.4
        rad = 0 if k == 0 else 10 + 5 * k
        nx, ny = x + rad * math.cos(ang), y + rad * math.sin(ang)
        if all((nx - a) ** 2 + (ny - b) ** 2 > (2 * r) ** 2 for a, b in tags):
            tags.append((nx, ny))
            return nx, ny
    tags.append((x, y))
    return x, y


def _tag(draw, x, y, text, color, font):
    w = draw.textlength(text, font=font)
    r = max(15, w / 2 + 6)
    draw.ellipse([x - r, y - 15, x + r, y + 15], fill="white", outline=color, width=3)
    draw.text((x - w / 2, y - 11), text, fill="black", font=font)


def render_sheet(frame: Dict[str, Any], image_path: str, out_path: str) -> None:
    font, small = _font(19), _font(15)
    sheet = Image.new("RGB", (CAM_W + BEV_W, BEV_H), "white")
    cam = Image.open(image_path).convert("RGB")
    sc = CAM_W / cam.width
    cam = cam.resize((CAM_W, CAM_H), Image.LANCZOS)
    dc = ImageDraw.Draw(cam, "RGBA")
    bev = Image.new("RGB", (BEV_W, BEV_H), "#f3f4f6")
    db = ImageDraw.Draw(bev, "RGBA")

    for poly in frame["I"]:
        db.polygon([tuple(q) for q in bev_xy(pts_of(poly))], fill=(255, 236, 179, 150))
    for poly in frame["X"]:
        db.polygon([tuple(q) for q in bev_xy(pts_of(poly))], fill=(120, 144, 156, 110), outline=(84, 110, 122, 255))
    for l in frame["L"]:
        if not l[0]:
            db.line([tuple(q) for q in bev_xy(densify(pts_of(l[4])))], fill=(170, 174, 181, 255), width=3)

    cam_tags, bev_tags, legend = [], [], []
    for i, u in enumerate(frame["U"]):
        color = PALETTE[i % len(PALETTE)]
        parts = [densify(pts_of(l[4])) for l in frame["L"] if l[5] == i]
        best_bev, best_cam = None, None
        for p in parts:
            q = bev_xy(p)
            db.line([tuple(a) for a in q], fill=color, width=6)
            d = q[-1] - q[-2]
            n = np.linalg.norm(d)
            if n > 1e-6:                                          # arrowhead = direction of travel
                d = d / n
                nrm = np.array([-d[1], d[0]])
                db.polygon([tuple(q[-1] + d * 4), tuple(q[-1] - d * 14 + nrm * 9), tuple(q[-1] - d * 14 - nrm * 9)], fill=color)
            inside = np.where((p[:, 0] > X_MIN + 2) & (p[:, 0] < X_MAX - 2) & (np.abs(p[:, 1]) < Y_ABS - 2))[0]
            if len(inside) and (best_bev is None or len(inside) > best_bev[0]):
                best_bev = (len(inside), q[inside[len(inside) // 2]])
            uv, depth = project(p, frame["cam"])
            ok = depth > 1.5
            vis = [k for k in range(len(p)) if ok[k] and 0 <= uv[k][0] * sc < CAM_W and CAM_H * 0.35 <= uv[k][1] * sc < CAM_H]
            if vis:
                for a in range(len(p) - 1):
                    if ok[a] and ok[a + 1]:
                        dc.line([tuple(uv[a] * sc), tuple(uv[a + 1] * sc)], fill=color, width=5)
                if best_cam is None or len(vis) > best_cam[0]:
                    best_cam = (len(vis), uv[vis[len(vis) // 2]] * sc)
        if best_bev is not None:                                  # one number per unit, on its longest visible part
            bx, by = _place(bev_tags, float(best_bev[1][0]), float(best_bev[1][1]))
            _tag(db, bx, by, str(i + 1), color, font)
        if best_cam is not None:
            cx_, cy_ = _place(cam_tags, float(best_cam[1][0]), float(best_cam[1][1]))
            _tag(dc, cx_, cy_, str(i + 1), color, font)
        legend.append((i + 1, color, u, len(parts)))

    for a in frame["A"]:
        cls, x, y, yaw = a[0], a[1] / 10, a[2] / 10, math.radians(a[3])
        vx, vy, w, ln = a[4] / 10, a[5] / 10, a[6] / 10, a[7] / 10
        col = {"p": (162, 59, 114), "v": (40, 90, 150)}.get(cls, (0, 137, 123))
        if len(a) > 10 and a[10] in ("k", "n") and cls != "p":      # parked vehicle / unattended cycle (nuScenes attribute)
            col = (135, 135, 135)
        c0 = bev_xy(np.array([[x, y]]))[0]
        if cls == "p":
            db.ellipse([c0[0] - 6, c0[1] - 6, c0[0] + 6, c0[1] + 6], fill=col + (255,), outline="white")
        else:
            cs, sn = math.cos(yaw), math.sin(yaw)
            corners = np.array([[ln / 2, w / 2], [ln / 2, -w / 2], [-ln / 2, -w / 2], [-ln / 2, w / 2]])
            g = np.stack([x + cs * corners[:, 0] - sn * corners[:, 1], y + sn * corners[:, 0] + cs * corners[:, 1]], axis=1)
            db.polygon([tuple(t) for t in bev_xy(g)], fill=col + (170,), outline=(20, 20, 20, 255))
        if math.hypot(vx, vy) > 0.5:                              # where it will be in 1 s
            e = bev_xy(np.array([[x + vx, y + vy]]))[0]
            db.line([tuple(c0), tuple(e)], fill=(0, 0, 0, 255), width=3)
            db.ellipse([e[0] - 3, e[1] - 3, e[0] + 3, e[1] + 3], fill=(0, 0, 0, 255))

    for t in frame["TL"]:
        c0 = bev_xy(np.array([[t[0] / 10, t[1] / 10]]))[0]
        db.rectangle([c0[0] - 5, c0[1] - 5, c0[0] + 5, c0[1] + 5], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
        uv, depth = project(np.array([[t[0] / 10, t[1] / 10]]), frame["cam"], z=t[2] / 10)
        if depth[0] > 2 and 0 <= uv[0][0] * sc < CAM_W and 0 <= uv[0][1] * sc < CAM_H:
            u, v = uv[0] * sc
            dc.rectangle([u - 9, v - 9, u + 9, v + 9], outline=(255, 255, 0, 255), width=2)

    ego = bev_xy(np.array([[2.04, 0.87], [2.04, -0.87], [-2.04, -0.87], [-2.04, 0.87]]) + np.array([1.0, 0.0]))
    db.polygon([tuple(t) for t in ego], fill=(23, 138, 47, 255), outline=(0, 60, 0, 255))
    o = bev_xy(np.array([[1.0, 0.0], [1.0 + frame["ego"][0], 0.0]]))
    db.line([tuple(o[0]), tuple(o[1])], fill=(0, 90, 0, 255), width=4)
    db.text((8, 6), "BEV  (forward = up, 64 x 46 m)   ego = green   black lines = 1 s of travel", fill="black", font=small)
    db.text((8, 26), "blue box = vehicle (grey = parked), purple dot = pedestrian, white square = signal pole", fill="black", font=small)
    db.text((8, 46), "yellow area = intersection, grey-blue = crosswalk, grey lines = not labelled", fill="black", font=small)
    db.text((8, 66), "one colour and number = one label unit (may be several lanes)", fill="black", font=small)

    sheet.paste(cam, (0, 0))
    sheet.paste(bev, (CAM_W, 0))
    ds = ImageDraw.Draw(sheet)
    t = frame["tpl"]
    y0 = CAM_H + 8
    ds.text((10, y0), "%s   sample %s   ego speed %.1f m/s (%.0f km/h)   scene min CV-TTC %.1f s"
            % (frame["s"], frame["t"][:8], frame["ego"][0], frame["ego"][0] * 3.6, frame["ttc"]), fill="black", font=small)
    ds.text((10, y0 + 22), "rule hits: intersection=%d (conflicting %d, signal poles %s)  merge/weave=%d  ped-near-lane=%d (n=%d)  occlusion=%d (low-vis %d)"
            % (t["I"][0], t["I"][1], "yes" if t["I"][2] else "no", t["M"][0], t["P"][0], t["P"][1], t["O"][0], t["O"][1]),
            fill="black", font=small)
    for j, (num, color, u, n_parts) in enumerate(legend):
        x = 10 + (j // 10) * 318
        y = y0 + 50 + (j % 10) * 21
        ds.rectangle([x, y + 3, x + 16, y + 15], fill=color)
        kind = {"l": "lane", "c": "conn", "m": "lane+conn"}[u[2]]
        ds.text((x + 22, y), "%d %s%s%s%s  ttc %s" % (num, kind, " x%d" % n_parts if n_parts > 1 else "", "/int" if u[3] else "",
                                                   " EGO" if u[1] else "", ("%.1f" % u[4]) if u[4] < 9.95 else "-"),
                fill="black", font=small)
    sheet.save(out_path, quality=88)


# ----------------------------------------------------------------------------- text digest
def _seg_dist(p, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    den = (ab ** 2).sum(1)
    den[den == 0] = 1e-9
    t = np.clip(((p - a) * ab).sum(1) / den, 0, 1)
    d = np.linalg.norm(a + ab * t[:, None] - p, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), ab[k] / max(np.linalg.norm(ab[k]), 1e-9)


def _side(y: float) -> str:
    return "left" if y > 1.0 else ("right" if y < -1.0 else "centre")


def _where(p) -> str:
    x, y = p
    fb = "ahead" if x >= 0 else "behind"
    return "%.0f m %s, %.0f m %s" % (abs(x), fb, abs(y), _side(y)) if abs(y) > 1.0 else "%.0f m %s" % (abs(x), fb)


def _lane_words(line) -> str:
    d = line[-1] - line[0]
    ang = math.degrees(math.atan2(d[1], d[0]))
    if abs(ang) < 30:
        head = "same direction as ego"
    elif abs(ang) > 150:
        head = "oncoming (toward ego)"
    else:
        head = "heading left across ego's view" if ang > 0 else "heading right across ego's view"
    d0, d1 = line[1] - line[0], line[-1] - line[-2]
    turn = math.degrees(math.atan2(d0[0] * d1[1] - d0[1] * d1[0], d0[0] * d1[0] + d0[1] * d1[1]))
    shape = "straight" if abs(turn) < 20 else ("curves left %.0f deg" % turn if turn > 0 else "curves right %.0f deg" % -turn)
    return "from (%s) to (%s); %s; %s" % (_where(line[0]), _where(line[-1]), head, shape)


def _attr_words(cls: str, attr: str) -> str:
    """nuScenes attribute of a road user, as a tag for the digest ('' when there is nothing worth saying)."""
    if cls == "vehicle":
        return {"k": " [parked]", "s": " [stopped, not parked]"}.get(attr, "")
    if cls in ("bicycle", "motorcycle"):
        return {"r": " [with rider]", "n": " [no rider]"}.get(attr, "")
    if cls == "pedestrian":
        return {"s": " [standing]", "l": " [sitting or lying]"}.get(attr, "")
    return ""


def frame_digest(k: int, f: Dict[str, Any]) -> str:
    t = f["tpl"]
    out = ["FRAME %03d | %s | sample %s | location %s | ego speed %.1f m/s" % (k, f["s"], f["t"], f["loc"], f["ego"][0]),
           "Selection rules that fired: intersection-conflict=%d merge/weave=%d pedestrian-near-lane=%d occlusion=%d | "
           "map has traffic-signal poles near the intersection ahead: %s"
           % (t["I"][0], t["M"][0], t["P"][0], t["O"][0], "yes" if t["I"][2] else "no")]
    actors = []
    for a in f["A"]:
        v = np.array([a[4] / 10.0, a[5] / 10.0])
        actors.append({"cls": CLS.get(a[0], "other"), "p": np.array([a[1] / 10.0, a[2] / 10.0]), "v": v,
                       "speed": float(np.linalg.norm(v)), "vis": a[8], "large": a[9], "attr": a[10] if len(a) > 10 else ""})
    for i, u in enumerate(f["U"]):
        lines = [densify(pts_of(l[4])) for l in f["L"] if l[5] == i]
        lines.sort(key=lambda p: -len(p))
        on, near, coming = [], [], []
        for a in actors:
            d, tangent = min((_seg_dist(a["p"], line) for line in lines), key=lambda r: r[0])
            desc = a["cls"] + (" (large)" if a["large"] else "") + _attr_words(a["cls"], a["attr"])
            if a["speed"] < 0.5:
                mot = "stationary"
            else:
                cosang = float(a["v"] @ tangent) / a["speed"]
                rel = "along the lane" if cosang > 0.7 else ("against the lane" if cosang < -0.7 else "crossing the lane direction")
                mot = "%.1f m/s %s" % (a["speed"], rel)
            vis = "" if a["vis"] in ("3", "4") else " [mostly occluded in sensors]"
            if d <= 1.6:
                on.append("%s at %s, %s%s" % (desc, _where(a["p"]), mot, vis))
            elif d <= 4.0:
                drift = ""
                if a["speed"] >= 0.5:                              # is it coming toward the unit or leaving it?
                    d2 = min(_seg_dist(a["p"] + a["v"] * 0.5, line)[0] for line in lines)
                    drift = ", closing on the lane" if d2 < d - 0.15 else (", moving away from it" if d2 > d + 0.15 else "")
                near.append("%s %.1f m from centreline at %s, %s%s%s" % (desc, d, _where(a["p"]), mot, drift, vis))
            elif a["speed"] >= 0.8:
                for tt in (1.0, 2.0, 3.0):
                    if min(_seg_dist(a["p"] + a["v"] * tt, line)[0] for line in lines) <= 2.0:
                        coming.append("%s now at %s moving %.1f m/s would reach this unit in ~%.0f s%s"
                                      % (desc, _where(a["p"]), a["speed"], tt, vis))
                        break
        kind = {"l": "lane", "c": "connector", "m": "lane + connector"}[u[2]]
        tags = [kind if len(lines) == 1 else "%s, %d parallel/continuing lanes merged" % (kind, len(lines))] + \
               (["inside intersection"] if u[3] else []) + (["EGO IS ON THIS LANE"] if u[1] else [])
        ttc = "none within 10 s" if u[4] >= 9.95 else "%.1f s" % u[4]
        out.append("Unit %d [%s] %s | constant-velocity TTC on this unit: %s" % (i + 1, ", ".join(tags), _lane_words(lines[0]), ttc))
        for extra in lines[1:4]:
            out.append("    also: " + _lane_words(extra))
        out.append("    on unit: " + ("; ".join(on) if on else "nobody"))
        out.append("    beside (1.6-4 m): " + ("; ".join(near[:6]) if near else "nobody"))
        if coming:
            out.append("    approaching: " + "; ".join(coming[:5]))
    return "\n".join(out) + "\n"


def tl_strip(f: Dict[str, Any], image_path: str, out_path: str) -> bool:
    img = Image.open(image_path).convert("RGB")
    crops = []
    for t in sorted(f["TL"], key=lambda q: q[0]):
        if not (30 <= t[0] <= 900):                                # 3 m .. 90 m ahead
            continue
        uv, depth = project(np.array([[t[0] / 10.0, t[1] / 10.0]]), f["cam"], z=t[2] / 10.0)
        u, v = uv[0]
        if depth[0] <= 2 or not (0 <= u < img.width and 0 <= v < img.height):
            continue
        half = int(np.clip(900.0 / max(depth[0], 6.0), 28, 110))
        crop = img.crop([int(u - half), int(v - half), int(u + half), int(v + half)]).resize((220, 220), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        d.rectangle([0, 0, 219, 219], outline="yellow", width=2)
        d.text((5, 3), "%.0f m ahead, %.0f m %s" % (t[0] / 10.0, abs(t[1] / 10.0), _side(t[1] / 10.0)), fill="yellow", font=_font(13))
        crops.append(crop)
        if len(crops) == 8:
            break
    if not crops:
        return False
    strip = Image.new("RGB", (220 * len(crops), 220), "black")
    for i, c in enumerate(crops):
        strip.paste(c, (220 * i, 0))
    strip.save(out_path, quality=92)
    return True


def run_sheets(cfg: Dict[str, Any], src: NuscSource = None) -> Dict[str, Any]:
    src = src or NuscSource(cfg["nuscenes"]["dataroot"], cfg["nuscenes"]["version"])
    out_dir = os.path.join(cfg["output_dir"], "review_sheets")
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for k, row in enumerate(curated(cfg)):
        rec = load_record(cfg, row)
        sample = src.nusc.get("sample", rec["sample_token"])
        f = frame_payload(rec, cam_from_ego(src, sample), VIEW)
        image_path = os.path.join(cfg["nuscenes"]["dataroot"], rec["cam_front"])
        base = "%03d_%s_%s" % (k, rec["scene_name"], rec["sample_token"][:8])
        render_sheet(f, image_path, os.path.join(out_dir, base + ".jpg"))
        with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(frame_digest(k, f))
        has_tl = tl_strip(f, image_path, os.path.join(out_dir, base + "_tl.jpg"))
        index.append({"k": k, "scene": rec["scene_name"], "sample_token": rec["sample_token"], "n_targets": len(f["U"]),
                      "base": base, "has_tl": has_tl, "img": rec["cam_front"], "unit_ids": [u[0] for u in f["U"]]})
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return {"frames": len(index), "dir": out_dir}
