"""Tiny synthetic nuScenes-format dataset, used ONLY by the tests.

It lets the test suite drive the real nuscenes-devkit loaders (tables + map expansion)
without the real data. It is not, and must never be used as, research data.

Layout (metres, global frame): a 4-way intersection occupying [990,1010]x[990,1010].
  L_E_in   eastbound approach  (900,998)   -> (990,998)
  L_E_in2  eastbound 2nd lane  (900,994.5) -> (990,994.5)
  L_E_out  eastbound exit      (1010,998)  -> (1100,998)     <- two connectors merge here
  L_N_in   northbound approach (1002,900)  -> (1002,990)
  L_N_out  northbound exit     (1002,1010) -> (1002,1100)
  C_E, C_E2 (merge), C_N       connectors through the intersection
"""
import json
import math
import os

from PIL import Image

LOCATION = "singapore-onenorth"
LANES = {  # token: (kind, start, end)
    "L_E_in": ("lane", (900.0, 998.0), (990.0, 998.0)),
    "L_E_in2": ("lane", (900.0, 994.5), (990.0, 994.5)),
    "L_E_out": ("lane", (1010.0, 998.0), (1100.0, 998.0)),
    "L_N_in": ("lane", (1002.0, 900.0), (1002.0, 990.0)),
    "L_N_out": ("lane", (1002.0, 1010.0), (1002.0, 1100.0)),
    "C_E": ("lane_connector", (990.0, 998.0), (1010.0, 998.0)),
    "C_E2": ("lane_connector", (990.0, 994.5), (1010.0, 998.0)),
    "C_N": ("lane_connector", (1002.0, 990.0), (1002.0, 1010.0)),
}
CONNECTIVITY = {
    "L_E_in": {"incoming": [], "outgoing": ["C_E"]},
    "L_E_in2": {"incoming": [], "outgoing": ["C_E2"]},
    "C_E": {"incoming": ["L_E_in"], "outgoing": ["L_E_out"]},
    "C_E2": {"incoming": ["L_E_in2"], "outgoing": ["L_E_out"]},
    "L_E_out": {"incoming": ["C_E", "C_E2"], "outgoing": []},
    "L_N_in": {"incoming": [], "outgoing": ["C_N"]},
    "C_N": {"incoming": ["L_N_in"], "outgoing": ["L_N_out"]},
    "L_N_out": {"incoming": ["C_N"], "outgoing": []},
}
N_SAMPLES = 6
DT = 0.5
T0_US = 1_600_000_000_000_000


def _attrs(cat, vx, vy):
    """nuScenes-style attribute tokens: moving vehicles are vehicle.moving, stationary ones vehicle.parked."""
    if "pedestrian" in cat:
        return ["attr_pedestrian_moving"]
    if "vehicle" in cat or "car" in cat or "truck" in cat:
        return ["attr_vehicle_moving"] if (vx or vy) else ["attr_vehicle_parked"]
    return []


def _yaw_quat(yaw):
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


class _MapBuilder:
    def __init__(self):
        self.nodes, self.polygons, self.lines = [], [], []

    def polygon(self, coords):
        toks = []
        for x, y in coords:
            tok = "node_%d" % len(self.nodes)
            self.nodes.append({"token": tok, "x": x, "y": y})
            toks.append(tok)
        ptok = "poly_%d" % len(self.polygons)
        self.polygons.append({"token": ptok, "exterior_node_tokens": toks, "holes": []})
        return ptok

    def line(self, coords):
        toks = []
        for x, y in coords:
            tok = "node_%d" % len(self.nodes)
            self.nodes.append({"token": tok, "x": x, "y": y})
            toks.append(tok)
        ltok = "line_%d" % len(self.lines)
        self.lines.append({"token": ltok, "node_tokens": toks})
        return ltok


def _lane_rect(a, b, half=1.75):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    nx, ny = -dy / n * half, dx / n * half
    return [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]


def _write_map(root):
    mb = _MapBuilder()
    lanes, connectors, arcs = [], [], {}
    for tok, (kind, a, b) in LANES.items():
        ptok = mb.polygon(_lane_rect(a, b))
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        arcs[tok] = [{"start_pose": [a[0], a[1], yaw], "end_pose": [b[0], b[1], yaw], "shape": "LSL",
                      "radius": 999.999, "segment_length": [0.0, length, 0.0]}]
        if kind == "lane":
            lanes.append({"token": tok, "polygon_token": ptok, "lane_type": "CAR",
                          "from_edge_line_token": "", "to_edge_line_token": "",
                          "left_lane_divider_segments": [], "right_lane_divider_segments": []})
        else:
            connectors.append({"token": tok, "polygon_token": ptok})
    inter = mb.polygon([(990, 990), (1010, 990), (1010, 1010), (990, 1010)])
    approach = mb.polygon([(900, 992), (990, 992), (990, 1004), (900, 1004)])
    crossing = mb.polygon([(984, 992), (988, 992), (988, 1002), (984, 1002)])
    tl_line = mb.line([(989, 1003), (989, 1004)])
    obj = {
        "version": "1.3", "canvas_edge": [2000.0, 2000.0],
        "polygon": mb.polygons, "line": mb.lines, "node": mb.nodes,
        "drivable_area": [{"token": "da_0", "polygon_tokens": [inter, approach]}],
        "road_segment": [
            {"token": "rs_int", "polygon_token": inter, "is_intersection": True, "drivable_area_token": "da_0"},
            {"token": "rs_app", "polygon_token": approach, "is_intersection": False, "drivable_area_token": "da_0"},
        ],
        "road_block": [], "lane": lanes,
        "ped_crossing": [{"token": "pc_0", "polygon_token": crossing, "road_segment_token": "rs_app"}],
        "walkway": [], "stop_line": [], "carpark_area": [], "road_divider": [], "lane_divider": [],
        "traffic_light": [{"token": "tl_0", "line_token": tl_line, "traffic_light_type": "VERTICAL",
                           "from_road_block_token": "", "items": [],
                           "pose": {"tx": 989.0, "ty": 1003.5, "tz": 5.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}}],
        "arcline_path_3": arcs, "connectivity": CONNECTIVITY, "lane_connector": connectors,
    }
    os.makedirs(os.path.join(root, "maps", "expansion"), exist_ok=True)
    with open(os.path.join(root, "maps", "expansion", LOCATION + ".json"), "w") as fh:
        json.dump(obj, fh)
    Image.new("L", (8, 8), 0).save(os.path.join(root, "maps", "fake.png"))  # devkit asserts the mask file exists


# Actors: (instance, category, x0, y0, vx, vy, size(w,l,h), visibility)
ACTORS = [
    ("crossing_car", "vehicle.car", 1002.0, 975.0, 0.0, 23.0 / 6.0, [1.9, 4.5, 1.6], "4"),
    ("merging_car", "vehicle.car", 985.0, 994.5, 6.0, 0.0, [1.9, 4.5, 1.6], "4"),
    ("curb_ped", "human.pedestrian.adult", 985.0, 995.5, 0.0, 0.0, [0.6, 0.6, 1.7], "4"),
    ("hidden_ped_a", "human.pedestrian.adult", 1000.0, 1006.0, 0.0, 0.0, [0.6, 0.6, 1.7], "1"),
    ("hidden_ped_b", "human.pedestrian.adult", 1004.0, 1006.0, 0.0, 0.0, [0.6, 0.6, 1.7], "2"),
    ("hidden_parked", "vehicle.car", 1006.0, 1012.0, 0.0, 0.0, [1.9, 4.5, 1.6], "1"),   # parked: must not count
    ("barrier", "movable_object.barrier", 980.0, 1000.0, 0.0, 0.0, [0.5, 2.0, 1.0], "4"),
]
EGO = (966.0, 998.0, 6.0, 0.0)  # x0, y0, vx, vy


def write_fake_nuscenes(root, version="v1.0-mini"):
    """Create the dataset under `root` and return the list of sample tokens."""
    _write_map(root)
    table = os.path.join(root, version)
    os.makedirs(table, exist_ok=True)
    cats = sorted({a[1] for a in ACTORS})
    category = [{"token": "cat_" + c, "name": c, "description": "", "index": i} for i, c in enumerate(cats)]
    visibility = [{"token": str(i), "level": "v%d" % i, "description": ""} for i in range(1, 5)]
    sensor = [{"token": "sen_lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
              {"token": "sen_cam", "channel": "CAM_FRONT", "modality": "camera"}]
    calibrated = [{"token": "cs_lidar", "sensor_token": "sen_lidar", "translation": [0, 0, 1.8],
                   "rotation": [1, 0, 0, 0], "camera_intrinsic": []},
                  {"token": "cs_cam", "sensor_token": "sen_cam", "translation": [1.5, 0, 1.5],
                   "rotation": [1, 0, 0, 0], "camera_intrinsic": [[1000, 0, 800], [0, 1000, 450], [0, 0, 1]]}]
    log = [{"token": "log_0", "logfile": "fake", "vehicle": "fake", "date_captured": "2020-01-01", "location": LOCATION}]
    maps = [{"token": "map_0", "log_tokens": ["log_0"], "category": "semantic_prior", "filename": "maps/fake.png"}]
    samples, sample_data, ego_pose, anns = [], [], [], []
    stoks = ["sample_%d" % i for i in range(N_SAMPLES)]
    os.makedirs(os.path.join(root, "samples", "CAM_FRONT"), exist_ok=True)
    for i, tok in enumerate(stoks):
        t = i * DT
        ts = T0_US + int(round(t * 1e6))
        samples.append({"token": tok, "timestamp": ts, "scene_token": "scene_0",
                        "prev": stoks[i - 1] if i > 0 else "", "next": stoks[i + 1] if i < N_SAMPLES - 1 else ""})
        ex, ey = EGO[0] + EGO[2] * t, EGO[1] + EGO[3] * t
        for ch, cs, fmt in (("LIDAR_TOP", "cs_lidar", "pcd"), ("CAM_FRONT", "cs_cam", "jpg")):
            ptok = "pose_%s_%d" % (ch, i)
            ego_pose.append({"token": ptok, "timestamp": ts, "rotation": _yaw_quat(0.0), "translation": [ex, ey, 0.0]})
            fname = "samples/%s/fake__%s__%d.%s" % (ch, ch, ts, fmt)
            sample_data.append({"token": "sd_%s_%d" % (ch, i), "sample_token": tok, "ego_pose_token": ptok,
                                "calibrated_sensor_token": cs, "timestamp": ts, "fileformat": fmt,
                                "is_key_frame": True, "height": 900, "width": 1600, "filename": fname,
                                "prev": "sd_%s_%d" % (ch, i - 1) if i > 0 else "",
                                "next": "sd_%s_%d" % (ch, i + 1) if i < N_SAMPLES - 1 else ""})
            if ch == "CAM_FRONT":
                Image.new("RGB", (160, 90), (90, 110, 130)).save(os.path.join(root, fname))
        for name, _cat, x0, y0, vx, vy, size, vis in ACTORS:
            yaw = math.atan2(vy, vx) if (vx or vy) else 0.0
            anns.append({"token": "ann_%s_%d" % (name, i), "sample_token": tok, "instance_token": "inst_" + name,
                         "visibility_token": vis, "attribute_tokens": _attrs(_cat, vx, vy),
                         "translation": [x0 + vx * t, y0 + vy * t, 0.8], "size": size, "rotation": _yaw_quat(yaw),
                         "prev": "ann_%s_%d" % (name, i - 1) if i > 0 else "",
                         "next": "ann_%s_%d" % (name, i + 1) if i < N_SAMPLES - 1 else "",
                         "num_lidar_pts": 50, "num_radar_pts": 2})
    instance = [{"token": "inst_" + a[0], "category_token": "cat_" + a[1], "nbr_annotations": N_SAMPLES,
                 "first_annotation_token": "ann_%s_0" % a[0], "last_annotation_token": "ann_%s_%d" % (a[0], N_SAMPLES - 1)}
                for a in ACTORS]
    scene = [{"token": "scene_0", "log_token": "log_0", "nbr_samples": N_SAMPLES, "first_sample_token": stoks[0],
              "last_sample_token": stoks[-1], "name": "scene-fake", "description": "synthetic test fixture"}]
    attribute = [{"token": "attr_" + n.replace(".", "_"), "name": n, "description": ""}
                 for n in ("vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving")]
    tables = {"category": category, "attribute": attribute, "visibility": visibility, "instance": instance,
              "sensor": sensor, "calibrated_sensor": calibrated, "ego_pose": ego_pose, "log": log,
              "scene": scene, "sample": samples, "sample_data": sample_data, "sample_annotation": anns, "map": maps}
    for name, rows in tables.items():
        with open(os.path.join(table, name + ".json"), "w") as fh:
            json.dump(rows, fh)
    return stoks
