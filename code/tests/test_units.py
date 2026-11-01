import math

import numpy as np

from drn.consensus import fleiss_kappa, majority
from drn.export import assign_splits
from drn.lanes import point_polyline_dist, to_ego_frame
from drn.source import coarse_class, resample_track
from drn.ttc import pair_ttc


def test_pair_ttc_head_on():
    t, pt = pair_ttc(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 1.0,
                     np.array([100.0, 0.0]), np.array([-10.0, 0.0]), 1.0)
    assert abs(t - 4.9) < 1e-9          # gap 100 m minus 2 m of radii, closing at 20 m/s
    assert np.allclose(pt, [50.0, 0.0])


def test_pair_ttc_crossing_and_miss():
    hit = pair_ttc(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 1.0,
                   np.array([50.0, -50.0]), np.array([0.0, 10.0]), 1.0)
    assert hit is not None and 4.8 < hit[0] < 5.0
    miss = pair_ttc(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 1.0,
                    np.array([50.0, -80.0]), np.array([0.0, 10.0]), 1.0)
    assert miss is None


def test_pair_ttc_parallel_receding_overlap():
    assert pair_ttc(np.zeros(2), np.array([5.0, 0.0]), 1.0, np.array([0.0, 4.0]), np.array([5.0, 0.0]), 1.0) is None
    assert pair_ttc(np.zeros(2), np.array([-5.0, 0.0]), 1.0, np.array([10.0, 0.0]), np.array([5.0, 0.0]), 1.0) is None
    assert pair_ttc(np.zeros(2), np.zeros(2), 1.0, np.array([1.0, 0.0]), np.zeros(2), 1.0) is None   # touching, static
    assert pair_ttc(np.zeros(2), np.array([2.0, 0.0]), 1.0, np.array([1.0, 0.0]), np.zeros(2), 1.0)[0] == 0.0


def test_pair_ttc_margin_does_not_make_neighbours_collide():
    # 1.0 m apart, radii 0.3 each, margin 0.5: inside the margin but not touching and not closing.
    side_by_side = pair_ttc(np.zeros(2), np.array([1.0, 0.0]), 0.3, np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.3, 0.5)
    assert side_by_side is None
    closing = pair_ttc(np.zeros(2), np.zeros(2), 0.3, np.array([0.0, 1.0]), np.array([0.0, -1.0]), 0.3, 0.5)
    assert abs(closing[0] - 0.4) < 1e-9                    # 0.4 m gap between the bare discs at 1 m/s


def test_lane_ttc_ignores_pedestrian_pairs():
    from drn.ttc import lane_min_ttc
    lane = {"lane_token": "l", "pts": np.array([[0.0, 0.0], [50.0, 0.0]])}
    ego = {"x": -100.0, "y": 50.0, "vx": 0.0, "vy": 0.0, "size": [1.7, 4.0, 1.5]}
    cfg = {"horizon_s": 10.0, "margin_m": 0.5, "lane_dist_m": 3.0}

    def actor(name, cls, x, vx):
        return {"instance_token": name, "cls": cls, "x": x, "y": 0.0, "vx": vx, "vy": 0.0, "size": [0.6, 0.6, 1.7]}

    peds = [actor("p1", "pedestrian", 10.0, 1.0), actor("p2", "pedestrian", 20.0, -1.0)]
    assert lane_min_ttc([lane], ego, peds, cfg)["per_lane"]["l"]["min_ttc_s"] == 10.0
    mixed = [actor("p1", "pedestrian", 10.0, 0.0), actor("c1", "vehicle", 30.0, -5.0)]
    res = lane_min_ttc([lane], ego, mixed, cfg)["per_lane"]["l"]
    assert abs(res["min_ttc_s"] - (20.0 - 1.1) / 5.0) < 1e-9 and res["pair"] == ["p1", "c1"]


def test_fleiss_kappa_reference_values():

    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    rows = [[c for c, k in enumerate(counts) for _ in range(k)] for counts in table]
    assert abs(fleiss_kappa(rows) - 0.210) < 5e-4
    assert abs(fleiss_kappa([[0, 0, 0], [1, 1, 1], [4, 4, 4]]) - 1.0) < 1e-12
    assert fleiss_kappa([]) is None
    assert fleiss_kappa([[2, 2, 2], [2, 2, 2]]) is None   # one category only: undefined


def test_majority_needs_strict_majority():
    assert majority([2, 2, 3]) == 2
    assert majority([1, 2, 3]) is None
    assert majority([4, 4, 4]) == 4


def test_resample_track_masks_instead_of_extrapolating():
    ts = np.array([10.0, 10.5, 11.0])
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    hist, hmask, fut, fmask = resample_track(ts, xy, 10.5, 10, 20, 30)
    assert hist.shape == (20, 2) and fut.shape == (30, 2)
    assert hmask.sum() == 6 and fmask.sum() == 5          # 0.5 s observed each side
    assert np.allclose(hist[-1], [5.0, 0.0]) and np.allclose(hist[-2], [4.0, 0.0])
    assert np.allclose(fut[0], [6.0, 0.0]) and np.allclose(fut[4], [10.0, 0.0])
    assert np.allclose(fut[5:], 0.0)


def test_geometry_helpers():
    ego = {"x": 10.0, "y": 5.0, "yaw": math.pi / 2}
    assert np.allclose(to_ego_frame(np.array([[10.0, 15.0]]), ego), [[10.0, 0.0]])   # straight ahead
    assert np.allclose(to_ego_frame(np.array([[0.0, 5.0]]), ego), [[0.0, 10.0]])     # to the left
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert abs(point_polyline_dist(np.array([5.0, 3.0]), line) - 3.0) < 1e-9
    assert abs(point_polyline_dist(np.array([13.0, 4.0]), line) - 5.0) < 1e-9


def test_classes_and_splits():
    assert coarse_class("vehicle.bus.rigid") == "vehicle"
    assert coarse_class("vehicle.bicycle") == "bicycle"
    assert coarse_class("human.pedestrian.child") == "pedestrian"
    assert coarse_class("movable_object.barrier") is None
    scenes = ["s%d" % i for i in range(20)]
    sp = assign_splits(scenes, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert sorted(sp) == sorted(scenes)
    assert [list(sp.values()).count(k) for k in ("train", "val", "test")] == [14, 3, 3]
    assert sp == assign_splits(scenes[::-1], {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)


def test_loose_ttc_counts_any_approaching_pair():
    from drn.ttc import lane_loose_ttc
    lane = {"lane_token": "l", "pts": np.array([[0.0, 0.0], [50.0, 0.0]])}
    far = {"lane_token": "far", "pts": np.array([[0.0, 40.0], [50.0, 40.0]])}
    ego = {"x": -100.0, "y": 50.0, "vx": 0.0, "vy": 0.0, "size": [1.7, 4.0, 1.5]}
    cfg = {"horizon_s": 10.0, "loose": {"max_pair_m": 30.0, "lane_dist_m": 3.0}}

    def actor(name, cls, x, y, vx):
        return {"instance_token": name, "cls": cls, "x": x, "y": y, "vx": vx, "vy": 0.0, "size": [1.8, 4.5, 1.5]}

    # Oncoming cars in neighbouring lanes (3.5 m apart sideways): they never touch, but the gap closes at 10 m/s.
    cars = [actor("a", "vehicle", 10.0, 0.0, 5.0), actor("b", "vehicle", 30.0, 3.5, -5.0)]
    rng = float(np.hypot(20.0, 3.5))
    res = lane_loose_ttc([lane, far], ego, cars, cfg)
    assert abs(res["l"] - rng * rng / (20.0 * 10.0)) < 1e-9 and res["far"] == 10.0
    assert lane_loose_ttc([lane], ego, [cars[0], actor("b", "vehicle", 30.0, 3.5, 5.0)], cfg)["l"] == 10.0    # same speed: not closing
    assert lane_loose_ttc([lane], ego, [cars[0], actor("b", "vehicle", 60.0, 3.5, -5.0)], cfg)["l"] == 10.0   # farther than max_pair_m
    peds = [actor("p1", "pedestrian", 10.0, 0.0, 1.0), actor("p2", "pedestrian", 20.0, 0.0, -1.0)]
    assert lane_loose_ttc([lane], ego, peds, cfg)["l"] == 10.0


def test_time_consensus_and_hybrid():
    from drn.ttc import consensus_time, hybrid_ttc
    assert consensus_time([2.4, 2.0, 3.0], ["est", "est", "calc"]) == (2.4, "est")
    assert consensus_time([4.0, None, 4.4], ["calc", "none", "calc"]) == (4.2, "calc")
    assert consensus_time([4.0, None, None], ["est", "none", "none"]) == (None, "none")      # one voice is not a time
    assert hybrid_ttc(1.2, 2.4, "est", 10.0) == 2.4          # estimated: the estimate replaces the computed value
    assert hybrid_ttc(1.2, 4.2, "calc", 10.0) == 1.2         # computation accepted: the computed value stands
    assert hybrid_ttc(None, None, "none", 10.0) == 10.0 and hybrid_ttc(12.0, None, "none", 10.0) == 10.0
