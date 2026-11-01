import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drn.compat import load_config  # noqa: E402
from fake_nusc import write_fake_nuscenes  # noqa: E402


@pytest.fixture(scope="session")
def fake_cfg(tmp_path_factory):
    """Config pointing at the synthetic fixture dataset (tests only)."""
    root = tmp_path_factory.mktemp("fake_nuscenes")
    out = tmp_path_factory.mktemp("drn_out")
    write_fake_nuscenes(str(root))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(here, "config.yaml"))
    cfg["nuscenes"] = {"dataroot": str(root), "version": "v1.0-mini"}
    cfg["output_dir"] = str(out)
    cfg["extraction"]["min_gap_s"] = 1.0
    # Pin the tunable defaults the assertions depend on, so retuning config.yaml does not break tests.
    cfg["label_roi"] = {"x_min": -10.0, "x_max": 45.0, "y_abs": 18.0, "min_length_m": 5.0}
    cfg["templates"]["occlusion"]["radius_m"] = 40.0
    cfg["label_units"] = {"chain": True, "ego_path": False, "parallel": True, "movement": True,
                          "movement_end_dist_m": 5.0, "movement_heading_deg": 30.0}
    cfg["extraction"].update({"require_template": True, "min_label_units": 2})
    return cfg
