"""Small helpers: config loading and a matplotlib shim needed by nuscenes-devkit."""
import os
from typing import Any, Dict

import yaml


def patch_matplotlib() -> None:
    """nuscenes-devkit calls plt.style.use('seaborn-whitegrid'), removed in matplotlib>=3.6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.style as mstyle
    if "seaborn-whitegrid" not in mstyle.library and "seaborn-v0_8-whitegrid" in mstyle.library:
        mstyle.library["seaborn-whitegrid"] = mstyle.library["seaborn-v0_8-whitegrid"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["nuscenes"]["dataroot"] = os.path.expanduser(cfg["nuscenes"]["dataroot"])
    cfg["output_dir"] = os.path.expanduser(cfg["output_dir"])
    return cfg


def out_path(cfg: Dict[str, Any], *parts: str) -> str:
    p = os.path.join(cfg["output_dir"], *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
