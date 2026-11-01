"""Shared fields and validation for annotation entries."""
from typing import Any, Dict, List

SIGNALS = ("red", "yellow", "green", "none", "unclear")
TTC_BASES = ("calc", "est", "none")


def parse_times(ttc: Any, basis: Any, keys: List[str], where: str) -> Dict[str, Any]:
    """Validate time-to-conflict answers and their calculation basis."""
    if ttc is None and basis is None:
        return {}
    if not isinstance(ttc, dict) or not isinstance(basis, dict) or sorted(ttc) != sorted(keys) or sorted(basis) != sorted(keys):
        raise SystemExit("%s: ttc and ttc_basis must cover exactly the labelled units" % where)
    out_t, out_b = {}, {}
    for k in keys:
        t, b = ttc[k], basis[k]
        if b not in TTC_BASES or (t is None) != (b == "none"):
            raise SystemExit("%s: ttc_basis must be calc/est/none, and null goes with none only" % where)
        if t is not None and not (0.0 <= float(t) <= 10.0):
            raise SystemExit("%s: ttc must be between 0 and 10 seconds" % where)
        out_t[k], out_b[k] = (None if t is None else round(float(t), 2)), b
    return {"ttc": out_t, "ttc_basis": out_b}
