"""Replay the reported operating points from their saved controls and physical gun exit."""

import argparse
import json

import numpy as np

from column import CACHE, HERE, POINT, prepare
from fields import CASES, solve
from observe import observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    archive = json.loads((HERE / "results.json").read_text())
    assert json.loads((POINT / "gun.json").read_text())["sha256"] == archive["gun_sha256"]
    CACHE.mkdir(parents=True, exist_ok=True)
    if any(not (CACHE / f"{key}.npz").exists() for key in ["columns", "mechanical", "electric"]):
        prepare()
    with np.load(CACHE / "electric.npz") as electric:
        assert str(electric["digest"]) == archive["electrostatic_digest"]
    for case in args.cases:
        ref = archive["cases"][case]["primary"]
        magnitude = abs(ref["ni"])
        level, padding = ref["level"], ref["padding"]
        if not (CACHE / f"{case}-{magnitude:g}-{level}-p{padding:g}.npz").exists():
            solve(case, magnitude, level, padding)
        measured, _ = observation(
            ref["controls"],
            case,
            ref["ni"],
            level,
            padding,
            ref["step_mm"],
            ref["hex_scale"],
            archive["source_count"],
            True,
        )
        for key in ("D95_nm", "alpha95_mrad", "transmitted_uA", "core_50pm_nA"):
            if not np.isclose(measured["surface"][key], ref["surface"][key], rtol=1e-8, atol=1e-8):
                raise ValueError(f"{case} replay differs for {key}: {measured['surface'][key]}")
        assert measured["D95_waist_bracket_nm"] == ref["D95_waist_bracket_nm"]
        print(case, measured["surface"], measured["D95_waist_bracket_nm"], flush=True)


if __name__ == "__main__":
    main()
