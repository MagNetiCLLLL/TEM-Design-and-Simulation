"""Bracket the D95 waist with propagated planes and measure the two-axis crossover topology."""

import argparse
import json
import numpy as np

from solve_fields import CACHE
from observe import observation
from polish import polish
from tune_objective import field_trial, report_surface


def verify(seed, name):
    row = json.loads((CACHE / seed / "metrics.json").read_text())
    ni, step = row["objective_ampere_turns"], row["max_step_mm"]
    fields = (json.loads(p.read_text()) for p in CACHE.glob("armco-*.json"))
    field = next(record for record in fields if record["field_digest"] == row["field_digest"])
    level = field["level"]
    padding, hex_scale = field["padding"], row["hexapole_scale"]
    fitted = polish(dict(case="armco", ni=ni, controls=row["controls"], hex_scale=hex_scale),
                    step, level, padding, name, cache=CACHE)
    q = fitted["controls"]
    mesh, tracker, base = field_trial(ni, level, step, hex_scale, padding)
    report_surface(mesh, tracker, base, q, ni, level, step, name, hex_scale, padding)
    measured, arrays = observation(q, "armco", ni, level, padding, step, hex_scale, len(tracker.ids), True, cache=CACHE)
    (CACHE / name / "focus.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
    np.savez_compressed(CACHE / name / "focus.npz", z_m=arrays[0], moments=arrays[1],
                        focus_z_m=arrays[2], D95_nm_RMS_nm_core_nA=arrays[3])
    print("Measured D95 waist bracket (nm)", measured["D95_waist_bracket_nm"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed")
    parser.add_argument("name")
    args = parser.parse_args()
    verify(args.seed, args.name)
