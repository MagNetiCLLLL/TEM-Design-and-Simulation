"""Close the first, scan-off TL12 waist against the independent two-lens front focus."""

import argparse
import json
import sys

import cupy as cp
import numpy as np

from joint_pivot_tuning import OUT, focal_planes
from joint_relay_search import NI, model_at
from normal_working_point import AUDIT
from trace_normal_joint_field import beam_moments

sys.path.insert(0, str(AUDIT))
from tl12_excitation import main as solve


def measure(amps, ni, half, step=0.0375):
    coarse, q = model_at(amps, ni, 1024, 0.12, half)
    root = min(focal_planes(coarse, q)["mini_and_objective"], key=lambda v: abs(v - 1490)) * 0.001
    del coarse
    z = root + np.array([-0.0002, 0.0, 0.0002])
    m, q = model_at(amps, ni, 32768, step, half, z)
    _, states = m.replay(q, z)
    live = states[0][1] == -1
    if not all(np.array_equal(live, loss == -1) for _, loss in states):
        raise ValueError("Interception inside waist measurement")
    rows = np.array([beam_moments(p, live, m.t.weights) for p, _ in states])
    cov = rows[:, 1] + rows[:, 2]
    grad = (cov[-1] - cov[0]) / 0.4
    if grad <= 0:
        raise ValueError("Selected current is on a maximum-radius branch")
    delta = -cov[1] / grad
    result = dict(
        amps=amps,
        objective_At=ni,
        half=half,
        step_mm=step,
        front_mm=root * 1000,
        offset_estimate_mm=delta,
        radial_cov_um_mrad=cov[1],
        growth_per_mm=grad,
        radius_um=rows[1, 0],
        principal_growth=rows[1, 3:].tolist(),
    )
    print(result, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("initial", type=float)
    p.add_argument(
        "--slope", type=float, required=True, help="Initial change in waist z (mm) per ampere"
    )
    p.add_argument("--ni", type=float, default=NI)
    p.add_argument("--tag", default="aligned-current")
    args = p.parse_args()
    a = args.initial
    slope = args.slope
    history = []
    for half in (8, 24):
        for iteration in range(7):
            solve([a], half)
            row = measure(a, args.ni, half)
            history.append(row)
            keep = OUT / (args.tag + ".json")
            keep.write_text(json.dumps(history, indent=2))
            dz = row["offset_estimate_mm"]
            if abs(dz) <= 0.018:
                break
            other = [r for r in history[:-1] if r["half"] == half and r["amps"] != a]
            if other:
                b = min(other, key=lambda r: abs(r["amps"] - a))
                slope = (b["offset_estimate_mm"] - dz) / (b["amps"] - a)
            if abs(slope) < 0.015 or not np.isfinite(slope):
                raise ValueError("A single stable waist branch is required")
            a = round(float(a - np.clip(dz / slope, -10, 10)), 5)
            cp.get_default_memory_pool().free_all_blocks()
        else:
            raise ValueError("Material-current iteration did not locate the waist")
