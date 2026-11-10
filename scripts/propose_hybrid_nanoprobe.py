"""Propose condenser zoom using a zero-current on-axis response, never qualify it.

The seed is an executed physical input snapshot. Actual original source arrays
provide the entrance phase-front curvature only; a normalized diagnostic ray
does not replace any source particle. Fresh full-source validation is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import root

from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.unified_field_runtime import capture_unified_field_runtime
from temsim.physics.round_lens_aberration_benchmark import _charge_over_momentum


@njit
def response(height, slope, strengths, step):
    for strength in strengths:
        k = np.sqrt(strength)
        sine = np.sin(k*step)/k if k else step
        cosine = np.cos(k*step)
        height, slope = cosine*height+sine*slope, -strength*sine*height+cosine*slope
    return np.array([height, slope])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=Path)
    parser.add_argument("--target-mrad", type=float, default=5.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads((args.seed/"instrument.json").read_text())
    state = InstrumentSnapshot.from_dict(record).restore()
    report = json.loads((args.seed/"report.json").read_text())
    with np.load(args.seed/"original_source.npz") as f:
        live = f["exit_alive"]
        xy = np.column_stack([f["exit_"+key][live] for key in ("x_m", "y_m")])
        slopes = np.column_stack([f["exit_"+key][live] for key in ("tx_rad", "ty_rad")])
        weight = f["exit_weight"][live]
    xy -= np.average(xy, axis=0, weights=weight)
    slopes -= np.average(slopes, axis=0, weights=weight)
    curvature = np.sum(weight[:, None]*xy*slopes)/np.sum(weight[:, None]*xy**2)
    grid = np.linspace(state.electron_gun.exit_plane_z_mm, 1030., 23201)*.001
    points = np.column_stack([np.zeros(len(grid)-1), np.zeros(len(grid)-1), (grid[1:]+grid[:-1])/2])
    runtime = capture_unified_field_runtime(state)
    keys = ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3")
    fields = np.array([runtime.snapshot.sample(points, source_ids=[key])["B_t"][:, 2] for key in keys])
    rigidity = .5*_charge_over_momentum(state.electron_gun.nominal_exit_energy_ev)
    def ray(factors):
        return response(1., curvature, (rigidity*(fields[0]+fields[1]*factors[0]+fields[2]*factors[1]))**2, grid[1]-grid[0])
    baseline = ray(np.ones(2))
    ratio = args.target_mrad/(report["planes"][0]["phase_space"]["alpha95_rad"]*1000)
    solution = root(lambda factors: (ray(factors)-baseline*ratio)/np.maximum(abs(baseline), 1e-8), np.ones(2), tol=1e-10)
    if not solution.success or np.linalg.norm(solution.fun) > 1e-7:
        raise ValueError(str(solution.message))
    controls = {}
    for index, key in enumerate(keys[1:]):
        lens = next(lens for lens in state.lenses if lens.key == key)
        value = lens.percent*solution.x[index]
        if not 0 < value <= lens.max_percent:
            raise ValueError("Condenser proposal outside physical control limits")
        controls[key] = {"percent": float(value)}
    output = dict(controls=controls, target_mrad=args.target_mrad,
        amplitude_ratio=float(ratio), comparison_plane_mm=1030.,
        input_curvature_per_m=float(curvature), seed=str(args.seed.resolve()),
        qualification="UNQUALIFIED_ZERO_CURRENT_FIRST_ORDER_PROPOSAL",
        model="On-axis Larmor transfer; full physical source, aberrations and aperture interception require fresh execution")
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
