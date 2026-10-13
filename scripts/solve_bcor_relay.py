"""Generate B-COR relay candidates without changing saved settings.

Aligned, constant-energy first-order Larmor optics only. Candidate adoption
requires the independent laboratory-frame CUDA validator. This does not solve
the finite-hexapole spherical/off-axial aberration correction problem.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from validate_image_corrector import ROOT, save


KEYS = ("image_tl11_lens", "image_tl12_lens", "image_tl21_lens",
        "image_tl22_lens", "image_tl31_lens", "image_tl32_lens", "image_adapter_lens")


class RelayModel:
    def __init__(self, state, step_mm):
        from temsim.physics.core import fields, interleaved_rk4_values, electron
        from temsim.optics.direct_alignment import _piecewise_endpoint_exact_grid
        parts = {p.key: p for p in state.image_corrector_system.components}
        self.lenses = [parts[k] for k in KEYS]
        self.original = np.array([p.percent for p in self.lenses])
        self.maximum = np.array([p.max_percent for p in self.lenses])
        self.planes = dict(sample=float(state.sample.z_mm),
            first_middle=(parts[KEYS[0]].z_mm+parts[KEYS[1]].z_mm)/2,
            middle=(parts[KEYS[2]].z_mm+parts[KEYS[3]].z_mm)/2,
            third_middle=(parts[KEYS[4]].z_mm+parts[KEYS[5]].z_mm)/2,
            **{f"hp{i}": float(parts[f"image_hp{i}_hexapole"].z_mm) for i in (1, 2, 3)},
            sad=float(parts["image_sad_plane"].z_mm))
        self.names = sorted((k for k in self.planes if k != "sample"), key=self.planes.get)
        targets = [self.planes[k] for k in self.names]
        grid = _piecewise_endpoint_exact_grid(self.planes["sample"], self.planes["sad"], step_mm, targets)
        self.z_m = np.ascontiguousarray(grid*1e-3)
        self.indices = np.asarray(np.searchsorted(grid, targets), dtype=np.int64)
        stages = interleaved_rk4_values(grid, .5*(grid[1:]+grid[:-1]))
        try:
            for p in self.lenses:
                p.percent = 0.
            self.fixed, sx, sy = fields(stages, state)
            if np.max(np.abs(sx))+np.max(np.abs(sy)) > 1e-10:
                raise ValueError("Candidate search requires aligned round fields")
            basis = []
            for p in self.lenses:
                p.percent = 100.
                basis.append(fields(stages, state)[0]-self.fixed)
                p.percent = 0.
            self.basis = np.asarray(basis)
        finally:
            for p, value in zip(self.lenses, self.original):
                p.percent = float(value)
        charge, momentum, _ = electron(state)
        self.factor = charge/(2*momentum)

    def maps(self, percent):
        from temsim.optics.direct_alignment import _rk4_axisymmetric_larmor_matrices
        g = np.ascontiguousarray(self.factor*(self.fixed+np.asarray(percent)@self.basis/100))
        raw = _rk4_axisymmetric_larmor_matrices(g, self.z_m, self.indices)
        maps = {key: raw[i] for i, key in enumerate(self.names)}
        maps["relay12"] = maps["hp2"]@np.linalg.inv(maps["hp1"])
        maps["relay23"] = maps["hp3"]@np.linalg.inv(maps["hp2"])
        return maps

    def residual(self, percent):
        m = self.maps(percent)
        return np.array([m["first_middle"][0, 1]/.01, m["middle"][0, 1]/.01,
            m["relay12"][0, 1]/.01, m["relay12"][0, 0]+1,
            m["relay23"][0, 1]/.01, m["relay23"][0, 0]+1, m["sad"][0, 1]/.01])


def main():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/bcor/candidates")
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--starts", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3 + B-COR", "No Energy Filter"))
    start = perf_counter()
    with threadpool_limits(1):
        model = RelayModel(state, args.step)
        seeds = [model.original, np.full(7, 60.), np.full(7, 45.)]
        rows = []
        for seed in seeds[:args.starts]:
            fit = least_squares(model.residual, seed, bounds=(np.full(7, .001), model.maximum),
                                max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            row = dict(cost=float(fit.cost), percent=dict(zip(KEYS, fit.x)),
                       residual=fit.fun, evaluations=int(fit.nfev))
            rows.append(row)
            print(row, flush=True)
        rows.sort(key=lambda r: r["cost"])
        fine = RelayModel(state, args.step/2)
        candidate = [rows[0]["percent"][k] for k in KEYS]
        save(args.output/"candidates.json", dict(status="CANDIDATES_ONLY", candidates=rows,
            planes_mm=model.planes, half_step_residual=fine.residual(candidate),
            half_step_maps=fine.maps(candidate), elapsed_s=perf_counter()-start,
            qualification="Position relay only; laboratory rotation and nonlinear correction not solved"))


if __name__ == "__main__":
    main()
