"""Locate candidate operating currents with multi-current FEM secants and native rays."""

import argparse
from functools import lru_cache
import json

import cupy as cp
import numpy as np
from scipy.optimize import brentq

from column_current import ACTIVE, OWNER, START, SURFACE
from joint_pivot_tuning import OUT, stations
from joint_relay_search import NI, model_at, mono_roots, surface_metrics, transfer
from normal_working_point import CACHE
from refocus_a2 import WORK
from relay_transfer import advance
from trace_normal_joint_field import beam_moments


class Search:
    def __init__(self, anchors, n, step, half=8):
        self.anchors = np.array(sorted(anchors), float)
        base = self.anchors[-1]
        self.m, self.q = model_at(base, NI, n, step, half)
        self.grids = []
        for a in self.anchors:
            p = (
                CACHE / "normal-joint-field.npz"
                if a == 10
                else OUT / f"tl12-{a:.10g}A-field-{half}.npz"
            )
            with np.load(p) as f:
                g = f["B_zyx_T"]
                stride = (g.shape[1] - 1) // (2 * half)
                self.grids.append(cp.asarray(np.ascontiguousarray(g[:, ::stride, ::stride])))
        self.obj = np.asarray(
            sorted(
                -float(p.name.split("-")[1]) for p in (WORK / "column").glob("armco-*-gap-fine-p8.json")
            )
        )
        self.mini = OWNER[ACTIVE] == "mini_condenser"
        self.iso_z = np.linspace(1.420, SURFACE, 1801)

    @lru_cache(maxsize=5)
    def bg(self, current):
        return self.m.m.base("armco", current, "gap-fine", 8)

    @staticmethod
    def bracket(vals, x):
        j = int(np.clip(np.searchsorted(vals, x, side="right") - 1, 0, len(vals) - 2))
        return j, (x - vals[j]) / (vals[j + 1] - vals[j])

    @lru_cache(maxsize=5)
    def objective(self, ni):
        j, f = self.bracket(self.obj, ni)
        m = self.m
        m.bg = (1 - f) * self.bg(self.obj[j]) + f * self.bg(self.obj[j + 1])
        iso = m.bg - m.m.group[0] + np.einsum("i,ijk->jk", self.q[self.mini], m.coils[self.mini])
        mat = transfer(self.iso_z, np.interp(self.iso_z, m.m.nodes, iso[:, 0]))
        d = -mat[-1, 2] * mat[:, 1] + mat[-1, 3] * mat[:, 0]
        candidates = np.r_[mono_roots(self.iso_z, d), mono_roots(self.iso_z, -d)]
        if not len(candidates):
            raise ValueError("No downstream front focal plane")
        target = min(candidates, key=lambda x: abs(x - 1.49))
        return m.field(self.q), target

    @lru_cache(maxsize=2500)
    def measure(self, a, ni):
        m = self.m
        j, f = self.bracket(self.anchors, a)
        m.t.tables[0][...] = (1 - f) * self.grids[j] + f * self.grids[j + 1]
        field, target = self.objective(ni)
        obs = stations(m, [target - 0.0006, target + 0.0006, SURFACE])
        data = advance(m.t, field, m.t.input, m.en, m.phi0, m.loss, 0, m.indices(obs))
        cov = []
        for phase, loss in data[:2]:
            r = beam_moments(phase, loss == -1, m.t.weights)
            cov.append(r[1] + r[2])
        sample = surface_metrics(m, *data[-1])
        return float(np.interp(target, obs[:2], cov)), cov[1] - cov[0], sample, float(target * 1000)

    def survey(self, ni_values, limits):
        records = []
        for ni in ni_values:
            tested = np.linspace(*limits, num=21)

            def error(a):
                return self.measure(a, ni)[0]

            es = np.asarray([error(float(a)) for a in tested])
            for j in np.flatnonzero(es[:-1] * es[1:] < 0):
                a = brentq(error, tested[j], tested[j + 1], xtol=0.005)
                cov, grad, s, front = self.measure(a, ni)
                row = dict(
                    tl12_A=a,
                    ni=ni,
                    front_mm=front,
                    radial_minimum=bool(grad > 0),
                    signed_growth=cov,
                    gradient=grad,
                    **s,
                )
                records.append(row)
                print(
                    f"NI {ni:g}  TL12 {a:.3f} A  minimum {grad>0}  D95 {s['D95_um']:.4f} um  "
                    f"drift {s['drift_D95_mm']:.4f} mm",
                    flush=True,
                )
            print(f"Finished NI {ni:g}", flush=True)
            (OUT / f"native-secant-{self.anchors[-1]:g}-{abs(ni_values[0]):g}.json").write_text(
                json.dumps(records, indent=2, allow_nan=False)
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchors", type=float, nargs="+", required=True)
    p.add_argument("--objective", type=float, nargs="+", required=True)
    p.add_argument("--limits", type=float, nargs=2, default=[-350.0, 350.0])
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--step", type=float, default=0.16)
    args = p.parse_args()
    Search(args.anchors, args.count, args.step).survey(args.objective, args.limits)
