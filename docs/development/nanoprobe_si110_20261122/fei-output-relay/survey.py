"""Screen 3D transmitted rays using the rebias-specific axial-field change."""

import argparse
import cupy as cp
import numpy as np
from scipy.interpolate import make_interp_spline
from aim import FACTOR, TOP, fast
from column_trace import crossings
from relay import (
    BASE,
    CACHE,
    HERE,
    WORK,
    BoreTracker,
    advance,
    covariance,
    electric_reference,
    layered,
    load_bias,
    optics_mesh,
    propagate,
    relaunch,
    source_packet,
    values,
    wq,
    write,
)
from supplies import field


class Survey:
    def __init__(self, step=0.12, reference=BASE, n=524288, table=None):
        self.m, _, _ = optics_mesh(step)
        self.t = BoreTracker(self.m, table or layered(reference), source_packet(relaunch(reference, n)))
        self.i = np.searchsorted(self.m.z, 1.04)
        ((_, p, loss),) = propagate(self.t, self.m, [self.i])
        ok = loss == -1
        self.p = p[ok]
        self.en = self.t.de[ok] + 300000
        self.ref = electric_reference(self.t)[ok]
        self.w = self.t.weights[ok] * 15
        self.zero = np.full(len(self.w), -1, np.int32)
        self.zz, self.b0 = field(reference)
        self.blank = np.zeros((len(self.m.nodes), 20))
        self.cache = {}

    def coeff(self, b):
        spl = make_interp_spline(self.zz, b - self.b0, k=5)
        fd = self.blank.copy()
        for i in range(6):
            fd[:, i] = spl(self.m.nodes, nu=i)
        return cp.asarray(fd)

    def run(self, b, target, detail=False, focus=False):
        m, t, w = self.m, self.t, self.w
        zs = (
            np.r_[np.arange(1120.0, 1598.7, 1.0), target, TOP * 1000] if detail else [target, TOP * 1000]
        )
        stops = np.unique(np.minimum(np.searchsorted(m.z, np.asarray(zs) * 0.001), len(m.z) - 1))
        fd = self.coeff(b)
        p = self.p.copy()
        prev = self.i
        hit = self.zero
        stats = []
        sizes = []
        for group in np.array_split(stops, 1 + len(stops) // 24):
            for p, hit in advance(t, fd, p, self.en, self.ref, hit, prev, group):
                if np.any(hit != -1):
                    raise ValueError("Screened rays intercepted after the handoff")
                stats.append(covariance(p, w))
                if detail:
                    sizes.append(
                        2e6
                        * wq(
                            np.linalg.norm(p[:, :2] - np.average(p[:, :2], weights=w, axis=0), axis=1), w
                        )
                    )
            prev = group[-1]
        z = m.z[stops] * 1000
        s = np.array(stats)
        growth = s[:, 2:].sum(axis=1) / np.sqrt(s[:, :2].sum(axis=1)) * 1000
        ret = dict(
            target_slope_mrad=float(np.interp(target, z, growth)),
            surface_slope_mrad=float(growth[-1]),
            D95_um=float(values(p, w)["D95_um"]),
        )
        if detail:
            ret.update(
                waists_mm=crossings(z, s[:, 2:].sum(axis=1)),
                x_waists_mm=crossings(z, s[:, 2]),
                y_waists_mm=crossings(z, s[:, 3]),
                upstream_D95=[
                    float(z[np.argmin(np.where(z > 1570, sizes, np.inf))]),
                    float(min(v for v, j in zip(sizes, z) if j > 1570)),
                ],
            )
        if focus:
            res = []
            xy = p[:, 0] + 1j * p[:, 1]
            xy -= np.average(xy, weights=w)
            vel = p[:, 2] + 1j * p[:, 3]
            vel -= np.average(vel, weights=w)
            v = vel - 1j * FACTOR * np.interp(TOP, self.zz, b) * xy
            for direction in [-1, 1]:
                zz = TOP + direction * np.arange(0.0, 9.001, 0.05) * 0.001
                M = fast(zz, np.interp(zz, self.zz, b))
                for zm, (a, bm, c, d) in zip(zz, M):
                    res.append((float(zm * 1000), 2e6 * wq(abs(a * xy + bm * v), w)))
            best = min(res, key=lambda row: row[1])
            ret["local_D95_waist_mm_um"] = best
            ret["focus_window_bound"] = bool(abs(best[0] - TOP * 1000) >= 8.999)
        return ret


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=float, default=1423.0)
    a = p.parse_args()
    m = Survey()
    out = []
    for path in sorted(CACHE.glob("global-*-fei-output-full-axis.npz")):
        state = WORK / (path.name.split("-fei-output")[0] + ".npz")
        b = load_bias(state)
        r = m.run(field(state)[1], a.target, True)
        r.update(field=state.stem, tl_A=b["probe_tl12_lens"], obj_A=b["objective_lens/physical-pack-4"])
        print(r, flush=True)
        out.append(r)
    write(HERE / "full-axis-survey.json", out)
