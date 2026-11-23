"""Find relay branches from the transmitted ensemble, then verify in the nonlinear field."""

import argparse
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from relay import BASE, HERE, OBJ, TL, WORK, axis, checkpoint, layered, load_bias, wq, write
from material_relay_branches import fast, FACTOR
from column_trace import crossings

TOP = 1.5991999975


class Screen:
    def __init__(self, base=BASE):
        with np.load(checkpoint()) as f:
            p, w, start = f["p"], f["w"], float(f["z"])
        self.z = np.unique(np.r_[np.arange(start, 1.610, 0.00012), TOP])
        self.w = w / w.sum()
        ph = p[:, :4] - np.average(p[:, :4], weights=w, axis=0)
        self.x = ph[:, 0] + 1j * ph[:, 1]
        self.slope = ph[:, 2] + 1j * ph[:, 3]
        with np.load(layered(BASE)) as f:
            self.b0 = CubicSpline(f["z_m"], f["layers"][0, :, 0, 2])(self.z.clip(max=TOP + 0.0015))
        self.frozen = self.b0 if base == BASE else axis(base, self.z)
        self.dt, self.do = [self.response(k) for k in ("tl", "objective")]
        drives = load_bias(base)
        self.t0, self.o0 = drives["probe_tl12_lens"], drives["objective_lens/physical-pack-4"]
        self.i = np.searchsorted(self.z, TOP)

    def response(self, k):
        with np.load(WORK / f"{BASE.stem}-{k}-bore-0.5mm.npz") as f:
            return CubicSpline(f["z"], f["coef"][:, 0, 2])(self.z)

    def optics(self, tl, obj, exact=None):
        field = (
            self.frozen + (tl - self.t0) * self.dt + (obj - self.o0) * self.do
            if exact is None
            else exact
        )
        v = self.slope - 1j * FACTOR * field[0] * self.x
        xx = self.w @ abs(self.x) ** 2
        vv = self.w @ abs(v) ** 2
        xv = self.w @ (self.x.conj() * v).real
        M = fast(self.z, field)
        a, b, c, d = M.T
        var = a * a * xx + 2 * a * b * xv + b * b * vv
        cov = a * c * xx + (a * d + b * c) * xv + b * d * vv
        return M, v, var, cov

    def describe(self, tl, obj, exact=None):
        M, v, var, cov = self.optics(tl, obj, exact)
        i = self.i
        xy = M[i, 0] * self.x + M[i, 1] * v
        roots = crossings(self.z * 1000, cov)
        return dict(
            tl_A=float(tl),
            obj_A=float(obj),
            roots=roots,
            D95_um=wq(abs(xy), self.w) * 2e6,
            slope_mrad=1000 * cov[i] / var[i] ** 0.5,
        )

    def select(self, target):
        limits = [(-340.0, 340.0), (-60.0, 65.0)]

        def find(ix, f):
            lo, hi = limits[ix]
            grid = np.linspace(lo, hi, 401)
            y = np.array([f(v) for v in grid])
            return [brentq(f, grid[i], grid[i + 1]) for i in np.where(y[1:] * y[:-1] < 0)[0]]

        def tl_cov(t):
            return np.interp(target / 1000, self.z, self.optics(t, OBJ)[-1])

        out = []
        for t in find(0, tl_cov):
            step = self.describe(t, OBJ)
            if not any(abs(r - target) < 0.2 for r in step["roots"]):
                continue
            options = []
            for o in find(1, lambda o: self.optics(t, o)[-1][self.i]):
                row = self.describe(t, o)
                if any(abs(r - TOP * 1000) < 0.12 for r in row["roots"]):
                    options.append(row)
            out.append(dict(tl_stage=step, focus_choices=options))
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=1423.0)
    parser.add_argument("--base", default=BASE.stem)
    a = parser.parse_args()
    model = Screen(WORK / (a.base + ".npz"))
    print("Baseline screen", model.describe(model.t0, model.o0))
    choices = model.select(a.target)
    for c in choices:
        print(c["tl_stage"])
        for f in c["focus_choices"]:
            if len([r for r in f["roots"] if 1420 < r < 1598]) <= 1:
                print("  ", f)
    write(HERE / "screen.json", dict(target_mm=a.target, base=a.base, branches=choices))
