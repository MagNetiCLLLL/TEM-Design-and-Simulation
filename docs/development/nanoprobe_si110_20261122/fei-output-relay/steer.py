"""Bracket relay settings before repeating the complete finite-source calculation."""

import argparse
import numpy as np
from numba import njit
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import CubicSpline
from aim import FACTOR, TOP, Screen
from relay import BASE, CACHE, HERE, OBJ, WORK, load_bias, write
from supplies import field
from survey import Survey
from pivot import CHANNELS, MIX


@njit(cache=True)
def dipole_map(z, b, drive):
    x = np.zeros(2, dtype=np.complex128)
    v = x.copy()
    rot = 1.0 + 0.0j
    out = np.zeros((len(z), 2), dtype=np.complex128)
    for i in range(len(z) - 1):
        dz = z[i + 1] - z[i]
        k = FACTOR * 0.5 * (b[i] + b[i + 1])
        a = k * dz
        c = np.cos(a)
        s = dz * np.sinc(a / np.pi)
        force = -2j * FACTOR * (drive[i] + drive[i + 1]) * 0.5 * rot.conjugate() * np.exp(-0.5j * a)
        x, v = (
            c * x + s * v + (0.5 * dz**2 * np.sinc(a / (2 * np.pi)) ** 2) * force,
            -k * k * s * x + c * v + s * force,
        )
        rot *= np.exp(1j * a)
        out[i + 1] = x * rot
    return out


class PivotEstimate:
    def __init__(self, z, path=BASE):
        self.z = z
        arr = []
        for k in CHANNELS:
            with np.load(WORK / f"{path.stem}-{k}-bore-0.5mm.npz") as f:
                c = f["coef"][:, 0]
                arr.append(CubicSpline(f["z"], c[:, 0] + 1j * c[:, 1])(z))
        self.drive = np.array(arr).T @ MIX

    def __call__(self, b):
        out = dipole_map(self.z, b, self.drive)
        r = abs(out) ** 2
        spl = CubicSpline(self.z, r.sum(axis=1) * 1e18)
        ix = np.where((self.z >= 1.420) & (self.z <= 1.5125))[0]
        j = ix[np.argmin(r.sum(axis=1)[ix])]
        if j in (ix[0], ix[-1]):
            return float(self.z[j] * 1000)
        return (
            1000
            * minimize_scalar(
                spl, bounds=(self.z[j - 1], self.z[j + 1]), method="bounded", options={"xatol": 2e-13}
            ).x
        )


def recent(o=OBJ):
    vals = {}
    for f in CACHE.glob("global-*-fei-output-full-axis.npz"):
        path = WORK / (f.name.split("-fei-output")[0] + ".npz")
        b = load_bias(path)
        if abs(b["objective_lens/physical-pack-4"] - o) < 1e-8:
            vals[b["probe_tl12_lens"]] = field(path)[1]
    return vals


def roots(grid, fn):
    vals = [fn(x) for x in grid]
    return [
        brentq(fn, x, y, xtol=1e-8)
        for i, (x, y) in enumerate(zip(grid, grid[1:]))
        if vals[i] * vals[i + 1] < 0
    ]


def describe(m, t, o, fn, pivot):
    b = fn(t, o)
    z = pivot.z
    return m.describe(t, o, CubicSpline(z, b)(m.z))


if __name__ == "__main__":
    from scipy.optimize import root_scalar

    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    bank = recent()
    z, base = field(BASE)
    piv = PivotEstimate(z)
    m = Screen()
    knots = sorted(bank)
    itp = PchipInterpolator(knots, [bank[t] for t in knots], axis=0)
    with np.load(WORK / f"{BASE.stem}-objective-bore-0.5mm.npz") as f:
        od = CubicSpline(f["z"], f["coef"][:, 0, 2])(z)

    # Interpolate only within tested TL currents. The objective tangent is a commissioning predictor.
    def get(t, o):
        return itp(t) + (o - OBJ) * od

    def prop(t, o):
        b = get(t, o)
        return m.optics(t, o, CubicSpline(z, b)(m.z))

    def derivative(t):
        return np.interp(piv(get(t, OBJ)) * 0.001, m.z, prop(t, OBJ)[-1])

    cand = roots(np.linspace(knots[0], knots[-1], 150), derivative)
    survey = Survey() if a.verify else None

    def full_cov(t):
        b = get(t, OBJ)
        return survey.run(b, piv(b))["target_slope_mrad"]

    out = []
    for t in cand:
        if a.verify:
            nearby = np.linspace(max(knots[0], t - 28), min(knots[-1], t + 28), 7)
            found = roots(nearby, full_cov)
            if not found:
                continue
            t = min(found, key=lambda x: abs(x - t))
        b = get(t, OBJ)
        q = piv(b)
        stage = survey.run(b, q, True) if a.verify else describe(m, t, OBJ, get, piv)
        waists = stage.get("waists_mm", stage.get("roots"))
        if not any(abs(x - q) < 1.0 for x in waists):
            continue

        def shoot(o):
            return prop(t, o)[-1][m.i]

        focuses = []
        for o in roots(np.linspace(-35.0, 36.0, 182), shoot):
            case = describe(m, t, o, get, piv)
            if not any(abs(x - TOP * 1000) < 0.15 for x in case["roots"]):
                continue
            q2 = piv(get(t, o))
            if a.verify:
                case.update(survey.run(get(t, o), q2, True))
            case["estimated_pivot_mm"] = q2
            focuses.append(case)
        row = dict(tl=t, baseline_obj=OBJ, pivot=q, stage=stage, focus=focuses)
        out.append(row)
        print("Screen", row, flush=True)
    write(HERE / ("commission-survey.json" if a.verify else "round-survey.json"), out)
