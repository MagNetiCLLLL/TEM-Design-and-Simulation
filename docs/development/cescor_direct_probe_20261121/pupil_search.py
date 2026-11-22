"""Trace transfer pencils across existing windings to propose aperture-limited operating points."""

import argparse
import json

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.stats import qmc
from scipy.optimize import least_squares

from column_current import FACTOR, OBJECTIVE_NI, OUT, POINT, SURFACE, baseline, fields, pencil_fields
from column import read_gun

MATCH = 0.820


@njit(cache=True)
def pencil(q, f, h, rad, region, y, v):
    clearance_ratio = 0.0
    counts = np.zeros(4, np.int64)
    for j in range(len(h)):
        bz = f[j, 0]
        for i in range(len(q)):
            bz += q[i] * f[j, i + 1]
        k = FACTOR * bz
        s, c = np.sin(k * h[j]), np.cos(k * h[j])
        yn = c * y + (s / k if abs(k) > 1e-15 else h[j]) * v
        v = -k * s * y + c * v
        counts[region[j]] += y * yn < 0
        clearance_ratio = max(clearance_ratio, max(abs(y), abs(yn)) / rad[j])
        y = yn
    return y, v, clearance_ratio, float(counts[1]), float(counts[2]), float(counts[3])


@njit(cache=True, parallel=True)
def batch(qs, f, h, rad, reg, y, v):
    out = np.empty((len(qs), 6))
    for i in prange(len(qs)):
        a = pencil(qs[i], f, h, rad, reg, y, v)
        for j in range(6):
            out[i, j] = a[j]
    return out


def pieces(step, ni=OBJECTIVE_NI):
    mesh, table, bg, cs = fields(step, [MATCH, SURFACE], ni)
    f = pencil_fields(mesh, table, bg, cs)
    z = mesh.nodes[1::2]
    c2, c3, adl, mini = mesh.intervals
    reg = np.select(
        [(z > c2) & (z < c3), (z >= c3) & (z < adl), (z >= mini) & (z < SURFACE - 0.00025)],
        [1, 2, 3],
        default=0,
    )
    return f, np.diff(mesh.z), mesh.radius, reg, z


def compress(f, reference, directions, mask):
    return np.ascontiguousarray(
        np.column_stack((f[mask, 0] + f[mask, 1:] @ reference, f[mask, 1:] @ directions))
    )


def pupil():
    p, *_ = read_gun(POINT)
    x, t = p[:, 0] + 1j * p[:, 1], p[:, 2] + 1j * p[:, 3]
    rr = np.quantile(abs(x), 0.95)
    return rr, rr * np.vdot(x, t).real / np.vdot(x, x).real


def upstream(power, seed):
    f, h, r, reg, z = pieces(0.6)
    us = z < MATCH
    original = baseline()
    ref = original.copy()
    ref[:8] = 0
    basis = np.eye(len(ref))[:, :8]
    ff = compress(f, ref, basis, us)
    lim = np.repeat([max(abs(original[:4])), max(abs(original[4:8]))], 4)
    q = (2 * qmc.Sobol(8, scramble=True, seed=seed).random_base2(power) - 1) * lim
    parameters = (ff, h[us], r[us], reg[us], *pupil())
    v = batch(q, *parameters)
    select = v[:, 3] <= 1
    print(
        "Upstream ratio range on branches with at most one crossing",
        np.quantile(v[select, 2], [0, 0.5, 0.95, 1]),
        flush=True,
    )
    choose = np.flatnonzero(select)
    high = choose[np.argsort(-v[choose, 2])[:96]]
    near = choose[np.argsort(abs(np.log(v[choose, 2] / 650)))[:96]]
    # Retain a range of transmitted currents and output curvatures for the relay.
    indices = [*high, *near]
    for t in np.geomspace(max(0.02, v[select, 2].min()), v[select, 2].max(), 40):
        indices.extend(choose[np.argsort(abs(np.log(v[choose, 2] / t)))[:12]])
    indices = np.unique(indices)
    out = OUT / "upstream.npz"
    np.savez(
        out,
        drives=q[indices],
        results=v[indices],
        power=power,
        seed=seed,
        limits_A=lim,
        best_ratio=v[select, 2].max(),
    )
    print("Highest ratios", v[high[:4]], flush=True)


def downstream(power, seed, ni):
    f, h, r, reg, z = pieces(0.7, ni)
    ds = z >= MATCH
    upstream_file = np.load(OUT / "upstream.npz")
    qq, vv = upstream_file["drives"], upstream_file["results"]
    u_high = np.argsort(-vv[:, 2])
    # Invert a convergent marginal ray from the physical surface. The fitted
    # marginal ray must pass every mechanical bore in both directions.
    dirs = np.zeros((21, 7))
    for j, (a, b) in enumerate([(8, 10), (10, 12), (12, 15), (15, 18), (18, 19), (19, 20), (20, 21)]):
        dirs[a:b, j] = 1
    scales = np.array([160, 160, 600, 1400, 1200, 1200, 1200])
    dirs *= scales
    # Representative upstream setting for the first broad inverse screen.
    ref = np.zeros(21)
    ref[:8] = qq[u_high[0]]
    ff = compress(f, ref, dirs, ds)[::-1].copy()
    other = (-h[ds][::-1].copy(), r[ds][::-1].copy(), reg[ds][::-1].copy(), 0.0, -0.030)
    d = 2 * qmc.Sobol(7, scramble=True, seed=seed).random_base2(power) - 1
    b = batch(d, ff, *other)
    accept = (b[:, 4] == 0) & (b[:, 5] == 0) & (b[:, 3] <= 1) & (b[:, 2] < 2.0)
    print("Inverse pencils with correct downstream topology near the bore", accept.sum(), flush=True)
    ind = np.flatnonzero(accept)
    if not len(ind):
        return
    samples = []
    # The aperture-clipped boundary is approximately 97.5% filled at the
    # 95th percentile for a strongly overfilled, locally uniform circular pupil.
    for n in np.unique(np.r_[u_high[:48], np.linspace(0, len(qq) - 1, 160, dtype=int)]):
        uv = vv[n]
        amp = min(1.0, 0.975 / max(uv[2], 1e-12))
        target = uv[:2] * amp
        sign = np.where(b[ind, 0] * target[0] < 0, -1.0, 1.0)
        score = np.hypot((sign * b[ind, 0] - target[0]) / 0.00012, (sign * b[ind, 1] - target[1]) / 0.0025)
        score += 10 * np.maximum(b[ind, 2] - 1.0, 0.0) + 50 * abs(b[ind, 3] + uv[3] - 1)
        for j in np.argsort(score)[:3]:
            samples.append((score[j] + 0.25 * abs(np.log(max(uv[2], 0.02) / 650.0)), n, ind[j], sign[j]))
    samples.sort()
    results = []
    for _, n, i, sign in samples[:200]:
        uv = vv[n]
        ref[:8] = qq[n]
        ff = compress(f, ref, dirs, ds)[::-1].copy()
        target = uv[:2] * min(1.0, 0.975 / max(uv[2], 1e-12))

        def residual(p):
            y, v, ratio, *_ = pencil(p, ff, *other)
            return np.r_[
                (sign * y - target[0]) / 0.0001,
                (sign * v - target[1]) / 0.001,
                3 * max(0.0, ratio - 0.99),
                0.002 * (p - d[i]),
            ]

        sol = least_squares(residual, d[i], bounds=(-1.8, 1.8), max_nfev=65)
        a = np.array(pencil(sol.x, ff, *other))
        if list(a[3:] + uv[3:]) != [1.0, 0.0, 0.0] or a[2] > 1.07 or np.linalg.norm(sol.fun[:2]) > 0.4:
            continue
        currents = ref + dirs @ sol.x
        fwd = pencil(currents, f, h, r, reg, *pupil())
        signed_radius, angle, ratio, *cs = fwd
        shrink = min(1.0, 0.975 / max(ratio, 1e-12))
        results.append(
            dict(
                currents_A=currents.tolist(),
                upstream_ratio=float(uv[2]),
                radius_um=signed_radius * shrink * 1e6,
                alpha_mrad=abs(angle) * shrink * 1e3,
                max_ratio=ratio,
                crossings=cs,
                objective_At=ni,
            )
        )
        if len(results) >= 24:
            break
    path = OUT / f"relay-{abs(ni)}.json"
    path.write_text(json.dumps(results, indent=2))
    print("Matched marginal-ray candidates", len(results), flush=True)
    if results:
        print(
            [
                (
                    round(a["alpha_mrad"], 3),
                    round(a["max_ratio"], 2),
                    round(a["radius_um"], 2),
                    a["crossings"],
                )
                for a in results[:4]
            ],
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["upstream", "downstream"])
    p.add_argument("--power", type=int, default=16)
    p.add_argument("--seed", type=int, default=1673)
    p.add_argument("--objective-ni", type=int, default=OBJECTIVE_NI)
    a = p.parse_args()
    set_num_threads(12)
    OUT.mkdir(exist_ok=True, parents=True)
    if a.stage == "upstream":
        upstream(a.power, a.seed)
    else:
        downstream(a.power, a.seed, a.objective_ni)
