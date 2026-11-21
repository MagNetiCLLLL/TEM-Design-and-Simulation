"""Explore common relay settings with zero-current transfer pencils and explicit branch constraints."""

import argparse
import json

import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares, root
from scipy.stats import qmc

from column import CACHE, FACTOR, Mesh, SURFACE, POINT, read_gun


@njit(cache=True)
def one(q, fields, h, reg, radii, radius, slope):
    y, v = 1.0, slope
    ex = 0.0
    counts = np.zeros(4, np.int64)
    for j, step in enumerate(h):
        b = fields[j, 0]
        for i in range(len(q)):
            b += q[i] * fields[j, i + 1]
        k = FACTOR * b
        c, s = np.cos(k * step), np.sin(k * step)
        yn = c * y + (s / k if abs(k) > 1e-14 else step) * v
        v = -k * s * y + c * v
        if y * yn < 0.0:
            counts[reg[j]] += 1
        ex = max(ex, radius * max(abs(y), abs(yn)) / radii[j])
        y = yn
    return y * radius * 1e6, v * radius * 1e3, ex, float(counts[1]), float(counts[2]), float(counts[3])


@njit(parallel=True, cache=True)
def batch(controls, fields, h, reg, radii, radius, slope):
    result = np.empty((len(controls), 6))
    for i in prange(len(controls)):
        vals = one(controls[i], fields, h, reg, radii, radius, slope)
        for j in range(6):
            result[i, j] = float(vals[j])
    return result


def inputs(mesh, case, ni, level, padding):
    pos, _, _, _, _ = read_gun(POINT)
    x, t = pos[:, 0] + 1j * pos[:, 1], pos[:, 2] + 1j * pos[:, 3]
    slope = (np.vdot(x, t) / np.vdot(x, x)).real
    r95 = np.quantile(abs(x), 0.95)
    base = mesh.base(case, ni, level, padding)
    basis = np.concatenate((mesh.group[1:4], mesh.transfers))
    f = np.ascontiguousarray(
        np.column_stack([(base - mesh.transfers.sum(axis=0))[1::2, 0]] + [v[1::2, 0] for v in basis])
    )
    mid = mesh.nodes[1::2]
    c2, c3, adl, mini = mesh.intervals
    reg = np.where(
        (mid > c2) & (mid < c3),
        1,
        np.where((mid >= c3) & (mid < adl), 2, np.where((mid >= mini) & (mid < SURFACE - 0.00025), 3, 0)),
    )
    return f, np.diff(mesh.z), reg, mesh.radius, r95, slope


def main(args):
    mesh = Mesh(args.step, [SURFACE])
    parameters = inputs(mesh, args.case, args.ni, args.level, 8)
    # Each multiplier preserves its lens's original winding ratios. All cases share the selected C1 and relay.
    limits = np.array(
        [
            [0.60, 1.55],
            [0.15, 4.0],
            [0.08, 1.8],
            [0.15, 3.6],
            [0.15, 3.6],
            [0.15, 3.6],
            [0.15, 3.6],
            [0.35, 2.00],
        ]
    )
    q = qmc.scale(qmc.Sobol(d=len(limits), scramble=True, seed=7723).random_base2(args.power), *limits.T)
    values = batch(q, *parameters)
    topology = (values[:, 3] == 1) & (values[:, 4] == 0) & (values[:, 5] == 0)
    score = (
        abs(values[:, 0]) / 120 + abs(abs(values[:, 1]) - 30) / 8 + 12 * np.maximum(0, values[:, 2] - 0.97)
    )
    order = np.argsort(np.where(topology, score, score + 100))
    print(
        "Reach among direct pencil branches",
        topology.sum(),
        np.max(abs(values[topology & (values[:, 2] <= 1), 1]), initial=0),
        flush=True,
    )
    candidates = []
    if args.global_fit:
        # Underfilled branches may require simultaneous relay and condenser changes.
        bounds = np.array(
            [
                [0.25, 2.5],
                [0.05, 8.0],
                [0.05, 2.1],
                [0.05, 6.0],
                [0.08, 5.0],
                [0.08, 5.0],
                [0.08, 5.0],
                [0.08, 4.0],
            ]
        )
        for j in order[:90]:
            seed = q[j].copy()
            direction = np.sign(values[j, 1]) or 1.0

            def residuals(q):
                r, a, ratio, c2, c3, mini = one(q, *parameters)
                return np.r_[r / 400.0, a / 30.0 - direction, max(0.0, ratio - 0.96), 0.008 * (q - seed)]

            sol = least_squares(
                residuals, seed, bounds=bounds.T, xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=140
            )
            final = sol.x
            r, a, ratio, c2, c3, mini = one(final, *parameters)
            if [c2, c3, mini] == [1, 0, 0] and ratio <= 1.05:
                candidates.append(
                    dict(
                        controls=final.tolist(),
                        alpha=a,
                        ratio=ratio,
                        mismatch_um=r,
                        distance=float(np.linalg.norm(final - 1)),
                    )
                )
    for j in order[:350]:
        seed = q[j].copy()
        direction = 1 if values[j, 1] > 0 else -1
        # Preserve each trial's common relay where possible. Fallback to the last transfer lens.
        for knob in (1, 0, 6, 5, 4):

            def residual(pair):
                proposal = seed.copy()
                proposal[[knob, 2]] = pair
                r, a, *_ = one(proposal, *parameters)
                return r / 1000.0, a / 30 - direction

            sol = root(residual, seed[[knob, 2]], tol=1e-10)
            if max(abs(np.asarray(residual(sol.x)))) > 1e-5:
                continue
            final = seed.copy()
            final[[knob, 2]] = sol.x
            if np.any(final <= 0.03) or np.any(final > 3.6):
                continue
            r, a, ratio, c2, c3, mini = one(final, *parameters)
            if [c2, c3, mini] != [1, 0, 0] or ratio > 1.07:
                continue
            candidates.append(
                dict(controls=final.tolist(), alpha=a, ratio=ratio, distance=float(np.linalg.norm(final - 1)))
            )
            break
    candidates.sort(
        key=lambda x: (
            abs(abs(x["alpha"]) - 30) > 0.3 or abs(x.get("mismatch_um", 0)) > 3.0,
            abs(abs(x["alpha"]) - 30) if abs(abs(x["alpha"]) - 30) > 0.3 else 0.0,
            x["ratio"] > 0.85,
            x["distance"],
            x["ratio"],
        )
    )
    (CACHE / f"branches-{args.case}-{args.ni:g}.json").write_text(json.dumps(candidates, indent=2))
    print("Candidates", len(candidates), candidates[:2], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="photo")
    p.add_argument("--ni", type=float, default=3000.0)
    p.add_argument("--step", type=float, default=0.75)
    p.add_argument("--level", default="fine")
    p.add_argument("--power", type=int, default=16)
    p.add_argument("--global-fit", action="store_true")
    main(p.parse_args())
