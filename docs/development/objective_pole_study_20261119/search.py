"""Fast, zero-current Larmor pencil proposals; never used to measure a physical probe."""

import argparse
import json

import numpy as np
from numba import njit
from scipy.optimize import root

from column import CACHE, FACTOR, Mesh, POINT, START, SURFACE, read_gun


@njit(cache=True)
def pencil(q, fields, steps, regions, radii, radius95, slope):
    y, v = 1.0, slope
    counts = np.zeros(4, np.int64)
    excursion = 0.0
    for j, h in enumerate(steps):
        k = FACTOR * (
            fields[j, 0]
            + q[0] * fields[j, 1]
            + q[1] * fields[j, 2]
            + q[2] * fields[j, 3]
            + (q[3] - 1.0) * fields[j, 4]
        )
        c, s = np.cos(k * h), np.sin(k * h)
        next_y = c * y + (s / k if abs(k) > 1e-12 else h) * v
        v = -k * s * y + c * v
        if y * next_y < 0:
            counts[regions[j]] += 1
        excursion = max(excursion, radius95 * max(abs(y), abs(next_y)) / radii[j])
        y = next_y
    return y, v, counts, excursion


def design(case, ni, level, padding, args):
    mesh = Mesh(args.step, [SURFACE])
    phase, _, _, _, _ = read_gun(POINT)
    x, t = phase[:, 0] + 1j * phase[:, 1], phase[:, 2] + 1j * phase[:, 3]
    slope = (np.vdot(x, t) / np.vdot(x, x)).real
    r95 = np.quantile(abs(x), 0.95)
    base = mesh.base(case, ni, level, padding)
    f = np.ascontiguousarray(
        np.column_stack([base[1::2, 0]] + [v[1::2, 0] for v in mesh.group[1:4]] + [mesh.relay[1::2, 0]])
    )
    h = np.diff(mesh.z)
    c2, c3, adl, mini = mesh.intervals
    mid = mesh.nodes[1::2]
    regions = np.where(
        (mid > c2) & (mid < c3), 1, np.where((mid >= c3) & (mid < adl), 2, np.where(mid >= mini, 3, 0))
    )

    def run(q):
        return pencil(q, f, h, regions, mesh.radius, r95, slope)

    target = np.tan(0.030) / r95
    options = {}
    for anchor in args.c2:
        for initial_c3 in args.relay_seeds if args.relay_search else args.c3:
            for initial_mini in args.mini:
                seed = [initial_c3, initial_mini]

                def controls(pair):
                    if args.relay_search:
                        return np.array([anchor, args.fixed_c3, pair[1], pair[0]])
                    return np.array([anchor, pair[0], pair[1], args.relay])

                first = run(controls(seed))
                for direction in (np.sign(first[1]) or 1.0, -(np.sign(first[1]) or 1.0)):

                    def residual(pair):
                        y, v, _, _ = run(controls(pair))
                        return [y / 600.0, v / target - direction]

                    result = root(residual, seed, tol=1e-10)
                    vals = controls(result.x)
                    if (
                        max(abs(np.asarray(residual(result.x)))) > 0.0001
                        or np.any(vals < 0.05)
                        or np.any(vals > 3.5)
                    ):
                        continue
                    y, v, count, excursion = run(vals)
                    # The final zero itself may fall in the last numerical cell.
                    trim = np.searchsorted(mesh.z, SURFACE - 0.00025)
                    _, _, interior, _ = pencil(
                        vals, f[:trim], h[:trim], regions[:trim], mesh.radius[:trim], r95, slope
                    )
                    row = dict(
                        controls=vals.tolist(),
                        endpoint_gain=v,
                        clearance_ratio_r95=excursion,
                        crosses=[int(t) for t in interior[1:]],
                        distance=float(np.linalg.norm(vals - 1)),
                    )
                    options[tuple(vals.round(5))] = row
    result = sorted(
        options.values(),
        key=lambda r: (
            r["crosses"] != [1, 0, 0],
            abs(r["crosses"][0] - 1) + r["crosses"][1] + r["crosses"][2],
            max(0.0, r["clearance_ratio_r95"] - 1),
            r["distance"],
        ),
    )
    target = CACHE / f"search-{case}-{ni:g}.json"
    target.write_text(json.dumps(result, indent=2))
    print(case, len(result), result[:4], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", default=["photo", "narrow", "wide", "photo_iron_tube"])
    p.add_argument("--ni", type=float, default=3000)
    p.add_argument("--level", default="fine")
    p.add_argument("--padding", type=float, default=8)
    p.add_argument("--step", type=float, default=0.4)
    p.add_argument("--relay-search", action="store_true")
    p.add_argument(
        "--relay-seeds",
        nargs="+",
        type=float,
        default=[0.15, 0.35, 0.55, 0.8, 1.0, 1.15, 1.4, 1.7, 2.0, 2.4, 2.9],
    )
    p.add_argument("--fixed-c3", type=float, default=1.0)
    p.add_argument("--relay", type=float, default=1.0)
    p.add_argument("--c2", nargs="+", type=float, default=[1.0, 0.94, 1.06, 0.86, 1.14])
    p.add_argument("--c3", nargs="+", type=float, default=[0.85, 0.96, 1.0, 1.08, 1.2])
    p.add_argument("--mini", nargs="+", type=float, default=[0.7, 0.9, 1.0, 1.15, 1.4])
    args = p.parse_args()
    for case in args.cases:
        design(case, args.ni, args.level, args.padding, args)
