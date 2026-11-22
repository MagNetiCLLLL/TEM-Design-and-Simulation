"""Search the existing column for overfilled apertures on a direct-focus branch."""

import argparse
import json

import numpy as np
from numba import set_num_threads
from scipy.stats import qmc
from scipy.optimize import differential_evolution

from column_current import OUT, baseline
from pupil_search import batch, pieces, pupil


def search(ni, power, seed):
    f, h, r, g, _ = pieces(0.85, ni)
    original = baseline()
    # Sample closed subwindings over the historical peak magnitude for each
    # lens. Include both weaker settings and the multi-kiloampere limit in
    # the archived recipe, so a weakly intercepted sample is informative.
    limits = np.repeat(
        [
            max(abs(original[:4])),
            max(abs(original[4:8])),
            200.0,
            max(abs(original[12:15])),
            max(abs(original[15:18])),
            max(abs(original[18:21])),
        ],
        [4, 4, 4, 3, 3, 3],
    )
    q = 2 * qmc.Sobol(21, scramble=True, seed=seed).random_base2(power) - 1
    q *= limits
    # Bias half the population toward less excitation in the three relay groups.
    q[1::2, 12:] *= 0.18
    # The second search family holds the C1/C2 sleeve within a direct branch
    # found by the aperture screen and varies the complete downstream relay.
    up = np.load(OUT / "upstream.npz")
    q[2::4, :8] = up["drives"][np.arange(len(q[2::4])) % len(up["drives"])]
    v = batch(q, f, h, r, g, *pupil())
    ok = np.all(v[:, 3:] == [1, 0, 0], axis=1)
    angle = abs(v[:, 1]) * np.minimum(1.0, 0.975 / np.maximum(v[:, 2], 1e-15)) * 1000
    radius = abs(v[:, 0]) * np.minimum(1.0, 0.975 / np.maximum(v[:, 2], 1e-15)) * 1e6
    direct = np.flatnonzero(ok)
    if not len(direct):
        raise ValueError("No direct proposals on this bounded screen")
    highest = direct[np.argsort(-v[direct, 2])[:12]]
    angle_band = direct[
        np.argsort(abs(angle[direct] - 30) / 5 + radius[direct] / 500 - 0.1 * np.log(v[direct, 2]))[:20]
    ]
    near = direct[(angle[direct] > 25) & (angle[direct] < 35)]
    sel = np.unique(np.r_[highest, angle_band, near[np.argsort(-v[near, 2])[:16]]])
    results = [
        dict(
            currents_A=q[i].tolist(),
            objective_At=ni,
            ratio=v[i, 2],
            angle_mrad=angle[i],
            radius_um=radius[i],
            pencil_crossings=v[i, 3:].tolist(),
        )
        for i in sel
    ]
    (OUT / f"overfill-{abs(ni)}.json").write_text(json.dumps(results, indent=2))
    doc = dict(
        objective_At=ni,
        seed=seed,
        power=power,
        screened=len(v),
        direct=len(direct),
        max_ratio_direct=v[highest[0], 2],
        direct_25_to_35mrad_count=len(near),
        max_ratio_in_angle_band=np.max(v[near, 2], initial=0.0),
        max_ratio_any_branch=np.max(v[:, 2]),
        limits_A=limits.tolist(),
    )
    (OUT / f"overfill-summary-{abs(ni)}.json").write_text(json.dumps(doc, indent=2))
    print(doc, flush=True)


def evolve(ni, seed, iterations):
    """Seek high physical aperture filling while preserving the three branch counts."""
    f, h, r, g, _ = pieces(0.7, ni)
    args = (f, h, r, g, *pupil())
    bounds = np.array(json.loads((OUT / f"overfill-summary-{abs(ni)}.json").read_text())["limits_A"])
    previous = json.loads((OUT / f"overfill-{abs(ni)}.json").read_text())
    matched = json.loads((OUT / f"pencils-{abs(ni)}.json").read_text())
    rng = np.random.default_rng(seed)
    base = np.array([v["currents_A"] for v in previous + matched]) / bounds
    init = base[rng.integers(len(base), size=256)] + rng.normal(0, 0.035, (256, len(bounds)))
    init[: len(base)] = base
    retained = []

    def cost(raw):
        qs = np.atleast_2d(raw.T) * bounds
        v = batch(qs, *args)
        ratio = np.maximum(v[:, 2], 1e-15)
        fill = np.minimum(1.0, 0.975 / ratio)
        alpha = abs(v[:, 1]) * 1000 * fill
        radius = abs(v[:, 0]) * 1e6 * fill
        shifts = abs(v[:, 3:] - [1, 0, 0]).sum(axis=1)
        # Grow aperture filling along a direct branch; retain off-target
        # candidates too so the current-versus-angle tradeoff is measurable.
        score = 4 * shifts - np.log(ratio) + 0.14 * abs(alpha - 30) / 10 + 0.07 * np.log1p(radius / 50)
        direct = np.flatnonzero(shifts == 0)
        if len(direct):
            for i in {direct[np.argmax(ratio[direct])], direct[np.argmin(score[direct])]}:
                retained.append(
                    dict(
                        currents_A=qs[i].tolist(),
                        ratio=ratio[i],
                        angle_mrad=alpha[i],
                        radius_um=radius[i],
                        pencil_crossings=v[i, 3:].tolist(),
                        objective_At=ni,
                        score=score[i],
                    )
                )
        return score

    differential_evolution(
        cost,
        [(-1.0, 1.0)] * len(bounds),
        init=np.clip(init, -1, 1),
        seed=seed,
        maxiter=iterations,
        updating="deferred",
        vectorized=True,
        polish=False,
        tol=1e-10,
        popsize=12,
    )
    retained.sort(key=lambda row: -row["ratio"])
    unique = []
    for row in retained:
        if unique and abs(unique[-1]["ratio"] - row["ratio"]) < 1e-8:
            continue
        unique.append(row)
    (OUT / f"overfill-evolved-{abs(ni)}.json").write_text(json.dumps(unique, indent=2))
    print(
        "Largest direct aperture ratio, angle and radius",
        [(a["ratio"], a["angle_mrad"], a["radius_um"]) for a in unique[:3]],
        flush=True,
    )


def relay_gain(ni, seed, iterations):
    """Search relay gain by section so high fill can be explored before final focus."""
    f, h, r, g, z = pieces(0.55, ni)
    y, v = pupil()
    limits = np.array(json.loads((OUT / f"overfill-summary-{abs(ni)}.json").read_text())["limits_A"])
    parts = [(0, 8, 0.82), (8, 15, 1.335), (15, 18, 1.499)]
    starts = [np.array(x["currents_A"]) for x in json.loads((OUT / f"pencils-{abs(ni)}.json").read_text())]
    retained = []
    for order, q in enumerate(starts):
        q = q.copy()
        for a, b, end in parts:
            mask = z < end
            keep = q.copy()
            keep[a:b] = 0
            ff = np.ascontiguousarray(
                np.column_stack((f[mask, 0] + f[mask, 1:] @ keep, f[mask, 1 + a : 1 + b]))
            )
            arg = (ff, h[mask], r[mask], g[mask], y, v)

            def loss(pop):
                vals = batch(np.atleast_2d(pop.T), *arg)
                penalty = (
                    6 * np.maximum(0, vals[:, 3] - 1)
                    if end < 0.915
                    else 6 * (abs(vals[:, 3] - 1) + vals[:, 4])
                )
                return penalty - np.log(1e-15 + np.hypot(vals[:, 0], 0.080 * vals[:, 1]))

            result = differential_evolution(
                loss,
                list(zip(-limits[a:b], limits[a:b])),
                x0=q[a:b],
                seed=seed + order + 17 * a,
                vectorized=True,
                updating="deferred",
                polish=False,
                maxiter=iterations,
                tol=1e-8,
                popsize=24,
            )
            q[a:b] = result.x
            val = batch(q[a:b][None, :], *arg)[0]
            retained.append(
                dict(
                    currents_A=q.tolist(),
                    stage_end_m=end,
                    objective_At=ni,
                    pupil_position_m=val[0],
                    pencil_slope=val[1],
                    max_ratio=val[2],
                    crossings=val[3:].tolist(),
                )
            )
            print(order, end, val, flush=True)
    (OUT / f"staged-{abs(ni)}.json").write_text(json.dumps(retained, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ni", type=int, default=-24000)
    p.add_argument("--seed", type=int, default=802)
    p.add_argument("--power", type=int, default=18)
    p.add_argument("--evolve", type=int, default=0)
    p.add_argument("--staged", type=int, default=0)
    a = p.parse_args()
    set_num_threads(13)
    if a.staged:
        relay_gain(a.ni, a.seed, a.staged)
    elif a.evolve:
        evolve(a.ni, a.seed, a.evolve)
    else:
        search(a.ni, a.power, a.seed)
