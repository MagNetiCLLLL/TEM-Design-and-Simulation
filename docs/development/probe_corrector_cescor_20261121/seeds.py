"""Locate direct paraxial branches; only a finite-current replay can accept a proposal."""

import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from numba import set_num_threads

from compare import ARCHIVE, WORK, Mesh, NI, SURFACE, inputs, shared_inputs
from branches import batch, one


def search(table, output, power, ni):
    set_num_threads(12)
    m = Mesh(0.8, [SURFACE], shared_inputs(ni))
    ref = inputs(m, "armco", NI, "gap-fine", 8)
    archived = np.array(json.loads((ARCHIVE / "after/focus.json").read_text())["controls"])
    print("Archived transfer pencil", one(archived, *ref), flush=True)
    m.group[0] -= m.transfers[1] + m.transfers[2]
    m.transfers[1:3] = 0.0
    pars = inputs(m, "armco", ni, "gap-fine", 8)
    with np.load(table) as data:
        j = len(data["xy_m"]) // 2
        pars[0][:, 0] += np.interp(m.nodes[1::2] - 0.45, data["z_local_m"], data["B_zyx_T"][:, j, j, 2])
    cols = [0, 1, 2, 3, 6, 7]
    lo = np.array([0.18, 0.06, 0.03, 0.08, 0.08, 0.18])
    hi = np.array([4.5, 32.0, 4.5, 12.0, 8.0, 3.0])
    samples = qmc.Sobol(6, scramble=True, seed=1261).random_base2(power)
    drives = np.ones((len(samples), 8))
    drives[:, cols] = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * samples)
    v = batch(drives, *pars)
    ok = (v[:, 3] == 1) & (v[:, 4] == 0) & (v[:, 5] == 0)
    print(
        "Maximum angle on direct aperture-admitted pencils",
        np.max(abs(v[ok & (v[:, 2] <= 1), 1])),
        flush=True,
    )
    print("Largest aperture-admitted angle on any branch", np.max(abs(v[v[:, 2] <= 1, 1])), flush=True)
    score = abs(v[:, 0]) / 450 + abs(abs(v[:, 1]) - 30) / 8 + 8 * np.maximum(v[:, 2] - 0.97, 0)
    rank = np.argsort(np.where(ok, score, score + 100))
    high = np.argsort(np.where(ok & (v[:, 2] < 1.15), -abs(v[:, 1]), 1e10))
    any_branch = np.argsort(score)
    high_any = np.argsort(np.where(v[:, 2] < 1.15, -abs(v[:, 1]), 1e10))
    indices = np.column_stack((rank[:100], high[:100], any_branch[:100], high_any[:100])).ravel()
    print("Direct pencils, best initial", int(ok.sum()), v[indices[0]], flush=True)
    choices = []
    fallback = []
    for i in indices[:300]:
        seed = drives[i]
        sign = np.sign(v[i, 1]) or 1.0

        def residual(w):
            q = seed.copy()
            q[cols] = w
            r, a, ratio, *_ = one(q, *pars)
            return np.r_[
                r / 350, (a - 30 * sign) / 27, max(0, ratio - 0.96) * 1.2, 0.007 * (w - seed[cols])
            ]

        fit = least_squares(residual, seed[cols], bounds=(lo, hi), max_nfev=120)
        q = seed.copy()
        q[cols] = fit.x
        val = one(q, *pars)
        if val[2] < 1.08:
            fallback.append(dict(controls=q.tolist(), pencil=list(val)))
        if (
            [int(k) for k in val[3:]] != [1, 0, 0]
            or val[2] > 1.035
            or abs(val[0]) > 30
            or abs(abs(val[1]) - 30) > 1.0
        ):
            continue
        if any(np.linalg.norm(q - np.array(old["controls"])) < 0.10 for old in choices):
            continue
        choices.append(dict(controls=q.tolist(), pencil=list(val)))
        if len(choices) == 14:
            break
    choices.sort(key=lambda entry: (entry["pencil"][2], abs(entry["pencil"][0])))
    if not choices:
        fallback.sort(
            key=lambda entry: abs(entry["pencil"][0]) / 350
            + abs(abs(entry["pencil"][1]) - 30) / 27
            + 0.05 * sum(abs(a - b) for a, b in zip(entry["pencil"][3:], [1, 0, 0]))
        )
        choices = fallback[:12]
    output.write_text(json.dumps(choices, indent=2))
    print("Candidate count", len(choices), choices[:2], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    from pathlib import Path

    p.add_argument("table", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--power", type=int, default=16)
    p.add_argument("--objective-ni", type=float, default=NI)
    args = p.parse_args()
    search(args.table, args.output, args.power, args.objective_ni)
