"""Fit conjugate planes and shoot the aperture marginal ray backwards from the sample."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from numba import set_num_threads

from relay_working_point import GROUPS, batch, data, predict
from relay_transfer import RADIUS, REF, save


def controls(full=False):
    q = np.array(REF["currents_A"])
    directions = []
    for group in GROUPS:
        v = np.zeros_like(q)
        v[group] = q[group]
        directions.append(v)
    # Independently supplied, complete closed subwindings in the saved coil model.
    # Differential currents change lens shape while preserving the physical turns.
    pairs = [
        (4, 6, 500),
        (8, 11, 320),
        (12, 14, 650),
        (15, 17, 820),
        (15, 16, 650),
        (18, 20, 850),
        (18, 19, 700),
    ]
    if full:
        pairs.extend(
            [
                (0, 3, 700),
                (1, 3, 700),
                (2, 3, 700),
                (4, 7, 500),
                (5, 7, 500),
                (9, 11, 320),
                (10, 11, 320),
                (13, 14, 650),
            ]
        )
    for left, right, amps in pairs:
        v = np.zeros_like(q)
        v[left] = amps
        v[right] = -amps
        directions.append(v)
    return np.column_stack(directions)


def solve(args):
    seed_record = json.loads(args.settings.read_text())
    start = np.array(seed_record["currents_A"])
    directions = controls(args.full)
    _, wind, p = data(
        args.table, args.pupil * 0.001, args.step, np.column_stack((start, directions)), args.objective_ni
    )
    base, basis, h, rad, region, planes, kb, kj, source = p
    pars = (
        np.ascontiguousarray(base + basis[:, 0]),
        np.ascontiguousarray(basis[:, 1:]),
        h,
        rad,
        region,
        planes,
        kb + kj[:, 0],
        np.ascontiguousarray(kj[:, 1:]),
        source,
    )
    n = directions.shape[1]
    calibration = None
    if args.calibration:
        delta_fields = []
        delta_end = []
        delta_current = []
        for path in args.calibration:
            _, other, p2 = data(path, args.pupil * 0.001, args.step, objective_ni=args.objective_ni)
            delta_fields.append(p2[0] - base)
            delta_end.append(p2[-3] - kb)
            delta_current.append([other[k] - wind[k] for k in ("probe_tl21_lens", "probe_tl22_lens")])
        calibration = np.linalg.solve(np.asarray(delta_current), np.asarray(delta_fields)).T * np.array(
            [70.0, 35.0]
        )
        ends = np.linalg.solve(np.asarray(delta_current), np.asarray(delta_end)).T * np.array([70.0, 35.0])
        pars = (
            pars[0],
            np.ascontiguousarray(np.column_stack((pars[1], calibration))),
            *pars[2:7],
            np.ascontiguousarray(np.column_stack((pars[7], ends))),
            pars[8],
        )
        n += 2
    alpha_target = RADIUS / np.tan(0.03)

    def residual(v, sign):
        if args.signed:
            return np.array(
                [
                    args.relay_weight * v[0] / 0.04,
                    args.relay_weight * v[1] / 0.06,
                    v[12] / 0.00008,
                    (v[13] - sign * 0.030) / 0.015,
                    2 * max(0.0, 1.015 - v[4]),
                    3 * max(0.0, v[5] - 0.99),
                ]
            )
        return np.array(
            [
                v[0] / 0.018,
                v[1] / 0.025,
                (v[10] - sign * alpha_target) / 0.004,
                v[11] / 0.5,
                max(0.0, 1.04 - v[4]) * 2,
                3 * max(0.0, v[5] - 0.99),
            ]
        )

    def loss(x, sign, seed):
        return np.r_[residual(predict(x, *pars), sign), 2e-5 * (x - seed)]

    rng = qmc.Sobol(n, scramble=True, seed=args.seed)
    x = (rng.random_base2(args.power) - 0.5) * 2
    scales = [0.20, 0.24, 1.3, 1.2, 0.8, 0.8, 0.6, 0.9, 1.0, 0.8, 0.7, 0.8, 0.8]
    if args.full:
        scales += [0.45, 0.45, 0.45, 0.5, 0.5, 0.65, 0.65, 0.7]
    if calibration is not None:
        scales += [0.75, 0.75]
    x *= np.array(scales)
    set_num_threads(8)
    vs = batch(x, *pars)
    topology = np.abs(vs[:, 6:9] - [1, 0, 0]).sum(axis=1)
    col = 13 if args.signed else 10
    score = np.linalg.norm(np.array([residual(v, 1 if v[col] > 0 else -1) for v in vs]), axis=1)
    starts = [np.zeros(n)]
    starts.extend(x[i] for i in np.argsort(score + 2.5 * topology)[:28])
    starts.extend(x[i] for i in np.argsort(score)[:15])
    best = None
    found = 0
    for no, seed in enumerate(starts):
        initial = predict(seed, *pars)
        sign = np.sign(initial[col]) or 1
        for s in [sign, -sign]:
            fit = least_squares(
                loss,
                seed,
                args=(s, seed),
                bounds=(-3.0, 3.0),
                max_nfev=110,
                ftol=2e-10,
                xtol=3e-10,
                gtol=2e-10,
            )
            v = predict(fit.x, *pars)
            err = np.linalg.norm(residual(v, s))
            crossings = v[6:9].astype(int)
            rating = err + abs(crossings - [1, 0, 0]).sum() + max(0.0, abs(v[2] - 0.03) * 20)
            cur = start + directions @ fit.x[: directions.shape[1]]
            di = fit.x[-2:] * np.array([70.0, 35.0]) if calibration is not None else np.zeros(2)
            record = dict(
                score=float(rating),
                paraxial_residual=float(err),
                currents_A=cur.tolist(),
                objective_At=args.objective_ni,
                B_hp_mm=v[0] * 1000,
                B_pupil_mm=v[1] * 1000,
                edge_mrad=v[2] * 1000,
                focus_projection_um=v[3] * 1e6,
                foil_fill=v[4],
                vignetting=v[5],
                crossings=crossings.tolist(),
                edge_spot_um=v[9] * 1e6,
                foil_to_sample_B_mm=v[10] * 1000,
                tl21_candidate_A=wind["probe_tl21_lens"] + di[0],
                tl22_candidate_A=wind["probe_tl22_lens"] + di[1],
            )
            if best is None or rating < best:
                best = rating
                save(args.name, record)
                print(no, "best", rating, err, crossings, v[:6], flush=True)
            if err < 0.018 and np.array_equal(crossings, [1, 0, 0]):
                save(args.name + "-candidate-" + str(found), record)
                found += 1
                if found >= 3:
                    return


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("name")
    a.add_argument("--table", type=Path, required=True)
    a.add_argument("--settings", type=Path, required=True)
    a.add_argument("--step", type=float, default=0.35)
    a.add_argument("--seed", type=int, default=501)
    a.add_argument("--power", type=int, default=14)
    a.add_argument("--pupil", type=float, default=1536.570)
    a.add_argument("--objective-ni", type=float, default=REF["objective_At"])
    a.add_argument("--signed", action="store_true")
    a.add_argument("--relay-weight", type=float, default=1.0)
    a.add_argument("--calibration", type=Path, nargs=2)
    a.add_argument("--full", action="store_true")
    solve(a.parse_args())
