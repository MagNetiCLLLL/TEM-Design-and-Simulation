"""Retune the full fixed gun-exit distribution at a stationary, physical specimen plane."""

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from column import CACHE, Mesh, SURFACE, metrics
from particles import Tracker


def waist_proposal(phase, weights):
    x, t = phase[:, :2], phase[:, 2:4]
    x = x - np.average(x, axis=0, weights=weights)
    t = t - np.average(t, axis=0, weights=weights)
    rms = -np.sum(weights[:, None] * x * t) / np.sum(weights[:, None] * t * t)
    speed = np.sqrt(np.sum(weights[:, None] * t * t) / weights.sum())
    spread = np.sqrt(np.sum(weights[:, None] * (x + rms * t) ** 2) / weights.sum()) / speed
    equal = np.all(weights == weights[0])
    index = int(np.ceil(0.95 * len(x))) - 1

    def span(z):
        r = x + z * t
        sq = r[:, 0] ** 2 + r[:, 1] ** 2
        if equal:
            return float(np.sqrt(np.partition(sq, index)[index]))
        from temsim.physics.beam_statistics import _weighted_quantile

        return float(np.sqrt(_weighted_quantile(sq, weights, 0.95)))

    delta = max(5e-9, 2 * spread)
    grid = np.linspace(rms - delta, rms + delta, 25)
    minimum = np.argmin([span(z) for z in grid])
    left, right = grid[max(0, minimum - 1)], grid[min(len(grid) - 1, minimum + 1)]
    sol = minimize_scalar(span, bounds=[left, right], method="bounded", options={"xatol": 3e-13})
    return float(sol.x), float(rms)


def fit(mesh, tracker, case, ni, level, pad, start, knob, count, max_iter, verbose=True):
    base = mesh.base(case, ni, level, pad)
    q = np.array(start, dtype=float)
    weights = tracker.weights[:count]
    history = []

    def run(q):
        data, loss, peak, margin = tracker.run(mesh.field(base, q), count)
        live = loss == -1
        if live.sum() < 100:
            raise ValueError("Fewer than 100 transmitted pilot packets")
        rows = data[-1, live]
        w = weights[live]
        result = metrics(rows, w, tracker.current)
        d95, rms = waist_proposal(rows, w)
        result.update(
            waist_proposal_nm=1e9 * d95,
            rms_proposal_nm=1e9 * rms,
            alive=int(live.sum()),
            population=len(live),
            peak_mm=float(peak[live].max() * 1000),
            margin_mm=float(margin[live].min() * 1000),
        )
        return result

    def errors(res):
        return np.array([(res["alpha95_mrad"] - 30.0) / 30.0, res["waist_proposal_nm"] / 1e7])

    best = None
    for iteration in range(max_iter):
        row = run(q)
        error = errors(row)
        history.append(dict(controls=q.tolist(), **row))
        if verbose:
            print(
                case,
                iteration,
                "alpha,waist_nm,D95_nm,current_uA",
                *[
                    round(row[k], 5)
                    for k in ["alpha95_mrad", "waist_proposal_nm", "D95_nm", "transmitted_uA"]
                ],
                flush=True,
            )
        if best is None or np.linalg.norm(error) < best[0]:
            best = (np.linalg.norm(error), q.copy(), row)
        if abs(row["alpha95_mrad"] - 30) < 0.09 and abs(row["waist_proposal_nm"]) < 0.07:
            break
        step1 = max(1e-6, min(0.001, max(abs(error)) * 0.005))
        step2 = max(2e-7, min(0.0003, max(abs(error)) * 0.002))
        jac = []
        for col, step in ((knob, step1), (2, step2)):
            trial = q.copy()
            trial[col] += step
            jac.append((errors(run(trial)) - error) / step)
        delta = np.linalg.lstsq(np.array(jac).T, -error, rcond=1e-10)[0]
        cap = min(1.0, 0.15 / max(1e-14, abs(delta[0])), 0.08 / max(1e-14, abs(delta[1])))
        for scale in [1.0, 0.5, 0.2, 0.07]:
            trial = q.copy()
            trial[[knob, 2]] += delta * (cap * scale)
            if np.any(trial <= 0.025) or np.any(trial > 3.8):
                continue
            new = run(trial)
            if np.linalg.norm(errors(new)) < np.linalg.norm(error):
                q = trial
                break
        else:
            # The D95 quantile can change identity at the waist. Adjust focus by a bracketing search later.
            break
    return best[1], best[2], history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="photo")
    p.add_argument("--ni", type=float, default=3000.0)
    p.add_argument("--level", default="fine")
    p.add_argument("--padding", type=float, default=8)
    p.add_argument("--step", type=float, default=0.2)
    p.add_argument("--count", type=int, default=32768)
    p.add_argument("--knob", type=int, default=0, choices=[0, 1, 3, 4, 5, 6])
    p.add_argument("--controls", type=float, nargs="+")
    p.add_argument("--seed-case", default="photo")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument(
        "--reference", help="Continue the reference paraxial map onto this case before physical refitting."
    )
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--hex-scale", type=float, default=1.0)
    p.add_argument("--tag", default="trial")
    args = p.parse_args()
    assert args.count >= 1000
    m = Mesh(args.step, [SURFACE])
    m.hex *= args.hex_scale
    t = Tracker(m)
    if args.reference:
        from branches import inputs, one
        from scipy.optimize import least_squares

        ref = json.loads((CACHE / args.reference).read_text())
        init = np.array(ref["controls"])
        old = inputs(m, ref["case"], ref["ni"], ref["level"], ref["padding"])
        new = inputs(m, args.case, args.ni, args.level, args.padding)
        target = np.array(one(init, *old)[:2])
        for fraction in np.linspace(0, 1, 21)[1:]:
            fields = (1 - fraction) * old[0] + fraction * new[0]

            def diff(pair):
                q = init.copy()
                q[[args.knob, 2]] = pair
                return (np.array(one(q, fields, *old[1:])[:2]) - target) / [1000.0, 30.0]

            sol = least_squares(
                diff, init[[args.knob, 2]], bounds=(0.02, 5.0), gtol=1e-12, xtol=1e-12, ftol=1e-12
            )
            init[[args.knob, 2]] = sol.x
        print("Continued optical proposal", init.tolist(), diff(init[[args.knob, 2]]), flush=True)
    else:
        init = (
            args.controls
            or json.loads((CACHE / f"branches-{args.seed_case}-{args.ni:g}.json").read_text())[
                args.seed_index
            ]["controls"]
        )
    result = fit(
        m, t, args.case, args.ni, args.level, args.padding, init, args.knob, args.count, args.iterations
    )
    (CACHE / f"fit-{args.case}-{args.tag}.json").write_text(
        json.dumps(
            dict(
                case=args.case,
                ni=args.ni,
                level=args.level,
                padding=args.padding,
                step_mm=args.step,
                knob=args.knob,
                count=args.count,
                hex_scale=args.hex_scale,
                controls=result[0].tolist(),
                metrics=result[1],
                history=result[2],
            ),
            indent=2,
        )
    )
