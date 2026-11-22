"""Bracket the angular and longitudinal errors independently with transported electrons."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from finite_probe import OUT, SURFACE, bundle, source_rate, survey


def run(a):
    q0 = np.array(json.loads(a.seed.read_text())[a.index]["currents_A"])
    rate = source_rate()
    setups = {}
    found = []

    def sample(q):
        for s in (a.step, a.step / 2, a.step / 4):
            if s not in setups:
                setups[s] = bundle(s, [SURFACE], a.ni, a.count, rate)
            try:
                row = survey(setups[s], q)
                row.update(currents_A=q.tolist(), objective_At=a.ni, max_step_mm=s)
                return row
            except ValueError as exc:
                if "Unresolved boundary" not in str(exc):
                    raise
        raise ValueError("Unresolved grazing trajectory after refinement")

    xknob = np.zeros(len(q0))
    xknob[a.slice[0] : a.slice[1]] = 1.0
    focus = np.zeros(len(q0))
    focus[18:21] = 1.0
    metric = "drift_" + a.metric + "_m"
    if a.local:
        q = q0.copy()
        row = sample(q)

        def mismatch(v):
            return np.array([(v["alpha95_mrad"] - 30), v[metric] * 1000])

        for k in range(9):
            err = mismatch(row)
            print(
                k, row["alpha95_mrad"], row[metric] * 1e6, row["D95_nm"], row["transmitted_uA"], flush=True
            )
            (OUT / (a.name + ".json")).write_text(json.dumps([row], indent=2))
            if abs(err[0]) < 0.002 and abs(row[metric]) < 2e-9:
                break
            steps = np.maximum(
                [0.0002, 0.00003], np.minimum([0.12, 0.035], np.linalg.norm(err) * np.array([0.5, 0.12]))
            )
            jac = []
            for knob, dx in zip((xknob, focus), steps):
                jac.append((mismatch(sample(q + dx * knob)) - err) / dx)
            delta = np.linalg.lstsq(np.array(jac).T, -err, rcond=1e-9)[0]
            delta *= min(1.0, 2.5 / max(abs(delta), default=1.0))
            for scale in (1.0, 0.5, 0.2, 0.06):
                attempt = q + scale * (delta[0] * xknob + delta[1] * focus)
                trial = sample(attempt)
                if np.linalg.norm(mismatch(trial)) < np.linalg.norm(err):
                    q, row = attempt, trial
                    break
            else:
                break
        (OUT / (a.name + ".json")).write_text(json.dumps([row], indent=2))
        return

    def focus_at(gain):
        trials = {}

        def row_at(x):
            x = float(x)
            if x not in trials:
                trials[x] = sample(q0 + xknob * gain + focus * x)
            return trials[x]

        start = row_at(0.0)
        delta = max(0.006, min(4.0, abs(start[metric]) / 0.00017))
        for span in (delta, 2 * delta, 4 * delta, 8 * delta, 16 * delta, 32 * delta):
            for x in (-span, span):
                row = row_at(x)
                if np.signbit(row[metric]) != np.signbit(start[metric]):
                    lo, hi = sorted((0.0, x))
                    root = brentq(lambda k: row_at(k)[metric], lo, hi, xtol=0.000006)
                    row_at(root)
                    result = min(trials.values(), key=lambda v: abs(v[metric]))
                    result.update(gain_A=float(gain))
                    found.append(result)
                    (OUT / (a.name + ".json")).write_text(json.dumps(sorted(found, key=rank), indent=2))
                    print(
                        round(gain, 4),
                        result["alpha95_mrad"],
                        result[metric] * 1e6,
                        result["D95_nm"],
                        result["transmitted_uA"],
                        flush=True,
                    )
                    return result["alpha95_mrad"] - 30.0
        raise ValueError("No bounded Mini Condenser focus root")

    tested = {}

    def angle(x):
        x = float(x)
        if x not in tested:
            tested[x] = focus_at(x)
        return tested[x]

    def rank(row):
        return abs(row["alpha95_mrad"] - 30) / 0.3 + abs(row[metric]) / 0.00003

    angle(0)
    for x in a.gains:
        try:
            angle(x)
        except ValueError as exc:
            if not any(
                reason in str(exc)
                for reason in ("No bounded Mini Condenser focus root", "Unresolved grazing trajectory")
            ):
                raise
            print("Excluded current change", x, str(exc), flush=True)
            continue
        ordered = sorted(tested)
        pairs = [(l, r) for l, r in zip(ordered[:-1], ordered[1:]) if tested[l] * tested[r] <= 0]
        if pairs:
            l, r = pairs[0]
            root = brentq(angle, l, r, xtol=0.0005)
            angle(root)
            break
    found.sort(key=rank)
    (OUT / (a.name + ".json")).write_text(json.dumps(found, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("seed", type=Path)
    p.add_argument("--ni", type=int, default=-24000)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--count", type=int, default=4096)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--slice", type=int, nargs=2, default=[6, 8])
    p.add_argument("--metric", choices=["D95", "RMS"], default="D95")
    p.add_argument("--gains", nargs="+", type=float, default=[-8, -20, -45, 12, 24, -70, 55])
    p.add_argument("--local", action="store_true")
    run(p.parse_args())
