"""Adjust condenser, transfer, and Mini Condenser currents against actual surface measurements."""

import argparse
import json
from pathlib import Path
import numpy as np

from finite_probe import OUT, SURFACE, bundle, source_rate, survey


def run(args):
    source = json.loads(args.seed.read_text())
    ref = np.array(source[args.index]["currents_A"])
    rate = source_rate()
    meshes = {args.step: bundle(args.step, [SURFACE], args.ni, args.count, rate)}
    patterns = np.zeros((21, 9))
    for j, (lo, hi) in enumerate(
        [(4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 18), (18, 19), (19, 20), (20, 21)]
    ):
        patterns[lo:hi, j] = 1.0
    # Physical ampere changes for the nine complete winding groups.
    patterns *= np.array([22.0, 22.0, 10.0, 10.0, 30.0, 40.0, 24.0, 24.0, 24.0])

    def evaluate(x):
        q = ref + patterns @ x
        for scale in (1.0, 0.5, 0.25, 0.125):
            dz = args.step * scale
            if dz not in meshes:
                meshes[dz] = bundle(dz, [SURFACE], args.ni, args.count, rate)
            try:
                m = survey(meshes[dz], q)
                return dict(**m, currents_A=q.tolist(), objective_At=args.ni, max_step_mm=dz)
            except ValueError as exc:
                if "Unresolved boundary" not in str(exc):
                    raise
        raise ValueError("Mechanical grazing trajectory remains unresolved after step refinement")

    def errors(v):
        return np.array(
            [
                (v["alpha95_mrad"] - 30) / 0.8,
                v["drift_D95_m"] / 0.00020,
                v["drift_RMS_m"] / 0.00020 * args.rms,
            ]
        )

    x = np.zeros(patterns.shape[1])
    best = evaluate(x)
    history = [best]
    for n in range(args.iterations):
        e = errors(best)
        if (
            abs(e[0]) < 0.015
            and abs(best["drift_D95_m"]) < 1e-8
            and (not args.rms or abs(best["drift_RMS_m"]) < 1e-7)
        ):
            break
        success = False
        for requested in (0.08, 0.02, 0.004, 0.2):
            step = min(requested, max(0.000015, np.linalg.norm(e) * 0.085))
            jac = []
            for direction in np.eye(len(x)):
                up, down = (errors(evaluate(x + sign * step * direction)) for sign in (1, -1))
                jac.append((up - down) / (2 * step))
            j = np.array(jac).T
            dx = np.linalg.lstsq(j, -e, rcond=1e-5)[0]
            dx *= min(1.0, 1.4 / max(abs(dx).max(), 1e-12))
            for scale in (1.0, 0.5, 0.2, 0.08, 0.02):
                trial = x + dx * scale
                row = evaluate(trial)
                if np.linalg.norm(errors(row)) < np.linalg.norm(e):
                    x = trial
                    best = row
                    success = True
                    break
            if success:
                break
        if not success:
            break
        history.append(best)
        print(
            n,
            best["alpha95_mrad"],
            best["drift_D95_m"] * 1e6,
            best["drift_RMS_m"] * 1e6,
            best["D95_nm"],
            best["transmitted_uA"],
            flush=True,
        )
        (OUT / (args.name + ".json")).write_text(json.dumps([best], indent=2))
    (OUT / (args.name + ".json")).write_text(json.dumps([best], indent=2))
    (OUT / (args.name + "-history.json")).write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("seed", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--step", type=float, default=0.3)
    p.add_argument("--count", type=int, default=4096)
    p.add_argument("--ni", type=int, default=-24000)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--rms", type=float, default=0.0)
    run(p.parse_args())
