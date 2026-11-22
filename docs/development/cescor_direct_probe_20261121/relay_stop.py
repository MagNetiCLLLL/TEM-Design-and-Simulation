"""Refocus deliberately overfilled relay candidates without introducing another specimen-side waist."""

import argparse
import json
from pathlib import Path
import numpy as np
from numba import set_num_threads
from scipy.optimize import least_squares
from scipy.stats import qmc

from column_current import OUT
from pupil_search import batch, pencil, pieces, pupil


def main(args):
    ni = args.ni
    f, h, r, g, z = pieces(0.55, ni)
    paths = json.loads((args.seed or OUT / f"staged-{abs(ni)}.json").read_text())
    selected = paths if args.seed else [v for v in paths if v["stage_end_m"] == 1.499]
    results = []
    for idx, source in enumerate(selected):
        # Adjust the complete TL12 split windings and the three existing Mini
        # windings. Hold the strong upstream aperture illumination for this arm.
        q = np.array(source["currents_A"])
        q[18:] = 0.0
        directions = np.zeros((21, 6))
        directions[15:, :] = np.diag([args.transfer_range] * 3 + [1600] * 3)
        ff = np.ascontiguousarray(np.column_stack((f[:, 0] + f[:, 1:] @ q, f[:, 1:] @ directions)))
        pars = (ff, h, r, g, *pupil())
        guesses = 2 * qmc.Sobol(6, scramble=True, seed=4882 + idx).random_base2(16) - 1
        vals = batch(guesses, *pars)
        scale = np.minimum(1.0, 0.975 / np.maximum(vals[:, 2], 1e-15))
        expected = [args.pencil_c2, 0, 0]
        topology = np.all(vals[:, 3:] == expected, axis=1)
        ids = np.flatnonzero(topology)
        print(
            idx,
            "direct",
            len(ids),
            "largest mechanical angle",
            max(abs(vals[ids, 1]) * 1000 * scale[ids], default=0),
            flush=True,
        )
        for target in (30.0, 33.0, 35.0, 38.0) if args.seed else (30.0, 33.0, 35.0):
            error = (
                abs(vals[ids, 0]) * scale[ids] / 0.00045
                + abs(abs(vals[ids, 1]) * 1000 * scale[ids] - target) / 3
            )
            matched = 0
            for i in ids[np.argsort(error)[:50]]:
                sign = np.sign(vals[i, 1])

                def residual(x):
                    y, v, ratio, *cross = pencil(x, *pars)
                    fill = min(1.0, 0.975 / max(1e-12, ratio))
                    return np.r_[
                        y * fill / 0.00010,
                        (v * fill * 1000 - sign * target) / 3,
                        0.00015 * (x - guesses[i]),
                    ]

                sol = least_squares(residual, guesses[i], bounds=(-1.8, 1.8), max_nfev=90)
                y, v, ratio, *cross = pencil(sol.x, *pars)
                fill = min(1.0, 0.975 / max(1e-12, ratio))
                if (
                    cross != expected
                    or abs(y * fill) > 0.000006
                    or abs(abs(v) * 1000 * fill - target) > 0.2
                ):
                    continue
                result = dict(
                    currents_A=(q + directions @ sol.x).tolist(),
                    objective_At=ni,
                    ratio=ratio,
                    angle_mrad=abs(v) * 1000 * fill,
                    radius_um=y * fill * 1e6,
                    pencil_crossings=cross,
                )
                results.append(result)
                matched += 1
                if matched >= args.keep:
                    break
    (OUT / (args.name + ".json")).write_text(json.dumps(results, indent=2))
    print("Candidate currents", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=Path)
    p.add_argument("--ni", type=int, default=-24000)
    p.add_argument("--name", default="stop-pencils")
    p.add_argument("--transfer-range", type=float, default=450)
    p.add_argument("--keep", type=int, default=4)
    p.add_argument("--pencil-c2", type=int, default=1)
    set_num_threads(12)
    main(p.parse_args())
