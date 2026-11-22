"""Continue each proposed direct branch to the target angle using the whole column."""

import argparse
import json

import numpy as np
from scipy.optimize import least_squares

from column_current import OUT, OBJECTIVE_NI
from pupil_search import pieces, pupil, pencil


def main(ni):
    f, h, r, g, _ = pieces(0.4, ni)
    args = (f, h, r, g, *pupil())
    rows = json.loads((OUT / f"relay-{abs(ni)}.json").read_text())
    found = []
    for k, seed in enumerate(rows):
        q = np.array(seed["currents_A"])
        start = np.asarray(pencil(q, *args))
        sign = np.sign(start[1]) or 1
        # Independent complete closed windings. C2/C3 set the angular gain,
        # while the Mini Condenser and transfer lens recover surface focus.
        directions = np.zeros((len(q), 8))
        for j, (left, right, amp) in enumerate(
            [
                (4, 6, 60),
                (6, 8, 60),
                (8, 10, 20),
                (10, 12, 20),
                (12, 15, 60),
                (15, 18, 120),
                (18, 21, 40),
                (20, 21, 40),
            ]
        ):
            directions[left:right, j] = amp
        a0 = abs(start[1]) / max(1.0, start[2] / 0.975)
        for ang in np.linspace(a0, 0.030, 9)[1:]:

            def residual(x):
                y, v, ratio, *_ = pencil(q + directions @ x, *args)
                fill = min(1.0, 0.975 / max(1e-15, ratio))
                return np.r_[y * fill / 0.0001, (v * fill * sign - ang) / 0.003, 0.00003 * x]

            sol = least_squares(
                residual,
                np.zeros(directions.shape[1]),
                bounds=(-2.0, 2.0),
                max_nfev=65,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-11,
            )
            q = q + directions @ sol.x
        y, v, ratio, *cross = pencil(q, *args)
        fill = min(1.0, 0.975 / max(1e-15, ratio))
        if cross == [1.0, 0.0, 0.0] and abs(v * fill * sign - 0.03) < 0.002 and abs(y * fill) < 0.00002:
            found.append(
                dict(
                    currents_A=q.tolist(),
                    objective_At=ni,
                    ratio=ratio,
                    angle_mrad=v * fill * 1000,
                    radius_um=y * fill * 1e6,
                    pencil_crossings=cross,
                    seed=k,
                )
            )
    found.sort(key=lambda x: -x["ratio"])
    (OUT / f"pencils-{abs(ni)}.json").write_text(json.dumps(found, indent=2))
    print("Direct matched pencils", len(found), "ratios", [round(s["ratio"], 2) for s in found], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objective-ni", type=int, default=OBJECTIVE_NI)
    main(p.parse_args().objective_ni)
