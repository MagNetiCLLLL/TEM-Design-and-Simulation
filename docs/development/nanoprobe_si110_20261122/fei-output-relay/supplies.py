"""Cache converged common-circuit solutions and directly evaluated round fields for the survey."""

import argparse
import numpy as np
from scipy.interpolate import CubicSpline
from unittest.mock import patch
from relay import (
    BASE,
    CACHE,
    OBJ,
    TL,
    WORK,
    Circuit,
    Global,
    Vacuum,
    apply_supply_changes,
    digest,
    groups,
    load_bias,
    newton,
    sha,
)
from aim import Screen


def field(path):
    out = CACHE / (path.stem + "-fei-output-full-axis.npz")
    if not out.exists():
        zz = np.linspace(0.448, 1.612, 2911)
        v = Vacuum(path)
        bz = v.direct(np.column_stack((zz * 0, zz * 0, zz)))[:, 2]
        np.savez(out, z=zz, b=bz, sha256=sha(path))
    with np.load(out) as f:
        return f["z"], f["b"]


class Supplies:
    def __init__(self):
        self.circuit = None
        self.bias = load_bias(BASE)
        self.history = [BASE]
        for path in WORK.glob("global-????????????????.npz"):
            b = load_bias(path)
            fixed = set(b) - set(groups(b)["tl"]) - set(groups(b)["objective"])
            if path != BASE and all(b[k] == self.bias[k] for k in fixed):
                self.history.append(path)

    def get(self, t, o):
        b = apply_supply_changes(self.bias, {"tl": float(t), "objective": float(o)})
        path = WORK / ("global-" + digest(b)[:16] + ".npz")
        if not path.exists():

            def distance(f):
                d = load_bias(f)
                return (
                    abs(d["probe_tl12_lens"] - t) / 60 + abs(d["objective_lens/physical-pack-4"] - o) / 2
                )

            seed = min(self.history, key=distance)
            if self.circuit is None:
                self.circuit = Circuit(Global(9))
            with patch.object(newton, "groups", groups):
                self.circuit.solve(b, seed, path, predictor=BASE)
        if load_bias(path) != b:
            raise ValueError("Supply or material solution mismatch")
        if path not in self.history:
            self.history.append(path)
        return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("settings", nargs="+")
    a = p.parse_args()
    sup = Supplies()
    screen = Screen()
    for val in a.settings:
        t, o = val.split(",")
        t = TL if t == "ref" else float(t)
        o = OBJ if o == "ref" else float(o)
        path = sup.get(t, o)
        z, b = field(path)
        r = screen.describe(t, o, CubicSpline(z, b)(screen.z))
        print("Exact DC survey", path.stem, r, flush=True)
