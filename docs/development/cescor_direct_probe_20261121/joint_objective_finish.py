"""Find specimen-plane D95 focus by changing only the physical objective excitation."""

import argparse
from functools import lru_cache
import json

import numpy as np
from scipy.optimize import brentq

from joint_pivot_tuning import OUT
from joint_relay_search import NI, model_at, surface_metrics, switch_objective
from refocus_a2 import WORK
from relay_transfer import SURFACE, advance


class Objective:
    def __init__(self, args):
        self.args = args
        self.m, self.q = model_at(args.amps, NI, args.count, args.step, args.half)
        self.values = np.array(
            sorted(
                -float(p.name.split("-")[1]) for p in (WORK / "column").glob("armco-*-gap-fine-p8.json")
            )
        )
        self.log = []

    @lru_cache(maxsize=6)
    def bg(self, k):
        return self.m.m.base("armco", k, "gap-fine", 8)

    @lru_cache(maxsize=150)
    def run(self, ni):
        ni = round(ni, 4)
        m = self.m
        if self.args.exact:
            switch_objective(m, ni)
        else:
            j = int(np.clip(np.searchsorted(self.values, ni) - 1, 0, len(self.values) - 2))
            a, b = self.values[j : j + 2]
            if ni < a or ni > b:
                raise ValueError("Objective prediction requires two bounding material solutions")
            ratio = (ni - a) / (b - a)
            m.bg = (1 - ratio) * self.bg(a) + ratio * self.bg(b)
        f = m.field(self.q)
        ((p, loss),) = advance(m.t, f, m.t.input, m.en, m.phi0, m.loss, 0, m.indices([SURFACE]))
        s = surface_metrics(m, p, loss)
        self.log.append(dict(objective_At=ni, **s))
        filename = f"objective-{self.args.amps:g}A-{self.args.count}-{self.args.step:g}-{self.args.half}-{'exact' if self.args.exact else 'secant'}"
        (OUT / (filename + ".json")).write_text(json.dumps(self.log, indent=2, allow_nan=False))
        print(
            f"Objective {ni:+.4f} At  D95 {s['D95_um']:.6f} um  drift {s['drift_D95_mm']:.8f} mm",
            flush=True,
        )
        return s["drift_D95_mm"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amps", type=float, required=True)
    p.add_argument("--ni", type=float, nargs="+", required=True)
    p.add_argument("--step", type=float, default=0.12)
    p.add_argument("--count", type=int, default=4096)
    p.add_argument("--half", type=int, default=8)
    p.add_argument("--root", action="store_true")
    p.add_argument("--exact", action="store_true")
    args = p.parse_args()
    obj = Objective(args)
    for ni in args.ni:
        obj.run(ni)
    if args.root:
        a, b = sorted(args.ni)
        x = brentq(obj.run, a, b, xtol=0.00011, maxiter=45)
        print("Material-current root proposal", x, flush=True)
