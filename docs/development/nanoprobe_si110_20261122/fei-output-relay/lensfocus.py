"""Screen objective-only focus, measuring D95 through the specimen and the radial waist."""

import argparse
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from relay import BASE, CACHE, HERE, WORK, load_bias, sample, write
from steer import PivotEstimate
from supplies import field
from survey import Survey

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="global-d356e950fb0db241")
    p.add_argument("--reference", default=BASE.stem)
    p.add_argument("--n", type=int, default=524288)
    p.add_argument("--step", type=float, default=0.12)
    p.add_argument("--outer", action="store_true")
    p.add_argument("amps", nargs="+", type=float)
    a = p.parse_args()
    path = WORK / (a.field + ".npz")
    z, b = field(path)
    o = load_bias(path)["objective_lens/physical-pack-4"]
    reference = WORK / (a.reference + ".npz")
    m = Survey(a.step, reference, a.n, sample(reference) if a.outer else None)
    tangent = WORK / f"{path.stem}-objective-bore-0.5mm.npz"
    with np.load(tangent if tangent.exists() else WORK / f"{BASE.stem}-objective-bore-0.5mm.npz") as f:
        delta = CubicSpline(f["z"], f["coef"][:, 0, 2])(z)
    # Replace the small-signal estimate by exact per-setting axial fields where available.
    known = {}
    for other in CACHE.glob("global-*-fei-output-full-axis.npz"):
        k = WORK / (other.name.split("-fei-output")[0] + ".npz")
        bb = load_bias(k)
        if bb["probe_tl12_lens"] == load_bias(path)["probe_tl12_lens"]:
            known[bb["objective_lens/physical-pack-4"]] = field(k)[1]
    q = PivotEstimate(z)
    rows = []
    for current in a.amps:
        if current in known:
            v = known[current]
        elif len(known) > 1 and min(known) < current < max(known):
            keys = sorted(known)
            v = PchipInterpolator(keys, [known[k] for k in keys], axis=0)(current)
        else:
            v = b + (current - o) * delta
        r = m.run(v, q(v), True, True)
        r.update(current_A=current, exact_axis=current in known, reference=a.reference)
        print(
            current,
            "D95",
            round(r["D95_um"], 3),
            "D95 waist",
            r["local_D95_waist_mm_um"],
            "RMS",
            r["waists_mm"],
            "surface growth",
            round(r["surface_slope_mrad"], 3),
            flush=True,
        )
        rows.append(r)
    suffix = "-outer" if a.outer else ""
    write(HERE / f"objective-survey-{a.field}-{a.reference}-{a.n}{suffix}.json", rows)
