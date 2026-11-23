"""Register the million-primary waist, then place the D95 focus on the specimen."""

from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
import numpy as np
from relay import CACHE, HERE, OBJ, WORK, load_bias, write
from supplies import field
from steer import PivotEstimate
from survey import Survey
from aim import TOP


def run():
    path = WORK / "global-ff8bb76b91abe5dd.npz"
    m = Survey(0.09, path, 1048576)
    z, b = field(path)
    q = PivotEstimate(z, path)
    slopes = []
    for key in ("tl", "objective"):
        with np.load(WORK / f"{path.stem}-{key}-bore-0.5mm.npz") as f:
            slopes.append(CubicSpline(f["z"], f["coef"][:, 0, 2])(z))

    def bfield(t, o):
        return b + (t + 319.1) * slopes[0] + (o - OBJ) * slopes[1]

    def reg(t):
        f = bfield(t, OBJ)
        return m.run(f, q(f))["target_slope_mrad"]

    t = round(brentq(reg, -323.0, -317.0, xtol=0.005), 3)
    at_stage = bfield(t, OBJ)
    stage = m.run(at_stage, q(at_stage), True, True)
    out = []
    for o in np.arange(-1.30, -0.769, 0.025):
        f = bfield(t, o)
        s = m.run(f, q(f), False, True)
        out.append((o, s["local_D95_waist_mm_um"][0], s["D95_um"]))
    choices = []
    for (o1, z1, d1), (o2, z2, d2) in zip(out, out[1:]):
        if (z1 - TOP * 1000) * (z2 - TOP * 1000) > 0:
            continue
        if z2 != z1:
            choices.append((min(d1, d2), float(o1 + (o2 - o1) * (TOP * 1000 - z1) / (z2 - z1))))
    print("TL12 registration", t, stage, "objectives", out, flush=True)
    if not choices:
        raise ValueError("D95 focus requires another objective-current bracket")
    o = round(min(choices)[1], 4)
    final = bfield(t, o)
    result = dict(
        tl_A=t,
        objective_A=o,
        stage=stage,
        objective_survey=out,
        final=m.run(final, q(final), True, True),
    )
    write(HERE / "million-primary-refinement.json", result)
    print("Selected", result, flush=True)


if __name__ == "__main__":
    run()
