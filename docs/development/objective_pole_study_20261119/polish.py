"""Place the percentile waist with Mini, using actually propagated flanking planes."""

import argparse
import json

import numpy as np
from scipy.optimize import brentq

from column import CACHE, Mesh, SURFACE, metrics
from particles import Tracker
from tune import waist_proposal


def resolve_percentile(phase, w, estimate):
    """Find competing D95 cusps around the coarse waist with conservative rank bounds."""
    assert np.all(w == w[0])
    xy = phase[:, :2] - phase[:, :2].mean(axis=0)
    t = phase[:, 2:4] - phase[:, 2:4].mean(axis=0)
    xy = xy + estimate * t
    a = (xy * xy).sum(axis=1)
    b = 2 * (xy * t).sum(axis=1)
    c = (t * t).sum(axis=1)
    radius = np.sqrt(a)
    excursion = 2e-6 * np.sqrt(c)
    low = np.maximum(0.0, radius - excursion)
    high = radius + excursion
    k = (19 * len(w) + 19) // 20 - 1
    l = np.partition(low, k)[k]
    h = np.partition(high, k)[k]
    possible = (high >= l) & (low <= h)
    k -= int(np.count_nonzero(high < l))
    a, b, c = a[possible], b[possible], c[possible]

    def value(d):
        return np.partition(a + d * (b + d * c), k)[k]

    grid = np.arange(-2000.0, 2000.1, 2.0) * 1e-9
    values = np.array([value(d) for d in grid])
    # Refine all low local minima, keeping the actual empirical quantile.
    minima = np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])) + 1
    if not len(minima):
        return estimate + grid[values.argmin()]
    from scipy.optimize import minimize_scalar

    results = [
        minimize_scalar(value, bounds=(grid[j - 1], grid[j + 1]), method="bounded", options={"xatol": 1e-15})
        for j in minima[np.argsort(values[minima])[:12]]
    ]
    return estimate + min(results, key=lambda v: v.fun).x


def polish(f, step, level, padding, tag, c2=None, cache=CACHE):
    q = np.array(f["controls"])
    if c2 is not None:
        q[0] = c2
    flank = 0.25e-9
    m = Mesh(step, [SURFACE - flank, SURFACE, SURFACE + flank], cache)
    m.hex *= f.get("hex_scale", 1.0)
    t = Tracker(m)
    base = m.base(f["case"], f["ni"], level, padding)
    trace = {}

    def run(mini):
        if mini not in trace:
            trial = q.copy()
            trial[2] = mini
            phase, lo, pk, ma = t.run(m.field(base, trial))
            ok = lo == -1
            w = t.weights[ok]
            stats = [metrics(x[ok], w, t.current) for x in phase]
            dz, rm = waist_proposal(phase[1, ok], w)
            dz = resolve_percentile(phase[1, ok], w, dz)
            diff = stats[2]["D95_nm"] - stats[0]["D95_nm"]
            trace[mini] = (dz * 1e9, diff, stats[1])
        return trace[mini]

    center = q[2]
    # Find the smooth branch near the broad D95 minimum, then center the actual three-plane bracket.
    for _ in range(8):
        dz, _, _ = run(center)
        if abs(dz) < 0.04:
            break
        shift = max(3e-10, min(3e-7, abs(dz) * 1e-10))
        derivative = (run(center + shift)[0] - dz) / shift
        if abs(derivative) < 1e5:
            break
        move = np.clip(-dz / derivative, -0.00008, 0.00008)
        trial = center + move
        if abs(run(trial)[0]) >= abs(dz):
            break
        center = trial
    preferred = center
    for width in (2e-9, 2e-8, 2e-7, 2e-6, 1e-5, 6e-5):
        left, right = center - width, center + width
        if run(left)[1] * run(right)[1] > 0:
            continue
        preferred = brentq(lambda u: run(u)[1], left, right, xtol=5e-15, rtol=1e-14)
        break
    q[2] = preferred
    dz, diff, stat = run(preferred)
    out = dict(
        case=f["case"],
        ni=f["ni"],
        level=level,
        padding=padding,
        step_mm=step,
        hex_scale=f.get("hex_scale", 1.0),
        controls=q.tolist(),
        surface=stat,
        predicted_offset_nm=dz,
        native_flank_D95_difference_nm=diff,
        focus_flanks_nm=[-0.25, 0.25],
        source_count=32768,
    )
    (cache / f"fit-{f['case']}-{tag}.json").write_text(json.dumps(out, indent=2))
    print(f["case"], q[2], dz, diff, stat, flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit")
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--level", default="finer")
    p.add_argument("--padding", type=float, default=8.0)
    p.add_argument("--tag", default="final")
    p.add_argument("--c2", type=float)
    a = p.parse_args()
    polish(json.loads((CACHE / a.fit).read_text()), a.step, a.level, a.padding, a.tag, a.c2)
