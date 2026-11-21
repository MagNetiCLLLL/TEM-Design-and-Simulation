"""Native through-focus and independent finite-current two-axis waist observations."""

import argparse
import json

import numpy as np

from column import CACHE, Mesh, SURFACE, metrics
from particles import Tracker


def axis_crossings(z, moments, lo, hi):
    ix = np.flatnonzero((z >= lo) & (z <= hi))
    result = []
    for k in (3, 4):
        a = moments[ix, k]
        j = np.flatnonzero((a[:-1] < 0) & (a[1:] >= 0))
        result.append([[float(z[ix[i]] * 1000), float(z[ix[i + 1]] * 1000)] for i in j])
    return result


def observation(q, case, ni, level, padding, step, hex_scale, prefix, topology, cache=CACHE):
    inner = np.arange(-250.0, 250.01, 0.5)
    outer = np.r_[-np.geomspace(110, 300000, 145), np.geomspace(110, 300000, 145)]
    # Uniform outer spacing resolves competing percentile minima of aberrated probes.
    offsets = np.unique(np.r_[inner, outer, np.arange(-140000.0, 140001.0, 500.0), 0.0])
    focus = SURFACE + offsets * 1e-9
    if topology:
        regular = np.linspace(0.45, SURFACE, int(np.ceil((SURFACE - 0.45) * 1000 / step)) + 1)
        coarse = regular[:: max(1, round(0.75 / step))]
        coarse = coarse[(coarse >= 0.647) & (coarse <= SURFACE - 0.00031)]
        observe = np.unique(np.r_[coarse, focus])
    else:
        observe = focus
    m = Mesh(step, observe, cache)
    m.hex *= hex_scale
    t = Tracker(m)
    field = m.field(m.base(case, ni, level, padding), q)
    index = np.searchsorted(m.z, SURFACE)
    _, loss, peak, margin = t.run(field, prefix, keep_history=False, record_probe=True, surface_index=index)
    ix = np.searchsorted(observe, focus)
    profiles = t.probes[ix]
    moments = t.moments
    live = t.surface_loss == -1
    surface = t.surface_phase[live]
    w = t.weights[:prefix][live]
    stat = metrics(surface, w, t.current)
    j = np.argmin(profiles[:, 0])
    sc = int(np.flatnonzero(offsets == 0)[0])
    assert np.isclose(stat["D95_nm"], profiles[sc, 0], atol=1e-6)
    neighbors = offsets[max(0, j - 1) : min(len(offsets), j + 2)]
    row = dict(
        case=case,
        ni=ni,
        controls=q,
        level=level,
        padding=padding,
        step_mm=step,
        hex_scale=hex_scale,
        source_count=len(live),
        surface=stat,
        D95_min_nm=float(profiles[j, 0]),
        D95_waist_offset_nm=float(offsets[j]),
        D95_waist_bracket_nm=neighbors.tolist(),
        rms_waist_offset_nm=float(offsets[np.argmin(profiles[:, 1])]),
        maximum_core_nA=float(profiles[:, 2].max()),
        max_surviving_radius_mm=float(peak[live].max() * 1000),
        focus_counts_stable=bool(np.all(moments[ix, 0] == moments[ix[0], 0])),
        min_clearance_mm=float(margin[live].min() * 1000),
        losses={},
        D95_surface_minus_min_nm=float(stat["D95_nm"] - profiles[j, 0]),
    )
    lost = t.surface_loss >= 0
    if np.any(lost):
        owners = m.keys[m.owner[t.surface_loss[lost]]]
        row["losses"] = {str(k): int(n) for k, n in zip(*np.unique(owners, return_counts=True))}
    if topology:
        c2, c3, adl, mini = m.intervals
        row["x_y_waist_brackets_mm"] = {
            k: axis_crossings(observe, moments, left, right)
            for k, left, right in (
                ("C2_to_C3", c2, c3),
                ("C3_to_ADL", c3, adl),
                ("Mini_to_sample", mini, SURFACE + 0.0003),
            )
        }
    return row, (
        observe,
        moments,
        focus,
        profiles,
        t.surface_phase,
        t.surface_loss,
        t.ids[:prefix],
        t.weights[:prefix],
        t.de[:prefix],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit")
    p.add_argument("--level")
    p.add_argument("--step", type=float, default=0.125)
    p.add_argument("--prefix", type=int, default=32768)
    p.add_argument("--topology", action="store_true")
    p.add_argument("--tag", default="observed")
    p.add_argument("--padding", type=float, default=8.0)
    args = p.parse_args()
    f = json.loads((CACHE / args.fit).read_text())
    row, arrays = observation(
        f["controls"],
        f["case"],
        f["ni"],
        args.level or f["level"],
        args.padding,
        args.step,
        f.get("hex_scale", 1.0),
        args.prefix,
        args.topology,
    )
    path = CACHE / f"probe-{f['case']}-{args.tag}"
    path.with_suffix(".json").write_text(json.dumps(row, indent=2))
    np.savez_compressed(
        path.with_suffix(".npz"),
        **dict(
            zip(
                (
                    "z_m",
                    "moments",
                    "focus_z_m",
                    "focus_D95_RMS_nm_core_nA",
                    "phase",
                    "loss_step",
                    "ray_id",
                    "weight",
                    "energy_offset_ev",
                ),
                arrays,
            )
        ),
    )
    print(row, flush=True)
