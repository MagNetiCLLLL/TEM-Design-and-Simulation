"""Replay the narrow-source reference in the wide-bore evaluator as a transport control."""

import json
from pathlib import Path

import numpy as np

from column_trace import BoreTracker, covariance, crossings, optics_mesh, propagate, values
from gun_images import HERE
from volume import BASE, CACHE


if __name__ == "__main__":
    previous = json.loads((HERE.parent / "c1-zoom-near-c2-focused/tracking.json").read_text())
    with np.load(previous["gun_source"]) as d:
        source = tuple(d[k] for k in ("phase", "weight", "ids", "energy_offset_ev")) + (
            float(d["emitted_A"]),
        )
    if len(source[2]) < 1000:
        raise ValueError("Invalid archived original population")
    m, st, _ = optics_mesh(0.12)
    table = CACHE / (BASE.stem + "-column-2880um-nested.npz")
    t = BoreTracker(m, table, source)
    stations = st[::3]
    if stations[-1] != st[-1]:
        stations = np.r_[stations, st[-1]]
    zz = []
    cov = []
    for j, p, loss in propagate(t, m, stations):
        assert np.all(loss == -1)
        zz.append(m.z[j] * 1000)
        cov.append(covariance(p, source[1])[2:].sum())
    result = dict(
        originals=len(source[2]),
        field=table.name,
        roots_mm=crossings(np.asarray(zz), np.asarray(cov)),
        previous_mm=previous["waists_mm"],
        measured=values(p, source[1] * 15),
        previous_surface=previous["surface"],
    )
    (HERE / "narrow-replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)
