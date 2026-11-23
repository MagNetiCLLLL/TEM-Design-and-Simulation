"""Measure the finite-emittance gun waist with fixed cohorts and both transverse axes."""

import argparse
import json

import numpy as np
from scipy.signal import find_peaks
from numba import set_num_threads, get_num_threads

from gun_images import HERE, GUN, magnetic_spline, read_gun
from volume import BASE, CACHE
from transport import transport


def recover(folder):
    rec = json.loads((folder / "result.json").read_text())
    path = CACHE / rec["gun_packet"]
    name = rec["field"]
    with np.load(path) as d, np.load(folder / "trajectories.npz") as col:
        ids = np.flatnonzero(d["label"] == 0)
        np.testing.assert_array_equal(ids, col["ids"])
        ph = d["phase"][ids]
        w = d["weight_uA"][ids]
        keep = col["loss"] == -1
        sel = np.searchsorted(ids, col["selected_ids"])
        wanted = d["end"][ids]
        fine_roots = json.loads(str(d["step_check"]))
    mag = magnetic_spline(
        GUN / "gun-volume-field.npz" if name == BASE.stem else CACHE / (name + "-gun.npz")
    )
    ef, metal = read_gun(50)
    z = np.unique(
        np.r_[
            np.linspace(0.0, 0.04, 401),
            np.arange(0.0405, 0.1001, 0.0005),
            np.arange(0.101, 0.4501, 0.001),
            0.45,
        ]
    )
    set_num_threads(min(14, get_num_threads()))
    ev, end, xy = transport(ph, ef, mag, metal, z, len(ph), 0.5)
    assert np.all(ev == -1)
    np.testing.assert_allclose(end, wanted, rtol=0, atol=1e-12)
    curves = []
    report = {"paraxial_half_step": fine_roots}
    for key, mask in (("gun_exit", np.ones(len(w), bool)), ("surface", keep)):
        p = xy[mask].astype(float)
        ww = w[mask] / w[mask].sum()
        p -= np.einsum("n,nzc->zc", ww, p)
        var = np.einsum("n,nzc,nzc->zc", ww, p, p)
        widths = np.sqrt(np.column_stack((var, var.sum(axis=1))))
        desc = []
        for a in widths.T:
            idx = find_peaks(-a)[0]
            desc.append([dict(z_mm=float(z[i] * 1000), rms_um=float(a[i] * 1e6)) for i in idx])
        report[key] = desc
        curves.append(widths)
    np.savez_compressed(
        folder / "gun-paths.npz", z_mm=z * 1000, xy_m=xy[sel], rms_m=curves, selected_ids=ids[sel]
    )
    (folder / "gun-waists.json").write_text(json.dumps(report, indent=2) + "\n")
    print(folder.name, report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    recover(HERE / p.parse_args().folder)
