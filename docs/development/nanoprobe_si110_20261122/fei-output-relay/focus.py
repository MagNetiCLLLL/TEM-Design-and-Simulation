"""Resolve both sides of the specimen focus with the local material field and actual ray IDs."""

import argparse
import json
import numpy as np
from types import SimpleNamespace
import cupy as cp
from relay import (
    CACHE,
    HERE,
    WORK,
    BoreTracker,
    Vacuum,
    advance,
    electric_reference,
    optics_mesh,
    sha,
    source_packet,
    values,
    wq,
    write,
)
from volume import basis
from material_relay_column import source_and_electrodes
from aim import TOP


def local_field(path):
    out = CACHE / (path.stem + "-sample-focus-volume.npz")
    if out.exists():
        return out
    vac = Vacuum(path)
    z = np.arange(1.578, 1.6161, 0.0002)
    degree = 5
    radius = 0.0015
    theta = np.arange(12) * np.pi / 6
    xy = np.vstack(
        (
            np.zeros((1, 2)),
            *[r * np.column_stack((np.cos(theta), np.sin(theta))) for r in (0.35, 0.69, 1.0)],
        )
    )
    p = np.column_stack((np.tile(xy * radius, (len(z), 1)), np.repeat(z, len(xy))))
    b = vac.symmetric(p).reshape(len(z), len(xy), 3)
    co = np.empty((len(z), 21, 3))
    co[:, 0] = b[:, 0]
    co[:, 1:] = np.einsum("ij,zjk->zik", np.linalg.pinv(basis(xy, degree)[:, 1:]), b - b[:, :1])
    np.savez_compressed(out, z_m=z, coeff=co, scale_m=radius, degree=degree, field_sha256=sha(path))
    return out


def mesh(z):
    nodes = np.empty(len(z) * 2 - 1)
    nodes[::2] = z
    nodes[1::2] = (z[1:] + z[:-1]) / 2
    base = source_and_electrodes().electric.base_field
    j = np.clip(np.searchsorted(base.z, nodes, side="right") - 1, 0, len(base.z) - 2)
    dz = np.diff(base.z)[j]
    electric = (base.r, base.voltage, j, (nodes - base.z[j]) / dz, dz)
    return SimpleNamespace(z=z, nodes=nodes, electric=electric, radius=np.full(len(z) - 1, 0.0015))


def run(path, folder):
    data = json.loads((folder / "result.json").read_text())
    packet = source_packet(CACHE / data["gun_packet"])
    m, _, _ = optics_mesh(0.12)
    with np.load(folder / "trajectories.npz") as rec:
        np.testing.assert_array_equal(packet[2], rec["ids"])
        ok = rec["loss"] == -1
        p = rec["surface_phase"][ok]
        weights = rec["weights_uA"][ok]
        ids = packet[2][ok]
        aper = np.maximum(
            np.linalg.norm(rec["aperture_phase"][ok, :2], axis=1),
            np.linalg.norm(rec["aperture_rear_phase"][ok, :2], axis=1),
        )
    from particles import Tracker

    en = 300000 + packet[3][ok]
    ref = electric_reference(Tracker(m, source=packet))[ok]
    src = (p, weights / 15, ids, en - 300000, 15e-6)
    field_table = local_field(path)
    half = ids < data["emitted_particles"] // 2

    def half_d95(phase):
        xy = phase[half, :2]
        offset = xy - np.average(xy, weights=weights[half], axis=0)
        return 2e6 * wq(np.linalg.norm(offset, axis=1), weights[half])

    rows = []
    for direction in (-1, 1):
        z = TOP + direction * np.arange(0, 14.001, 0.025) * 0.001
        lm = mesh(z)
        t = BoreTracker(lm, field_table, src)
        fd = cp.zeros((len(lm.nodes), 20))
        ph = p.copy()
        lost = np.full(len(p), -1, np.int32)
        last = 0
        for grp in np.array_split(np.arange(1, len(z)), 24):
            for idx, (ph, lost) in zip(grp, advance(t, fd, ph, en, ref, lost, last, grp), strict=True):
                if np.any(lost != -1):
                    raise ValueError("Focus diagnostic exceeded validated volume")
                xy = ph[:, :2] - np.average(ph[:, :2], weights=weights, axis=0)
                r = np.linalg.norm(xy, axis=1)
                v = ph[:, 2:4] - np.average(ph[:, 2:4], weights=weights, axis=0)
                proj = (xy * v).sum(axis=1)
                rows.append(
                    [
                        z[idx] * 1000,
                        2e6 * wq(r, weights),
                        1e6 * np.sqrt(np.average(r * r, weights=weights)),
                        np.average(proj, weights=weights),
                        np.average(proj < 0, weights=weights),
                        half_d95(ph),
                    ]
                )
            last = grp[-1]
    xy = p[:, :2] - np.average(p[:, :2], weights=weights, axis=0)
    v = p[:, 2:4] - np.average(p[:, 2:4], weights=weights, axis=0)
    r = np.linalg.norm(xy, axis=1)
    dot = (xy * v).sum(axis=1)
    rows.append(
        [
            TOP * 1000,
            2e6 * wq(r, weights),
            1e6 * np.sqrt(np.average(r * r, weights=weights)),
            np.average(dot, weights=weights),
            np.average(dot < 0, weights=weights),
            half_d95(p),
        ]
    )
    arr = np.array(sorted(rows))
    roots = np.where((arr[:-1, 3] < 0) & (arr[1:, 3] >= 0))[0]
    rms_roots = [
        (arr[i, 0] - arr[i, 3] * (arr[i + 1, 0] - arr[i, 0]) / (arr[i + 1, 3] - arr[i, 3]))
        for i in roots
    ]
    best = arr[np.argmin(arr[:, 1])]
    best_half = arr[np.argmin(arr[:, 5])]
    chief = np.r_[np.average(p[:, 2:4], weights=weights, axis=0), 1.0]
    dirs = np.column_stack((p[:, 2:4], np.ones(len(p))))
    angle = np.arctan2(np.linalg.norm(np.cross(dirs, chief), axis=1), dirs @ chief) * 1000
    bands = []
    for threshold in (45.0, 47.5, 49.0):
        band = aper >= threshold * 1e-6
        if not np.any(band):
            continue
        inward = band & (dot < 0)
        bands.append(
            dict(
                inner_radius_um=threshold,
                outer_radius_um=50.0,
                n=int(band.sum()),
                median_mrad=wq(angle[band], weights[band], 0.5),
                range_mrad=[angle[band].min(), angle[band].max()],
                inward_current_fraction=np.average(dot[band] < 0, weights=weights[band]),
                inward_count=int(inward.sum()),
                inward_median_mrad=wq(angle[inward], weights[inward], 0.5) if np.any(inward) else None,
                inward_range_mrad=[angle[inward].min(), angle[inward].max()] if np.any(inward) else None,
            )
        )
    res = dict(
        specimen_z_mm=TOP * 1000,
        minimum_D95_z_mm=best[0],
        minimum_D95_um=best[1],
        first_half_D95_focus=dict(
            emitted=data["emitted_particles"] // 2, z_mm=best_half[0], D95_um=best_half[5]
        ),
        at_search_boundary=bool(best[0] in arr[[0, -1], 0]),
        rms_waists_mm=rms_roots,
        at_specimen=values(p, weights),
        inward_current_fraction=np.average(dot < 0, weights=weights),
        aperture_rim=bands,
        rim_maximum_um=float(max(aper) * 1e6),
        outermost_ray=dict(
            source_id=int(ids[np.argmax(aper)]),
            angle_mrad=float(angle[np.argmax(aper)]),
            radially_inward=bool(dot[np.argmax(aper)] < 0),
        ),
        continuation="Vacuum diagnostic on both sides. Material scattering excluded after the specimen surface.",
    )
    np.savez_compressed(
        folder / "through-focus.npz",
        z_mm=arr[:, 0],
        D95_um=arr[:, 1],
        rms_um=arr[:, 2],
        radial_covariance_m=arr[:, 3],
        inward_current_fraction=arr[:, 4],
    )
    write(folder / "focus.json", res)
    print("Focus", res, flush=True)


if __name__ == "__main__":
    q = argparse.ArgumentParser()
    q.add_argument("field")
    q.add_argument("name")
    a = q.parse_args()
    run(WORK / (a.field + ".npz"), HERE / a.name)
