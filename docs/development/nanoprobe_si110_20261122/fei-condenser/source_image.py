"""Measure C1 imaging of the real paraxial gun crossover independently of the current envelope."""

import argparse
import json

import numpy as np
from scipy.signal import find_peaks

from gun_images import GUN, HERE, magnetic_spline, read_gun
from transport import REST_EV, field_at, transport
from volume import BASE, CACHE
from column_trace import BoreTracker, optics_mesh, propagate


def measure(folder):
    rec = json.loads((folder / "result.json").read_text())
    ef, metal = read_gun(50)
    name = rec["field"]
    mag = magnetic_spline(
        GUN / "gun-volume-field.npz" if name == BASE.stem else CACHE / (name + "-gun.npz")
    )
    with np.load(CACHE / rec["gun_packet"]) as d:
        root = json.loads(str(d["gun_image"]))[0]["z_mm"] * 0.001
    delta = 0.1e-6
    angle = 1e-6
    init = np.zeros((8, 6))
    init[:, 2] = root
    for i in range(2):
        init[2 * i, i] = delta
        init[2 * i + 1, i] = -delta
    for i, p in enumerate(init):
        k = 0.2 + field_at(*p[:3], ef)[0]
        mom = np.sqrt(k / REST_EV * (2 + k / REST_EV))
        p[5] = mom
        if i >= 4:
            p[3 + (i - 4) // 2] = (1 if i % 2 == 0 else -1) * mom * np.sin(angle)
            p[5] = mom * np.cos(angle)
    ev, end, _ = transport(init, ef, mag, metal, np.array([0.45]), 0, 0.5)
    assert np.all(ev == -1)
    phase = np.column_stack((end[:, :2], end[:, 3:5] / end[:, 5, None], end[:, 6]))
    u2 = (end[:, 3:6] ** 2).sum(axis=1)
    en = REST_EV * u2 / (1 + np.sqrt(1 + u2))
    m, obs, _ = optics_mesh(0.075)
    obs = obs[(m.z[obs] >= 0.45) & (m.z[obs] <= 0.65)]
    t = BoreTracker(m, CACHE / rec["field_table"], (phase, np.ones(8), np.arange(8), en - 300000.0, 0.0))
    A = []
    B = []
    z = m.z[obs]
    for _, s, loss in propagate(t, m, obs):
        assert np.all(loss == -1)
        der = s[0::2, :2] - s[1::2, :2]
        A.append(der[:2].T / (2 * delta))
        B.append(der[2:].T / (2 * angle))
    A = np.array(A)
    B = np.array(B)
    norm = np.linalg.norm(B, axis=(1, 2))
    images = []
    for i in find_peaks(-norm)[0]:
        for j in (i - 1, i):
            diff = B[j + 1] - B[j]
            f = -np.sum(B[j] * diff) / np.sum(diff * diff)
            if not 0 <= f <= 1:
                continue
            bs = np.linalg.svd(B[j] + f * diff, compute_uv=False)
            ma = np.linalg.svd(A[j] + f * (A[j + 1] - A[j]), compute_uv=False)
            images.append(
                dict(
                    z_mm=1000 * (z[j] + f * (z[j + 1] - z[j])),
                    B_residual_nm_per_urad=(1000 * bs).tolist(),
                    absolute_magnification=ma.tolist(),
                )
            )
    result = dict(
        object_plane_mm=root * 1000,
        search_mm=[450, 650],
        position_step_nm=delta * 1e9,
        angular_step_urad=angle * 1e6,
        images=images,
    )
    (folder / "c1-image.json").write_text(json.dumps(result, indent=2) + "\n")
    print(folder.name, result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    measure(HERE / p.parse_args().folder)
