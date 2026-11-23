"""Trace signed orthogonal aperture perturbations in the fixed-anode gun FEM."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
GUN = HERE.parent / "fei-gun-control"
sys.path.insert(0, str(GUN))
from gun_field import study_field_reader
from transport import REST_EV, transport


def magnetic_spline(path):
    with np.load(path) as data:
        return np.ascontiguousarray(
            CubicSpline(data["z_m"], data["coeff"], axis=0).c.transpose(1, 2, 3, 0)
        )


def read_gun(v):
    ef, receipt = study_field_reader()("anode", True, v)
    metal = np.array(
        [[r[k] for k in ("start_m", "stop_m", "inner_m", "outer_m")] for r in receipt["stops"]]
    )
    return ef, metal


def axial_images(ef, metal, magnetic, scale=0.5, angle=1e-6):
    phase = np.zeros((4, 6))
    mom = np.sqrt((0.2 / REST_EV) * (2 + 0.2 / REST_EV))
    phase[:, 3:5] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) * mom * np.sin(angle)
    phase[:, 5] = mom * np.cos(angle)
    z = np.unique(np.r_[np.linspace(0.0, 0.04, 4001), np.arange(0.0401, 0.45, 0.0001), 0.45])
    events, end, xy = transport(phase, ef, magnetic, metal, z, 4, scale)
    if np.any(events != -1):
        raise ValueError(f"Central fan not transmitted: {events}")
    a = ((xy[0] - xy[1]) / (2 * angle)).astype(float)
    b = ((xy[2] - xy[3]) / (2 * angle)).astype(float)
    # Before the first condenser, lens rotation is very small. Reject a mere
    # crossing of one projection using the orthogonal fan and the full norm.
    roots = []
    for i in np.flatnonzero(a[1:, 0] * a[:-1, 0] < 0):
        t = -a[i, 0] / (a[i + 1, 0] - a[i, 0])
        at = (1 - t) * np.stack((a[i], b[i])) + t * np.stack((a[i + 1], b[i + 1]))
        roots.append(
            dict(
                z_mm=float((z[i] + t * (z[i + 1] - z[i])) * 1000),
                residual_xy_nm_per_urad=float(np.linalg.norm(at) * 1000),
            )
        )
    return roots, end


if __name__ == "__main__":
    magnetic = magnetic_spline(GUN / "gun-volume-field.npz")
    results = {}
    for voltage in [float(v) for v in sys.argv[1:]] or [50, 500, 1000, 1200, 1500, 1800, 2000]:
        ef, metal = read_gun(voltage)
        roots, _ = axial_images(ef, metal, magnetic)
        results[str(voltage)] = roots
        print(voltage, roots, flush=True)
    (HERE / "gun-images.json").write_text(json.dumps(results, indent=2) + "\n")
