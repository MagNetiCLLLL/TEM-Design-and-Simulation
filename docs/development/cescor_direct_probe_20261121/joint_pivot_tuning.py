"""Locate and align the three downstream planes using the joint material field."""

import argparse
import json

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from column_current import SURFACE
from downstream_pivot import DrivenTrace
from normal_working_point import CACHE, operating_point
from objective_pivot_sensitivity import closest_pivot, summary
from scan_coil_waist_origin import nearest
from scan_waist_audit import locate_planes


OUT = CACHE / "three-plane-tuning"
OUT.mkdir(exist_ok=True, parents=True)
LOWER_EXIT = 1.420


def save(name, data):
    (OUT / (name + ".json")).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def stations(m, z):
    return m.m.z[np.unique(nearest(m.m.z, np.asarray(z)))]


def focal_planes(m, q):
    observed = stations(m, np.r_[np.arange(1.420, SURFACE, 0.0005), SURFACE])
    roots = locate_planes(m, q, observed)
    return {key: [float(v * 1000) for v in values] for key, values in roots.items()}


def scan_response(m, q, mix, focus_mm, *, calibrate=False):
    target = focus_mm * 0.001
    planes = stations(m, np.r_[np.arange(LOWER_EXIT, SURFACE, 0.001), target, SURFACE])
    trace = DrivenTrace(m)
    field = m.field(q)
    if calibrate:
        basis, _, _ = trace.response(field, np.eye(4), [planes[np.argmin(abs(planes - target))]], 1.0)
        position = basis[0, :2]
        if np.linalg.cond(position[:, 2:]) > 1e5:
            raise ValueError("The two lower scan channels cannot independently set the pivot")
        mix = np.vstack([mix[:2], np.linalg.solve(position[:, 2:], -position[:, :2] @ mix[:2])])
    response, even, masks = trace.response(field, mix, planes, 1.0)
    norm = np.linalg.norm(response[:, :2], axis=(1, 2))
    minima = np.flatnonzero((norm[1:-1] < norm[:-2]) & (norm[1:-1] < norm[2:])) + 1
    fits = []
    spline = CubicHermiteSpline(planes, response[:, :2], response[:, 2:4], axis=0)
    for i in minima:
        if i < 2 or i + 2 >= len(planes):
            continue
        lo, hi = max(0, i - 6), min(len(planes), i + 7)
        zone = slice(lo, hi)
        (pivot,) = closest_pivot(planes[zone], response[zone], columns=(slice(None),))
        value = np.vstack([spline(pivot), spline(pivot, nu=1)])
        separate = []
        for j in (0, 1):
            axis_norm = np.linalg.norm(response[:, :2, j], axis=1)
            own = (
                np.flatnonzero((axis_norm[1:-1] < axis_norm[:-2]) & (axis_norm[1:-1] < axis_norm[2:]))
                + 1
            )
            if len(own):
                best = min(own, key=lambda k: abs(k - i))
                if 3 <= best < len(planes) - 3:
                    window = slice(best - 3, best + 4)
                    (result,) = closest_pivot(planes[window], response[window], columns=(j,))
                    separate.append(float(result * 1000))
        fits.append(dict(z_mm=float(pivot * 1000), axis_minima_mm=separate, **summary(value)))
    slopes = np.vstack([spline(target), spline(target, nu=1)])
    trace.set_drive(np.zeros(4))
    return dict(
        command_urad=1.0,
        mix=mix.tolist(),
        candidates=fits,
        front_focus_response=summary(slopes),
        physical_A_per_command_urad=(trace.current_matrix @ mix * 0.001).tolist(),
        full_rank_at_surface=summary(response[-1]),
        current_A=m.t.current * m.t.weights[masks[-1]].sum(),
        even_max_position_nm=float(np.max(np.abs(even[:, :2])) * 1e9),
        specimen_surviving=int(masks[-1].sum()),
        sampled_mm=(planes[[0, -1]] * 1000).tolist(),
    )


def baseline():
    m, q, ref = operating_point()
    mix = np.array(ref["after"]["downstream_calibration"]["station_command_matrix"])
    front = focal_planes(m, q)
    print("Baseline front focal candidates (mm)", front, flush=True)
    target = min(front["mini_and_objective"], key=lambda z: abs(z - 1490))
    result = scan_response(m, q, mix, target)
    save("baseline", dict(front_mm=front, pivot=result))
    print("Measured pivots", result["candidates"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline"])
    args = parser.parse_args()
    baseline()
