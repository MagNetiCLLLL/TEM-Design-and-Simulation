"""Locate real illumination-side objective coma pivots and measure both transverse components."""

import argparse
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

from column_current import SURFACE, fields
from particles import Tracker
from relay_transfer import REF, advance, save
import cupy as cp


def optical_plane(step_mm, entry, ring, displacement, slope, objective_ni):
    # The optical target is determined solely by the working objective at the
    # actual specimen coordinate. No HP1/HP2 coordinates occur in this calculation.
    grid = np.unique(np.r_[np.arange(max(entry + 0.0001, 1.30), SURFACE, 0.0002), SURFACE])
    mesh, _, _, _ = fields(step_mm, np.r_[entry, grid], objective_ni)
    t = Tracker(mesh)
    owners = np.where(mesh.owner >= 0, mesh.keys[mesh.owner], "")
    objective_bore = np.char.find(owners.astype(str), "objective") >= 0
    t.rad = cp.asarray(np.where(objective_bore, mesh.radius**2, np.inf))
    t.ep.fill(0)
    only = mesh.base("armco", objective_ni, "gap-fine", 8) - mesh.group[0]
    f = cp.asarray(np.column_stack((only, np.zeros_like(mesh.hex))))

    def trace(phase, zs):
        n = len(phase)
        states = advance(
            t,
            f,
            np.column_stack((phase, np.zeros(n))),
            np.full(n, 300000.0),
            np.zeros(n),
            np.full(n, -1, dtype=np.int32),
            np.searchsorted(mesh.z, entry),
            np.searchsorted(mesh.z, zs),
        )
        if any(np.any(loss != -1) for _, loss in states):
            raise ValueError("An objective calibration ray touched hardware")
        return np.array([p[:, :4] for p, _ in states])

    dx = np.array([1e-7, 1e-7, 1e-6, 1e-6])
    basis = np.diag(dx)
    calibration = trace(np.vstack((basis, -basis)), grid)
    transfer = ((calibration[:, :4] - calibration[:, 4:]) / (2 * dx[:, None])).transpose(0, 2, 1)
    inv = np.linalg.inv(transfer[-1])
    backward = transfer @ inv
    phi = np.arange(24) * 2 * np.pi / 24
    # Define pupil azimuths and scan height in Cartesian specimen coordinates.
    # Use the ring average of the odd-in-field, even-in-pupil position error.
    # Its quadratic coefficient is proportional to the coma vector.
    pupil = np.zeros((len(phi) + 1, 4))
    pupil[1:, 2:4] = ring * np.column_stack((np.cos(phi), np.sin(phi)))
    steps = np.array([displacement, displacement, slope, slope])
    field = np.r_[np.diag(steps), -np.diag(steps)]
    rays = ((pupil[None, :, :] + field[:, None, :]) @ inv.T).reshape(-1, 4)
    spot = trace(rays, [SURFACE])[0, :, :2].reshape(8, len(pupil), 2)
    response = (spot[:, 1:].mean(axis=1) - spot[:, 0]) / ring**2
    response = ((response[:4] - response[4:]) / (2 * steps[:, None])).T
    valid = grid < SURFACE - 0.0003
    zs = grid[valid]
    transforms = backward[valid]

    def evaluate(matrix):
        # A central field ray reaches unit specimen displacement with a slope
        # set by its crossing of the axis at this candidate plane.
        tilt = -np.linalg.solve(matrix[:2, 2:4], matrix[:2, :2])
        return response[:, :2] + response[:, 2:] @ tilt

    coefficients = np.array([evaluate(m) for m in transforms])
    radial = np.trace(coefficients, axis1=1, axis2=2) / 2
    skew = (coefficients[:, 1, 0] - coefficients[:, 0, 1]) / 2
    tilt_response = complex(np.trace(response[:, 2:]) / 2, (response[1, 2] - response[0, 3]) / 2)
    # A real axial pivot changes meridional tilt. The perpendicular component
    # of the coma vector is invariant to this tilt and stays finite even when
    # the real coma coefficient has a pole at a conjugate plane.
    perpendicular = ((radial + 1j * skew) * np.conj(tilt_response) / abs(tilt_response)).imag
    invariant = np.median(perpendicular)
    scatter = np.diff(np.percentile(perpendicular, [10, 90])).item()
    # Interpolate the smooth transfer map before forming its pivot quotient.
    # The quotient also changes sign at a singular conjugate and cannot be
    # interpolated as if it were continuous there.
    transfer_at = CubicSpline(zs, transforms, axis=0)

    def radial_at(z):
        return np.trace(evaluate(transfer_at(z))) / 2

    brackets = np.flatnonzero(radial[:-1] * radial[1:] < 0)
    roots = []
    for i in brackets:
        # A change of sign across a pole of the imaging condition is not a coma zero.
        root = brentq(radial_at, zs[i], zs[i + 1], xtol=2e-14)
        coefficient = evaluate(transfer_at(root))
        at_root = complex(np.trace(coefficient) / 2, (coefficient[1, 0] - coefficient[0, 1]) / 2)
        consistency = (at_root * np.conj(tilt_response) / abs(tilt_response)).imag - invariant
        if abs(at_root.real) > 0.001 or abs(consistency) > max(0.03, 5 * scatter):
            continue
        roots.append(dict(z_mm=1000 * root, skew=at_root.imag, radial_bracket=radial[i : i + 2].tolist()))
    return dict(
        z_mm=(1000 * zs).tolist(),
        radial_coma=radial.tolist(),
        skew_coma=skew.tolist(),
        perpendicular_median=float(np.median(perpendicular)),
        perpendicular_p10_p90=np.percentile(perpendicular, [10, 90]).tolist(),
        coma_response=response.tolist(),
        roots=roots,
        specimen_z_mm=1000 * SURFACE,
        objective_At=objective_ni,
        entry_m=entry,
        ring_mrad=1000 * ring,
        scan_um=displacement * 1e6,
        tilt_mrad=slope * 1000,
        step_mm=step_mm,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--step", type=float, default=0.075)
    p.add_argument("--entry", type=float, default=0.45)
    p.add_argument("--ring", type=float, default=0.00024)
    p.add_argument("--amplitude", type=float, default=0.06)
    p.add_argument("--objective-ni", type=float, default=REF["objective_At"])
    p.add_argument("--null-bounds", type=float, nargs=2, metavar=("LOW_AT", "HIGH_AT"))
    a = p.parse_args()

    def at(ni):
        return optical_plane(a.step, a.entry, a.ring, a.amplitude * 1e-6, a.amplitude * 1e-4, ni)

    if a.null_bounds:
        cache = {}

        def sample(strength):
            key = round(float(strength), 1)
            if key not in cache:
                cache[key] = at(-key)
                print(key, cache[key]["perpendicular_median"], cache[key]["roots"], flush=True)
            return cache[key]["perpendicular_median"]

        brentq(sample, *a.null_bounds, xtol=0.11)
        result = min(cache.values(), key=lambda v: abs(v["perpendicular_median"]))
        if not result["roots"]:
            raise ValueError(
                "The vector response has no confirmed finite radial pupil in the search interval"
            )
    else:
        result = at(a.objective_ni)
    print(result["roots"], min(result["radial_coma"]), max(result["radial_coma"]), flush=True)
    save(a.name, result)
