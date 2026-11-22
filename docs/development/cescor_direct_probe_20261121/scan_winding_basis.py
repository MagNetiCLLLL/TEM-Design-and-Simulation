"""Sample the installed four AC winding channels on the registered corrector grid."""

from hashlib import sha256
import json

import numpy as np
from numba import njit, prange
from scipy.interpolate import make_interp_spline

from temsim.physics.compiled_transverse_coil import pack_transverse_potential, transverse_b_one
from temsim.physics.transverse_coil_field import TransverseCoilPotential
from temsim.transverse_current import validate_transverse_current_inputs
from temsim.transverse_winding import transverse_winding_definition

from condenser_normal import FIELD
from relay_transfer import PARTS, WORK


@njit(parallel=True)
def sample(points, packed):
    nodes, weights, geometry, axis_zero = packed
    result = np.empty_like(points)
    for i in prange(len(points)):
        ok, value = transverse_b_one(points[i], nodes, weights, geometry, 0, len(nodes),
                                     0, len(geometry), axis_zero)
        result[i] = value if ok else np.full(3, np.nan)
    return result


def grid():
    with np.load(FIELD) as field:
        return field["z_local_m"], field["xy_m"]


def axial_grid(z, centers):
    # Preserve every FEM plane. Resolve the saddle fringes at 0.125 mm while
    # retaining the original piecewise-linear corrector field between planes.
    fine = z[:-1][np.min(np.abs(z[:-1, None] - centers), axis=1) < .075]
    return np.unique(np.r_[z, *(fine + t * (z[1]-z[0]) for t in (.25, .5, .75))])


def expand(data, coarse_z, coarse_xy, z, xy):
    for axis, start, end in ((2, coarse_xy, xy), (1, coarse_xy, xy), (0, coarse_z, z)):
        data = make_interp_spline(start, data, axis=axis)(end)
    return np.ascontiguousarray(data)


def capture():
    part = PARTS["ac_deflector"]
    w = transverse_winding_definition(part)
    electrical = validate_transverse_current_inputs(part)
    if np.any(electrical["current_bias_a"]) or part.get("ideal_transverse_current_segments"):
        raise ValueError("The archived unbiased four-channel AC winding is required")
    z, xy = grid()
    centers = .001 * np.asarray(part["interaction_centers_local_z_mm"])
    distance = np.min(np.abs(z[:, None] - centers), axis=1)
    stride = np.select([distance < .05, distance < .12, distance < .25], [1, 3, 8], default=20)
    index = np.unique(np.r_[0, np.flatnonzero(np.arange(len(z)) % stride == 0), len(z)-1])
    coarse_z, coarse_xy = z[index], xy[::4]
    z = axial_grid(z, centers)
    distance = np.min(np.abs(z[:, None] - centers), axis=1)
    zz, yy, xx = np.meshgrid(coarse_z, coarse_xy, coarse_xy, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    rng = np.random.default_rng(211121)
    checks = np.column_stack((rng.integers(1, len(xy)-1, (192, 2)),
                              np.r_[rng.choice(np.where(distance < .075)[0], 144),
                                    rng.integers(0, len(z)-1, 48)]))
    x_test, y_test, z_test = checks.T
    probes = np.column_stack((xy[x_test], xy[y_test], (z[z_test]+z[z_test+1])*.5))
    basis, errors = [], []
    for station in range(2):
        coils = tuple(c for c in w["coils"] if c["station_index"] == station)
        for axis in range(2):
            potential = TransverseCoilPotential(f"ac_deflector:{station}", coils,
                            w["turn_count_per_coil"], tuple(np.eye(2)[axis]))
            packed = pack_transverse_potential(potential)
            values = sample(points, packed).reshape(*zz.shape, 3)
            exact = potential.field_at_global_positions_t(probes)
            table = expand(values, coarse_z, coarse_xy, z, xy)
            checked = .5*(table[z_test,y_test,x_test] + table[z_test+1,y_test,x_test])
            error = np.linalg.norm(checked-exact, axis=1)
            relative = error / np.linalg.norm(exact, axis=1)
            if not np.isfinite(values).all() or not np.isfinite(relative).all() or relative.max() > .005:
                raise ValueError(f"AC winding interpolation error too large: {relative.max():.4g}")
            basis.append(values)
            errors.append(float(relative.max()))
            print("AC winding", station, axis, "max relative field error", errors[-1], flush=True)
    np.savez_compressed(WORK / "downstream-scan-windings.npz", coarse_z=coarse_z, coarse_xy=coarse_xy,
        basis_T_per_A=basis, command_matrix_A_per_mrad=electrical["matrix_a_per_unit"],
        max_check_relative=errors, checks_per_winding=len(checks),
        winding_sha256=sha256(json.dumps(part, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    capture()
