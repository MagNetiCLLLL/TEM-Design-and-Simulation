"""Evaluate the captured all-Z magnetic gun field on a refined vacuum table.

The production source, electrostatics, time integrator, contact and exit event
solver still execute. The table has a separate execution identity, native
out-of-domain fallback, and off-grid comparisons to the winding evaluator.
"""

from hashlib import sha256
from math import factorial

import numpy as np
from numba import njit
from scipy.interpolate import make_interp_spline

from temsim import input_io
from temsim.physics.gun_field_environment import instrument_gun_field_context
from temsim.vacuum import bind_gun_environment


@njit(cache=True)
def evaluate(points, coefficients, start, dz, radius):
    result = np.zeros((len(points), 3))
    for row in range(len(points)):
        x, y, z = points[row]
        x /= radius
        y /= radius
        i = min(len(coefficients) - 1, max(0, int((z - start) / dz)))
        t = (z - start) / dz - i
        for component in range(3):
            value = 0.0
            for a in range(coefficients.shape[1] - 1, -1, -1):
                v = 0.0
                for b in range(coefficients.shape[2] - 1, -1, -1):
                    p = 0.0
                    for k in range(coefficients.shape[4] - 1, -1, -1):
                        p = p * t + coefficients[i, a, b, component, k]
                    v = v * y + p
                value = value * x + v
            result[row, component] = value
    return result


class GunField:
    def __init__(self, native, stop, spacing_mm):
        self.native = native
        self.start, self.stop, self.radius = -0.001, stop, 0.050e-3
        zz = np.linspace(self.start, stop, int(np.ceil((stop - self.start) / (0.001 * spacing_mm))) + 1)
        self.dz = zz[1] - zz[0]
        xy = np.cos(np.arange(5) * np.pi / 4)
        xx, yy = np.meshgrid(xy, xy, indexing="ij")
        shape = (len(zz), len(xy), len(xy), 3)
        points = np.empty(shape)
        points[..., 0] = self.radius * xx
        points[..., 1] = self.radius * yy
        points[..., 2] = zz[:, None, None]
        p = points.reshape(-1, 3)
        fields = np.concatenate(
            [native.field_at_global_positions_t(part) for part in np.array_split(p, max(1, len(p) // 2000))]
        )
        inverse = np.linalg.inv(np.polynomial.polynomial.polyvander(xy, 4))
        coeff = np.einsum("ai,zijc,bj->zabc", inverse, fields.reshape(shape), inverse)
        spline = make_interp_spline(zz, coeff, k=5)
        self.coef = np.ascontiguousarray(
            np.stack([spline(zz[:-1], nu=k) * self.dz**k / factorial(k) for k in range(6)], axis=-1)
        )
        rng = np.random.default_rng(73001)
        test = rng.uniform(-1.0, 1.0, (2048, 3))
        test[:, :2] *= self.radius
        test[::3, :2] *= 0.01
        test[:, 2] = self.start + (test[:, 2] + 1) / 2 * (self.stop - self.start)
        actual = native.field_at_global_positions_t(test)
        error = np.linalg.norm(actual - evaluate(test, self.coef, self.start, self.dz, self.radius), axis=1)
        self.error = float(np.max(error))
        if np.any(error > 2e-13 + 1e-11 * np.linalg.norm(actual, axis=1)):
            raise ValueError(f"Independent gun winding-field interpolation check failed: {self.error:g} T")
        self.calls = self.fallback_points = 0
        self.max_radius = 0.0

    def field_at_global_positions_t(self, positions):
        points = np.asarray(positions, dtype=float)
        xyz = points.reshape(-1, 3)
        selected = (
            (np.max(abs(xyz[:, :2]), axis=1) <= self.radius)
            & (xyz[:, 2] >= self.start)
            & (xyz[:, 2] <= self.stop)
        )
        out = np.empty_like(xyz)
        if selected.any():
            out[selected] = evaluate(xyz[selected], self.coef, self.start, self.dz, self.radius)
        if not selected.all():
            out[~selected] = self.native.field_at_global_positions_t(xyz[~selected])
            self.fallback_points += int((~selected).sum())
        self.calls += 1
        self.max_radius = max(self.max_radius, float(np.max(np.hypot(xyz[:, 0], xyz[:, 1]), initial=0.0)))
        return out.reshape(points.shape)

    def receipt(self):
        return dict(
            schema="study-magnetic-gun-table-v1",
            coefficient_sha256=sha256(self.coef).hexdigest(),
            z_bounds_m=[self.start, self.stop],
            spacing_mm=1000 * self.dz,
            xy_halfwidth_m=self.radius,
            check_points=2048,
            max_check_error_t=self.error,
            evaluation_calls=self.calls,
            native_fallback_points=self.fallback_points,
            maximum_executed_radius_m=self.max_radius,
        )


@input_io.using_state_inputs
def trace_gun(state, spacing_mm=0.5):
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source

    gun = state.electron_gun
    require_physical_gun_source(gun)
    bind_gun_environment(state)
    with instrument_gun_field_context(state) as native:
        table = GunField(native, (gun.exit_plane_z_mm + 3) * 0.001, spacing_mm)
        print("Gun winding-field holdout maximum (T)", table.error, flush=True)
        gun._instrument_magnetic_field = table
        gun._instrument_magnetic_identity = None  # Never populate or consult the native transport cache.
        result = gun.trace_to_exit()
        return result, table.receipt()
