"""Opt-in exact native coefficient aggregation on aligned observation planes.

This is a sampling backend, never a new field model. Common on-axis round
fields are summed as F(r squared,z); aligned compact Az fields are summed as
their existing XY polynomials. Every source retains its own strict domain.
Tilted/off-axis sources and non-planar point sets use the original strict
compiled sampler. No tiny field, coefficient or wave tail is discarded.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.potential_native_sampler import compile_native_sampler


@njit(cache=True, fastmath=False)
def _plane_values(points, radial, axial, polynomial, gradient, hessian):
    potential, magnetic = np.zeros_like(points), np.zeros_like(points)
    for index in range(len(points)):
        x, y, z = points[index]
        squared = x*x+y*y
        f, fz, bz = 0., 0., 0.
        for degree in range(len(radial)-1, -1, -1):
            f = f*squared+radial[degree]
            fz = fz*squared+axial[degree]
            bz = bz*squared+2*(degree+1)*radial[degree]
        az, dx, dy = 0., 0., 0.
        for ix in range(4):
            for iy in range(4-ix):
                value = polynomial[ix, iy]
                az += value*x**ix*y**iy
                if ix:
                    dx += ix*value*x**(ix-1)*y**iy
                if iy:
                    dy += iy*value*x**ix*y**(iy-1)
        potential[index, 0] = -y*f
        potential[index, 1] = x*f
        potential[index, 2] = az
        magnetic[index, 0] = -x*fz+dy
        magnetic[index, 1] = -y*fz-dx
        magnetic[index, 2] = bz
        for axis in range(3):
            potential[index, axis] += (gradient[axis]+hessian[axis, 0]*x
                +hessian[axis, 1]*y+hessian[axis, 2]*z)
    return potential, magnetic


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FrozenPlanePotentialSampler:
    """Detached immutable inputs, with a strict compiled fallback."""
    snapshot: object
    fallback: object
    aligned: bool
    backend: str = "numba-aligned-plane-strict"

    def __init__(self, *args, **kwargs):
        raise TypeError("Use compile_plane_sampler with an immutable potential snapshot")

    @property
    def snapshot_digest(self):
        return self.fallback.snapshot_digest

    def require_snapshot(self, snapshot):
        self.fallback.require_snapshot(snapshot)

    def sample_jet(self, positions_m, source_ids=None):
        return self.fallback.sample_jet(positions_m, source_ids)

    def sample(self, positions_m, source_ids=None):
        raw = np.asarray(positions_m)
        if raw.dtype.kind not in "iuf" or raw.shape[-1:] != (3,) or not np.isfinite(raw).all():
            raise ValueError("Potential coordinates must be finite real XYZ metres")
        points = np.ascontiguousarray(raw, dtype=float).reshape(-1, 3)
        if (source_ids is not None or not self.aligned or not len(points)
                or not np.all(points[:, 2] == points[0, 2])):
            return self.fallback.sample(positions_m, source_ids)
        z = float(points[0, 2])
        maximum_radius = float(np.max(np.hypot(points[:, 0], points[:, 1])))
        radial, axial, polynomial = np.zeros(5), np.zeros(5), np.zeros((4, 4))
        for row in self.snapshot.contributions:
            provider = row.potential
            round_lens = type(provider) is HighOrderRoundLensPotential
            native = provider.native if round_lens else provider.base.field
            local_z = z-native.registration.origin_array_m[2]
            lower, upper = np.asarray(native.native_support_mm)*.001
            if not round_lens and not lower <= local_z <= upper:
                continue  # Explicit compact model zero, never an FEM extrapolation.
            if (round_lens and provider.axial_domain == "native_support"
                    and not lower < local_z < upper):
                raise ValueError(f"Potential source {row.source_id}: outside the strict native potential domain")
            radius = provider.valid_radius_m if round_lens else native.radial_support_m
            if math.isclose(maximum_radius, radius, rel_tol=64*np.finfo(float).eps, abs_tol=0.):
                # Floating rotations can change a norm by one ulp. This is
                # only an algorithm dispatch margin, never a domain tolerance:
                # the original providers decide exact boundary membership.
                return self.snapshot.sample(positions_m)
            if maximum_radius > radius:
                if round_lens and provider.remote_domain!="strict":
                    # The common native path certifies each original local
                    # point and retains the identical R-order evaluation.
                    return self.fallback.sample(positions_m,source_ids)
                raise ValueError(f"Potential source {row.source_id}: outside the strict native potential domain")
            if round_lens:
                derivatives = provider.axial_derivatives(local_z, provider.radial_order)
                for degree in range((provider.radial_order+1)//2):
                    coefficient = (-1.)**degree/(2.**(2*degree+1)*math.factorial(degree)*math.factorial(degree+1))
                    radial[degree] += coefficient*derivatives[2*degree]
                    axial[degree] += coefficient*derivatives[2*degree+1]
            else:
                envelope = (math.exp(-.5*((local_z-native.center_z_m)/native.sigma_m)**2)
                            if native.envelope_kind == "gaussian" else 1.)
                rotation = native.registration.rotation_array
                for ix, iy, value in native.polynomial_terms:
                    for x0 in range(ix+1):
                        for x1 in range(iy+1):
                            polynomial[x0+x1, ix+iy-x0-x1] += (envelope*value
                                *math.comb(ix, x0)*rotation[0, 0]**x0*rotation[1, 0]**(ix-x0)
                                *math.comb(iy, x1)*rotation[0, 1]**x1*rotation[1, 1]**(iy-x1))
        if not all(np.isfinite(value).all() for value in (radial, axial, polynomial)):
            raise ValueError("Aligned-plane coefficient aggregation exceeds the finite numerical range")
        a, b = _plane_values(points, radial, axial, polynomial,
                            self.snapshot.gauge.gradient_t_m, self.snapshot.gauge.hessian_t)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Aligned-plane potential exceeds the finite numerical range")
        return {"A_t_m": a.reshape(raw.shape), "B_t": b.reshape(raw.shape),
                "phi_v": np.zeros(raw.shape[:-1]), "E_v_per_m": np.zeros(raw.shape)}


def compile_plane_sampler(snapshot):
    """Select coefficient aggregation only for exact common +Z axes at X=Y=0."""
    fallback = compile_native_sampler(snapshot)
    aligned = True
    for row in snapshot.contributions:
        provider = row.potential
        native = provider.native if type(provider) is HighOrderRoundLensPotential else provider.base.field
        registration = native.registration
        rotation = registration.rotation_array
        if (np.any(registration.origin_array_m[:2] != 0.)
                or not np.array_equal(rotation[:, 2], (0., 0., 1.))
                or not np.array_equal(rotation[2, :], (0., 0., 1.))
                or np.max(abs(rotation.T@rotation-np.eye(3))) > 8*np.finfo(float).eps
                or abs(np.linalg.det(rotation)-1.) > 8*np.finfo(float).eps):
            aligned = False
    result = object.__new__(FrozenPlanePotentialSampler)
    for name, value in (("snapshot", snapshot), ("fallback", fallback), ("aligned", aligned),
                        ("backend", "numba-aligned-plane-strict")):
        object.__setattr__(result, name, value)
    return result
