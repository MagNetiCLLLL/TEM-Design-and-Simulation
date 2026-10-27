"""Opt-in exact native coefficient aggregation on aligned observation planes.

This is a sampling backend, never a new field model. Common on-axis round
fields are summed as F(r squared,z); aligned compact Az fields are summed as
their existing XY polynomials. Every source retains its own strict domain.
Tilted/off-axis sources and non-planar point sets use the original strict
compiled sampler. No tiny field, coefficient or wave tail is discarded.
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
from threading import RLock

import numpy as np
from numba import njit

from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.potential_native_sampler import compile_native_sampler, _frozen


_MAXIMUM_CACHED_PLANES = 16
# These are dispatch bounds, not field cutoffs. Outside this conservative
# finite-arithmetic certificate the original ordered provider sum is retained.
_COEFFICIENT_CERTIFICATE_LIMIT = 1e100


@njit(cache=True, fastmath=False)
def _plane_coefficients(z, packed, projected_offsets, projected_terms):
    """Evaluate the retained axial laws once per source and exact global Z."""
    (kinds, rotations, origins, radii, limits, domains, orders, gaussian_offsets,
     gaussian_terms, polynomial_offsets, polynomial_terms, envelopes,
     centers, sigmas, coefficients) = packed
    radial, axial, polynomial = np.zeros(5), np.zeros(5), np.zeros((4, 4))
    active = np.zeros(len(kinds), dtype=np.bool_)
    safe = True
    for source in range(len(kinds)):
        local_z = z-origins[source, 2]
        lower, upper = limits[source]
        if kinds[source] == 1:
            if local_z < lower or local_z > upper:
                continue
        elif domains[source] == 1 and (local_z <= lower or local_z >= upper):
            return radial, axial, polynomial, active, safe, source
        active[source] = True
        if kinds[source] == 0:
            maximum = 2*orders[source]+1
            derivatives = np.zeros(10)
            for term in range(gaussian_offsets[source], gaussian_offsets[source+1]):
                amplitude, center, sigma = gaussian_terms[term]
                u = (local_z-center)/sigma
                gaussian = amplitude*math.exp(-.5*u*u)
                previous, current = 1., u
                derivatives[0] += gaussian
                for degree in range(1, maximum+1):
                    derivatives[degree] += (-1.)**degree/sigma**degree*current*gaussian
                    previous, current = current, u*current-degree*previous
            for degree in range(orders[source]+1):
                radial[degree] += coefficients[degree]*derivatives[2*degree]
                axial[degree] += coefficients[degree]*derivatives[2*degree+1]
            for degree in range(maximum+1):
                if not math.isfinite(derivatives[degree]) or abs(derivatives[degree]) > _COEFFICIENT_CERTIFICATE_LIMIT:
                    safe = False
        else:
            envelope = 1.
            if envelopes[source]:
                u = (local_z-centers[source])/sigmas[source]
                envelope = math.exp(-.5*u*u)
            for term in range(projected_offsets[source], projected_offsets[source+1]):
                row = projected_terms[term]
                value = envelope*row[2]
                # Keep the existing scalar polynomial expansion order. Only
                # its Z-independent factors were captured ahead of sampling.
                for factor in range(3, 9):
                    value *= row[factor]
                polynomial[int(row[0]), int(row[1])] += value
    return radial, axial, polynomial, active, safe, -1


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
    _projected_offsets: np.ndarray
    _projected_terms: np.ndarray
    _remote_domain: tuple
    _static_certificate: bool
    _coefficient_cache: object
    _cache_lock: object
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

    def _coefficients(self, z):
        # The sampler belongs to one detached numerical snapshot. Distinct
        # floats, including nextafter one-sided RK faces, keep distinct keys.
        with self._cache_lock:
            previous = self._coefficient_cache.get(z)
            if previous is not None:
                self._coefficient_cache.move_to_end(z)
                return previous
        try:
            radial, axial, polynomial, active, safe, invalid = _plane_coefficients(
                z, self.fallback._packed, self._projected_offsets, self._projected_terms)
        except (OverflowError, ZeroDivisionError):
            return None  # Let the original ordered evaluator decide admission.
        if invalid >= 0:
            raise ValueError(f"Potential source {self.fallback.source_ids[invalid]}: outside the strict native potential domain")
        safe = safe and all(np.isfinite(value).all() for value in (radial, axial, polynomial))
        result = (_frozen(radial), _frozen(axial), _frozen(polynomial), _frozen(active, bool), safe)
        with self._cache_lock:
            result = self._coefficient_cache.setdefault(z, result)
            self._coefficient_cache.move_to_end(z)
            while len(self._coefficient_cache) > _MAXIMUM_CACHED_PLANES:
                self._coefficient_cache.popitem(last=False)
        return result

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
        if (not self._static_certificate or maximum_radius > 1.
                or abs(z) > _COEFFICIENT_CERTIFICATE_LIMIT):
            return self.fallback.sample(positions_m, source_ids)
        coefficients = self._coefficients(z)
        if coefficients is None or not coefficients[-1]:
            # Aggregation must not turn individually overflowing opposite
            # fields into an apparently admissible zero. For radius <= 1 m,
            # bounded native derivatives/polynomials and gauge coefficients
            # certify all ordered field intermediates far below float64 max.
            return self.fallback.sample(positions_m, source_ids)
        radial, axial, polynomial, active, _ = coefficients
        for index, radius in enumerate(self.fallback._packed[3]):
            if not active[index]:
                continue
            if math.isclose(maximum_radius, radius, rel_tol=64*np.finfo(float).eps, abs_tol=0.):
                # Floating rotations can change a norm by one ulp. This is
                # only an algorithm dispatch margin, never a domain tolerance:
                # the original providers decide exact boundary membership.
                return self.snapshot.sample(positions_m)
            if maximum_radius > radius:
                if self._remote_domain[index]:
                    # The common native path certifies each original local
                    # point and retains the identical R-order evaluation.
                    return self.fallback.sample(positions_m,source_ids)
                raise ValueError(f"Potential source {self.fallback.source_ids[index]}: outside the strict native potential domain")
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
    projected_offsets, projected_terms, remote_domain = [0], [], []
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
        round_lens = type(provider) is HighOrderRoundLensPotential
        remote_domain.append(round_lens and provider.remote_domain != "strict")
        if not round_lens:
            for ix, iy, value in native.polynomial_terms:
                for x0 in range(ix+1):
                    for x1 in range(iy+1):
                        projected_terms.append((x0+x1, ix+iy-x0-x1, value,
                            math.comb(ix, x0), rotation[0, 0]**x0, rotation[1, 0]**(ix-x0),
                            math.comb(iy, x1), rotation[0, 1]**x1, rotation[1, 1]**(iy-x1)))
        projected_offsets.append(len(projected_terms))
    result = object.__new__(FrozenPlanePotentialSampler)
    static_certificate = (np.all(np.abs(fallback._packed[6]) <= 4)
        and np.all(np.abs(fallback._packed[10][:, 2]) <= _COEFFICIENT_CERTIFICATE_LIMIT)
        and np.all(np.abs(snapshot.gauge.gradient_t_m) <= _COEFFICIENT_CERTIFICATE_LIMIT)
        and np.all(np.abs(snapshot.gauge.hessian_t) <= _COEFFICIENT_CERTIFICATE_LIMIT))
    for name, value in (("snapshot", snapshot), ("fallback", fallback), ("aligned", aligned),
                        ("backend", "numba-aligned-plane-strict"),
                        ("_projected_offsets", _frozen(projected_offsets, np.int64)),
                        ("_projected_terms", _frozen(np.asarray(projected_terms).reshape(-1, 9))),
                        ("_remote_domain", tuple(remote_domain)),
                        ("_static_certificate", bool(static_certificate)),
                        ("_coefficient_cache", OrderedDict()), ("_cache_lock", RLock())):
        object.__setattr__(result, name, value)
    return result
