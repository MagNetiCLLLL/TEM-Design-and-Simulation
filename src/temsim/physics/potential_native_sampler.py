"""Opt-in strict native potential sampling, without a different physical model.

Only frozen high-order Gaussian round fields and compact native Az multipoles
are admitted. All coefficients and registrations are detached, ordered and
bound to the snapshot numerical digest. Numba uses strict floating point, no
parallel reduction and no fastmath. This replaces sampling overhead only;
integrator tolerances, field order, gauge, ownership and domain gates remain.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


BACKEND = "numba-native-strict"
SCHEMA = "frozen-native-potential-sampling-v1"


def _frozen(values, dtype=float):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@njit(cache=True, fastmath=False)
def _sample_values(points, selected, packed):
    """Allocation-free inner loop; identical local Z reuses exact derivatives."""
    (kinds, rotations, origins, radii, limits, domains, orders, gaussian_offsets,
     gaussian_terms, polynomial_offsets, polynomial_terms, envelopes,
     centers, sigmas, coefficients) = packed
    potential, magnetic = np.zeros((len(points), 3)), np.zeros((len(points), 3))
    cached_z = np.full(len(kinds), np.nan)
    axial = np.zeros((len(kinds), 10))
    for point_index in range(len(points)):
        for source in range(len(kinds)):
            if not selected[source]:
                continue
            rotation = rotations[source]
            dx, dy, dz = (points[point_index, 0]-origins[source, 0],
                          points[point_index, 1]-origins[source, 1],
                          points[point_index, 2]-origins[source, 2])
            x = dx*rotation[0, 0]+dy*rotation[1, 0]+dz*rotation[2, 0]
            y = dx*rotation[0, 1]+dy*rotation[1, 1]+dz*rotation[2, 1]
            z = dx*rotation[0, 2]+dy*rotation[1, 2]+dz*rotation[2, 2]
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return potential, magnetic, source, 3
            lower, upper = limits[source]
            if kinds[source] == 1:
                if z < lower or z > upper:
                    continue
            elif domains[source] == 1 and (z <= lower or z >= upper):
                return potential, magnetic, source, 1
            if math.hypot(x, y) > radii[source]:
                return potential, magnetic, source, 1
            if kinds[source] == 0:
                order = orders[source]
                maximum = 2*order+1
                if z != cached_z[source]:
                    for degree in range(maximum+1):
                        axial[source, degree] = 0.
                    for term in range(gaussian_offsets[source], gaussian_offsets[source+1]):
                        amplitude, center, sigma = gaussian_terms[term]
                        u = (z-center)/sigma
                        gaussian = amplitude*math.exp(-.5*u*u)
                        previous, current = 1., u
                        axial[source, 0] += gaussian
                        for degree in range(1, maximum+1):
                            axial[source, degree] += (-1.)**degree/sigma**degree*current*gaussian
                            previous, current = current, u*current-degree*previous
                    cached_z[source] = z
                squared = x*x+y*y
                f, fs, fz = 0., 0., 0.
                for n in range(order+1):
                    coefficient = coefficients[n]
                    power = squared**n
                    f += coefficient*power*axial[source, 2*n]
                    fz += coefficient*power*axial[source, 2*n+1]
                    if n:
                        fs += n*coefficient*squared**(n-1)*axial[source, 2*n]
                a0, a1, a2 = -y*f, x*f, 0.
                b0, b1, b2 = -x*fz, -y*fz, 2*f+2*squared*fs
            else:
                envelope = 1.
                if envelopes[source]:
                    u = (z-centers[source])/sigmas[source]
                    envelope = math.exp(-.5*u*u)
                p, px, py = 0., 0., 0.
                for term in range(polynomial_offsets[source], polynomial_offsets[source+1]):
                    ix, iy, value = polynomial_terms[term]
                    ix, iy = int(ix), int(iy)
                    p += value*x**ix*y**iy
                    if ix:
                        px += ix*value*x**(ix-1)*y**iy
                    if iy:
                        py += iy*value*x**ix*y**(iy-1)
                a0, a1, a2 = 0., 0., p*envelope
                b0, b1, b2 = py*envelope, -px*envelope, 0.
            if not (math.isfinite(a0) and math.isfinite(a1) and math.isfinite(a2)
                    and math.isfinite(b0) and math.isfinite(b1) and math.isfinite(b2)):
                return potential, magnetic, source, 3
            for axis in range(3):
                potential[point_index, axis] += rotation[axis, 0]*a0+rotation[axis, 1]*a1+rotation[axis, 2]*a2
                magnetic[point_index, axis] += rotation[axis, 0]*b0+rotation[axis, 1]*b1+rotation[axis, 2]*b2
    return potential, magnetic, -1, 0


@njit(cache=True, fastmath=False)
def _evaluate(points, selected, packed, derivatives):
    (kinds, rotations, origins, radii, limits, domains, orders, gaussian_offsets,
     gaussian_terms, polynomial_offsets, polynomial_terms, envelopes,
     centers, sigmas, coefficients) = packed
    count = len(points)
    potential = np.zeros((count, 3))
    magnetic = np.zeros((count, 3))
    jacobian = np.zeros((count if derivatives else 0, 3, 3))
    hessian = np.zeros((count if derivatives else 0, 3, 3, 3))
    for point_index in range(count):
        for source in range(len(kinds)):
            if not selected[source]:
                continue
            rotation = rotations[source]
            local = np.zeros(3)
            for axis in range(3):
                for coordinate in range(3):
                    local[axis] += (points[point_index, coordinate]-origins[source, coordinate])*rotation[coordinate, axis]
            if not np.isfinite(local).all():
                return potential, magnetic, jacobian, hessian, source, 3
            x, y, z = local
            lower, upper = limits[source]
            if kinds[source] == 1:
                if derivatives and (z == lower or z == upper):
                    return potential, magnetic, jacobian, hessian, source, 2
                if z < lower or z > upper:
                    continue  # Known compact zero, without evaluating a radial polynomial.
            elif domains[source] == 1 and (z <= lower or z >= upper):
                return potential, magnetic, jacobian, hessian, source, 1
            if math.hypot(x, y) > radii[source]:
                return potential, magnetic, jacobian, hessian, source, 1
            local_a, local_b = np.zeros(3), np.zeros(3)
            local_jac, local_hess = np.zeros((3, 3)), np.zeros((3, 3, 3))
            if kinds[source] == 0:
                order = orders[source]
                maximum = 2*order+(2 if derivatives else 1)
                axial = np.zeros(maximum+1)
                for term in range(gaussian_offsets[source], gaussian_offsets[source+1]):
                    amplitude, center, sigma = gaussian_terms[term]
                    u = (z-center)/sigma
                    gaussian = amplitude*math.exp(-.5*u*u)
                    previous, current = 1., u
                    axial[0] += gaussian
                    for degree in range(1, maximum+1):
                        axial[degree] += (-1.)**degree/sigma**degree*current*gaussian
                        previous, current = current, u*current-degree*previous
                squared = x*x+y*y
                f, fs, fss, fz, fsz, fzz = 0., 0., 0., 0., 0., 0.
                for n in range(order+1):
                    coefficient = coefficients[n]
                    power = squared**n
                    f += coefficient*power*axial[2*n]
                    fz += coefficient*power*axial[2*n+1]
                    if n:
                        fs += n*coefficient*squared**(n-1)*axial[2*n]
                    if derivatives:
                        if n:
                            fsz += n*coefficient*squared**(n-1)*axial[2*n+1]
                        if n >= 2:
                            fss += n*(n-1)*coefficient*squared**(n-2)*axial[2*n]
                        fzz += coefficient*power*axial[2*n+2]
                local_a[0], local_a[1] = -y*f, x*f
                local_b[0], local_b[1], local_b[2] = -x*fz, -y*fz, 2*f+2*squared*fs
                if derivatives:
                    direction = np.array((-y, x, 0.))
                    cross = np.zeros((3, 3)); cross[0, 1], cross[1, 0] = -1., 1.
                    gradient = np.array((2*x*fs, 2*y*fs, fz))
                    second = np.empty((3, 3))
                    second[0, 0], second[1, 1], second[2, 2] = 2*fs+4*x*x*fss, 2*fs+4*y*y*fss, fzz
                    second[0, 1] = second[1, 0] = 4*x*y*fss
                    second[0, 2] = second[2, 0] = 2*x*fsz
                    second[1, 2] = second[2, 1] = 2*y*fsz
                    for i in range(3):
                        for j in range(3):
                            local_jac[i, j] = cross[i, j]*f+direction[i]*gradient[j]
                            for k in range(3):
                                local_hess[i, j, k] = (cross[i, j]*gradient[k]+cross[i, k]*gradient[j]
                                                       +direction[i]*second[j, k])
            else:
                envelope, first, second = 1., 0., 0.
                if envelopes[source]:
                    u = (z-centers[source])/sigmas[source]
                    envelope = math.exp(-.5*u*u)
                    first = -envelope*u/sigmas[source]
                    second = envelope*(u*u-1)/sigmas[source]**2
                p, px, py, pxx, pxy, pyy = 0., 0., 0., 0., 0., 0.
                for term in range(polynomial_offsets[source], polynomial_offsets[source+1]):
                    ix, iy, value = polynomial_terms[term]
                    ix, iy = int(ix), int(iy)
                    p += value*x**ix*y**iy
                    if ix:
                        px += ix*value*x**(ix-1)*y**iy
                    if iy:
                        py += iy*value*x**ix*y**(iy-1)
                    if derivatives:
                        if ix >= 2:
                            pxx += ix*(ix-1)*value*x**(ix-2)*y**iy
                        if ix and iy:
                            pxy += ix*iy*value*x**(ix-1)*y**(iy-1)
                        if iy >= 2:
                            pyy += iy*(iy-1)*value*x**ix*y**(iy-2)
                local_a[2] = p*envelope
                local_b[0], local_b[1] = py*envelope, -px*envelope
                if derivatives:
                    local_jac[2, 0], local_jac[2, 1], local_jac[2, 2] = px*envelope, py*envelope, p*first
                    local_hess[2, 0, 0], local_hess[2, 1, 1] = pxx*envelope, pyy*envelope
                    local_hess[2, 0, 1] = local_hess[2, 1, 0] = pxy*envelope
                    local_hess[2, 0, 2] = local_hess[2, 2, 0] = px*first
                    local_hess[2, 1, 2] = local_hess[2, 2, 1] = py*first
                    local_hess[2, 2, 2] = p*second
            if not np.isfinite(local_a).all() or not np.isfinite(local_b).all():
                return potential, magnetic, jacobian, hessian, source, 3
            for i in range(3):
                value_a, value_b = 0., 0.
                for a in range(3):
                    value_a += rotation[i, a]*local_a[a]
                    value_b += rotation[i, a]*local_b[a]
                potential[point_index, i] += value_a
                magnetic[point_index, i] += value_b
            if derivatives:
                if not np.isfinite(local_jac).all() or not np.isfinite(local_hess).all():
                    return potential, magnetic, jacobian, hessian, source, 3
                for i in range(3):
                    for j in range(3):
                        value_j = 0.
                        for a in range(3):
                            for b in range(3):
                                value_j += rotation[i, a]*local_jac[a, b]*rotation[j, b]
                        jacobian[point_index, i, j] += value_j
                        for k in range(3):
                            value_h = 0.
                            for a in range(3):
                                for b in range(3):
                                    for c in range(3):
                                        value_h += rotation[i, a]*local_hess[a, b, c]*rotation[j, b]*rotation[k, c]
                            hessian[point_index, i, j, k] += value_h
    return potential, magnetic, jacobian, hessian, -1, 0


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FrozenNativePotentialSampler:
    """Detached compiled coefficients; never infer compatibility from equal B."""
    snapshot_digest: str
    source_ids: tuple[str, ...]
    _packed: tuple
    _gauge_gradient: np.ndarray
    _gauge_hessian: np.ndarray
    _remote_providers: tuple
    backend: str = BACKEND

    def __init__(self, *args, **kwargs):
        raise TypeError("Use compile_native_sampler with a validated potential snapshot")

    def require_snapshot(self, snapshot):
        if type(snapshot) is not ElectromagneticPotentialSnapshot or snapshot.numerical_digest != self.snapshot_digest:
            raise ValueError("Native sampler requires its exact captured potential numerical digest")

    def _sample(self, positions_m, source_ids, *, derivatives):
        raw = np.asarray(positions_m)
        if raw.dtype.kind not in "iuf" or raw.shape[-1:] != (3,) or not np.isfinite(raw).all():
            raise ValueError("Potential coordinates must be finite real XYZ metres")
        points = np.ascontiguousarray(raw, dtype=float)
        selected = np.ones(len(self.source_ids), dtype=np.bool_)
        if source_ids is not None:
            if (not isinstance(source_ids, (tuple, list))
                    or any(type(value) is not str or not value or value != value.strip() for value in source_ids)
                    or len(set(source_ids)) != len(source_ids)):
                raise ValueError("Selected source_ids must be unique nonempty identifiers")
            unknown = set(source_ids)-set(self.source_ids)
            if unknown:
                raise ValueError("Unknown potential source: "+", ".join(sorted(unknown)))
            selected = np.array([value in source_ids for value in self.source_ids], dtype=np.bool_)
        packed=self._packed
        radii=None
        local_cache={}
        for source_index,provider in self._remote_providers:
            if not selected[source_index]:
                continue
            registration=provider.registration
            frame=(registration.origin_global_m,registration.rotation_local_to_global)
            if frame not in local_cache:
                local=registration.positions_global_to_local_m(points)
                local_cache[frame]=(local,np.hypot(local[...,0],local[...,1]))
            local,radius=local_cache[frame]
            outside=radius>provider.valid_radius_m
            if outside.any():
                # The same Python certificate also checks that the unchanged
                # polynomial naturally rounds to zero. Only this call's already
                # certified radial checks are relaxed; coefficients are unchanged.
                provider.sample_jet(points[outside])
                if radii is None:
                    radii=packed[3].copy()
                radii[source_index]=np.inf
        if radii is not None:
            packed=(*packed[:3],radii,*packed[4:])
        if derivatives:
            a, b, jac, hess, source, error = _evaluate(points.reshape(-1, 3), selected, packed, True)
        else:
            a, b, source, error = _sample_values(points.reshape(-1, 3), selected, packed)
        if error:
            reason = {1: "outside the strict native potential domain",
                      2: "Two-sided potential derivatives are undefined on a hard axial face",
                      3: "Native potential evaluation exceeds the finite numerical range"}[error]
            raise ValueError(f"Potential source {self.source_ids[source]}: {reason}")
        a += self._gauge_gradient+points.reshape(-1, 3)@self._gauge_hessian.T
        shape = points.shape[:-1]
        if derivatives:
            jac += self._gauge_hessian
            result = {"A_t_m": a.reshape((*shape, 3)), "jac_A_t": jac.reshape((*shape, 3, 3)),
                      "hess_A_t_per_m": hess.reshape((*shape, 3, 3, 3))}
        else:
            result = {"phi_v": np.zeros(shape), "E_v_per_m": np.zeros_like(points),
                      "A_t_m": a.reshape(points.shape), "B_t": b.reshape(points.shape)}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Combined native potential exceeds the finite numerical range")
        return result

    def sample(self, positions_m, source_ids=None):
        return self._sample(positions_m, source_ids, derivatives=False)

    def sample_jet(self, positions_m, source_ids=None):
        """Exact native coefficient derivatives; hard axial faces are excluded."""
        return self._sample(positions_m, source_ids, derivatives=True)


def compile_native_sampler(snapshot):
    """Capture a strictly selected, static native CPU sampling backend.

    Unsupported providers are rejected, never skipped or converted from B.
    No field order, coefficients, material solution or ownership is altered.
    """
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        raise ValueError("Native sampling requires an immutable potential snapshot")
    kinds, rotations, origins, radii, limits, domains, orders = [], [], [], [], [], [], []
    gaussian_offsets, polynomial_offsets = [0], [0]
    gaussians, polynomials, envelopes, centers, sigmas = [], [], [], [], []
    remote=[]
    for source_index,row in enumerate(snapshot.contributions):
        provider = row.potential
        if row.kind != "magnetic" or type(provider) not in (HighOrderRoundLensPotential, AnalyticPotentialDomain):
            raise ValueError(f"{row.source_id}: unsupported native potential sampling provider")
        # Round-trip through the exact adapter also rejects overridden native
        # laws and detaches caller-owned registrations/provenance.
        provider = type(provider).from_dict(provider.to_dict())
        if type(provider) is HighOrderRoundLensPotential:
            field = provider.native
            kinds.append(0); radii.append(provider.valid_radius_m)
            domains.append(int(provider.axial_domain == "native_support"))
            orders.append((provider.radial_order-1)//2)
            gaussians.extend(field.terms_t_m)
            envelopes.append(0); centers.append(0.); sigmas.append(1.)
            if provider.remote_domain!="strict":
                remote.append((source_index,provider))
        else:
            field = provider.base.field
            if type(field) is not FrozenPosedMultipole:
                raise ValueError(f"{row.source_id}: only compact native Az domains are admitted")
            if field.dynamic or provider.base.provenance.get("dynamic_drive_frozen", False):
                raise ValueError(f"{row.source_id}: frozen dynamic sources are not admitted for static native sampling")
            kinds.append(1); radii.append(field.radial_support_m)
            domains.append(0); orders.append(0)
            polynomials.extend(field.polynomial_terms)
            envelopes.append(int(field.envelope_kind == "gaussian"))
            centers.append(field.center_z_m); sigmas.append(field.sigma_m)
        rotations.append(field.registration.rotation_array)
        origins.append(field.registration.origin_array_m)
        limits.append(np.asarray(field.native_support_mm)*.001)
        gaussian_offsets.append(len(gaussians)); polynomial_offsets.append(len(polynomials))
    packed = (_frozen(kinds, np.int64), _frozen(rotations).reshape(-1, 3, 3), _frozen(origins).reshape(-1, 3),
        _frozen(radii), _frozen(limits).reshape(-1, 2), _frozen(domains, np.int64), _frozen(orders, np.int64),
        _frozen(gaussian_offsets, np.int64), _frozen(gaussians).reshape(-1, 3),
        _frozen(polynomial_offsets, np.int64), _frozen(polynomials).reshape(-1, 3),
        _frozen(envelopes, np.int64), _frozen(centers), _frozen(sigmas),
        _frozen([(-1.)**n/(2.**(2*n+1)*math.factorial(n)*math.factorial(n+1)) for n in range(5)]))
    result = object.__new__(FrozenNativePotentialSampler)
    for key, value in (("snapshot_digest", snapshot.numerical_digest),
                       ("source_ids", tuple(row.source_id for row in snapshot.contributions)),
                       ("_packed", packed), ("_gauge_gradient", _frozen(snapshot.gauge.gradient_t_m)),
                       ("_gauge_hessian", _frozen(snapshot.gauge.hessian_t)), ("backend", BACKEND)):
        object.__setattr__(result, key, value)
    object.__setattr__(result,"_remote_providers",tuple(remote))
    return result


__all__ = ("compile_native_sampler", "FrozenNativePotentialSampler")
