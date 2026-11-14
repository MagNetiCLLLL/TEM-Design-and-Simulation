"""Air-core finite annular winding, integrated from circular current loops.

The rectangular winding cross-section has uniform azimuthal current density.
Copper and the surrounding space have relative permeability one; pole pieces,
ferromagnetic enhancement, saturation and empirical aberrations are absent.
Gauss-Legendre quadrature integrates radius and axial position. Each circular
loop uses complete elliptic integrals, with a regular near-axis expansion only
where its first omitted term is below binary64 precision. The full axial tails
are retained. The winding volume is excluded from vacuum particle queries.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
from numbers import Real

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0
from scipy.special import ellipe, ellipk

from temsim.immutable_json import json_digest
from temsim.physics.lens_field_provider import CoordinateRegistration


SCHEMA = "air-core-annular-coil-potential-v1"
_CROSS = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 0.)))


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _axis_polynomials():
    rows = [np.array([1.])]
    for n in range(12):
        rows.append(poly.polysub(poly.polymul([1., 0., 1.], poly.polyder(rows[-1])),
                                  poly.polymul([0., 3.+2*n], rows[-1])))
    return tuple(rows)


_AXIS_POLYNOMIALS = _axis_polynomials()
# H(m)=((1-m/2)K(m)-E(m)); its m**2 factor is removed before evaluation.
_K_COEFFICIENTS = [(math.comb(2*n, n)/4.**n)**2 for n in range(26)]
_H_COEFFICIENTS = np.array([np.pi/2*(_K_COEFFICIENTS[n]*(1+1/(2*n-1))
                                     -.5*_K_COEFFICIENTS[n-1]) for n in range(2, 26)])


def _elliptic_h(m):
    """H/m**2 and its first two derivatives without small-m subtraction."""
    out = [np.empty_like(m) for _ in range(3)]
    small = m < .05
    for n in range(3):
        out[n][small] = poly.polyval(m[small], poly.polyder(_H_COEFFICIENTS, n))
    t = m[~small]
    if t.size:
        k, e = ellipk(t), ellipe(t)
        kp = (e/(1-t)-k)/(2*t)
        ep = (e-k)/(2*t)
        kpp = (k/4-(1-2*t)*kp)/(t*(1-t))
        epp = (ep-kp)/(2*t)-(e-k)/(2*t*t)
        h = (1-t/2)*k-e
        hp = -k/2+(1-t/2)*kp-ep
        hpp = -kp+(1-t/2)*kpp-epp
        out[0][~small] = h/t**2
        out[1][~small] = hp/t**2-2*h/t**3
        out[2][~small] = hpp/t**2-4*hp/t**3+6*h/t**4
    return out


def _multiply(a, b):
    v, r, z, rr, rz, zz = a
    w, s, t, ss, st, tt = b
    return (v*w, r*w+v*s, z*w+v*t, rr*w+2*r*s+v*ss,
            rz*w+r*t+z*s+v*st, zz*w+2*z*t+v*tt)


def _compose(a, value, first, second):
    _, r, z, rr, rz, zz = a
    return (value, first*r, first*z, first*rr+second*r*r,
            first*rz+second*r*z, first*zz+second*z*z)


def _power(a, exponent):
    v = a[0]
    return _compose(a, v**exponent, exponent*v**(exponent-1),
                    exponent*(exponent-1)*v**(exponent-2))


@dataclass(frozen=True, slots=True)
class AnnularCoilPotential:
    source_id: str
    inner_radius_m: float
    outer_radius_m: float
    length_m: float
    ampere_turns: float
    center_z_m: float = 0.
    registration: CoordinateRegistration = CoordinateRegistration()
    quadrature_order: int = 12
    digest: str = field(init=False)
    _loops: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id or self.source_id.strip() != self.source_id:
            raise ValueError("Annular coil source_id must be a nonempty trimmed string")
        for name in ("inner_radius_m", "outer_radius_m", "length_m", "ampere_turns", "center_z_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not 0 < self.inner_radius_m < self.outer_radius_m or self.length_m <= 0:
            raise ValueError("Annular winding requires 0 < inner radius < outer radius and positive length")
        if type(self.registration) is not CoordinateRegistration:
            raise ValueError("Annular winding requires a rigid CoordinateRegistration")
        if type(self.quadrature_order) is not int or not 4 <= self.quadrature_order <= 64:
            raise ValueError("Annular winding quadrature order must be an integer in [4, 64]")
        nodes, weights = leggauss(self.quadrature_order)
        radii = self.inner_radius_m+(nodes+1)*(self.outer_radius_m-self.inner_radius_m)/2
        axial = self.center_z_m+nodes*self.length_m/2
        loops = tuple((float(a), float(z), float(self.ampere_turns*wa*wz/4))
                      for a, wa in zip(radii, weights) for z, wz in zip(axial, weights))
        object.__setattr__(self, "_loops", loops)
        object.__setattr__(self, "digest", json_digest(self.to_metadata()))

    @property
    def lens_key(self):
        return self.source_id

    @property
    def fingerprint(self):
        return self.digest

    def to_metadata(self):
        return dict(schema=SCHEMA, source_id=self.source_id, inner_radius_m=self.inner_radius_m,
            outer_radius_m=self.outer_radius_m, length_m=self.length_m, ampere_turns=self.ampere_turns,
            center_z_m=self.center_z_m, registration=asdict(self.registration),
            quadrature_order=self.quadrature_order, permeability_h_per_m=float(mu_0), relative_permeability=1.,
            current_law="Uniform azimuthal ampere-turn density over the radial-axial winding rectangle",
            method="Tensor Gauss-Legendre circular loops; complete elliptic-integral vector potential",
            axial_domain="all_z", domain="Finite vacuum XYZ excluding closed winding volume",
            gauge="Coulomb; local A=(-y*f,x*f,0)", derivative_order=2,
            units=dict(position="m", A="T m", B="T", ampere_turns="A turn"),
            empirical_aberrations_applied=False, pole_piece_magnetism=False, wave_admission=False,
            numerical_scope="Quadrature convergence must be tested; no geometric iron-field or coherent qualification")

    def to_dict(self):
        return dict(metadata=self.to_metadata(), digest=self.digest)

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"metadata", "digest"}:
            raise ValueError("Invalid annular coil snapshot")
        m = record["metadata"]
        if not isinstance(m, Mapping) or m.get("schema") != SCHEMA:
            raise ValueError("Unknown annular coil schema")
        try:
            result = cls(**{key: m[key] for key in ("source_id", "inner_radius_m", "outer_radius_m", "length_m",
                "ampere_turns", "center_z_m", "quadrature_order")}, registration=CoordinateRegistration(**m["registration"]))
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid annular coil coefficients") from error
        if result.digest != record["digest"] or json_digest(m) != result.digest:
            raise ValueError("Annular coil snapshot identity mismatch")
        return result

    def _local(self, positions_m):
        points = np.asarray(positions_m)
        if points.dtype.kind not in "iuf" or points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Annular coil positions must be finite real XYZ metres")
        local = self.registration.positions_global_to_local_m(np.asarray(points, float))
        radius = np.hypot(local[..., 0], local[..., 1])
        inside = ((radius >= self.inner_radius_m) & (radius <= self.outer_radius_m)
                  & (np.abs(local[..., 2]-self.center_z_m) <= self.length_m/2))
        if inside.any():
            raise ValueError("Vacuum field query intersects the excluded copper winding volume")
        return local

    def require_global_box(self, positions_m):
        """Conservatively reject a local enclosing box intersecting copper."""
        points = np.asarray(positions_m, float).reshape(-1, 3)
        if not points.size or not np.isfinite(points).all():
            raise ValueError("Annular coil swept box requires finite XYZ points")
        from itertools import product
        global_corners = np.array(tuple(product(*zip(points.min(axis=0), points.max(axis=0)))))
        local = self.registration.positions_global_to_local_m(global_corners)
        low, high = local.min(axis=0), local.max(axis=0)
        nearest = np.maximum(np.maximum(low[:2], -high[:2]), 0.)
        furthest = np.maximum(np.abs(low[:2]), np.abs(high[:2]))
        if (high[2] >= self.center_z_m-self.length_m/2 and low[2] <= self.center_z_m+self.length_m/2
                and np.linalg.norm(nearest) <= self.outer_radius_m and np.linalg.norm(furthest) >= self.inner_radius_m):
            raise ValueError("Annular coil swept box may intersect excluded copper winding")

    def axial_derivatives(self, local_z_m, maximum_order=0):
        if type(maximum_order) is not int or not 0 <= maximum_order <= 12:
            raise ValueError("Annular axis derivative order must be an integer in [0, 12]")
        z = np.asarray(local_z_m)
        if z.dtype.kind not in "iuf" or not np.isfinite(z).all():
            raise ValueError("Annular axis coordinates must be finite real metres")
        z = np.asarray(z, float)
        unique, inverse = np.unique(z, return_inverse=True)
        values = np.zeros((maximum_order+1, len(unique)))
        a, centre, current = np.asarray(self._loops).T
        for start in range(0, len(unique), 512):
            u = (unique[start:start+512, None]-centre)/a
            for n in range(maximum_order+1):
                values[n, start:start+512] = np.sum(mu_0*current/(2*a**(n+1))
                    *poly.polyval(u, _AXIS_POLYNOMIALS[n])/(1+u*u)**(n+1.5), axis=1)
        result = values[:, inverse].reshape((maximum_order+1, *z.shape))
        if not np.isfinite(result).all():
            raise ValueError("Annular axis derivative exceeded finite numerical range")
        return result

    def axis_bz_t(self, local_z_m):
        return self.axial_derivatives(local_z_m, 0)[0]

    def _near_axis(self, local):
        distance_z = np.maximum(np.abs(local[:, 2]-self.center_z_m)-self.length_m/2, 0.)
        return np.hypot(local[:, 0], local[:, 1]) < .01*np.hypot(self.inner_radius_m, distance_z)

    def _axis_coefficients(self, local, derivative_order):
        d = self.axial_derivatives(local[:, 2], 10+derivative_order)
        s = np.sum(local[:, :2]**2, axis=1)
        f, fs, fss, fz, fsz, fzz = (np.zeros(len(local)) for _ in range(6))
        for n in range(6):
            c = (-1.)**n/(2.**(2*n+1)*math.factorial(n)*math.factorial(n+1))
            f += c*s**n*d[2*n]
            if n:
                fs += n*c*s**(n-1)*d[2*n]
            if n >= 2:
                fss += n*(n-1)*c*s**(n-2)*d[2*n]
            if derivative_order:
                fz += c*s**n*d[2*n+1]
                if n:
                    fsz += n*c*s**(n-1)*d[2*n+1]
            if derivative_order >= 2:
                fzz += c*s**n*d[2*n+2]
        return f, fs, fss, fz, fsz, fzz

    def _coefficients(self, local):
        result = np.zeros((6, len(local)))
        near = self._near_axis(local)
        if near.any():
            result[:, near] = self._axis_coefficients(local[near], 2)
        points = local[~near]
        if len(points):
            r, z = np.hypot(points[:, 0], points[:, 1]), points[:, 2]
            total = np.zeros((6, len(points)))
            zero, one = np.zeros_like(r), np.ones_like(r)
            for a, centre, current in self._loops:
                t = z-centre
                q = ((a+r)**2+t*t, 2*(a+r), 2*t, 2*one, zero, 2*one)
                m = _multiply((4*a*r, 4*a*one, zero, zero, zero, zero), _power(q, -1))
                h = _compose(m, *_elliptic_h(m[0]))
                values = _multiply(_power(q, -1.5), h)
                total += (8*mu_0*current*a*a/np.pi)*np.asarray(values)
            f, fr, fz, frr, frz, fzz = total
            result[:, ~near] = (f, fr/(2*r), (frr-fr/r)/(4*r*r), fz, frz/(2*r), fzz)
        return result

    def _sample_local(self, local, include_a):
        result_b = np.zeros_like(local)
        result_a = np.zeros_like(local) if include_a else None
        if self.ampere_turns == 0.:
            return result_a, result_b
        near = self._near_axis(local)
        if near.any():
            point = local[near]
            f, fs, _, fz, _, _ = self._axis_coefficients(point, 1)
            x, y = point[:, 0], point[:, 1]
            result_b[near] = np.column_stack((-x*fz, -y*fz, 2*f+2*(x*x+y*y)*fs))
            if include_a:
                result_a[near] = np.column_stack((-y*f, x*f, np.zeros(len(x))))
        point = local[~near]
        if len(point):
            r, z = np.hypot(point[:, 0], point[:, 1]), point[:, 2]
            br, bz, f = np.zeros(len(point)), np.zeros(len(point)), np.zeros(len(point))
            for a, centre, current in self._loops:
                t = z-centre
                q, delta = (a+r)**2+t*t, (a-r)**2+t*t
                m = 4*a*r/q
                k, e = ellipk(m), ellipe(m)
                factor = mu_0*current/(2*np.pi*np.sqrt(q))
                br += factor*t/r*(-k+(a*a+r*r+t*t)/delta*e)
                bz += factor*(k+(a*a-r*r-t*t)/delta*e)
                if include_a:
                    f += 8*mu_0*current*a*a/np.pi*q**-1.5*_elliptic_h(m)[0]
            result_b[~near] = np.column_stack((br*point[:, 0]/r, br*point[:, 1]/r, bz))
            if include_a:
                result_a[~near] = np.column_stack((-point[:, 1]*f, point[:, 0]*f, np.zeros(len(point))))
        if not np.isfinite(result_b).all() or (include_a and not np.isfinite(result_a).all()):
            raise ValueError("Annular elliptic field exceeded the finite numerical range")
        return result_a, result_b

    def field_at_global_positions_t(self, positions_m):
        local = self._local(positions_m)
        flat = local.reshape(-1, 3)
        result = np.empty_like(flat)
        for start in range(0, len(flat), 4096):
            result[start:start+4096] = self._sample_local(flat[start:start+4096], False)[1]
        return self.registration.vectors_local_to_global(result.reshape(local.shape))

    def sample(self, positions_m):
        local = self._local(positions_m)
        flat = local.reshape(-1, 3)
        a, b = np.empty_like(flat), np.empty_like(flat)
        for start in range(0, len(flat), 4096):
            a[start:start+4096], b[start:start+4096] = self._sample_local(flat[start:start+4096], True)
        return dict(A_t_m=self.registration.vectors_local_to_global(a.reshape(local.shape)),
                    B_t=self.registration.vectors_local_to_global(b.reshape(local.shape)))

    def sample_jet(self, positions_m):
        local = self._local(positions_m)
        shape = local.shape[:-1]
        points = local.reshape(-1, 3)
        f, fs, fss, fz, fsz, fzz = self._coefficients(points)
        x, y = points[:, 0], points[:, 1]
        direction = points@_CROSS.T
        gradient = np.column_stack((2*x*fs, 2*y*fs, fz))
        h = np.empty((len(points), 3, 3))
        h[:, 0, 0], h[:, 1, 1] = 2*fs+4*x*x*fss, 2*fs+4*y*y*fss
        h[:, 0, 1] = h[:, 1, 0] = 4*x*y*fss
        h[:, 0, 2] = h[:, 2, 0] = 2*x*fsz
        h[:, 1, 2] = h[:, 2, 1] = 2*y*fsz
        h[:, 2, 2] = fzz
        a = direction*f[:, None]
        jac = _CROSS*f[:, None, None]+np.einsum("ni,nj->nij", direction, gradient)
        hess = (np.einsum("ij,nk->nijk", _CROSS, gradient)+np.einsum("ik,nj->nijk", _CROSS, gradient)
                +np.einsum("ni,njk->nijk", direction, h))
        rot = self.registration.rotation_array
        return dict(A_t_m=(a@rot.T).reshape((*shape, 3)),
            jac_A_t=np.einsum("ai,nij,bj->nab", rot, jac, rot).reshape((*shape, 3, 3)),
            hess_A_t_per_m=np.einsum("ai,nijk,bj,ck->nabc", rot, hess, rot, rot).reshape((*shape, 3, 3, 3)))


__all__ = ("AnnularCoilPotential",)
