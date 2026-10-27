"""Opt-in near-axis vacuum expansion of an existing round-lens axial field.

For an axisymmetric, static, current-free vacuum region, regularity gives

    A_phi = sum_n (-1)^n r^(2n+1) B0^(2n)(z) / (2^(2n+1) n! (n+1)!).

The unchanged Gaussian B0 is captured from a native analytical field. B is
always the curl of this SAME truncated potential. The divergence vanishes
identically, but curl(B) has a declared truncation residual. A finite series
does not reconstruct pole geometry or certify its useful radial aperture.

This independent adapter is never selected by a production preset. It has no
paraxial ``slope_derivative`` shortcut, empirical Cs/Cc, or wave admission.
The existing first-order adapter and propagation remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np

from temsim.immutable_json import json_digest
from .analytic_potential import AnalyticMagneticPotential
from .lens_field_provider import FrozenAnalyticField


SCHEMA = "high-order-round-lens-potential-v1"
_CROSS = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 0.)))


def _finite_radius(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError("High-order round-lens validity radius must be a positive finite scalar")
    return float(value)


@dataclass(frozen=True, slots=True)
class HighOrderRoundLensPotential:
    """Immutable Gaussian coefficients, radial expansion order and domain.

    ``radial_order`` is the largest power in A_phi, not the ray aberration
    order. Order 1 reproduces the native near-axis field, order 3 includes the
    field terms needed for third-order ray aberrations, and higher orders
    permit an independent field-truncation study. All lengths are SI except
    the captured native support, whose existing unit is explicitly mm.
    """

    native: FrozenAnalyticField
    radial_order: int = 3
    valid_radius_m: float | None = None
    axial_domain: str = "native_support"
    remote_domain: str = "strict"
    digest: str = field(init=False)

    def __post_init__(self):
        if type(self.native) is not FrozenAnalyticField:
            raise ValueError("High-order round lens requires an exact frozen native round field")
        if type(self.radial_order) is not int or self.radial_order not in {1, 3, 5, 7, 9}:
            raise ValueError("A_phi radial order must be one of 1, 3, 5, 7, 9")
        if type(self.axial_domain) is not str or self.axial_domain not in {"native_support", "gaussian_all_z"}:
            raise ValueError("High-order axial domain must be native_support or gaussian_all_z")
        from .gaussian_remote_domain import POLICY
        if type(self.remote_domain) is not str or self.remote_domain not in ("strict",POLICY) or (self.remote_domain!= "strict" and self.axial_domain!="gaussian_all_z"):
            raise ValueError("Remote Gaussian qualification requires the explicit all-z Gaussian domain")
        native = AnalyticMagneticPotential(self.native).field
        radius = _finite_radius(native.radial_support_m if self.valid_radius_m is None else self.valid_radius_m)
        if radius > native.radial_support_m:
            raise ValueError("High-order validity radius cannot exceed the captured native aperture")
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "valid_radius_m", radius)
        object.__setattr__(self, "digest", json_digest(self.to_metadata()))

    @classmethod
    def from_native(cls, native, *, radial_order=3, valid_radius_m=None, axial_domain="native_support",remote_domain="strict"):
        return cls(native, radial_order, valid_radius_m, axial_domain,remote_domain)

    @property
    def source_id(self):
        return self.native.lens_key

    @property
    def lens_key(self):
        return self.source_id

    @property
    def registration(self):
        return self.native.registration

    @property
    def native_support_mm(self):
        return self.native.native_support_mm

    def to_metadata(self):
        result = {"schema": SCHEMA, "source_id": self.source_id,
            "native_field_fingerprint": self.native.fingerprint,
            "axial_field": "unchanged captured Gaussian B0(z)", "radial_order_A_phi": self.radial_order,
            "gauge": "regular local azimuthal; A=(-y F(r_squared,z), x F(r_squared,z), 0)",
            "coefficient_law": "(-1)^n B0^(2n)/(2^(2n+1) n! (n+1)!)",
            "units": {"position": "m", "A": "T m", "B": "T", "curl_B": "T/m"},
            "domain": {"local_radius_m": [0., self.valid_radius_m],
                       "local_z_m": [value*.001 for value in self.native_support_mm]
                                     if self.axial_domain == "native_support" else [None, None]},
            "axial_domain": self.axial_domain,
            "native_support_mm": list(self.native_support_mm),
            "outside_domain": "reject; no zero fill; gaussian_all_z explicitly evaluates the native analytical Gaussian",
            "derivative_order": 2, "derivative_domain": "smooth interior of declared cylindrical domain",
            "maxwell_scope": "divergence-free curl; finite-order vacuum curl residual is reported, not zeroed",
            "approximation": "near-axis vacuum series; radial and integration convergence required separately",
            "wave_admission": False, "default_propagation": False,
            "manual_aberrations": "not included; derived coefficients are observations only"}
        if self.remote_domain!="strict":
            from .gaussian_remote_domain import metadata
            result["remote_domain"]=metadata()
            result["outside_domain"]="reject unless the explicit Gaussian binary64 remote-roundoff certificate admits the point; no zero fill"
            result["derivative_domain"]="native cylinder and independently certified remote points; the union is not convex"
        return result

    def to_dict(self):
        result = {"schema": SCHEMA, "native": AnalyticMagneticPotential(self.native).to_dict(),
                "radial_order": self.radial_order, "valid_radius_m": self.valid_radius_m,
                "axial_domain": self.axial_domain, "digest": self.digest}
        if self.remote_domain!="strict":
            result["remote_domain"]=self.remote_domain
        return result

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping)
                or set(record)-{"remote_domain"} != {"schema", "native", "radial_order", "valid_radius_m", "axial_domain", "digest"}
                or record["schema"] != SCHEMA):
            raise ValueError("Invalid high-order round-lens snapshot schema")
        native = AnalyticMagneticPotential.from_dict(record["native"]).field
        result = cls(native, record["radial_order"], record["valid_radius_m"], record["axial_domain"],record.get("remote_domain","strict"))
        if result.digest != record["digest"]:
            raise ValueError("High-order round-lens snapshot identity mismatch")
        return result

    def _local(self, positions_m):
        raw = np.asarray(positions_m)
        if raw.dtype.kind not in "iuf" or raw.shape[-1:] != (3,) or not np.isfinite(raw).all():
            raise ValueError("High-order potential positions must be finite real XYZ metres")
        local = self.registration.positions_global_to_local_m(np.asarray(raw, float))
        if not np.isfinite(local).all():
            raise ValueError("High-order potential local coordinates exceed the finite range")
        lower, upper = np.asarray(self.native_support_mm)*.001
        if self.axial_domain == "native_support" and (np.any(local[..., 2] <= lower) or np.any(local[..., 2] >= upper)):
            raise ValueError("Position outside the strict high-order round-lens validity domain")
        outside=np.hypot(local[...,0],local[...,1])>self.valid_radius_m
        if outside.any():
            if self.remote_domain=="strict":
                raise ValueError("Position outside the strict high-order round-lens validity domain")
            from .gaussian_remote_domain import require_remote_domain
            require_remote_domain(self.native,local[outside],self.radial_order)
        return local

    def require_global_box(self,positions_m):
        """Admit a complete global AABB for canonical swept-domain checks."""
        if self.remote_domain=="strict":
            points=np.asarray(positions_m)
            if points.dtype.kind not in "iuf" or points.shape[-1:]!=(3,) or not points.size or not np.isfinite(points).all():
                raise ValueError("Round-lens swept box requires finite global XYZ points")
            points=np.asarray(points,dtype=float).reshape(-1,3)
            from itertools import product
            corners=np.asarray(tuple(product(*zip(points.min(axis=0),points.max(axis=0)))))
            self._local(corners)
            return
        from .gaussian_remote_domain import require_global_box
        require_global_box(self.native,positions_m,self.radial_order,self.valid_radius_m)

    def axial_derivatives(self, local_z_m, maximum_order=0):
        """Analytic B0 derivatives per metre, using probabilists' Hermite polynomials."""
        if type(maximum_order) is not int or not 0 <= maximum_order <= 12:
            raise ValueError("Gaussian axial derivative order must be an integer in [0, 12]")
        z = np.asarray(local_z_m)
        lower, upper = np.asarray(self.native_support_mm)*.001
        if (z.dtype.kind not in "iuf" or not np.isfinite(z).all()
                or (self.axial_domain == "native_support" and (np.any(z <= lower) or np.any(z >= upper)))):
            raise ValueError("Axial derivative positions must lie inside the captured support")
        z = np.asarray(z, float)
        result = np.zeros((maximum_order+1, *z.shape))
        for amplitude, centre, sigma in self.native.terms_t_m:
            u = (z-centre)/sigma
            gaussian = amplitude*np.exp(-.5*u*u)
            previous, current = np.ones_like(u), u.copy()
            result[0] += gaussian
            for order in range(1, maximum_order+1):
                hermite = current
                result[order] += ((-1.)**order/sigma**order)*hermite*gaussian
                previous, current = current, u*current-order*previous
        if not np.isfinite(result).all():
            raise ValueError("Gaussian derivative expansion exceeded finite numerical range")
        return result

    def _coefficient_jets(self, local, derivative_order):
        order = (self.radial_order-1)//2
        d = self.axial_derivatives(local[..., 2], 2*order+derivative_order)
        s = np.sum(local[..., :2]**2, axis=-1)
        f, fs, fss, fz, fsz, fzz = (np.zeros_like(s) for _ in range(6))
        for n in range(order+1):
            coefficient = (-1.)**n/(2.**(2*n+1)*math.factorial(n)*math.factorial(n+1))
            power = s**n
            f += coefficient*power*d[2*n]
            if n:
                fs += n*coefficient*s**(n-1)*d[2*n]
            if n >= 2:
                fss += n*(n-1)*coefficient*s**(n-2)*d[2*n]
            if derivative_order:
                fz += coefficient*power*d[2*n+1]
                if n:
                    fsz += n*coefficient*s**(n-1)*d[2*n+1]
            if derivative_order >= 2:
                fzz += coefficient*power*d[2*n+2]
        return f, fs, fss, fz, fsz, fzz

    def sample(self, positions_m):
        local = self._local(positions_m)
        x, y = local[..., 0], local[..., 1]
        f, fs, _, fz, _, _ = self._coefficient_jets(local, 1)
        a = np.stack((-y*f, x*f, np.zeros_like(f)), axis=-1)
        b = np.stack((-x*fz, -y*fz, 2*f+2*(x*x+y*y)*fs), axis=-1)
        return self._checked_values(local,{"A_t_m": self.registration.vectors_local_to_global(a),
                "B_t": self.registration.vectors_local_to_global(b)})

    def _checked_values(self,local,result):
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Retained round-lens polynomial exceeded the finite numerical range")
        remote=np.hypot(local[...,0],local[...,1])>self.valid_radius_m
        if remote.any() and any(np.any(value[remote]!=0.) for value in result.values()):
            raise ValueError("Certified remote point did not underflow in the unchanged retained polynomial")
        return result

    def field_at_global_positions_t(self, positions_m):
        """Actual curl field for full Lorentz transport, without a paraxial hook."""
        return self.sample(positions_m)["B_t"]

    def sample_jet(self, positions_m):
        local = self._local(positions_m)
        x, y = local[..., 0], local[..., 1]
        f, fs, fss, fz, fsz, fzz = self._coefficient_jets(local, 2)
        direction = local@_CROSS.T
        gradient = np.stack((2*x*fs, 2*y*fs, fz), axis=-1)
        h = np.empty((*local.shape[:-1], 3, 3))
        h[..., 0, 0], h[..., 1, 1] = 2*fs+4*x*x*fss, 2*fs+4*y*y*fss
        h[..., 0, 1] = h[..., 1, 0] = 4*x*y*fss
        h[..., 0, 2] = h[..., 2, 0] = 2*x*fsz
        h[..., 1, 2] = h[..., 2, 1] = 2*y*fsz
        h[..., 2, 2] = fzz
        a = direction*f[..., None]
        jac = _CROSS*f[..., None, None]+np.einsum("...i,...j->...ij", direction, gradient)
        hess = (np.einsum("ij,...k->...ijk", _CROSS, gradient)
                +np.einsum("ik,...j->...ijk", _CROSS, gradient)
                +np.einsum("...i,...jk->...ijk", direction, h))
        rotation = self.registration.rotation_array
        return self._checked_values(local,{"A_t_m": a@rotation.T,
                "jac_A_t": np.einsum("ai,...ij,bj->...ab", rotation, jac, rotation),
                "hess_A_t_per_m": np.einsum("ai,...ijk,bj,ck->...abc", rotation, hess, rotation, rotation)})

    def maxwell_residual(self, positions_m):
        """Exact analytical truncation residual; no residual projection/removal."""
        local = self._local(positions_m)
        n = (self.radial_order-1)//2
        d = self.axial_derivatives(local[..., 2], 2*n+2)[-1]
        s = np.sum(local[..., :2]**2, axis=-1)
        factor = (-1.)**(n+1)*s**n*d/(2.**(2*n+1)*math.factorial(n)*math.factorial(n+1))
        # curl(B)_phi = -c_n r^(2n+1) B0^(2n+2).
        curl = np.stack((-local[..., 1]*factor, local[..., 0]*factor, np.zeros_like(factor)), axis=-1)
        return {"div_B_t_per_m": np.zeros(local.shape[:-1]),
                "curl_B_t_per_m": curl@self.registration.rotation_array.T}


__all__ = ("HighOrderRoundLensPotential",)
