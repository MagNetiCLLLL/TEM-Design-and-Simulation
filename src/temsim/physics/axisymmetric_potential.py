"""Opt-in preservation of an axisymmetric FEM azimuthal vector potential.

This adapter retains the triangular P1 basis used by ``prepared_mesh``. Its
curl is a new represented B evaluator, not the historical finite-difference
nodal B followed by independent component interpolation. Nothing here changes
the legacy propagator or grants coherent-wave admission.

In local cylindrical coordinates A=(0,A_phi,0), Br=-d_z A_phi and
Bz=d_r A_phi+A_phi/r. Potentials are C0; their first derivatives are defined
within each triangle and by the declared one-sided convention on interfaces.
There is no globally smooth derivative or Hessian claim. A=0 at the axis is
mandatory. Axis values use the regular limit of the adjacent axis triangle;
at axial mesh vertices this is the right-z limit (left at the final node).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import math
from numbers import Real

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.lens_field_provider import CoordinateRegistration


SCHEMA = "axisymmetric-magnetic-potential-v1"
_BASIS = "triangular-P1-r-z; diagonal lower-left to upper-right"
_GAUGE = "axisymmetric-azimuthal-local; Ar=Az=0; regular axis"
_REGISTRATION_ACCESSORS = {name: getattr(CoordinateRegistration, name) for name in
    ("__post_init__", "origin_array_m", "rotation_array", "positions_global_to_local_m", "vectors_local_to_global")}


def _array(value, name, ndim):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a real {ndim}-dimensional array")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    # An owning array's writeable=False can be undone. Immutable byte storage
    # also protects snapshots from input mutation, views and deepcopy callers.
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(result.shape)


def _array_identity(value):
    return {"shape": list(value.shape), "dtype": "float64",
            "sha256": sha256(value.tobytes(order="C")).hexdigest()}


@dataclass(frozen=True, slots=True, eq=False)
class AxisymmetricMagneticPotential:
    """Frozen SI potential coefficients and one consistent piecewise curl.

    ``source_scale`` is a captured multiplier, not permission to rescale a
    nonlinear circuit. Nonlinear FEM callers must capture its solved operating
    point with scale=1; solve/current ownership remains with the source owner.
    """

    r_m: np.ndarray
    z_m: np.ndarray
    a_phi_tm: np.ndarray
    registration: CoordinateRegistration = field(default_factory=CoordinateRegistration)
    provenance: Mapping = field(default_factory=dict)
    source_id: str = "axisymmetric_fem"
    source_scale: float = 1.0
    digest: str = field(init=False)

    def __post_init__(self):
        r, z, a = (_array(value, name, ndim) for value, name, ndim in (
            (self.r_m, "r_m", 1), (self.z_m, "z_m", 1), (self.a_phi_tm, "a_phi_tm", 2)))
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            widths = tuple(np.diff(axis) for axis in (r, z))
            reciprocal = tuple(1./width for width in widths)
        if (min(r.size, z.size) < 2 or r[0] != 0.
                or any(not np.isfinite(width).all() or np.any(width <= 0.) for width in widths)
                or any(not np.isfinite(value).all() for value in reciprocal)):
            raise ValueError("Potential axes must increase strictly, with at least two nodes and r_m[0]=0")
        if a.shape != (len(r), len(z)):
            raise ValueError("a_phi_tm shape must match the radial and axial nodes")
        if np.any(a[0] != 0.):
            raise ValueError("Regular azimuthal potential requires exactly zero A_phi on the axis")
        if type(self.registration) is not CoordinateRegistration:
            raise ValueError("Potential registration must be an exact CoordinateRegistration without overridden behavior")
        if any(getattr(CoordinateRegistration, name) is not original
               for name, original in _REGISTRATION_ACCESSORS.items()):
            raise ValueError("Overridden coordinate registration requires an explicit potential adapter")
        registration = CoordinateRegistration(self.registration.origin_global_m,
                                              self.registration.rotation_local_to_global)
        if not isinstance(self.provenance, Mapping) or any(not isinstance(k, str) for k in self.provenance):
            raise ValueError("Potential provenance must be a mapping with string keys")
        if (not isinstance(self.source_id, str) or not self.source_id
                or self.source_id.strip() != self.source_id or any(ord(c) < 32 for c in self.source_id)):
            raise ValueError("Potential source_id must be a nonempty identifier")
        if (isinstance(self.source_scale, (bool, np.bool_)) or not isinstance(self.source_scale, Real)
                or not math.isfinite(self.source_scale)):
            raise ValueError("Potential source_scale must be a finite real number")
        for name, value in (("r_m", r), ("z_m", z), ("a_phi_tm", a),
                            ("registration", registration),
                            ("source_scale", float(self.source_scale)),
                            ("provenance", freeze_json(self.provenance))):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "digest", json_digest(self._metadata()))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @classmethod
    def from_solution(cls, solution, **kwargs):
        """Capture an already solved FEM result; never run or infer a solve."""
        return cls(solution.r_m, solution.z_m, solution.a_phi_tm, **kwargs)

    def _metadata(self):
        return {
            "schema": SCHEMA, "source_id": self.source_id, "source_scale": self.source_scale,
            "units": {"position": "m", "A": "T m", "B": "T"},
            "gauge": _GAUGE, "basis": _BASIS,
            "derivative_capability": "first derivatives within a triangle; no global Hessian",
            "interface_policy": "right cell; lower-right triangle on diagonal; final cell at outer faces",
            "axis_policy": "A=0; Br=0; Bz=2*d_r(A_phi) in right-z axis triangle, left at final node",
            "outside_domain": "reject; no zero fill or extrapolation",
            "wave_admission": False,
            "scope": "preserved potential representation; not a validated propagation operator or Maxwell certificate",
            "domain": {"local_radius_m": [0., float(self.r_m[-1])],
                       "local_z_m": [float(self.z_m[0]), float(self.z_m[-1])]},
            "registration": {"origin_global_m": list(self.registration.origin_global_m),
                             "rotation_local_to_global": [list(row) for row in self.registration.rotation_local_to_global]},
            "arrays": {name: _array_identity(getattr(self, name)) for name in ("r_m", "z_m", "a_phi_tm")},
            "provenance": thaw_json(self.provenance),
        }

    def to_metadata(self):
        """Return detached JSON metadata with array identities, not coefficients."""
        return {**self._metadata(), "digest": self.digest}

    def to_dict(self):
        """Return a self-contained JSON snapshot without external file access."""
        return {"metadata": self.to_metadata(),
                "coefficients": {name: getattr(self, name).tolist() for name in ("r_m", "z_m", "a_phi_tm")}}

    @classmethod
    def from_dict(cls, value):
        """Restore and verify identities, units, basis, domain and gauge."""
        if not isinstance(value, Mapping) or set(value) != {"metadata", "coefficients"}:
            raise ValueError("Invalid potential snapshot fields")
        meta, arrays = value["metadata"], value["coefficients"]
        if not isinstance(meta, Mapping) or meta.get("schema") != SCHEMA:
            raise ValueError("Unknown potential snapshot schema")
        if not isinstance(arrays, Mapping) or set(arrays) != {"r_m", "z_m", "a_phi_tm"}:
            raise ValueError("Invalid potential snapshot coefficients")
        try:
            result = cls(**arrays, registration=CoordinateRegistration(**meta["registration"]),
                         provenance=meta["provenance"], source_id=meta["source_id"], source_scale=meta["source_scale"])
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid potential snapshot metadata") from error
        if json_digest(meta) != json_digest(result.to_metadata()):
            raise ValueError("Potential snapshot identity or representation metadata mismatch")
        return result

    def sample(self, positions_m):
        """Return A_t_m and B_t, both shaped (...,3), in global coordinates.

        Querying outside the closed local cylindrical domain is an error even
        for zero excitation. On non-axis interfaces a deterministic one-sided
        derivative is returned; it is not a claim of derivative continuity.
        """
        positions = np.asarray(positions_m)
        if positions.dtype.kind not in "iuf" or positions.shape[-1:] != (3,) or not np.isfinite(positions).all():
            raise ValueError("Potential query requires finite real XYZ positions in metres")
        shape = positions.shape
        rotation = np.asarray(self.registration.rotation_local_to_global, dtype=np.float64)
        local = (np.asarray(positions, dtype=np.float64).reshape(-1, 3)
                 -np.asarray(self.registration.origin_global_m, dtype=np.float64)) @ rotation
        radius = np.hypot(local[:, 0], local[:, 1])
        axial = local[:, 2]
        if np.any(radius > self.r_m[-1]) or np.any(axial < self.z_m[0]) or np.any(axial > self.z_m[-1]):
            raise ValueError("Potential query is outside the represented cylindrical domain")
        i = np.minimum(np.searchsorted(self.r_m, radius, side="right")-1, len(self.r_m)-2)
        j = np.minimum(np.searchsorted(self.z_m, axial, side="right")-1, len(self.z_m)-2)
        dr, dz = self.r_m[i+1]-self.r_m[i], self.z_m[j+1]-self.z_m[j]
        u, v = (radius-self.r_m[i])/dr, (axial-self.z_m[j])/dz
        a, b = self.a_phi_tm[i, j], self.a_phi_tm[i+1, j]
        c, d = self.a_phi_tm[i+1, j+1], self.a_phi_tm[i, j+1]
        lower = u >= v
        a_r = np.where(lower, (b-a)/dr, (c-d)/dr)
        a_z = np.where(lower, (c-b)/dz, (d-a)/dz)
        a_phi = a+a_r*(radius-self.r_m[i])+a_z*(axial-self.z_m[j])
        inverse = np.divide(1., radius, out=np.zeros_like(radius), where=radius > 0.)
        br, bz = -a_z, a_r+a_phi*inverse
        axis = radius == 0.
        # The triangle sharing the open axis edge has A_phi=r*A(1,j+1)/r[1].
        # Selecting this side at axis vertices avoids an invented nonzero Br
        # from a different triangle that touches the axis only at that vertex.
        bz[axis] = 2*self.a_phi_tm[1, j[axis]+1]/self.r_m[1]
        br[axis] = 0.
        a_phi[axis] = 0.
        cosine, sine = local[:, 0]*inverse, local[:, 1]*inverse
        potential = np.column_stack((-sine*a_phi, cosine*a_phi, np.zeros_like(radius)))
        magnetic = np.column_stack((cosine*br, sine*br, bz))
        with np.errstate(over="ignore", invalid="ignore"):
            result = {"A_t_m": (self.source_scale*potential@rotation.T).reshape(shape),
                      "B_t": (self.source_scale*magnetic@rotation.T).reshape(shape)}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Potential sampling exceeds the finite numerical range")
        return result


__all__ = ("AxisymmetricMagneticPotential",)
