"""Explicit smooth, axis-regular reconstruction of a solved A-phi potential.

This is a NEW representation, not the preserved triangular P1 basis. In a
declared inner cylinder, non-axis FEM nodal values A_phi/r are fitted to a
polynomial in s=(r/fit_radius)^2, independently at each original axial node.
An interpolating axial spline then defines A_phi=r*a(s,z). Its analytic curl
defines B. No independently stored B values, expected aberrations, peak-field
normalization, source retuning, or outside-domain extrapolation are used.

The caller must establish that the fit cylinder is vacuum and source free.
This adapter cannot infer material/current regions from a potential array.
FEM mesh, remote boundary, fit window/degree, axial spline and trajectory
convergence remain independent qualifications. A smooth fit is not itself a
Maxwell, natural-aberration, coherent-wave or instrument certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Real

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential


SCHEMA = "regular-axisymmetric-magnetic-potential-v1"


def _frozen(array):
    value = np.asarray(array, dtype=np.float64)
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(value.shape)


def _identity(array):
    return {"shape": list(array.shape), "dtype": "float64",
            "sha256": sha256(array.tobytes(order="C")).hexdigest()}


@dataclass(frozen=True, slots=True, eq=False)
class RegularAxisymmetricMagneticPotential:
    source: AxisymmetricMagneticPotential
    fit_radius_m: float
    radial_degree: int = 3
    axial_degree: int = 3
    valid_radius_m: float | None = None
    vacuum_domain_note: str = ""
    knots_m: np.ndarray = field(init=False, repr=False)
    coefficients_t: np.ndarray = field(init=False, repr=False)
    fit_diagnostics: object = field(init=False, repr=False)
    digest: str = field(init=False)

    def __post_init__(self):
        if type(self.source) is not AxisymmetricMagneticPotential:
            raise ValueError("Regular reconstruction requires an exact preserved FEM potential")
        for value in (self.fit_radius_m, self.valid_radius_m if self.valid_radius_m is not None else self.fit_radius_m):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError("Fit and valid radii must be finite positive lengths")
        fit_radius = float(self.fit_radius_m)
        valid_radius = fit_radius if self.valid_radius_m is None else float(self.valid_radius_m)
        if not valid_radius <= fit_radius <= self.source.r_m[-1]:
            raise ValueError("Valid cylinder must lie inside the radial fit window and original FEM domain")
        if type(self.radial_degree) is not int or not 1 <= self.radial_degree <= 6:
            raise ValueError("Radial polynomial degree must be an integer from one to six")
        if type(self.axial_degree) is not int or self.axial_degree not in (3, 5):
            raise ValueError("Axial interpolating spline degree must be three or five")
        if (type(self.vacuum_domain_note) is not str or not self.vacuum_domain_note.strip()
                or self.vacuum_domain_note.strip() != self.vacuum_domain_note):
            raise ValueError("An explicit source-free vacuum-cylinder justification is required")
        source = AxisymmetricMagneticPotential.from_dict(self.source.to_dict())
        indices = np.flatnonzero((source.r_m > 0.) & (source.r_m <= fit_radius))
        if len(indices) < self.radial_degree+3 or len(source.z_m) <= self.axial_degree:
            raise ValueError("Reconstruction needs at least degree+3 non-axis radial nodes and enough axial nodes")
        radius = source.r_m[indices]
        values = source.a_phi_tm[indices]/radius[:, None]
        design = np.polynomial.polynomial.polyvander((radius/fit_radius)**2, self.radial_degree)
        coefficients, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
        if rank != self.radial_degree+1:
            raise ValueError("Radial reconstruction is rank deficient")
        residual = design@coefficients-values
        spline = make_interp_spline(source.z_m, coefficients.T, k=self.axial_degree, axis=0)
        if not np.isfinite(spline.c).all():
            raise ValueError("Potential reconstruction produced non-finite coefficients")
        diagnostics = {"radial_node_count": len(indices),
            "first_non_axis_node_m": float(radius[0]), "last_fit_node_m": float(radius[-1]),
            "radial_design_condition": float(singular[0]/singular[-1]),
            "A_phi_over_r_fit_rms_t": float(np.sqrt(np.mean(residual**2))),
            "A_phi_over_r_fit_max_t": float(np.max(np.abs(residual)))}
        for name, value in (("source", source), ("fit_radius_m", fit_radius),
                            ("valid_radius_m", valid_radius), ("knots_m", _frozen(spline.t)),
                            ("coefficients_t", _frozen(spline.c)),
                            ("fit_diagnostics", freeze_json(diagnostics))):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "digest", json_digest(self._metadata()))

    @classmethod
    def from_potential(cls, source, **kwargs):
        return cls(source, **kwargs)

    @property
    def registration(self):
        return self.source.registration

    def _metadata(self):
        return {"schema": SCHEMA, "source_digest": self.source.digest,
            "source_id": self.source.source_id, "source_scale": self.source.source_scale,
            "representation": "radial polynomial in r^2 for A_phi/r; axial interpolating spline",
            "axis_policy": "same even radial polynomial extrapolated to r=0; A_phi=0 exactly",
            "gauge": "axisymmetric azimuthal local; Ar=Az=0",
            "units": {"position": "m", "A": "T m", "B": "T"},
            "fit_radius_m": self.fit_radius_m, "radial_degree": self.radial_degree,
            "axial_degree": self.axial_degree,
            "domain": {"local_radius_m": [0., self.valid_radius_m],
                       "local_z_m": [float(self.source.z_m[0]), float(self.source.z_m[-1])]},
            "vacuum_domain_note": self.vacuum_domain_note,
            "outside_domain": "reject; no extrapolation or zero continuation",
            "fit_diagnostics": thaw_json(self.fit_diagnostics),
            "coefficient_identity": {"knots_m": _identity(self.knots_m),
                                     "coefficients_t": _identity(self.coefficients_t)},
            "derivative_capability": "analytic first derivatives of the reconstructed scalar potential",
            "static": True, "wave_admission": False, "production_admission": False,
            "scope": "explicit reconstruction; requires independent field, fit and transport convergence"}

    def to_metadata(self):
        return {**self._metadata(), "digest": self.digest}

    def to_dict(self):
        return {"metadata": self.to_metadata(), "source": self.source.to_dict(),
                "coefficients": {"knots_m": self.knots_m.tolist(),
                                 "coefficients_t": self.coefficients_t.tolist()}}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, dict) or set(record) != {"metadata", "source", "coefficients"}:
            raise ValueError("Invalid regular potential record")
        try:
            meta = record["metadata"]
            result = cls(AxisymmetricMagneticPotential.from_dict(record["source"]),
                fit_radius_m=meta["fit_radius_m"], radial_degree=meta["radial_degree"],
                axial_degree=meta["axial_degree"], valid_radius_m=meta["domain"]["local_radius_m"][1],
                vacuum_domain_note=meta["vacuum_domain_note"])
            if json_digest(record) != json_digest(result.to_dict()):
                raise ValueError("Regular potential identity or coefficient mismatch")
            return result
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Invalid regular potential metadata") from error

    def sample(self, positions_m):
        points = np.asarray(positions_m)
        if points.dtype.kind not in "iuf" or points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Potential query requires finite real XYZ positions in metres")
        shape = points.shape
        rotation = self.registration.rotation_array
        local = (np.asarray(points, float).reshape(-1, 3)-self.registration.origin_array_m)@rotation
        x, y, z = local.T
        radius = np.hypot(x, y)
        if (np.any(radius > self.valid_radius_m) or np.any(z < self.source.z_m[0])
                or np.any(z > self.source.z_m[-1])):
            raise ValueError("Potential query is outside the reconstructed vacuum cylinder")
        spline = BSpline.construct_fast(self.knots_m, self.coefficients_t, self.axial_degree, extrapolate=False)
        values, axial = spline(z), spline(z, nu=1)
        s = (radius/self.fit_radius_m)**2
        powers = s[:, None]**np.arange(self.radial_degree+1)
        f = np.sum(values*powers, axis=1)
        fz = np.sum(axial*powers, axis=1)
        sf_s = np.sum(values*powers*np.arange(self.radial_degree+1), axis=1)
        potential = np.column_stack((-y*f, x*f, np.zeros_like(f)))
        magnetic = np.column_stack((-x*fz, -y*fz, 2*(f+sf_s)))
        scale = self.source.source_scale
        result = {"A_t_m": (scale*potential@rotation.T).reshape(shape),
                  "B_t": (scale*magnetic@rotation.T).reshape(shape)}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Regular potential sample exceeds the finite numerical range")
        return result


__all__ = ("RegularAxisymmetricMagneticPotential",)
