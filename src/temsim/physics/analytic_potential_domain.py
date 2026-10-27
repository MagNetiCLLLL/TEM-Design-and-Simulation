"""Opt-in axial coverage for the existing native analytical potentials.

Round-lens Gaussian tails are evaluated analytically, not set to zero. Native
Az multipoles instead have an explicitly compact axial envelope. Their zero
exterior is a property of that simplified model, not evidence about hardware
fringes. No unknown FEM or imported-map exterior is admitted here.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from temsim.immutable_json import json_digest
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.lens_field_provider import FrozenAnalyticField
from temsim.physics.posed_lens_wave import vector_potential_jet


SCHEMA = "native-analytical-potential-domain-v1"


@dataclass(frozen=True, slots=True)
class AnalyticPotentialDomain:
    """Explicitly select a model-defined axial extension of an exact adapter.

    The round model retains its local cylindrical radial validity bound at all
    axial positions. A compact multipole retains that bound in its active slab;
    outside the slab A and B are explicitly zero without evaluating its radial
    polynomial. At an axial face sample() returns the native inside value.
    sample_jet() rejects the face because a two-sided derivative is undefined.
    This interface does not change a legacy provider or grant wave admission.
    """

    base: AnalyticMagneticPotential

    def __post_init__(self):
        if type(self.base) is not AnalyticMagneticPotential:
            raise ValueError("Axial extension requires an exact native analytical potential; FEM and unknown exteriors are rejected")
        object.__setattr__(self, "base", AnalyticMagneticPotential.from_dict(self.base.to_dict()))
        dynamic = self.base.provenance.get("dynamic_drive_frozen", False)
        if type(dynamic) is not bool:
            raise ValueError("dynamic_drive_frozen must be a Boolean")

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @property
    def source_id(self):
        return self.base.source_id

    @property
    def digest(self):
        return json_digest(self.to_metadata())

    @property
    def _round(self):
        return type(self.base.field) is FrozenAnalyticField

    def to_metadata(self):
        field = self.base.field
        dynamic = bool(getattr(field, "dynamic", False) or self.base.provenance.get("dynamic_drive_frozen", False))
        return {"schema": SCHEMA, "source_id": self.source_id,
                "base_digest": self.base.digest, "base_metadata": self.base.to_metadata(),
                "units": {"position": "m", "A": "T m", "B": "T"},
                "axial_policy": "gaussian_all_z" if self._round else "native_compact_Az_zero_exterior",
                "radial_policy": "strict local radius at all z" if self._round else "strict local radius in the closed active slab",
                "local_radius_m": field.radial_support_m,
                "native_support_m": [value*1e-3 for value in field.native_support_mm],
                "known_zero_exterior": not self._round,
                "edge_value": "smooth Gaussian continuation" if self._round else "native inside value on the axial face",
                "derivative_order": 2,
                "derivative_domain": "all admitted positions" if self._round else "each open support side; axial faces excluded",
                "static": not dynamic, "dynamic_drive_frozen": dynamic,
                "wave_admission": False, "legacy_default_changed": False,
                "limitations": ["Native analytical approximation; no new Maxwell qualification",
                    "No unknown map or FEM exterior is filled",
                    "No automatic near-axis or phase-error qualification",
                    ("Gaussian tails retained; no axial truncation or tail error bound imposed" if self._round else
                     "Original hard axial envelope retained; physical fringe structure is not added")],
                }

    def to_dict(self):
        return {"metadata": self.to_metadata(), "base": self.base.to_dict(), "digest": self.digest}

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, Mapping) or set(record) != {"metadata", "base", "digest"}
                or not isinstance(record["metadata"], Mapping)
                or record["metadata"].get("schema") != SCHEMA):
            raise ValueError("Invalid native analytical domain record")
        result = cls(AnalyticMagneticPotential.from_dict(record["base"]))
        if result.digest != record["digest"] or json_digest(record["metadata"]) != json_digest(result.to_metadata()):
            raise ValueError("Native analytical domain identity mismatch")
        return result

    def _positions(self, positions_m, *, derivatives):
        points = np.asarray(positions_m)
        if points.dtype.kind not in "iuf" or points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Potential coordinates must be finite real XYZ metres")
        points = np.asarray(points, dtype=float)
        field = self.base.field
        local = field.registration.positions_global_to_local_m(points)
        if not np.isfinite(local).all():
            raise ValueError("Potential coordinates exceed the finite numerical range")
        lower, upper = np.asarray(field.native_support_mm)*1e-3
        inside = np.ones(points.shape[:-1], dtype=bool) if self._round else ((local[..., 2] >= lower) & (local[..., 2] <= upper))
        if np.any(inside & (np.hypot(local[..., 0], local[..., 1]) > field.radial_support_m)):
            raise ValueError("Position is outside the strict local analytical radial domain")
        if derivatives and not self._round and np.any((local[..., 2] == lower) | (local[..., 2] == upper)):
            raise ValueError("Two-sided potential derivatives are undefined on a hard axial face")
        return points, inside

    def _jet(self, positions_m, *, derivatives):
        points, active = self._positions(positions_m, derivatives=derivatives)
        shape = points.shape[:-1]
        result = {name: np.zeros((*shape, *tail)) for name, tail in
                  (("A_t_m", (3,)), ("jac_A_t", (3, 3)), ("hess_A_t_per_m", (3, 3, 3)))}
        flat = points.reshape(-1, 3)
        active_indices = np.flatnonzero(active.reshape(-1))
        # Do not evaluate an inactive radial polynomial: its exact model zero
        # is known even where the polynomial alone would overflow.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index in active_indices:
                values = vector_potential_jet((self.base.field,), flat[index])
                for name, value in zip(result, values):
                    result[name].reshape((len(flat), *value.shape))[index] = value
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Analytical potential evaluation exceeds the finite numerical range")
        return result

    def sample_jet(self, positions_m):
        """A, its Jacobian and Hessian on a differentiable support side."""
        return self._jet(positions_m, derivatives=True)

    def sample(self, positions_m):
        """The same potential and curl for particle and wave consumers.

        On a compact axial face the native inside field value is used; the
        interface has no ordinary two-sided derivative. This endpoint convention
        does not assert smooth physical fringes or a smooth Maxwell solution.
        """
        points, active = self._positions(positions_m, derivatives=False)
        flat, selection = points.reshape(-1, 3), active.reshape(-1)
        potential, magnetic = np.zeros_like(flat), np.zeros_like(flat)
        field = self.base.field
        # Sampling an entire wave plane need not construct per-pixel Hessians.
        # These are the same native potential/curl formulas used by _jet.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            selected = flat[selection]
            if self._round:
                local = field.registration.positions_global_to_local_m(selected)
                b0 = np.zeros(len(selected))
                for amplitude, center, sigma in field.terms_t_m:
                    b0 += amplitude*np.exp(-.5*((local[:, 2]-center)/sigma)**2)
                local_a = .5*b0[:, None]*np.column_stack((-local[:, 1], local[:, 0], np.zeros(len(selected))))
                potential[selection] = field.registration.vectors_local_to_global(local_a)
            else:
                potential[selection] = field.vector_potential_at_global_positions_t_m(selected)
            magnetic[selection] = field.field_at_global_positions_t(selected)
        if not np.isfinite(potential).all() or not np.isfinite(magnetic).all():
            raise ValueError("Analytical potential evaluation exceeds the finite numerical range")
        return {"A_t_m": potential.reshape(points.shape), "B_t": magnetic.reshape(points.shape)}
