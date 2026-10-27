"""Explicit potentials of the existing finite analytical magnetic models.

These are adapters, not higher-order Maxwell lens solutions. The round-lens
gauge and multipole polynomial are the same ones already used by the placed
wave operator. Sampling outside their declared validity domain is rejected.
No derivative across a hard support edge is claimed.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field as dataclass_field
from numbers import Real

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.posed_column_fields import FrozenPosedMultipole


SCHEMA = "native-analytical-potential-v1"
_NATIVE_METHODS = {
    FrozenAnalyticField: {"field_at_global_positions_t": FrozenAnalyticField.field_at_global_positions_t},
    FrozenPosedMultipole: {name: getattr(FrozenPosedMultipole, name) for name in
        ("field_at_global_positions_t", "vector_potential_at_global_positions_t_m", "vector_potential_jet",
         "_local", "_envelope", "_polynomial")},
}
_REGISTRATION_METHODS = {name: getattr(CoordinateRegistration, name) for name in
                         ("positions_global_to_local_m", "vectors_local_to_global")}


def _valid_identity(value):
    return type(value) is str and bool(value) and value == value.strip()


def _detached_native_field(value):
    """Keep native coefficient types without retaining caller-owned objects."""
    kind = type(value)
    if kind not in _NATIVE_METHODS or type(value.registration) is not CoordinateRegistration:
        raise ValueError("Analytical potentials require a known frozen native field and registration")
    if any(getattr(CoordinateRegistration, name) is not original
           for name, original in _REGISTRATION_METHODS.items()):
        raise ValueError("Overridden coordinate registration requires an explicit potential adapter")
    for name, original in _NATIVE_METHODS[kind].items():
        if (getattr(kind, name) is not original
                or getattr(getattr(value, name), "__func__", None) is not original):
            raise ValueError("Overridden native magnetic fields require an explicit potential adapter")
    if not _valid_identity(value.lens_key):
        raise ValueError("Analytical potential source identity must be a nonempty trimmed string")
    if kind is FrozenPosedMultipole and (
            not _valid_identity(value.component_key)
            or not all(_valid_identity(key) for key in value.drive_keys)):
        raise ValueError("Analytical multipole component and drive identities must be nonempty trimmed strings")
    coefficients = asdict(value)
    coefficients["registration"] = CoordinateRegistration(**coefficients["registration"])
    if kind is FrozenPosedMultipole:
        for name in ("center_z_m", "sigma_m", "radial_support_m", "reference_momentum",
                     "event_z_mm", "event_dx_rad", "event_dy_rad", "captured_time_s"):
            item = coefficients[name]
            if item is None and name in {"event_z_mm", "captured_time_s"}:
                continue
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real) or not np.isfinite(item):
                raise ValueError(f"Analytical multipole {name} requires a finite scalar")
            coefficients[name] = float(item)
        if type(coefficients["dynamic"]) not in {bool, np.bool_}:
            raise ValueError("Analytical multipole dynamic state requires a Boolean")
        coefficients["dynamic"] = bool(coefficients["dynamic"])
    captured = kind(**coefficients)
    # Reject non-JSON/non-finite event metadata now, not when saving later.
    json_digest(asdict(captured))
    return captured


@dataclass(frozen=True, slots=True)
class AnalyticMagneticPotential:
    field: object
    provenance: Mapping = dataclass_field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Potential provenance must be a mapping")
        object.__setattr__(self, "field", _detached_native_field(self.field))
        object.__setattr__(self, "provenance", freeze_json(self.provenance))

    @property
    def source_id(self):
        return self.field.lens_key

    @property
    def digest(self):
        return json_digest(self.to_metadata())

    def to_metadata(self):
        return {
            "schema": SCHEMA, "source_id": self.source_id,
            "kind": type(self.field).__name__, "field_fingerprint": self.field.fingerprint,
            "units": {"position": "m", "A": "T m", "B": "T"},
            "gauge": ("local A=(-y*B0/2,x*B0/2,0)" if type(self.field) is FrozenAnalyticField
                      else "local A=(0,0,polynomial(x,y)*envelope(z))"),
            "derivative_order": 2, "derivative_domain": "strict interior of axial support",
            "outside_domain": "reject; no extrapolation or automatic zero fill",
            "physical_model": "existing finite analytical law; no new Maxwell or propagation qualification",
            "wave_admission": False, "provenance": thaw_json(self.provenance),
        }

    def to_dict(self):
        return {"metadata": self.to_metadata(), "field": asdict(self.field), "digest": self.digest}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"metadata", "field", "digest"}:
            raise ValueError("Invalid analytical potential record")
        metadata = record["metadata"]
        if not isinstance(metadata, Mapping) or metadata.get("schema") != SCHEMA:
            raise ValueError("Unknown analytical potential schema")
        types = {kind.__name__: kind for kind in (FrozenAnalyticField, FrozenPosedMultipole)}
        if type(metadata.get("kind")) is not str or metadata["kind"] not in types:
            raise ValueError("Unknown analytical potential basis")
        if not isinstance(record["field"], Mapping):
            raise ValueError("Invalid analytical potential coefficients")
        values = dict(record["field"])
        try:
            if not isinstance(values["registration"], Mapping):
                raise ValueError("Invalid analytical potential registration")
            values["registration"] = CoordinateRegistration(**values["registration"])
            result = cls(types[metadata["kind"]](**values), metadata["provenance"])
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid analytical potential coefficients") from error
        if result.digest != record["digest"] or json_digest(metadata) != json_digest(result.to_metadata()):
            raise ValueError("Analytical potential identity mismatch")
        return result

    def _positions(self, positions_m):
        points = np.asarray(positions_m, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Potential positions must be finite XYZ metres")
        local = self.field.registration.positions_global_to_local_m(points)
        lower, upper = np.asarray(self.field.native_support_mm)*1e-3
        if (np.any(local[..., 2] <= lower) or np.any(local[..., 2] >= upper)
                or np.any(np.hypot(local[..., 0], local[..., 1]) > self.field.radial_support_m)):
            raise ValueError("Position is outside the strict analytical potential domain or on a hard axial edge")
        return points

    def sample_jet(self, positions_m):
        """A, Cartesian Jacobian and Hessian in the strict support interior."""
        from temsim.physics.posed_lens_wave import vector_potential_jet
        points = self._positions(positions_m)
        shape = points.shape[:-1]
        rows = [vector_potential_jet((self.field,), point) for point in points.reshape(-1, 3)]
        if not rows:
            return {"A_t_m": np.empty((*shape, 3)), "jac_A_t": np.empty((*shape, 3, 3)),
                    "hess_A_t_per_m": np.empty((*shape, 3, 3, 3))}
        return {name: np.asarray([row[index] for row in rows]).reshape((*shape, *tail))
                for index, (name, tail) in enumerate((("A_t_m", (3,)), ("jac_A_t", (3, 3)),
                                                       ("hess_A_t_per_m", (3, 3, 3))))}

    def sample(self, positions_m):
        jet = self.sample_jet(positions_m)
        gradient = jet["jac_A_t"]
        magnetic = np.stack((gradient[..., 2, 1]-gradient[..., 1, 2],
                             gradient[..., 0, 2]-gradient[..., 2, 0],
                             gradient[..., 1, 0]-gradient[..., 0, 1]), axis=-1)
        return {"A_t_m": jet["A_t_m"], "B_t": magnetic}
