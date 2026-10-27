"""Immutable composition of explicitly supported static potential snapshots.

This is a P1 representation boundary, not a new propagation operator. Sources
must already represent compatible independent contributions. This module does
not solve a coupled material problem, reconstruct a potential from B, infer
unknown exterior values, or admit these fields for coherent propagation.

Static electromagnetic gauge changes use A'=A+grad(chi), phi'=phi and
psi'=exp(i*q*chi/hbar)*psi with electron charge q=-e. This differs from a
numerical wave carrier change. The gauge is applied once after composition.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
import math
from numbers import Real

import numpy as np
from scipy.constants import e, hbar

from temsim.immutable_json import json_digest, thaw_json
from temsim.physics.physical_scene import PhysicalSceneSnapshot


SCHEMA = "electromagnetic-potential-snapshot-v1"
_PROVIDERS = {
    "gun_alignment": ("magnetic", "temsim.physics.gun_magnetic_potential", "GunMagneticPotential"),
    "axisymmetric_fem": ("magnetic", "temsim.physics.axisymmetric_potential", "AxisymmetricMagneticPotential"),
    "analytic_magnetic": ("magnetic", "temsim.physics.analytic_potential", "AnalyticMagneticPotential"),
    "analytic_domain": ("magnetic", "temsim.physics.analytic_potential_domain", "AnalyticPotentialDomain"),
    "high_order_round": ("magnetic", "temsim.physics.high_order_round_lens", "HighOrderRoundLensPotential"),
    "regular_axisymmetric": ("magnetic", "temsim.physics.regular_axisymmetric_magnetic_potential", "RegularAxisymmetricMagneticPotential"),
    "electric_snapshot": ("electric", "temsim.physics.electric_potential_snapshot", "ElectricPotentialSnapshot"),
}


def _identifier(value, name):
    if (not isinstance(value, str) or not value or value.strip() != value
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"{name} must be a nonempty identifier")
    return value


def _provider_class(tag):
    if not isinstance(tag, str) or tag not in _PROVIDERS:
        raise ValueError(f"Unsupported static potential provider tag: {tag!r}")
    _, module_name, name = _PROVIDERS[tag]
    return getattr(import_module(module_name), name)


def _provider_tag(potential, kind):
    # Match names first so optional adapters are not imported while unrelated
    # sources are captured. The final exact-type check rejects custom subclasses.
    for tag, (role, module, name) in _PROVIDERS.items():
        if type(potential).__module__ == module and type(potential).__name__ == name:
            if role != kind or type(potential) is not _provider_class(tag):
                break
            return tag
    raise ValueError(f"Unsupported {kind} potential provider; an exact qualified snapshot type is required")


def _positions(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape[-1:] != (3,) or not np.isfinite(array).all():
        raise ValueError("Potential coordinates must be finite real XYZ metres")
    return np.asarray(array, dtype=np.float64)


def _canonical_owner(scene, owner):
    """Resolve declared aliases, retaining separate coils and field channels."""
    current, seen = owner, set()
    while current not in seen:
        seen.add(current)
        host = scene.owner_for(current)
        if host is None and ":" in current:
            parent, suffix = current.rsplit(":", 1)
            parent_host = scene.owner_for(parent)
            host = f"{parent_host}:{suffix}" if parent_host is not None else None
        if host is None or host == current:
            return current
        current = host
    raise ValueError(f"Cyclic potential owner alias: {owner}")


def _immutable_array(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return np.frombuffer(np.asarray(array, dtype=np.float64).tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, slots=True, eq=False)
class StaticPolynomialGauge:
    """chi=c+g.r+r.H.r/2 in T m^2; H must be exactly symmetric."""

    constant_t_m2: float = 0.0
    gradient_t_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hessian_t: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    digest: str = field(init=False)

    def __post_init__(self):
        if (isinstance(self.constant_t_m2, (bool, np.bool_)) or not isinstance(self.constant_t_m2, Real)
                or not math.isfinite(self.constant_t_m2)):
            raise ValueError("Gauge constant must be a finite real value in T m^2")
        gradient = _immutable_array(self.gradient_t_m, (3,), "Gauge gradient")
        hessian = _immutable_array(self.hessian_t, (3, 3), "Gauge Hessian")
        if not np.array_equal(hessian, hessian.T):
            raise ValueError("Gauge Hessian must be symmetric")
        object.__setattr__(self, "constant_t_m2", float(self.constant_t_m2))
        object.__setattr__(self, "gradient_t_m", gradient)
        object.__setattr__(self, "hessian_t", hessian)
        object.__setattr__(self, "digest", json_digest(self.to_dict()))

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def to_dict(self):
        return {"schema": "static-polynomial-gauge-v1", "coordinates": "global_xyz_m",
                "constant_t_m2": self.constant_t_m2, "gradient_t_m": self.gradient_t_m.tolist(),
                "hessian_t": self.hessian_t.tolist(), "charge_convention": "electron q=-e"}

    @classmethod
    def from_dict(cls, record):
        keys = {"schema", "coordinates", "constant_t_m2", "gradient_t_m", "hessian_t", "charge_convention"}
        if (not isinstance(record, Mapping) or set(record) != keys
                or record["schema"] != "static-polynomial-gauge-v1"
                or record["coordinates"] != "global_xyz_m" or record["charge_convention"] != "electron q=-e"):
            raise ValueError("Invalid static polynomial gauge record")
        return cls(record["constant_t_m2"], record["gradient_t_m"], record["hessian_t"])

    def chi_t_m2(self, positions_m):
        positions = _positions(positions_m)
        result = (self.constant_t_m2+positions@self.gradient_t_m
                  +.5*np.einsum("...i,ij,...j->...", positions, self.hessian_t, positions))
        if not np.isfinite(result).all():
            raise ValueError("Gauge polynomial exceeds the finite numerical range")
        return result

    def vector_potential_shift_t_m(self, positions_m):
        result = self.gradient_t_m+_positions(positions_m)@self.hessian_t.T
        if not np.isfinite(result).all():
            raise ValueError("Gauge gradient exceeds the finite numerical range")
        return result

    def electron_phase_factor(self, positions_m):
        """Return exp(-i*e*chi/hbar), with the source phase otherwise retained."""
        angle = (-e/hbar)*self.chi_t_m2(positions_m)
        if not np.isfinite(angle).all():
            raise ValueError("Gauge phase exceeds the finite numerical range")
        return np.exp(1j*angle)


@dataclass(frozen=True, slots=True)
class PotentialContribution:
    """One frozen contribution and all atomic owners it already covers.

    A joint nonlinear solution lists every member channel in owner_ids. Linear
    independently solved bases instead claim their own distinct channel IDs.
    These are explicit coverage declarations, never inferred from a field norm.
    Atomic IDs can name separate coils such as host:0, so they need not be
    scene component keys. This generic record validates declared coverage;
    the production capture API supplies its connection to actual hardware.
    The containing snapshot resolves aliases explicitly declared by its scene;
    it never infers common ownership merely from shared material geometry.
    """

    source_id: str
    owner_ids: tuple[str, ...]
    potential: object
    kind: str

    def __post_init__(self):
        _identifier(self.source_id, "Potential source_id")
        if not isinstance(self.kind, str) or self.kind not in {"electric", "magnetic"}:
            raise ValueError("Potential kind must be electric or magnetic")
        if not isinstance(self.owner_ids, (tuple, list)) or not self.owner_ids:
            raise ValueError("Potential owner_ids must be a nonempty sequence")
        owners = tuple(_identifier(owner, "Potential atomic owner") for owner in self.owner_ids)
        if len(set(owners)) != len(owners):
            raise ValueError("Potential owner_ids contains duplicate atomic owners")
        object.__setattr__(self, "owner_ids", tuple(sorted(owners)))
        _provider_tag(self.potential, self.kind)

    def to_metadata(self):
        provider = self.potential
        metadata = provider.to_metadata() if hasattr(provider, "to_metadata") else thaw_json(provider.metadata)
        return {"source_id": self.source_id, "owner_ids": list(self.owner_ids), "kind": self.kind,
                "provider_type": _provider_tag(provider, self.kind), "provider_digest": provider.digest,
                "provider_metadata": metadata}

    def to_dict(self):
        return {"source_id": self.source_id, "owner_ids": list(self.owner_ids), "kind": self.kind,
                "provider_type": _provider_tag(self.potential, self.kind), "potential": self.potential.to_dict()}

    @classmethod
    def from_dict(cls, record):
        keys = {"source_id", "owner_ids", "kind", "provider_type", "potential"}
        if not isinstance(record, Mapping) or set(record) != keys:
            raise ValueError("Invalid potential contribution record")
        cls_provider = _provider_class(record["provider_type"])
        return cls(record["source_id"], record["owner_ids"], cls_provider.from_dict(record["potential"]), record["kind"])


@dataclass(frozen=True, slots=True, eq=False)
class ElectromagneticPotentialSnapshot:
    """One static scene, explicitly owned potentials, and a common gauge.

    physical_digest identifies declared scene inputs and coverage only; it is
    NOT a solved-field equivalence/cache key. numerical_digest also contains
    every retained provider representation and the final gauge. Unknown field
    regions stay unknown, even if a source's sampled amplitude would be zero.
    """

    scene: PhysicalSceneSnapshot
    contributions: tuple[PotentialContribution, ...]
    gauge: StaticPolynomialGauge = field(default_factory=StaticPolynomialGauge)
    physical_digest: str = field(init=False)
    numerical_digest: str = field(init=False)

    def __post_init__(self):
        if type(self.scene) is not PhysicalSceneSnapshot:
            raise ValueError("Potential snapshot requires a captured PhysicalSceneSnapshot")
        if not isinstance(self.contributions, (tuple, list)) or any(type(item) is not PotentialContribution for item in self.contributions):
            raise ValueError("Potential contributions must be declared immutable records")
        if type(self.gauge) is not StaticPolynomialGauge:
            raise ValueError("Potential snapshot requires a supported static gauge")
        rows = tuple(self.contributions)
        sources, coverage = set(), {}
        for row in rows:
            if row.source_id in sources:
                raise ValueError(f"Duplicate potential source_id: {row.source_id}")
            sources.add(row.source_id)
            for owner in row.owner_ids:
                canonical = _canonical_owner(self.scene, owner)
                key = row.kind, canonical
                if key in coverage:
                    raise ValueError(f"Duplicate {row.kind} potential ownership for {canonical}: {coverage[key]} and {row.source_id}")
                coverage[key] = row.source_id
        object.__setattr__(self, "contributions", rows)
        physical = json_digest({"scene": self.scene.digest, "coverage": sorted(
            (row.source_id, row.kind, row.owner_ids) for row in rows)})
        numerical = json_digest({"schema": SCHEMA, "physical_digest": physical,
            "providers": [(row.source_id, row.potential.digest) for row in rows], "gauge": self.gauge.digest})
        object.__setattr__(self, "physical_digest", physical)
        object.__setattr__(self, "numerical_digest", numerical)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @property
    def digest(self):
        return self.numerical_digest

    def to_metadata(self):
        return {"schema": SCHEMA, "scene_digest": self.scene.digest,
                "physical_digest": self.physical_digest, "numerical_digest": self.numerical_digest,
                "physical_digest_scope": "declared scene and atomic ownership only; not field equivalence",
                "static": True, "wave_admission": False,
                "scalar_potential_reference": "sum of captured cathode-rise contributions; no ground-reference conversion",
                "domain_policy": "every selected source must admit every query; no exterior zero fill",
                "composition_scope": "sum of already compatible independent sources; no new coupled material solve",
                "canonical_coverage": [{"source_id": row.source_id, "kind": row.kind,
                    "owner_ids": sorted(_canonical_owner(self.scene, owner) for owner in row.owner_ids)}
                    for row in self.contributions],
                "sources": [row.to_metadata() for row in self.contributions], "gauge": self.gauge.to_dict()}

    def to_dict(self):
        return {"schema": SCHEMA, "scene": self.scene.to_dict(),
                "contributions": [row.to_dict() for row in self.contributions], "gauge": self.gauge.to_dict(),
                "physical_digest": self.physical_digest, "numerical_digest": self.numerical_digest}

    @classmethod
    def from_dict(cls, record):
        keys = {"schema", "scene", "contributions", "gauge", "physical_digest", "numerical_digest"}
        if (not isinstance(record, Mapping) or set(record) != keys or record["schema"] != SCHEMA
                or not isinstance(record["contributions"], (tuple, list))):
            raise ValueError("Invalid electromagnetic potential snapshot record")
        result = cls(PhysicalSceneSnapshot.from_dict(record["scene"]),
            tuple(PotentialContribution.from_dict(row) for row in record["contributions"]),
            StaticPolynomialGauge.from_dict(record["gauge"]))
        if result.physical_digest != record["physical_digest"] or result.numerical_digest != record["numerical_digest"]:
            raise ValueError("Electromagnetic potential snapshot identity mismatch")
        return result

    def _selected(self, source_ids):
        if source_ids is None:
            return self.contributions
        if not isinstance(source_ids, (tuple, list)):
            raise ValueError("Selected source_ids must be a sequence of identifiers")
        selected = tuple(_identifier(item, "Selected potential source") for item in source_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("Selected source_ids must not repeat a source")
        unknown = set(selected)-{row.source_id for row in self.contributions}
        if unknown:
            raise ValueError("Unknown potential source: "+", ".join(sorted(unknown)))
        return tuple(row for row in self.contributions if row.source_id in selected)

    def sample_source(self, source_id, positions_m):
        """Return one source in its captured native gauge, without the final shift."""
        row, = self._selected((source_id,))
        return row.potential.sample(_positions(positions_m))

    def sample(self, positions_m, source_ids=None):
        """Sum selected sources then apply the common gauge exactly once.

        An explicit empty selection returns zero fields and the common pure
        gauge. This describes no selected sources, not known exterior vacuum.
        A failed source query publishes no partial sum.
        """
        positions = _positions(positions_m)
        result = {"phi_v": np.zeros(positions.shape[:-1]), "E_v_per_m": np.zeros_like(positions),
                  "A_t_m": np.zeros_like(positions), "B_t": np.zeros_like(positions)}
        for row in self._selected(source_ids):
            try:
                sample = row.potential.sample(positions)
            except ValueError as error:
                raise ValueError(f"Potential source {row.source_id}: {error}") from error
            names = ("phi_v", "E_v_per_m") if row.kind == "electric" else ("A_t_m", "B_t")
            for name in names:
                value = np.asarray(sample[name])
                if value.shape != result[name].shape or not np.isfinite(value).all():
                    raise ValueError(f"Source {row.source_id} returned invalid {name} values")
                result[name] += value
            # Each contribution has been consumed. Do not retain its bundle
            # or final array alias while evaluating the next source.
            del sample, value
        result["A_t_m"] += self.gauge.vector_potential_shift_t_m(positions)
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Combined potential exceeds the finite numerical range")
        return result


__all__ = ("PotentialContribution", "StaticPolynomialGauge", "ElectromagneticPotentialSnapshot")
