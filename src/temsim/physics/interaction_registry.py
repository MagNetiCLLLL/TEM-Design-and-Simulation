"""Immutable ownership metadata for physical propagation interactions.

The registry describes which physical response is executed, not how a solver
discretizes it. Coverage keys name atomic responses (for example one coil
basis, a joint circuit member, or a separately sourced residual). A reduced
map must claim every response it replaces. Observations never execute.

Distributed intervals conflict only on positive-length overlap, so adjacent
intervals are allowed. A point action is inside both endpoints of a covered
interval: a lens cannot reapply its total response at the interval exit.
This coverage rule is separate from a solver's before/after-plane convention.
Numerical half-steps can share an action ID and the same physical interval,
with positive fractions summing to one. No field or propagation is calculated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real

from temsim.immutable_json import json_digest


RECORD_SCHEMA = "temforge-interaction-record-v1"
REGISTRY_SCHEMA = "temforge-interaction-registry-v1"
REPRESENTATIONS = frozenset({
    "field", "reduced", "residual", "observation", "geometry_boundary",
    "material", "detector",
})


class InteractionOwnershipError(ValueError):
    """Invalid, ambiguous, or multiply executed interaction ownership."""


def _identifier(value, name):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(character) < 32 for character in value)):
        raise InteractionOwnershipError(f"{name} must be a nonempty identifier without surrounding whitespace or control characters")
    return value


def _identifiers(values, name, *, allow_empty=True):
    if not isinstance(values, (tuple, list)):
        raise InteractionOwnershipError(f"{name} must be a list or tuple of identifiers")
    result = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not result:
        raise InteractionOwnershipError(f"{name} must name at least one physical effect")
    if len(set(result)) != len(result):
        raise InteractionOwnershipError(f"{name} contains duplicate identifiers")
    return tuple(sorted(result))


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InteractionOwnershipError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (ValueError, OverflowError) as error:
        raise InteractionOwnershipError(f"{name} must be a finite real number") from error
    if not math.isfinite(converted):
        raise InteractionOwnershipError(f"{name} must be a finite real number")
    return converted


def _document(value, expected_schema, keys):
    if not isinstance(value, Mapping):
        raise InteractionOwnershipError("Interaction metadata must be a JSON object")
    if value.get("schema") != expected_schema:
        raise InteractionOwnershipError(f"Unsupported interaction metadata schema; expected {expected_schema!r}")
    missing, extra = keys - set(value), set(value) - keys
    if missing or extra:
        raise InteractionOwnershipError(
            f"Invalid interaction metadata fields; missing={sorted(missing)!r}, unknown={sorted(extra, key=str)!r}")


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """One declared response, or one numerical partition of that response.

    ``owner_id`` identifies physical ownership; logical controls and circuit
    membership are separate metadata. ``covered_effects`` is authoritative for
    conflicts: joint totals include member keys, and complete reduced maps
    include all replaced field and explicit-action keys. A residual claims its
    own response key and needs provenance; the residual label never permits
    it to execute a field-derived total a second time.
    """

    interaction_id: str
    owner_id: str
    effect_family: str
    representation: str
    interval_mm: tuple[float, float]
    covered_effects: tuple[str, ...] = ()
    source_digest: str = ""
    circuit_id: str | None = None
    logical_drive_ids: tuple[str, ...] = ()
    residual_provenance: str | None = None
    action_id: str | None = None
    execution_fraction: float = 1.0

    def __post_init__(self):
        for name in ("interaction_id", "owner_id", "effect_family", "representation"):
            _identifier(getattr(self, name), name)
        if self.representation not in REPRESENTATIONS:
            raise InteractionOwnershipError(f"{self.interaction_id}: unknown representation {self.representation!r}")
        if not isinstance(self.interval_mm, (tuple, list)) or len(self.interval_mm) != 2:
            raise InteractionOwnershipError(f"{self.interaction_id}: interval_mm needs a start and stop")
        interval = tuple(_finite_real(value, "interval_mm") for value in self.interval_mm)
        if interval[1] < interval[0]:
            raise InteractionOwnershipError(f"{self.interaction_id}: interval_mm must be ordered")
        coverage = _identifiers(self.covered_effects, "covered_effects")
        if not coverage:
            coverage = (f"{self.owner_id}/{self.effect_family}",)
        drives = _identifiers(self.logical_drive_ids, "logical_drive_ids")
        if not isinstance(self.source_digest, str):
            raise InteractionOwnershipError(f"{self.interaction_id}: source_digest must be a string")
        if self.source_digest:
            _identifier(self.source_digest, "source_digest")
        if self.circuit_id is not None:
            _identifier(self.circuit_id, "circuit_id")
        if self.representation == "residual":
            _identifier(self.residual_provenance, "residual_provenance")
        elif self.residual_provenance is not None:
            raise InteractionOwnershipError(f"{self.interaction_id}: residual provenance requires the residual representation")
        action = self.interaction_id if self.action_id is None else _identifier(self.action_id, "action_id")
        fraction = _finite_real(self.execution_fraction, "execution_fraction")
        if not 0. < fraction <= 1.:
            raise InteractionOwnershipError(f"{self.interaction_id}: execution_fraction must be in (0, 1]")
        object.__setattr__(self, "interval_mm", interval)
        object.__setattr__(self, "covered_effects", coverage)
        object.__setattr__(self, "logical_drive_ids", drives)
        object.__setattr__(self, "action_id", action)
        object.__setattr__(self, "execution_fraction", fraction)

    @property
    def executes(self):
        return self.representation != "observation"

    def validate(self):
        # Reconstructing also validates metadata restored by external callers.
        type(self).from_dict(self.to_dict())
        return self

    def to_dict(self):
        return {
            "schema": RECORD_SCHEMA,
            "interaction_id": self.interaction_id,
            "owner_id": self.owner_id,
            "effect_family": self.effect_family,
            "representation": self.representation,
            "interval_mm": list(self.interval_mm),
            "interval_unit": "mm",
            "covered_effects": list(self.covered_effects),
            "source_digest": self.source_digest,
            "circuit_id": self.circuit_id,
            "logical_drive_ids": list(self.logical_drive_ids),
            "residual_provenance": self.residual_provenance,
            "action_id": self.action_id,
            "execution_fraction": self.execution_fraction,
        }

    @classmethod
    def from_dict(cls, value):
        keys = {"schema", "interaction_id", "owner_id", "effect_family", "representation",
                "interval_mm", "interval_unit", "covered_effects", "source_digest", "circuit_id",
                "logical_drive_ids", "residual_provenance", "action_id", "execution_fraction"}
        _document(value, RECORD_SCHEMA, keys)
        if value["interval_unit"] != "mm":
            raise InteractionOwnershipError("Interaction intervals must explicitly use mm")
        return cls(**{key: item for key, item in value.items()
                      if key not in {"schema", "interval_unit"}})


def _overlaps(first, second):
    a, b = first
    c, d = second
    if a == b:
        return a == c if c == d else c <= a <= d
    if c == d:
        return a <= c <= b
    return max(a, c) < min(b, d)


@dataclass(frozen=True, slots=True)
class InteractionRegistry:
    """Validated immutable interaction set with deterministic JSON identity."""

    records: tuple[InteractionRecord, ...] = ()

    def __post_init__(self):
        if not isinstance(self.records, (tuple, list)):
            raise InteractionOwnershipError("Interaction registry records must be a list or tuple")
        if any(not isinstance(record, InteractionRecord) for record in self.records):
            raise InteractionOwnershipError("Interaction registry entries must be InteractionRecord values")
        object.__setattr__(self, "records", tuple(self.records))
        self.validate()

    def _physical_actions(self):
        groups = {}
        for record in self.records:
            if record.executes:
                groups.setdefault(record.action_id, []).append(record)
        actions = []
        for action_id, pieces in groups.items():
            first = replace(pieces[0], interaction_id=action_id, execution_fraction=1.)
            for piece in pieces[1:]:
                candidate = replace(piece, interaction_id=action_id, execution_fraction=1.)
                if candidate != first:
                    raise InteractionOwnershipError(f"Action {action_id!r} has incompatible numerical partitions; owner, coverage, interval and source must agree")
            total = math.fsum(piece.execution_fraction for piece in pieces)
            if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
                raise InteractionOwnershipError(f"Action {action_id!r} executes a total fraction of {total:.17g}; numerical partitions must sum to one")
            actions.append(first)
        return tuple(actions)

    def validate(self):
        ids = set()
        for record in self.records:
            if record.interaction_id in ids:
                raise InteractionOwnershipError(f"Duplicate interaction ID {record.interaction_id!r}")
            ids.add(record.interaction_id)
        owners = {}
        for action in self._physical_actions():
            for effect in action.covered_effects:
                for previous in owners.get(effect, ()):
                    if _overlaps(previous.interval_mm, action.interval_mm):
                        raise InteractionOwnershipError(
                            f"Effect {effect!r} is executed twice: {previous.action_id!r} "
                            f"({previous.representation}, owner {previous.owner_id!r}, interval {previous.interval_mm} mm) "
                            f"overlaps {action.action_id!r} ({action.representation}, owner {action.owner_id!r}, "
                            f"interval {action.interval_mm} mm)")
                owners.setdefault(effect, []).append(action)
        return self

    @property
    def digest(self):
        return json_digest(self.to_dict())

    @property
    def executing_action_ids(self):
        return tuple(action.action_id for action in self._physical_actions())

    def to_dict(self):
        return {"schema": REGISTRY_SCHEMA, "records": [record.to_dict() for record in self.records]}

    @classmethod
    def from_dict(cls, value):
        _document(value, REGISTRY_SCHEMA, {"schema", "records"})
        if not isinstance(value["records"], (tuple, list)):
            raise InteractionOwnershipError("Interaction registry records must be a JSON array")
        return cls(tuple(InteractionRecord.from_dict(record) for record in value["records"]))
