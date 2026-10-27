"""Immutable per-mode budget for accepted kinetic-expansion phase bounds.

This is an operator truncation bound, not a physical wave phase. The bound is
independent of mode weight. Numerical retries spend nothing until their final
operator result is accepted. Spatial/splitting and field errors are separate.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.potential_kinetic_expansion import validate_kinetic_order


SCHEMA = "potential-kinetic-phase-ledger-v1"


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _plane(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Kinetic phase plane must be finite metres")
    return float(value)


@dataclass(frozen=True, slots=True)
class KineticPhaseLedger:
    maximum_phase_rad: float
    events: tuple = ()
    binding: object = None

    def __post_init__(self):
        maximum = _number(self.maximum_phase_rad, "Maximum kinetic phase", True)
        binding = None if self.binding is None else freeze_json(self.binding)
        if binding is not None:
            if (not isinstance(binding, Mapping) or set(binding) !=
                    {"snapshot_digest", "energy_ev", "kinetic_order", "entrance_z_m"}
                    or type(binding["snapshot_digest"]) is not str or not binding["snapshot_digest"]):
                raise ValueError("Invalid kinetic phase source/energy/order binding")
            validate_kinetic_order(binding["kinetic_order"])
            _number(binding["energy_ev"], "Kinetic phase energy", True)
            _plane(binding["entrance_z_m"])
        events = freeze_json(self.events)
        if not isinstance(events, tuple) or (events and binding is None):
            raise ValueError("Kinetic phase history requires a bound source")
        expected = None if binding is None else binding["entrance_z_m"]
        for index, event in enumerate(events):
            if (not isinstance(event, Mapping) or set(event) !=
                    {"index", "lower_z_m", "upper_z_m", "accepted_bounds_rad"}
                    or type(event["index"]) is not int or event["index"] != index):
                raise ValueError("Invalid accepted kinetic phase event")
            lower, upper = _plane(event["lower_z_m"]), _plane(event["upper_z_m"])
            if upper <= lower or lower != expected:
                raise ValueError("Kinetic phase history must continue at the same observation plane")
            bounds = event["accepted_bounds_rad"]
            if not isinstance(bounds, tuple) or not bounds:
                raise ValueError("Kinetic phase event needs accepted residual bounds")
            for bound in bounds:
                _number(bound, "Accepted kinetic phase bound")
            expected = upper
        object.__setattr__(self, "maximum_phase_rad", maximum)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "binding", binding)
        if self.used_phase_rad > maximum:
            raise ValueError("Accepted kinetic truncation exceeds the cumulative phase budget")

    @property
    def used_phase_rad(self):
        return math.fsum(value for event in self.events for value in event["accepted_bounds_rad"])

    @property
    def remaining_phase_rad(self):
        return self.maximum_phase_rad-self.used_phase_rad

    @property
    def last_z_m(self):
        return (self.events[-1]["upper_z_m"] if self.events else
                None if self.binding is None else self.binding["entrance_z_m"])

    @property
    def digest(self):
        return json_digest(self._payload())

    def _payload(self):
        return {"schema": SCHEMA, "maximum_phase_rad": self.maximum_phase_rad,
                "binding": thaw_json(self.binding), "events": thaw_json(self.events)}

    def to_dict(self):
        return {**self._payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {
                "schema", "maximum_phase_rad", "binding", "events", "digest"} or value["schema"] != SCHEMA:
            raise ValueError("Invalid kinetic phase ledger record")
        result = cls(value["maximum_phase_rad"], tuple(value["events"]), value["binding"])
        if result.digest != value["digest"]:
            raise ValueError("Kinetic phase ledger digest mismatch")
        return result

    def require(self, snapshot_digest, energy_ev, kinetic_order, start_z_m):
        validate_kinetic_order(kinetic_order)
        energy_ev, start_z_m = _number(energy_ev, "Kinetic phase energy", True), _plane(start_z_m)
        binding = {"snapshot_digest": snapshot_digest, "energy_ev": energy_ev,
                   "kinetic_order": kinetic_order, "entrance_z_m": start_z_m}
        if self.binding is None:
            return KineticPhaseLedger(self.maximum_phase_rad, self.events, binding)
        if any(self.binding[key] != binding[key] for key in ("snapshot_digest", "energy_ev", "kinetic_order")):
            raise ValueError("Kinetic phase ledger source, energy or operator order changed")
        if float(start_z_m) != self.last_z_m:
            raise ValueError("Kinetic phase ledger cannot restart at a different observation plane")
        return self

    def append(self, lower_z_m, upper_z_m, accepted_bounds_rad):
        event = {"index": len(self.events), "lower_z_m": _plane(lower_z_m),
                 "upper_z_m": _plane(upper_z_m),
                 "accepted_bounds_rad": tuple(_number(x, "Accepted kinetic phase bound") for x in accepted_bounds_rad)}
        return KineticPhaseLedger(self.maximum_phase_rad, (*self.events, event), self.binding)
