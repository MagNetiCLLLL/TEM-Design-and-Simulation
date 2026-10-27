"""Read-only common support history at a conditional material branch.

The prefix certifies only the numerical projections already performed before
branching. It is not a bound on conditional material dynamics or individual
normalized histories. No remaining projection budget is issued to children.
"""
from collections.abc import Mapping
import math

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.potential_wave_support import NumericalSupportLedger


_SCHEMA = "inherited-common-potential-support-v1"
_SCOPE = "executed common-prefix projections only; no conditional material or branch error bound"
_POLICY = "read-only shared prefix; all descendant support projection disabled"
_FIELDS = {"schema", "ledger", "entrance_checkpoint_digest", "entrance_mode_id",
    "snapshot_digest", "entrance_z_mm", "entrance_weight_per_reference", "scope", "policy", "digest"}


def validate_inherited_support(record, options=None):
    """Validate a shared prefix and reject any attempted budget restart."""
    value = record.get("inherited_potential_support")
    if value is None:
        if record.get("support_prefix_digest") is not None:
            raise ValueError("Conditional history lost its inherited support prefix")
        return None
    if (not isinstance(value, Mapping) or set(value) != _FIELDS or value["schema"] != _SCHEMA
            or value["scope"] != _SCOPE or value["policy"] != _POLICY):
        raise ValueError("Invalid inherited potential support lineage")
    payload = {key: value[key] for key in value if key != "digest"}
    if json_digest(payload) != value["digest"]:
        raise ValueError("Inherited support lineage digest changed")
    ledger = NumericalSupportLedger.from_dict(value["ledger"])
    for key in ("entrance_z_mm", "entrance_weight_per_reference"):
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or (key == "entrance_weight_per_reference" and number < 0)):
            raise ValueError("Inherited support entrance coordinate must be finite and weight finite and nonnegative")
    # The ledger records metres while checkpoints may own native millimetres.
    # Accept either exact conversion of the same plane, without an epsilon
    # that could merge genuinely distinct adjacent floating-point planes.
    entrance = value["entrance_z_mm"]
    if any(event["z_m"] > entrance*.001 and event["z_m"]*1000. > entrance for event in ledger.events):
        raise ValueError("Inherited support projection occurs after the declared entrance")
    for key in ("entrance_checkpoint_digest", "entrance_mode_id", "snapshot_digest"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError("Inherited support lineage requires complete source identity")
    if record.get("snapshot_digest", value["snapshot_digest"]) != value["snapshot_digest"]:
        raise ValueError("Inherited support belongs to a different potential snapshot")
    if record.get("support_prefix_digest", value["digest"]) != value["digest"]:
        raise ValueError("Conditional history changed its common support prefix")
    options = {} if options is None else options
    if (record.get("support_ledger") is not None or options.get("support_ledger") is not None
            or "support_mode_weight" in options or "support_projection_policy" in options):
        raise ValueError("Support projection is disabled after material branching; the common-prefix budget cannot be reset")
    return freeze_json(thaw_json(value))


def capture_inherited_support(checkpoint, snapshot_digest):
    """Retire one active parent ledger into an immutable, nonspendable prefix."""
    existing = validate_inherited_support(checkpoint.record)
    if existing is not None:
        if existing["snapshot_digest"] != snapshot_digest:
            raise ValueError("Inherited support belongs to a different potential snapshot")
        return existing
    raw = checkpoint.record.get("support_ledger")
    if raw is None:
        return None
    if len(checkpoint.beam.modes) != 1:
        raise ValueError("An active potential support ledger must have one incident mode")
    if checkpoint.record.get("snapshot_digest", snapshot_digest) != snapshot_digest:
        raise ValueError("Potential support snapshot changed before material branching")
    ledger = NumericalSupportLedger.from_dict(raw)
    mode = checkpoint.beam.modes[0]
    payload = {"schema": _SCHEMA, "ledger": ledger.to_dict(),
        "entrance_checkpoint_digest": checkpoint.digest, "entrance_mode_id": mode.mode_id,
        "snapshot_digest": snapshot_digest, "entrance_z_mm": checkpoint.plane_z_mm,
        "entrance_weight_per_reference": mode.weight_per_reference_electron,
        "scope": _SCOPE, "policy": _POLICY}
    prefix = freeze_json({**payload, "digest": json_digest(payload)})
    return validate_inherited_support({"snapshot_digest": snapshot_digest,
        "inherited_potential_support": prefix})


def common_support_summary(record):
    """One accounting entry regardless of the number of descendant histories."""
    value = validate_inherited_support(record)
    if value is None:
        return None
    ledger = NumericalSupportLedger.from_dict(value["ledger"])
    return freeze_json({"prefix_digest": value["digest"], "accounting_multiplicity": 1,
        "probability_removed_per_reference": ledger.used_probability,
        "prefix_l2_per_reference": ledger.used_l2, "scope": _SCOPE,
        "remaining_budget_reissued": False})
