"""Fixed source-weight support allocations for the unified column executor.

The request limits are relative to each mode at its original column entrance.
Per-reference allocations are epsilon*sqrt(weight) and delta*weight. Physical
absorption and conditional material branches never issue a replacement budget.
"""
from collections.abc import Mapping
import math

from temsim.immutable_json import json_digest, thaw_json
from temsim.physics.potential_support_lineage import validate_inherited_support
from temsim.physics.potential_wave_support import NumericalSupportLedger


_SCHEMA = "unified-source-support-allocation-v1"
_KEY = "unified_support_allocation"
_SCOPE = "numerical support projections only; subsequent grid/domain propagation needs independent convergence"


def _allocation(value, l2_limit=None, probability_limit=None):
    keys = {"schema", "source_checkpoint_digest", "source_mode_id", "input_weight_per_reference",
        "relative_l2_limit", "probability_fraction_limit", "maximum_l2_per_reference",
        "maximum_probability_per_reference", "budget_issued", "scope", "digest"}
    if not isinstance(value, Mapping) or set(value) != keys or value["schema"] != _SCHEMA or value["scope"] != _SCOPE:
        raise ValueError("Invalid unified source support allocation")
    if json_digest({k: v for k, v in value.items() if k != "digest"}) != value["digest"]:
        raise ValueError("Unified source support allocation digest changed")
    for key in ("source_checkpoint_digest", "source_mode_id"):
        if type(value[key]) is not str or not value[key]:
            raise ValueError("Unified support requires its original source identity")
    numbers = ("input_weight_per_reference", "relative_l2_limit", "probability_fraction_limit",
               "maximum_l2_per_reference", "maximum_probability_per_reference")
    if any(type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0 for key in numbers):
        raise ValueError("Unified support allocations must be finite and nonnegative")
    weight = value["input_weight_per_reference"]
    expected_l2 = value["relative_l2_limit"]*math.sqrt(weight)
    expected_probability = value["probability_fraction_limit"]*weight
    if (value["maximum_l2_per_reference"] != expected_l2
            or value["maximum_probability_per_reference"] != expected_probability
            or type(value["budget_issued"]) is not bool
            or value["budget_issued"] != (expected_l2 > 0 and expected_probability > 0)):
        raise ValueError("Unified support allocation does not match its fixed original mode weight")
    if ((l2_limit is not None and value["relative_l2_limit"] != l2_limit)
            or (probability_limit is not None and value["probability_fraction_limit"] != probability_limit)):
        raise ValueError("Unified support limits changed; restart from the actual gun source")
    return value


def prepare_unified_support(context, mode, *, enabled, l2_limit, probability_limit,
                            source_digest, allow_allocate):
    """Validate or create one allocation, without resetting its spent ledger."""
    record = thaw_json(context)
    prefix = validate_inherited_support(record)
    if not enabled:
        if record.get("support_ledger") is not None or _KEY in record:
            raise ValueError("Unified numerical support policy changed to disabled")
        return record
    if _KEY not in record:
        if not allow_allocate or prefix is not None or record.get("support_ledger") is not None:
            raise ValueError("Unified continuation lost its fixed source support allocation")
        weight = mode.weight_per_reference_electron
        maximum_l2, maximum_probability = l2_limit*math.sqrt(weight), probability_limit*weight
        payload = {"schema": _SCHEMA, "source_checkpoint_digest": source_digest,
            "source_mode_id": mode.mode_id, "input_weight_per_reference": weight,
            "relative_l2_limit": l2_limit, "probability_fraction_limit": probability_limit,
            "maximum_l2_per_reference": maximum_l2,
            "maximum_probability_per_reference": maximum_probability,
            "budget_issued": maximum_l2 > 0 and maximum_probability > 0, "scope": _SCOPE}
        record[_KEY] = {**payload, "digest": json_digest(payload)}
        record["support_ledger"] = (NumericalSupportLedger(maximum_l2, maximum_probability).to_dict()
                                    if payload["budget_issued"] else None)
        record["unified_support_ledger_digest"] = (None if record["support_ledger"] is None
                                                   else json_digest(record["support_ledger"]))
    allocation = _allocation(record[_KEY], l2_limit, probability_limit)
    if allocation["source_checkpoint_digest"] != source_digest:
        raise ValueError("Unified support allocation belongs to another original source checkpoint")
    expected_mode = mode.mode_id if prefix is None else prefix["entrance_mode_id"]
    if prefix is None and record.get("material_kinetic_history") is not None:
        from temsim.physics.potential_material_phase_budget import validate_material_kinetic_record
        history = validate_material_kinetic_record(record)
        expected_mode = history["prefix"]["mode_id"]
        if allocation["budget_issued"]:
            raise ValueError("Material continuation lost its inherited numerical support prefix")
    if allocation["source_mode_id"] != expected_mode:
        raise ValueError("Unified support allocation belongs to another original mode")
    raw = record.get("support_ledger") if prefix is None else prefix["ledger"]
    if ("unified_support_ledger_digest" not in record
            or record["unified_support_ledger_digest"] != (None if raw is None else json_digest(raw))):
        raise ValueError("Unified numerical support history changed or lost its publication receipt")
    if allocation["budget_issued"]:
        if raw is None:
            raise ValueError("Unified continuation lost its consumed numerical support ledger")
        ledger = NumericalSupportLedger.from_dict(raw)
        if (ledger.maximum_l2 != allocation["maximum_l2_per_reference"]
                or ledger.maximum_probability != allocation["maximum_probability_per_reference"]):
            raise ValueError("Unified support ledger limits differ from the original allocation")
    elif raw is not None:
        raise ValueError("A zero unified source allocation cannot acquire a support ledger")
    return record


def stamp_unified_support(context):
    """Bind the accepted ledger at publication, after successful propagation."""
    record = thaw_json(context)
    if _KEY in record:
        _allocation(record[_KEY])
        prefix = validate_inherited_support(record)
        raw = record.get("support_ledger") if prefix is None else prefix["ledger"]
        record["unified_support_ledger_digest"] = None if raw is None else json_digest(raw)
    return record


def unified_support_summary(contexts):
    """Count each original mode allocation once, including shared descendants."""
    unique = {}
    for context in contexts.values():
        if _KEY not in context:
            continue
        allocation = _allocation(context[_KEY])
        prefix = validate_inherited_support(context)
        raw = context.get("support_ledger") if prefix is None else prefix["ledger"]
        ledger = None if raw is None else NumericalSupportLedger.from_dict(raw)
        if allocation["budget_issued"] != (ledger is not None):
            raise ValueError("Unified support summary lost its original ledger")
        if (context.get("unified_support_ledger_digest") != (None if raw is None else json_digest(raw))
                or (ledger is not None and (ledger.maximum_l2 != allocation["maximum_l2_per_reference"]
                    or ledger.maximum_probability != allocation["maximum_probability_per_reference"]))):
            raise ValueError("Unified support summary changed its fixed allocation or published ledger")
        row = {"allocation_digest": allocation["digest"], "source_mode_id": allocation["source_mode_id"],
            "source_weight_per_reference": allocation["input_weight_per_reference"],
            "maximum_l2_per_reference": allocation["maximum_l2_per_reference"],
            "maximum_probability_per_reference": allocation["maximum_probability_per_reference"],
            "used_l2_per_reference": 0. if ledger is None else ledger.used_l2,
            "used_probability_per_reference": 0. if ledger is None else ledger.used_probability,
            "ledger_digest": None if ledger is None else json_digest(ledger.to_dict()),
            "material_prefix_digest": None if prefix is None else prefix["digest"],
            "further_projection_allowed": prefix is None and allocation["budget_issued"]}
        key = allocation["digest"]
        if key in unique and unique[key] != row:
            raise ValueError("Unified descendants disagree about their common support history")
        unique[key] = row
    rows = tuple(unique[key] for key in sorted(unique))
    return {"schema": "unified-source-support-summary-v1", "scope": _SCOPE,
        "allocations": rows, "accounting": "each original allocation or inherited common prefix counted once",
        "input_weight_per_reference": math.fsum(row["source_weight_per_reference"] for row in rows),
        "maximum_l2_per_reference": math.sqrt(math.fsum(row["maximum_l2_per_reference"]**2 for row in rows)),
        "maximum_probability_per_reference": math.fsum(row["maximum_probability_per_reference"] for row in rows),
        "used_l2_per_reference": math.sqrt(math.fsum(row["used_l2_per_reference"]**2 for row in rows)),
        "used_probability_per_reference": math.fsum(row["used_probability_per_reference"] for row in rows),
        "normalization_applied": False}


__all__ = ("prepare_unified_support", "stamp_unified_support", "unified_support_summary")
