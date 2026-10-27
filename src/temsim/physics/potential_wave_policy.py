"""Explicit finite-step operator identity for potential-wave continuation."""
from collections.abc import Mapping
import math

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.potential_kinetic_expansion import validate_kinetic_order


def _historical_operator_policy(record):
    """Read actual legacy stage receipts; missing items retain old defaults."""
    rows = [record]
    rows.extend(row for row in record.get("stages", ()) if isinstance(row, Mapping))
    for mode in record.get("modes", ()):
        if isinstance(mode, Mapping):
            rows.extend(row for row in mode.get("stages", ()) if isinstance(row, Mapping))
    rows.extend(item for row in tuple(rows) for item in row.get("residual_steps", ())
                if isinstance(item, Mapping))
    values = [row[key] for row in rows for key in ("scalar_exact_strang", "exact_scalar_potential_strang")
              if key in row]
    if any(type(value) is not bool for value in values) or len(set(values)) > 1:
        raise ValueError("Historical scalar splitting receipts are invalid or inconsistent")
    result = {"scalar_splitting": values[0] if values else False,
              "quartic_scalar_splitting": False, "higher_order_scalar_splitting": False,
              "residual_exponential_method": "taylor", "maximum_chebyshev_group": 8}
    if any(row.get("residual_exponential_method", "taylor") != "taylor" for row in rows):
        raise ValueError("Non-Taylor execution requires its explicit operator policy receipt")
    for key, default, allowed, kind in (
        ("kinetic_order", 4, (2, 4), int),
        ("reference_quadrature", "midpoint", ("midpoint", "cf4"), str),
    ):
        values = [row[key] for row in rows if key in row]
        if (any(type(value) is not kind or value not in allowed for value in values)
                or len(set(values)) > 1):
            raise ValueError("Historical operator receipts are invalid or inconsistent")
        result[key] = values[0] if values else default
    return result


def potential_wave_operator_policy(options, previous_record=None):
    """Validate the requested split and refuse an unannounced resume change.

    Historical executed checkpoints did not contain the quartic split. Their
    absent receipt therefore denotes False, never permission to adopt True.
    An independent numerical experiment may start from a supplied entrance
    while retaining its original producer record separately as provenance.
    """
    scalar = options.get("scalar_splitting", False)
    quartic = options.get("quartic_scalar_splitting", False)
    higher = options.get("higher_order_scalar_splitting", False)
    if any(type(value) is not bool for value in (scalar, quartic, higher)):
        raise ValueError("Scalar splitting policies must be explicit booleans")
    order = validate_kinetic_order(options.get("kinetic_order", 4))
    reference = options.get("reference_quadrature", "midpoint")
    if reference not in ("midpoint", "cf4") or type(reference) is not str:
        raise ValueError("Unknown quadratic reference quadrature")
    if quartic and (not scalar or order != 4):
        raise ValueError("Quartic scalar splitting requires scalar_splitting=True and kinetic_order=4")
    if higher and (not scalar or order not in (6, 8)):
        raise ValueError("Higher-order scalar splitting requires scalar_splitting=True and kinetic_order=6 or 8")
    method = options.get("residual_exponential_method", "taylor")
    group = options.get("maximum_chebyshev_group", 8)
    if type(method) is not str or method not in ("taylor", "chebyshev"):
        raise ValueError("Residual exponential method must be taylor or chebyshev")
    if type(group) is not int or not 1 <= group <= 32:
        raise ValueError("Maximum Chebyshev group must be an explicit integer from 1 to 32")
    policy = {"kinetic_order": order, "reference_quadrature": reference,
              "scalar_splitting": scalar, "quartic_scalar_splitting": quartic,
              "higher_order_scalar_splitting": higher,
              "residual_exponential_method": method, "maximum_chebyshev_group": group}
    receipt = {"schema": "potential-wave-operator-policy-v3", "policy": policy,
               "digest": json_digest(policy)}
    if previous_record is not None:
        if "potential_operator_policy" in previous_record:
            previous = previous_record["potential_operator_policy"]
            previous = thaw_json(previous) if isinstance(previous, Mapping) else None
            if previous is not None and previous.get("schema") in (
                    "potential-wave-operator-policy-v1", "potential-wave-operator-policy-v2"):
                old = previous.get("policy")
                keys = {"kinetic_order", "reference_quadrature", "scalar_splitting", "quartic_scalar_splitting"}
                old_order = (2, 4)
                if previous["schema"] == "potential-wave-operator-policy-v2":
                    keys.add("higher_order_scalar_splitting")
                    old_order = (2, 4, 6, 8)
                if (set(previous) != {"schema", "policy", "digest"} or not isinstance(old, Mapping)
                        or set(old) != keys or old.get("kinetic_order") not in old_order
                        or previous["digest"] != json_digest(old)):
                    raise ValueError("Invalid historical potential operator policy")
                upgraded = thaw_json(potential_wave_operator_policy(old))["policy"]
                previous = {"schema": receipt["schema"], "policy": upgraded, "digest": json_digest(upgraded)}
            if previous != receipt:
                raise ValueError("Potential operator policy changed or is invalid; restart from the declared entrance")
        elif previous_record.get("snapshot_digest") is not None:
            if quartic or higher:
                raise ValueError("Historical executed checkpoints have higher kinetic scalar splitting disabled; restart from the declared entrance")
            if policy != _historical_operator_policy(previous_record):
                raise ValueError("Historical operator policy changed; restart from the declared entrance")
    return freeze_json(receipt)


_SUPPORT_POLICIES = ("sampling_retry", "step_start_and_retry", "step_boundaries_and_retry", "weighted_segment_and_retry", "preview_segment_and_retry", "preview_phase_segment_and_retry")
_PRESSURE_ALLOCATION = "remaining_for_admissible_grid"
_TRANSITION_SCHEMA = "explicit-support-pressure-policy-transition-v1"
_PREVIEW_TRANSITION_SCHEMA = "explicit-preview-segment-policy-transition-v1"
_PHASE_PREVIEW_TRANSITION_SCHEMA = "explicit-preview-phase-policy-transition-v1"


def _pressure_policy(transition_digest=None):
    payload = {"policy": "weighted_segment_and_retry", "trigger": "typed_resource_failure_only",
               "allocation": _PRESSURE_ALLOCATION}
    if transition_digest is not None:
        payload["transition_digest"] = transition_digest
    return {"schema": "potential-support-policy-v3", **payload, "digest": json_digest(payload)}


def _preview_segment_policy(transition_digest=None):
    payload = {"policy": "preview_segment_and_retry",
        "trigger": "segment_start_first_residual_and_typed_resource_failure",
        "allocation": _PRESSURE_ALLOCATION,
        "voluntary": "first_residual_at_segment_start_once",
        "voluntary_allocation": "original_one_percent_event_limit"}
    if transition_digest is not None:
        payload["transition_digest"] = transition_digest
    return {"schema": "potential-support-policy-v4", **payload, "digest": json_digest(payload)}


def _preview_phase_policy(transition_digest=None):
    payload = {"policy": "preview_phase_segment_and_retry",
        "trigger": "segment_start_first_residual_and_typed_resource_failure",
        "allocation": _PRESSURE_ALLOCATION,
        "voluntary": "first_residual_at_segment_start_once",
        "voluntary_allocation": "original_one_percent_event_limit",
        "phase_preparation": {"backend": "CPU",
            "maximum_pixels": "min(65536, 2 * propagation_maximum_pixels)",
            "maximum_working_bytes": "unchanged_host_limit",
            "propagation_grid_limit": "unchanged"},
        "representation_error": "estimated_not_certified",
        "representation_estimator": "two_dense_grids_shifted_relative_5percent_absolute_1e-7_norm_charged_v2",
        "representation_allocation": "shared_original_complex_l2_allowance"}
    if transition_digest is not None:
        payload["transition_digest"] = transition_digest
    return {"schema": "potential-support-policy-v5", **payload, "digest": json_digest(payload)}


def _validate_support_policy(previous):
    if not isinstance(previous, Mapping):
        raise ValueError("Invalid stored numerical support policy")
    version = previous.get("schema")
    if version in ("potential-support-policy-v4", "potential-support-policy-v5"):
        builder = _preview_phase_policy if version == "potential-support-policy-v5" else _preview_segment_policy
        expected = builder(previous.get("transition_digest"))
        if (thaw_json(previous) != expected
                or ("transition_digest" in previous and (
                    type(previous["transition_digest"]) is not str or not previous["transition_digest"]))):
            raise ValueError("Invalid stored numerical support policy")
        return previous
    fields = {"schema", "policy", "digest"}
    if version in ("potential-support-policy-v2", "potential-support-policy-v3"):
        fields.add("trigger")
    if version == "potential-support-policy-v3":
        fields.add("allocation")
        if "transition_digest" in previous:
            fields.add("transition_digest")
    payload = (previous.get("policy") if version == "potential-support-policy-v1" else
               {key: previous.get(key) for key in fields-{"schema", "digest"}})
    if (set(previous) != fields or version not in ("potential-support-policy-v1", "potential-support-policy-v2", "potential-support-policy-v3")
            or type(previous.get("policy")) is not str or previous["policy"] not in _SUPPORT_POLICIES[:4]
            or (version != "potential-support-policy-v1" and (previous["policy"] != "weighted_segment_and_retry"
                or previous["trigger"] != "typed_resource_failure_only"))
            or (version == "potential-support-policy-v3" and previous["allocation"] != _PRESSURE_ALLOCATION)
            or ("transition_digest" in previous and (type(previous["transition_digest"]) is not str or not previous["transition_digest"]))
            or previous["digest"] != json_digest(payload)):
        raise ValueError("Invalid stored numerical support policy")
    return previous


def _request_without_cache(request):
    if not isinstance(request, Mapping):
        raise ValueError("Support transition requires complete numerical request receipts")
    value = thaw_json(request)
    if isinstance(value.get("execution"), dict):
        value["execution"].pop("cache_directory", None)
    # Earlier requests predate this opt-in field. Its exact False default has
    # no execution effect; True remains explicit and cannot be normalized.
    if value.get("column_preview_phase_reselection") is False:
        value.pop("column_preview_phase_reselection")
    if value.get("source_nominal_energy_preview") is False:
        value.pop("source_nominal_energy_preview")
    return value


def representation_estimate_history(record, *, initialize=False):
    """Validate production estimate evidence against its unchanged source budget.

    Initialization is reserved for the actual source entrance or an explicit
    forward policy transition. A restored v5 record must carry its history,
    including an empty one; absence never grants another error allowance.
    """
    from temsim.physics.potential_representation_estimates import (
        RepresentationEstimateLedger, require_combined_support_allowance,
    )
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_support_lineage import validate_inherited_support
    from temsim.physics.unified_wave_support import _allocation
    allocation = _allocation(record.get("unified_support_allocation"))
    binding = {"source_checkpoint_digest": allocation["source_checkpoint_digest"],
        "source_mode_id": allocation["source_mode_id"],
        "support_allocation_digest": json_digest(allocation)}
    inherited = validate_inherited_support(record)
    strict = NumericalSupportLedger.from_dict(record.get("support_ledger") if inherited is None else inherited["ledger"])
    raw = record.get("representation_estimates")
    if raw is None:
        if not initialize or inherited is not None:
            raise ValueError("Preview phase continuation lost its representation estimate history")
        history = RepresentationEstimateLedger.for_support(strict, source_binding=binding)
    else:
        history = RepresentationEstimateLedger.from_dict(raw)
    history.require_source_binding(binding)
    require_combined_support_allowance(strict, history)
    return history


def _validate_transition_receipt(receipt):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.unified_wave_support import _allocation
    fields = {"schema", "previous_context", "previous_context_digest", "source_reference", "native_plane_m",
              "previous_implementation", "implementation", "previous_request", "request", "new_policy", "digest"}
    if (not isinstance(receipt, Mapping) or set(receipt) != fields
            or receipt["schema"] not in (_TRANSITION_SCHEMA, _PREVIEW_TRANSITION_SCHEMA, _PHASE_PREVIEW_TRANSITION_SCHEMA)
            or receipt["digest"] != json_digest({key: value for key, value in receipt.items() if key != "digest"})):
        raise ValueError("Invalid explicit support policy transition receipt")
    previous = receipt["previous_context"]
    phase_preview = receipt["schema"] == _PHASE_PREVIEW_TRANSITION_SCHEMA
    preview = receipt["schema"] in (_PREVIEW_TRANSITION_SCHEMA, _PHASE_PREVIEW_TRANSITION_SCHEMA)
    if (not isinstance(previous, Mapping) or json_digest(previous) != receipt["previous_context_digest"]
            or (not preview and "support_policy_transition" in previous)):
        raise ValueError("Support policy transition lost its original context")
    old_policy = _validate_support_policy(previous.get("potential_support_policy"))
    if phase_preview:
        if (old_policy["schema"] not in ("potential-support-policy-v3", "potential-support-policy-v4")
                or thaw_json(receipt["new_policy"]) != _preview_phase_policy()):
            raise ValueError("Preview phase transition requires an executed v3 or v4 prefix")
        validate_support_policy_transition(previous)
    elif preview:
        if (old_policy["schema"] != "potential-support-policy-v3"
                or thaw_json(receipt["new_policy"]) != _preview_segment_policy()):
            raise ValueError("Preview segment transition requires an executed resource-only v3 prefix")
        validate_support_policy_transition(previous)
    elif old_policy["schema"] != "potential-support-policy-v2" or thaw_json(receipt["new_policy"]) != _pressure_policy():
        raise ValueError("Only an explicit v2 to v3 support policy transition is supported")
    old_ledger = NumericalSupportLedger.from_dict(previous.get("support_ledger"))
    if not preview and any("pressure_allocation" in event for event in old_ledger.events):
        raise ValueError("A v2 support prefix cannot contain v3 pressure allocations")
    if previous.get("unified_support_ledger_digest") != json_digest(previous["support_ledger"]):
        raise ValueError("Support transition prefix lost its published ledger digest")
    allocation = _allocation(previous.get("unified_support_allocation"))
    if (old_ledger.maximum_l2 != allocation["maximum_l2_per_reference"]
            or old_ledger.maximum_probability != allocation["maximum_probability_per_reference"]):
        raise ValueError("Support transition cannot change the original source allocation")
    source = receipt["source_reference"]
    source_fields = {"checkpoint_digest", "manifest_digest", "source_checkpoint_digest", "mode_id", "mode_weight",
                     "array_descriptors", "previous_executor_identity", "executor_identity"}
    if not isinstance(source, Mapping) or set(source) != source_fields:
        raise ValueError("Support transition requires exact complete-checkpoint source references")
    for key in source_fields-{"mode_weight", "array_descriptors"}:
        if type(source[key]) is not str or not source[key]:
            raise ValueError("Support transition source references must be nonempty strings")
    weight = source["mode_weight"]
    exceeds_source = (type(weight) in (int, float) and math.isfinite(weight)
                      and weight > allocation["input_weight_per_reference"])
    geometry = previous.get("potential_geometry")
    if preview and isinstance(geometry, Mapping):
        if geometry.get("output_weight_per_reference") != weight:
            raise ValueError("Preview transition changed the exact published mode weight")
        if exceeds_source:
            # Preserve a previously admitted lossless-roundoff change exactly;
            # never normalize it away or enlarge its original source budget.
            from temsim.physics.wave_flux import FLUX_ATOL, FLUX_RTOL
            exceeds_source = not math.isclose(weight, allocation["input_weight_per_reference"],
                                              rel_tol=FLUX_RTOL, abs_tol=FLUX_ATOL)
    if (type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
            or exceeds_source
            or source["source_checkpoint_digest"] != allocation["source_checkpoint_digest"]
            or source["mode_id"] != allocation["source_mode_id"]
            or previous.get("unified_wave_identity") != source["previous_executor_identity"]):
        raise ValueError("Support transition source mode or executor identity changed")
    arrays = source["array_descriptors"]
    if (not isinstance(arrays, Mapping)
            or set(arrays) != {"amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"}):
        raise ValueError("Support transition must bind all original complex-wave arrays")
    for descriptor in arrays.values():
        if (not isinstance(descriptor, Mapping) or set(descriptor) != {"file", "sha256", "shape", "nbytes"}
                or type(descriptor["file"]) is not str or not descriptor["file"]
                or type(descriptor["sha256"]) is not str or len(descriptor["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in descriptor["sha256"])
                or type(descriptor["nbytes"]) is not int or descriptor["nbytes"] <= 0
                or not isinstance(descriptor["shape"], (tuple, list)) or not descriptor["shape"]
                or any(type(size) is not int or size <= 0 for size in descriptor["shape"])):
            raise ValueError("Support transition array descriptor is invalid")
    z = receipt["native_plane_m"]
    if (type(z) not in (int, float) or not math.isfinite(z)
            or any(event["z_m"] > z for event in old_ledger.events)):
        raise ValueError("Support transition must retain the exact accepted native plane")
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.electric_wave_reference import ElectricWaveHistory
    for key, kind in (("kinetic_phase_ledger", KineticPhaseLedger), ("electric_wave_history", ElectricWaveHistory)):
        if previous.get(key) is not None and kind.from_dict(previous[key]).last_z_m != z:
            raise ValueError("Support transition phase history does not end at its exact native plane")
    if previous.get("potential_geometry") is not None and previous["potential_geometry"].get("last_z_m") != z:
        raise ValueError("Support transition geometry history does not end at its exact native plane")
    for key in ("previous_implementation", "implementation"):
        if type(receipt[key]) is not str or not receipt[key]:
            raise ValueError("Support transition requires both implementation identities")
    old_request = _request_without_cache(receipt["previous_request"])
    new_request = _request_without_cache(receipt["request"])
    if phase_preview:
        old_segment = old_request.pop("column_preview_segment_resampling", False)
        if (old_request.pop("column_preview_phase_reselection", False) is not False
                or new_request.pop("column_preview_phase_reselection", False) is not True
                or new_request.pop("column_preview_segment_resampling", False) is not True
                or old_segment is not (old_policy["schema"] == "potential-support-policy-v4")
                or any(new_request.get(key) is not True for key in (
                    "column_exploratory_preview", "column_support_projection", "column_weighted_resampling"))
                or new_request.get("column_model") != "physical"):
            raise ValueError("Preview phase transition must explicitly enable phase preparation and segment resampling")
    elif preview:
        if (old_request.pop("column_preview_segment_resampling", False) is not False
                or new_request.pop("column_preview_segment_resampling", False) is not True
                or any(new_request.get(key) is not True for key in (
                    "column_exploratory_preview", "column_support_projection", "column_weighted_resampling"))
                or new_request.get("column_model") != "physical"):
            raise ValueError("Preview transition must explicitly enable only segment resampling in an existing physical preview")
    if json_digest(old_request) != json_digest(new_request):
        raise ValueError("Support allocation transition cannot change other numerical request parameters")
    return old_ledger


def _validate_preview_history_prefix(previous, record):
    """A new projection timing cannot erase previously accepted propagation."""
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.electric_wave_reference import ElectricWaveHistory
    from temsim.physics.potential_material_phase_budget import validate_material_kinetic_record
    material = validate_material_kinetic_record(record)
    for key, kind, invariants, inherited_key in (
        ("kinetic_phase_ledger", KineticPhaseLedger, ("maximum_phase_rad", "binding"), "ledger"),
        ("electric_wave_history", ElectricWaveHistory, ("maximum_forward_error", "reference"), "electric_history"),
    ):
        if previous.get(key) is None:
            continue
        original = kind.from_dict(previous[key]).to_dict()
        raw = record.get(key) if material is None else material["prefix"].get(inherited_key)
        current = kind.from_dict(raw).to_dict()
        events = original["events"]
        if (any(json_digest(current[name]) != json_digest(original[name]) for name in invariants)
                or len(current["events"]) < len(events)
                or json_digest(current["events"][:len(events)]) != json_digest(events)):
            raise ValueError("Preview transition altered its original phase or electric history")
    original_geometry = previous.get("potential_geometry")
    if original_geometry is None or material is not None:
        return
    geometry = record.get("potential_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Preview transition lost its original geometry history")
    for key in ("geometry_digest", "source_scene_geometry_digest", "policy", "entry_z_m", "normalization_repair"):
        if json_digest(geometry.get(key)) != json_digest(original_geometry.get(key)):
            raise ValueError("Preview transition changed its original geometry contract")
    for key in ("physical_losses", "axial_resource_retries"):
        old, new = original_geometry.get(key, ()), geometry.get(key, ())
        if len(new) < len(old) or json_digest(new[:len(old)]) != json_digest(old):
            raise ValueError("Preview transition changed its original geometry event prefix")
    for key in ("accepted_steps", "sampling_refinements", "series_error_log_bound"):
        if geometry.get(key, -1) < original_geometry.get(key, 0):
            raise ValueError("Preview transition reset its original cumulative propagation history")


def validate_support_policy_transition(record):
    """Check the immutable v2 prefix on every restored v3 continuation.

    A migrated policy binds its receipt digest, so losing the receipt is an
    error. New ledger-v4 pressure events separately bind every prior ledger.
    Historical events retain their original individual one-percent limits.
    """
    policy = record.get("potential_support_policy")
    receipt = record.get("support_policy_transition")
    if isinstance(policy, Mapping) and policy.get("schema") == "potential-support-policy-v5":
        _validate_support_policy(policy)
        representation_estimate_history(record)
    link = policy.get("transition_digest") if isinstance(policy, Mapping) else None
    if receipt is None and link is None:
        return None
    if receipt is None or link is None or not isinstance(receipt, Mapping) or receipt.get("digest") != link:
        raise ValueError("Support policy transition receipt or policy link is missing")
    original = _validate_transition_receipt(receipt)
    expected_policy = (_preview_phase_policy(link) if receipt["schema"] == _PHASE_PREVIEW_TRANSITION_SCHEMA
                       else _preview_segment_policy(link) if receipt["schema"] == _PREVIEW_TRANSITION_SCHEMA
                       else _pressure_policy(link))
    if thaw_json(policy) != expected_policy:
        raise ValueError("Active support policy differs from its explicit transition")
    previous = receipt["previous_context"]
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_support_lineage import validate_inherited_support
    inherited = validate_inherited_support(record)
    raw = record.get("support_ledger") if inherited is None else inherited.get("ledger")
    current = NumericalSupportLedger.from_dict(raw)
    prefix = current.events[:len(original.events)]
    if (len(current.events) < len(original.events) or json_digest(prefix) != json_digest(original.events)
            or current.maximum_l2 != original.maximum_l2 or current.maximum_probability != original.maximum_probability
            or json_digest(record.get("unified_support_allocation")) != json_digest(previous["unified_support_allocation"])):
        raise ValueError("Support transition altered its original ledger prefix or source budget")
    # Exactly the original source and physical operator survive the numerical
    # allocation change. Their propagation histories may only advance later.
    for key in ("snapshot_digest", "source_checkpoint_digest", "source_mode_id", "source_mode_weight_per_reference",
                "potential_operator_policy"):
        if key in previous and json_digest(record.get(key)) != json_digest(previous[key]):
            raise ValueError("Support transition changed a source or physical operator identity")
    if record.get("unified_wave_identity") != receipt["source_reference"]["executor_identity"]:
        raise ValueError("Support transition belongs to another field executor")
    new_events = current.events[len(original.events):]
    if any(event["z_m"] < receipt["native_plane_m"] for event in new_events):
        raise ValueError("Support transition cannot authorize earlier numerical projections")
    if receipt["schema"] in (_PREVIEW_TRANSITION_SCHEMA, _PHASE_PREVIEW_TRANSITION_SCHEMA):
        _validate_preview_history_prefix(previous, record)
    if receipt["schema"] == _PHASE_PREVIEW_TRANSITION_SCHEMA:
        original_history = representation_estimate_history(previous, initialize=True)
        history = representation_estimate_history(record)
        history.require_prefix(original_history)
    return receipt


def transition_support_policy(record, *, source_reference, native_plane_m,
        previous_implementation, implementation, previous_request, request):
    """Explicitly adopt pressure-target allocation without changing any wave.

    The caller must verify committed complete-checkpoint payload checksums and
    the unchanged physical snapshot before invoking this metadata-only gate.
    The old context, policy, full ledger and fixed source allocation remain
    embedded unchanged. No event, weight, budget or phase history is issued.
    """
    previous = thaw_json(record)
    receipt = {"schema": _TRANSITION_SCHEMA, "previous_context": previous,
        "previous_context_digest": json_digest(previous), "source_reference": thaw_json(source_reference),
        "native_plane_m": native_plane_m, "previous_implementation": previous_implementation,
        "implementation": implementation, "previous_request": thaw_json(previous_request),
        "request": thaw_json(request), "new_policy": _pressure_policy()}
    receipt["digest"] = json_digest(receipt)
    _validate_transition_receipt(receipt)
    updated = {**previous, "potential_support_policy": _pressure_policy(receipt["digest"]),
               "support_policy_transition": receipt,
               "unified_wave_identity": source_reference["executor_identity"]}
    validate_support_policy_transition(updated)
    return freeze_json(updated)


def transition_preview_segment_policy(record, *, source_reference, native_plane_m,
        previous_implementation, implementation, previous_request, request):
    """Enable future preview segment starts while retaining an executed v3 prefix.

    This metadata-only gate never changes an array, phase, weight or budget.
    The caller must verify the complete committed mixture, every original
    payload checksum and unchanged fields/geometry before applying it per mode.
    Previously linked transitions remain embedded and independently validated.
    Only the explicit segment-resampling flag may change in the numerical
    request, apart from the storage path and recorded implementation identity.
    """
    previous = thaw_json(record)
    receipt = {"schema": _PREVIEW_TRANSITION_SCHEMA, "previous_context": previous,
        "previous_context_digest": json_digest(previous), "source_reference": thaw_json(source_reference),
        "native_plane_m": native_plane_m, "previous_implementation": previous_implementation,
        "implementation": implementation, "previous_request": thaw_json(previous_request),
        "request": thaw_json(request), "new_policy": _preview_segment_policy()}
    receipt["digest"] = json_digest(receipt)
    _validate_transition_receipt(receipt)
    updated = {**previous, "potential_support_policy": _preview_segment_policy(receipt["digest"]),
               "support_policy_transition": receipt,
               "unified_wave_identity": source_reference["executor_identity"]}
    validate_support_policy_transition(updated)
    return freeze_json(updated)


def transition_preview_phase_policy(record, *, source_reference, native_plane_m,
        previous_implementation, implementation, previous_request, request):
    """Adopt explicit estimated phase preparation without resetting any prefix.

    The caller verifies the complete mixture, every array checksum and exact
    unchanged physical snapshot. Only the new phase flag and, for a v3 prefix,
    the segment-start flag may become true. Previously accepted complex waves,
    weights, strict projection limits and all physical histories remain intact.
    """
    previous = thaw_json(record)
    receipt = {"schema": _PHASE_PREVIEW_TRANSITION_SCHEMA, "previous_context": previous,
        "previous_context_digest": json_digest(previous), "source_reference": thaw_json(source_reference),
        "native_plane_m": native_plane_m, "previous_implementation": previous_implementation,
        "implementation": implementation, "previous_request": thaw_json(previous_request),
        "request": thaw_json(request), "new_policy": _preview_phase_policy()}
    receipt["digest"] = json_digest(receipt)
    _validate_transition_receipt(receipt)
    updated = {**previous, "potential_support_policy": _preview_phase_policy(receipt["digest"]),
               "support_policy_transition": receipt,
               "unified_wave_identity": source_reference["executor_identity"],
               "representation_estimates": representation_estimate_history(previous, initialize=True).to_dict()}
    validate_support_policy_transition(updated)
    return freeze_json(updated)


def potential_support_policy(options, previous_record=None):
    """Bind active projection timing; inherited material prefixes are read-only.

    The policy does not issue any budget. Historical active ledgers infer the
    executed timing from actual kernel stages, defaulting to sampling retries.
    """
    allowed = _SUPPORT_POLICIES
    selected = options.get("support_projection_policy", "sampling_retry")
    if type(selected) is not str or selected not in allowed:
        raise ValueError("Invalid numerical support projection policy")
    record = {} if previous_record is None else previous_record
    from temsim.physics.potential_support_lineage import validate_inherited_support
    prefix = validate_inherited_support(record, options)
    previous = record.get("potential_support_policy")
    if previous is not None:
        _validate_support_policy(previous)
        if previous["schema"] == "potential-support-policy-v5":
            representation_estimate_history(record)
    validate_support_policy_transition(record)
    old_active = record.get("support_ledger") is not None
    active = old_active or options.get("support_ledger") is not None
    if prefix is not None:
        # The prefix validator already forbids explicit projection options or
        # an active descendant ledger. Retain evidence, not permission.
        if previous is not None:
            return freeze_json(thaw_json(previous))
        return None
    if previous is not None and not old_active:
        raise ValueError("Saved numerical support policy lost its active ledger")
    if not active:
        return None
    if previous is None and old_active and record.get("snapshot_digest") is not None:
        rows = [record, *record.get("stages", ())]
        rows.extend(stage for mode in record.get("modes", ()) if isinstance(mode, Mapping)
                    for stage in mode.get("stages", ()))
        values = [row["support_projection_policy"] for row in rows
                  if isinstance(row, Mapping) and "support_projection_policy" in row]
        if any(type(value) is not str or value not in allowed for value in values) or len(set(values)) > 1:
            raise ValueError("Historical numerical support policies conflict")
        previous_policy = values[0] if values else "sampling_retry"
    else:
        previous_policy = None if previous is None else previous["policy"]
    if previous_policy is not None and selected != previous_policy:
        raise ValueError("Numerical support projection policy changed; restart from the declared entrance")
    if selected == "weighted_segment_and_retry":
        receipt = _pressure_policy(None if previous is None else previous.get("transition_digest"))
        if (previous is not None and thaw_json(previous) != receipt) or (
                previous is None and old_active and record.get("snapshot_digest") is not None):
            raise ValueError("Numerical support projection trigger changed or allocation changed; an explicit v2 to v3 transition is required")
        return freeze_json(receipt)
    if selected == "preview_segment_and_retry":
        receipt = _preview_segment_policy(None if previous is None else previous.get("transition_digest"))
        if (previous is not None and thaw_json(previous) != receipt) or (
                previous is None and old_active and record.get("snapshot_digest") is not None):
            raise ValueError("Preview segment timing requires an explicit v3 to v4 transition")
        return freeze_json(receipt)
    if selected == "preview_phase_segment_and_retry":
        receipt = _preview_phase_policy(None if previous is None else previous.get("transition_digest"))
        if (previous is not None and thaw_json(previous) != receipt) or (
                previous is None and old_active and record.get("snapshot_digest") is not None):
            raise ValueError("Preview phase preparation requires an explicit v3/v4 to v5 transition")
        return freeze_json(receipt)
    return freeze_json({"schema": "potential-support-policy-v1", "policy": selected,
                        "digest": json_digest(selected)})


__all__ = ("potential_wave_operator_policy", "potential_support_policy", "transition_support_policy",
           "transition_preview_segment_policy", "transition_preview_phase_policy", "validate_support_policy_transition",
           "representation_estimate_history")
