"""Kinetic operator budgets across admitted conditional material histories.

The original prefix is shared read-only. Each history continues the same total
phase allowance; a proved energy-changing collision closes one energy-bound
segment and allocates only its remaining allowance to the next. Bounds concern
unnormalized linear/contractive branch operators, not pointwise wave phase or
relative errors of normalized conditional states.
"""
from collections.abc import Mapping
import math

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.potential_phase_budget import KineticPhaseLedger


_SCHEMA = "material-kinetic-history-v1"
_ELECTRIC_SCHEMA = "material-electromagnetic-history-v2"
_SCOPE = "sum of local kinetic operator bounds on an unnormalized contractive history; no normalized conditional or pointwise phase guarantee"


def _end(ledger):
    return ledger.events[-1]["upper_z_m"] if ledger.events else ledger.binding["entrance_z_m"]


def _stamp(payload):
    return freeze_json({**payload, "digest": json_digest(payload)})


def _electric_history(value):
    from temsim.physics.electric_wave_reference import ElectricWaveHistory
    result = ElectricWaveHistory.from_dict(value)
    if result.dynamic_state is not None:
        raise ValueError("Dynamic material histories require a separately qualified time-energy scattering operator")
    return result


def _electric_energy(history, snapshot, plane):
    if snapshot is None:
        raise ValueError("Electric material histories require their actual captured potential")
    if history.last_z_m != plane:
        raise ValueError("Electric and kinetic material histories disagree about the executed plane")
    history.reference.require(snapshot)
    # WaveMode stores keV. Preserve its exact, deliberate unit conversion;
    # no arbitrary energy tolerance can hide an unrecorded collision.
    return float(history.reference.axis_energy(snapshot, plane))*.001*1000.


def _native_plane(z_mm, ledger, record=None):
    exact = _end(ledger)
    if record is not None and "material_kinetic_native_plane" in record:
        proof = record["material_kinetic_native_plane"]
        if (isinstance(proof, Mapping) and set(proof) == {"native_z_mm", "executed_z_m"}
                and proof["native_z_mm"] == z_mm and proof["executed_z_m"] == exact
                and z_mm*.001 == exact):
            return exact
        raise ValueError("Material checkpoint does not match its exact kinetic ledger plane")
    if z_mm != exact*1000.:
        raise ValueError("Material checkpoint does not match its exact kinetic ledger plane")
    return exact


def _transition(event, old_energy, new_energy, plane):
    if not isinstance(event, Mapping):
        raise ValueError("Kinetic energy changes require a recorded material collision")
    try:
        before, after, loss, z = (event[name] for name in
            ("input_energy_kev", "output_energy_kev", "energy_loss_ev", "z_mm"))
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in (before, after, loss, z))
                or loss <= 0 or after <= 0 or after != before-loss*.001
                or before*1000. != old_energy or after*1000. != new_energy
                or z*.001 != plane
                or event.get("kind") in (None, "real_zero_loss", "effective_absorption", "already_absorbed")):
            raise ValueError("Kinetic energy change does not match its actual material collision")
    except KeyError as error:
        raise ValueError("Kinetic energy changes require complete material collision evidence") from error


def validate_material_kinetic_record(record, options=None):
    """Reject removed prefixes, changed allocations, or restarted branch ledgers."""
    history = record.get("material_kinetic_history")
    if history is None:
        if record.get("material_kinetic_prefix_digest") is not None or "material_kinetic_native_plane" in record:
            raise ValueError("Material kinetic history lost its common prefix")
        if record.get("common_kinetic_prefix") is not None or record.get("kinetic_phase_histories") is not None:
            raise ValueError("Material kinetic histories require their separate per-mode execution contexts")
        return None
    if (not isinstance(history, Mapping) or set(history) !=
            {"schema", "prefix", "completed_energy_segments", "scope", "digest"}
            or history["schema"] not in (_SCHEMA, _ELECTRIC_SCHEMA) or history["scope"] != _SCOPE
            or json_digest({k: v for k, v in history.items() if k != "digest"}) != history["digest"]):
        raise ValueError("Material kinetic history identity changed")
    prefix = history["prefix"]
    electric = history["schema"] == _ELECTRIC_SCHEMA
    if not electric and record.get("electric_wave_history") is not None:
        raise ValueError("Material electric history lost its declared common prefix")
    prefix_keys = {"ledger", "checkpoint_digest", "mode_id", "entrance_z_mm", "digest"}
    if electric:
        prefix_keys.add("electric_history")
    if (not isinstance(prefix, Mapping) or set(prefix) !=
            prefix_keys
            or json_digest({k: v for k, v in prefix.items() if k != "digest"}) != prefix["digest"]
            or any(not isinstance(prefix[k], str) or not prefix[k] for k in ("checkpoint_digest", "mode_id"))):
        raise ValueError("Invalid common kinetic prefix")
    original = KineticPhaseLedger.from_dict(prefix["ledger"])
    if original.binding is None:
        raise ValueError("Material kinetic prefix must have executed source/energy binding")
    original.require(original.binding["snapshot_digest"], original.binding["energy_ev"],
                     original.binding["kinetic_order"], _native_plane(prefix["entrance_z_mm"], original))
    if record.get("material_kinetic_prefix_digest", prefix["digest"]) != prefix["digest"]:
        raise ValueError("Material kinetic common prefix changed")
    if record.get("snapshot_digest", original.binding["snapshot_digest"]) != original.binding["snapshot_digest"]:
        raise ValueError("Material kinetic history belongs to another potential snapshot")
    active = KineticPhaseLedger.from_dict(record.get("kinetic_phase_ledger"))
    completed = history["completed_energy_segments"]
    if not isinstance(completed, (tuple, list)):
        raise ValueError("Material kinetic completed segments must be a sequence")
    ledgers, used, electric_histories = [], [], []
    for entry in completed:
        entry_keys = {"ledger", "collision", "history_index"}
        if electric:
            entry_keys.update(("electric_history", "collision_phi_v"))
        if not isinstance(entry, Mapping) or set(entry) != entry_keys:
            raise ValueError("Invalid completed material kinetic segment")
        ledgers.append(KineticPhaseLedger.from_dict(entry["ledger"]))
        if electric:
            electric_histories.append(_electric_history(entry["electric_history"]))
    ledgers.append(active)
    if electric:
        original_electric = _electric_history(prefix["electric_history"])
        electric_histories.append(_electric_history(record.get("electric_wave_history")))
        electric_used = []
    for index, ledger in enumerate(ledgers):
        if ledger.binding is None or any(ledger.binding[k] != original.binding[k] for k in ("snapshot_digest", "kinetic_order")):
            raise ValueError("Material kinetic source or operator order changed")
        remaining = original.maximum_phase_rad-math.fsum(used)
        if ledger.maximum_phase_rad != remaining:
            raise ValueError("Material kinetic allowance was reset or increased")
        if index == 0:
            if (ledger.binding != original.binding
                    or ledger.events[:len(original.events)] != original.events):
                raise ValueError("Material kinetic history changed its executed common prefix")
        else:
            previous, entry = ledgers[index-1], completed[index-1]
            if (type(entry["history_index"]) is not int or entry["history_index"] < 0
                    or (index > 1 and entry["history_index"] <= completed[index-2]["history_index"])
                    or ledger.binding["entrance_z_m"] != _end(previous)):
                raise ValueError("Material kinetic energy segments are not consecutive")
            before = previous.binding["energy_ev"]
            if electric:
                prior = electric_histories[index-1]
                phi = entry["collision_phi_v"]
                if isinstance(phi, bool) or not isinstance(phi, (int, float)) or not math.isfinite(phi):
                    raise ValueError("Material electric collision potential is invalid")
                before = (prior.reference.entrance_energy_ev+(phi-prior.reference.entrance_phi_v))*.001*1000.
            _transition(entry["collision"], before, ledger.binding["energy_ev"], _end(previous))
        if electric:
            current_electric = electric_histories[index]
            if (current_electric.reference.snapshot_digest != original.binding["snapshot_digest"]
                    or current_electric.reference.entrance_energy_ev != ledger.binding["energy_ev"]
                    or current_electric.last_z_m != _end(ledger)):
                raise ValueError("Electric and kinetic material execution histories disagree")
            prior_used = math.expm1(math.fsum(math.log1p(value) for value in electric_used))
            allowance = ((original_electric.maximum_forward_error-prior_used)/(1.+prior_used)
                         if index else original_electric.maximum_forward_error)
            if current_electric.maximum_forward_error != allowance:
                raise ValueError("Material electric allowance was reset or increased")
            if index == 0:
                if (current_electric.reference != original_electric.reference
                        or current_electric.events[:len(original_electric.events)] != original_electric.events):
                    raise ValueError("Material electric common prefix changed")
            elif (current_electric.reference.entrance_z_m != _end(ledgers[index-1])
                    or current_electric.reference.entrance_phi_v != completed[index-1]["collision_phi_v"]):
                raise ValueError("Material electric collision reference does not match its physical plane")
            electric_used.append(current_electric.used_forward_error)
        used.append(ledger.used_phase_rad)
    if math.fsum(used) > original.maximum_phase_rad:
        raise ValueError("Material kinetic history exceeds its unchanged total allowance")
    if options is not None and "kinetic_phase_ledger" in options:
        if options["kinetic_phase_ledger"] != active:
            raise ValueError("Material kinetic active ledger cannot be reset or replaced")
    if electric and options is not None and "electric_wave_history" in options:
        if options["electric_wave_history"] != electric_histories[-1]:
            raise ValueError("Material electric active history cannot be reset or replaced")
    return freeze_json(thaw_json(history))


def capture_material_kinetic(checkpoint, snapshot_digest, *, snapshot=None):
    """Create one read-only prefix before copying into conditional histories."""
    existing = validate_material_kinetic_record(checkpoint.record)
    if existing is not None:
        active = KineticPhaseLedger.from_dict(checkpoint.record["kinetic_phase_ledger"])
        _native_plane(checkpoint.plane_z_mm, active, checkpoint.record)
        return kinetic_context(checkpoint.record)
    raw = checkpoint.record.get("kinetic_phase_ledger")
    if raw is None:
        return {}
    if len(checkpoint.beam.modes) != 1:
        raise ValueError("A common kinetic prefix requires one controlled incident mode")
    original = KineticPhaseLedger.from_dict(raw)
    if original.binding is None:
        raise ValueError("Material kinetic handoff requires an executed bound ledger")
    mode = checkpoint.beam.modes[0]
    plane = _native_plane(checkpoint.plane_z_mm, original)
    electric_raw = checkpoint.record.get("electric_wave_history")
    electric = None if electric_raw is None else _electric_history(electric_raw)
    reference_energy = mode.energy_kev*1000.
    if electric is not None:
        if _electric_energy(electric, snapshot, plane) != reference_energy:
            raise ValueError("Material entrance energy differs from its executed electric history")
        reference_energy = electric.reference.entrance_energy_ev
    original.require(snapshot_digest, reference_energy, original.binding["kinetic_order"], plane)
    prefix_data = {"ledger": original.to_dict(), "checkpoint_digest": checkpoint.digest,
        "mode_id": mode.mode_id, "entrance_z_mm": checkpoint.plane_z_mm}
    if electric is not None:
        prefix_data["electric_history"] = electric.to_dict()
    prefix = _stamp(prefix_data)
    history = _stamp({"schema": _SCHEMA if electric is None else _ELECTRIC_SCHEMA,
                      "prefix": prefix, "completed_energy_segments": (), "scope": _SCOPE})
    result = {"kinetic_phase_ledger": original.to_dict(), "material_kinetic_history": history,
              "material_kinetic_prefix_digest": prefix["digest"]}
    if electric is not None:
        result["electric_wave_history"] = electric.to_dict()
    validate_material_kinetic_record(result)
    return result


def kinetic_context(record):
    validate_material_kinetic_record(record)
    if record.get("kinetic_phase_ledger") is not None:
        KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    result = {key: thaw_json(record[key]) for key in
            ("kinetic_phase_ledger", "material_kinetic_history", "material_kinetic_prefix_digest",
             "material_kinetic_native_plane", "electric_wave_history", "potential_operator_policy",
             "unified_support_allocation", "unified_support_ledger_digest", "potential_support_policy",
             "representation_estimates",
             "material_domain_policy", "material_execution_identity") if key in record}
    if record.get("support_policy_transition") is not None:
        # A linked policy remains evidence after its active ledger retires to
        # a read-only material prefix. Keep the full immutable transition and
        # its bound source identities together across every conditional slice.
        for key in ("support_policy_transition", "unified_wave_identity", "snapshot_digest",
                    "source_checkpoint_digest", "source_mode_id", "source_mode_weight_per_reference"):
            if key in record:
                result[key] = thaw_json(record[key])
    return result


def stamp_material_kinetic_plane(record, native_z_mm):
    """Retain an adapter's actual native-mm boundary and exact executed SI node."""
    if record.get("kinetic_phase_ledger") is None:
        return record
    ledger = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    if native_z_mm*.001 != _end(ledger):
        raise ValueError("Material native boundary differs from the executed kinetic plane")
    return {**record, "material_kinetic_native_plane":
        {"native_z_mm": native_z_mm, "executed_z_m": _end(ledger)}}


def exact_material_kinetic_plane(record, native_z_mm):
    """Restore a validated material receipt's exact executed metre endpoint."""
    if validate_material_kinetic_record(record) is None:
        raise ValueError("Material native-plane restoration requires its kinetic lineage")
    ledger = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    return _native_plane(native_z_mm, ledger, record)


def prepare_material_kinetic(record, mode, z_mm, snapshot_digest, kinetic_order, *, snapshot=None):
    """Continue the active segment or prove an actual energy-changing collision."""
    history = validate_material_kinetic_record(record)
    if history is None:
        if record.get("kinetic_phase_ledger") is not None:
            raise ValueError("Material kinetic propagation requires an explicit common-prefix lineage")
        return {}, None
    active = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    prefix = history["prefix"]
    if mode.mode_id != prefix["mode_id"] and not mode.mode_id.startswith(prefix["mode_id"]+"/"):
        raise ValueError("Material kinetic mode does not descend from its declared entrance")
    completed = list(thaw_json(history["completed_energy_segments"]))
    electric = (_electric_history(record["electric_wave_history"])
                if history["schema"] == _ELECTRIC_SCHEMA else None)
    for item in completed:
        index = item["history_index"]
        if index >= len(mode.scattering_history) or json_digest(mode.scattering_history[index]) != json_digest(item["collision"]):
            raise ValueError("Material kinetic collision history changed")
        if electric is not None:
            from temsim.physics.electric_wave_reference import scalar_potential
            if snapshot is None:
                raise ValueError("Electric material histories require their actual captured potential")
            prior_electric = _electric_history(item["electric_history"])
            prior_electric.reference.require(snapshot)
            actual_phi = float(scalar_potential(snapshot, [0., 0., prior_electric.last_z_m]))
            if actual_phi != item["collision_phi_v"]:
                raise ValueError("Material collision potential differs from its captured field")
    plane = _native_plane(z_mm, active, record)
    before_energy = (active.binding["energy_ev"] if electric is None
                     else _electric_energy(electric, snapshot, plane))
    if before_energy != mode.energy_kev*1000.:
        index = len(mode.scattering_history)-1
        event = mode.scattering_history[index] if index >= 0 else None
        if completed and index <= completed[-1]["history_index"]:
            raise ValueError("Material kinetic collision was already consumed")
        _transition(event, before_energy, mode.energy_kev*1000., plane)
        active.require(snapshot_digest, active.binding["energy_ev"], kinetic_order, plane)
        entry = {"ledger": active.to_dict(), "collision": thaw_json(event), "history_index": index}
        if electric is not None:
            from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory
            reference = ElectrostaticWaveReference.capture(snapshot, plane, mode.energy_kev*1000.)
            entry.update(electric_history=electric.to_dict(), collision_phi_v=reference.entrance_phi_v)
            initial_electric = _electric_history(prefix["electric_history"])
            used_electric = math.expm1(math.fsum(math.log1p(_electric_history(row["electric_history"]).used_forward_error)
                                                   for row in (*completed, entry)))
            remaining_electric = (initial_electric.maximum_forward_error-used_electric)/(1.+used_electric)
            electric = ElectricWaveHistory(reference, remaining_electric)
        completed.append(entry)
        original = KineticPhaseLedger.from_dict(prefix["ledger"])
        remaining = original.maximum_phase_rad-math.fsum(KineticPhaseLedger.from_dict(row["ledger"]).used_phase_rad for row in completed)
        active = KineticPhaseLedger(remaining).require(snapshot_digest, mode.energy_kev*1000., kinetic_order, plane)
        history = _stamp({"schema": history["schema"], "prefix": prefix, "completed_energy_segments": completed, "scope": _SCOPE})
    reference_energy = mode.energy_kev*1000. if electric is None else electric.reference.entrance_energy_ev
    active.require(snapshot_digest, reference_energy, kinetic_order, _native_plane(z_mm, active, record))
    result = {"kinetic_phase_ledger": active.to_dict(), "material_kinetic_history": history,
              "material_kinetic_prefix_digest": prefix["digest"]}
    if electric is not None:
        result["electric_wave_history"] = electric.to_dict()
    if "material_kinetic_native_plane" in record:
        result["material_kinetic_native_plane"] = thaw_json(record["material_kinetic_native_plane"])
    validate_material_kinetic_record(result)
    return result, active


def require_contractive_collision(record, inside, distribution):
    """Spatially varying conditional normalization is outside this bound."""
    if validate_material_kinetic_record(record) is None:
        return
    import numpy as np
    inside = np.asarray(inside, dtype=bool)
    identity = distribution.absorbed_probability == 0 and all(
        channel.probability == (1. if channel.key == "real_zero_loss" else 0.) for channel in distribution.channels)
    if inside.size and not (np.all(inside) or not np.any(inside) or identity):
        raise ValueError("Cumulative kinetic phase accounting does not admit spatially varying conditional normalization")


def material_kinetic_summary(record):
    history = validate_material_kinetic_record(record)
    if history is None:
        return None
    original = KineticPhaseLedger.from_dict(history["prefix"]["ledger"])
    ledgers = [KineticPhaseLedger.from_dict(row["ledger"]) for row in history["completed_energy_segments"]]
    active = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    used = math.fsum([*(ledger.used_phase_rad for ledger in ledgers), active.used_phase_rad])
    result = {"common_prefix_digest": history["prefix"]["digest"],
        "common_prefix_used_phase_rad": original.used_phase_rad,
        "maximum_phase_rad": original.maximum_phase_rad, "history_used_phase_rad": used,
        "remaining_phase_rad": original.maximum_phase_rad-used, "scope": _SCOPE,
        "common_prefix_accounting": "once before branching; per-history totals are not summed across modes"}
    if history["schema"] == _ELECTRIC_SCHEMA:
        initial = _electric_history(history["prefix"]["electric_history"])
        electric = [_electric_history(row["electric_history"]) for row in history["completed_energy_segments"]]
        electric.append(_electric_history(record["electric_wave_history"]))
        used_electric = math.expm1(math.fsum(math.log1p(row.used_forward_error) for row in electric))
        result["electric_forward"] = {"maximum_error": initial.maximum_forward_error,
            "common_prefix_error": initial.used_forward_error, "history_error": used_electric,
            "scope": "composed added-electric forward amplitude bounds; common prefix counted once per history"}
    return freeze_json(result)
