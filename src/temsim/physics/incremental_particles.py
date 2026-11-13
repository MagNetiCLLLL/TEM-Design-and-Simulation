"""Strict append-only classical source reuse, distinct from axial restart.

The original low-count quadrature and native adaptive histories are immutable.
A replacement high-count result assigns its own 1/N measure, retains the old
kinematics, and integrates only new original IDs. Warm and cold whole-batch
adaptive Gun runs need not have identical timesteps; the lineage says so.
"""
from __future__ import annotations

from dataclasses import fields, replace
import json

import numpy as np

from temsim.immutable_json import json_digest

SCHEMA = "append-only-classical-particle-transport-v1"
NON_NESTED_SCHEMA = "classical-source-without-incremental-transport-authority-v1"
_PHASE_FIELDS = ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "ray_id", "weight")
_SURFACE_FIELDS = ("surface_position_m", "surface_direction", "surface_normal", "surface_energy_ev")


def _frozen(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def capture_emission_phase(gun, bundle):
    """Retain actual launch arrays, before the first force or energy update."""
    from temsim.physics.ray_identity import emission_reference
    reference = emission_reference(bundle, getattr(gun.emitter, "surface_model", None))
    return {**{name: _frozen(getattr(bundle, name)) for name in _PHASE_FIELDS},
            **{name: _frozen(getattr(bundle, name)) for name in _SURFACE_FIELDS if hasattr(bundle, name)},
            "position_m": reference["position_m"], "direction": reference["direction"],
            "normal": reference["normal"],
            "emission_time_s": _frozen(np.zeros(len(bundle.ray_id), dtype=np.float64))}


def _source_payload(gun):
    """Remove exactly one declared numerical source count, retaining all else."""
    payload = gun.to_dict()
    payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    emitter = payload["components"][gun.emitter.key]
    if type(emitter.get("ray_count")) is not int:
        raise ValueError("Source extension requires an explicit original particle count")
    del emitter["ray_count"]
    return payload


def _supported(state):
    from temsim.optics.electron_gun.emitter import EmissionSampling, HALTON_PREFIX_SAMPLING_SCHEMA
    gun = getattr(state, "electron_gun", None)
    emitter = getattr(gun, "emitter", None)
    plan = getattr(emitter, "sampling", None)
    if (type(plan) is not EmissionSampling or plan.schema != HALTON_PREFIX_SAMPLING_SCHEMA
            or plan.seed != 0 or getattr(gun, "type_key", None) != "cold_feg"):
        return False
    if any(getattr(emitter, key, None) is not None for key in
           ("surface_model", "coherence", "quadrature", "cdf_quadrature")):
        return False
    if any(getattr(emitter, key, 0) for key in ("_tuning_surface_probes", "_tuning_boundary_probes")):
        return False
    if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
        return False
    if bool(getattr(getattr(state, "nanopulser", None), "enabled", False)):
        return False
    for item in (*getattr(state, "stigmators", ()), *getattr(state, "deflectors", ()),
                 *getattr(state, "corrector_elements", ())):
        if bool(getattr(item, "enabled", False)) and any(bool(getattr(item, key, False))
                for key in ("scan_enabled", "wobble_enabled", "dynamic", "time_dependent")):
            return False
    if any(bool(getattr(state, key, False)) for key in
           ("space_charge_enabled", "collective_fields_enabled", "self_consistent_fields_enabled")):
        return False
    return True


def source_metadata(state):
    """Worker proof under the normal bound full-instrument Gun environment.

    The native cache key includes all executed electrodes, mechanical stops,
    solver tolerances, source calibration, and every magnetic contributor in
    the enforced Gun query envelope. Only its documented count leaves change.
    """
    if not _supported(state):
        # Retain the source-family identity for an honest future rejection or
        # explicit new-source calculation. This is never a growth certificate.
        try:
            from temsim.calculation_manifest import solver_source_identity
            gun = state.electron_gun
            record = {"schema": NON_NESTED_SCHEMA, "implementation": solver_source_identity(),
                      "source_parameters_digest": json_digest(_source_payload(gun)),
                      "original_count": int(gun.ray_count),
                      "growth_policy": "No append-only transport authority; declared source law retained"}
            record["column_input_digest"] = column_input_digest(state)
            return record
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    from temsim.calculation_manifest import solver_source_identity
    from temsim.optics.electron_gun.stable_halton_sampling import halton_prefix_diagnostics
    from temsim.physics.emission_timing import resolve_emission_time_s
    gun = state.electron_gun
    key = getattr(gun, "_cache_key", None)
    if not callable(key) or not getattr(gun, "_instrument_magnetic_identity", None):
        return None
    payload = json.loads(key(None))
    if payload.get("requested_count") != gun.ray_count:
        return None
    del payload["requested_count"]
    for owner in ("components", "executed_components"):
        emitter = payload[owner][gun.emitter.key]
        if emitter.get("ray_count") != gun.ray_count:
            return None
        del emitter["ray_count"]
    return {"schema": SCHEMA, "implementation": solver_source_identity(),
            "source_parameters_digest": json_digest(_source_payload(gun)),
            "gun_dependency_digest": json_digest(payload),
            "sampling": halton_prefix_diagnostics(gun.emitter),
            "emission_epoch_s": resolve_emission_time_s(state),
            "original_count": int(gun.ray_count),
            "reference_current_a": float(gun.emitted_current_a),
            "column_input_digest": column_input_digest(state)}


def column_input_digest(state):
    """Complete pinned inputs, omitting exactly the original source count."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.immutable_json import thaw_json
    payload = thaw_json(capture_instrument_snapshot(state).identity_payload())
    owners = [node for node in payload["graph"]["nodes"] if
              node.get("type") == "temsim.optics.electron_gun.emitter:ColdFieldEmitter"]
    if len(owners) != 1:
        raise ValueError("Source growth requires one declared original cold-FEG emitter")
    count = owners[0]["attributes"].get("ray_count")
    if type(count) is not int or count != state.electron_gun.ray_count:
        raise ValueError("Source graph and executed original count differ")
    owners[0]["attributes"]["ray_count"] = {"numerical_input": "original_particle_count"}
    return json_digest(payload)


def validate_particle_count_growth(previous_result, new_state):
    """Reject silent growth of historical batch-dependent emission laws.

    Selecting a different explicit source law is a fresh calculation, never a
    relabelled continuation. This guard needs no fields, emissions or CUDA.
    """
    simulation = getattr(previous_result, "simulation", previous_result)
    trace = getattr(simulation, "gun_trace", None)
    if trace is None:
        return
    old_count = len(trace.exit_bundle.ray_id)
    gun = new_state.electron_gun
    if old_count >= gun.ray_count:
        return
    sampling = getattr(gun.emitter, "sampling", None)
    record = getattr(trace, "incremental_source", None)
    previous_digest = record.get("source_parameters_digest") if isinstance(record, dict) else None
    old_state = getattr(previous_result, "state_snapshot", None)
    if previous_digest is None and old_state is not None:
        previous_digest = json_digest(_source_payload(old_state.electron_gun))
    if previous_digest is not None and previous_digest != json_digest(_source_payload(gun)):
        return  # Explicitly changed physical source: a fresh calculation.
    if sampling is not None and not _supported(new_state):
        old_inputs = record.get("column_input_digest") if isinstance(record, dict) else None
        if old_inputs is None and old_state is not None:
            # Capture a detached metadata clone; never migrate the raw result.
            from temsim.instrument_snapshot import encode_instrument, decode_instrument
            old_inputs = column_input_digest(decode_instrument(encode_instrument(old_state)))
        if old_inputs is not None and old_inputs == column_input_digest(new_state):
            raise ValueError("The source prefix is stable, but append-only transport reuse is unavailable for these dynamic, residual-medium, collective, or unsupported source settings. Keep the completed result, or explicitly start a new calculation; increasing the count cannot silently replace its original trajectories.")
        return  # Different physical/numerical inputs are a fresh experiment.
    if sampling is None:
        if previous_digest is None and any(getattr(gun.emitter, key, None) is not None
                for key in ("surface_model", "coherence", "quadrature", "cdf_quadrature")):
            return  # An explicit different family must validate its own count.
        raise ValueError("Cannot extend a historical batch-dependent source: increasing its particle count changes existing launch energies. In Physical Layout > Tip parameters choose 'Use nested source sampling (new calculation)', then Apply and start a new calculation; the historical result remains unchanged.")


def column_growth_prefix(previous_trace, current_trace):
    """Authorize count-only column growth after the strict Gun prefix proof."""
    before, after = (getattr(trace, "incremental_source", None)
                     for trace in (previous_trace, current_trace))
    if (not isinstance(before, dict) or not isinstance(after, dict)
            or not before.get("column_input_digest")
            or before["column_input_digest"] != after.get("column_input_digest")
            or after.get("reused_original_count") != len(previous_trace.exit_bundle.ray_id)):
        return 0
    return int(after["reused_original_count"])


def can_extend_particle_result(previous_result, new_state, *, target_z_mm=None):
    """Cheap candidate routing only; no field construction or source emission.

    Actual execution separately proves native dependency keys and every launch
    array. This predicate is deliberately conservative about source families,
    but field edits may route a candidate that the worker then rejects.
    """
    try:
        if not _supported(new_state):
            return False
        simulation = getattr(previous_result, "simulation", previous_result)
        trace = simulation.gun_trace
        record = trace.incremental_source
        from temsim.calculation_manifest import solver_source_identity
        return (isinstance(record, dict) and record.get("schema") == SCHEMA
                and record.get("implementation") == solver_source_identity()
                and record.get("source_parameters_digest") == json_digest(_source_payload(new_state.electron_gun))
                and trace.emission_phase is not None
                and 0 < len(trace.exit_bundle.ray_id) < new_state.electron_gun.ray_count
                and record.get("original_count") == len(trace.exit_bundle.ray_id))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def prove_emission_prefix(previous, current, old_count):
    """Exact original IDs, boundary, launch energy, clocks and source measure."""
    if not isinstance(previous, dict) or set(previous) != set(current):
        raise ValueError("Saved Gun lacks the complete original launch phase")
    count = len(current["ray_id"])
    if not 0 < old_count < count:
        raise ValueError("Source extension must append a nonempty new population")
    for name, value in current.items():
        old = np.asarray(previous[name])
        new = np.asarray(value)
        if name == "weight":
            if (not np.array_equal(old, np.full(old_count, 1./old_count))
                    or not np.array_equal(new, np.full(count, 1./count))):
                raise ValueError("Source extension requires explicit uniform original 1/N quadrature")
        elif old.dtype != new.dtype or not np.array_equal(old, new[:old_count], equal_nan=True):
            raise ValueError(f"Original source prefix changed: {name}")
    ids = np.asarray(current["ray_id"])
    if ids.dtype.kind not in "iu" or np.any(ids < 0) or len(np.unique(ids)) != count:
        raise ValueError("Appended original IDs must be unique nonnegative integers")


def _slice_bundle(bundle, start):
    from temsim.optics.electron_gun.base import EmissionBundle
    result = EmissionBundle(**{field.name: np.asarray(getattr(bundle, field.name))[start:].copy()
                               for field in fields(EmissionBundle)})
    for name in _SURFACE_FIELDS:
        if hasattr(bundle, name):
            object.__setattr__(result, name, np.asarray(getattr(bundle, name))[start:].copy())
    if hasattr(bundle, "emission_geometry_model"):
        object.__setattr__(result, "emission_geometry_model", bundle.emission_geometry_model)
    return result


def _join(old, tail, name, axis=0):
    before, after = getattr(old, name), getattr(tail, name)
    if before is None or after is None:
        if before is not after:
            raise ValueError(f"Source partitions disagree about {name}")
        return None
    return np.concatenate((before, after), axis=axis)


def merge_gun_traces(old, tail, phase, metadata):
    """Join untouched kinematics; assign only the replacement quadrature mass."""
    from temsim.optics.electron_gun.base import GunPlaneArrival
    from temsim.optics.electron_gun.tracing import _weighted_fwhm_from_standard_deviation
    if not np.array_equal(old.z_mm, tail.z_mm):
        raise ValueError("Gun axial observation meshes differ")
    old_count, count = len(old.exit_bundle.ray_id), len(phase["ray_id"])
    ratio = old_count/count
    bundle = replace(tail.exit_bundle, **{
        name: (np.asarray(phase["weight"]).copy() if name == "weight" else
               _join(old.exit_bundle, tail.exit_bundle, name))
        for name in (field.name for field in fields(tail.exit_bundle))})
    if not np.array_equal(bundle.ray_id, phase["ray_id"]):
        raise ValueError("Executed Gun IDs differ from the validated launch population")
    arrivals = []
    if [(p.key, p.name, p.z_mm) for p in old.plane_arrivals] != [(p.key, p.name, p.z_mm) for p in tail.plane_arrivals]:
        raise ValueError("Gun partition boundary lists differ")
    for before, after in zip(old.plane_arrivals, tail.plane_arrivals):
        arrivals.append(GunPlaneArrival(before.key, before.name, before.z_mm, **{
            name: _join(before, after, name) for name in ("time_s", "x_m", "y_m", "reached", "transmitted")}))
    current_fields = ("dpa_transmitted_current_a", "c1_transmitted_current_a", "monochromator_transmitted_current_a")
    currents = {}
    for name in current_fields:
        a, b = getattr(old, name), getattr(tail, name)
        if (a is None) != (b is None):
            raise ValueError("Gun partition current boundary lists differ")
        currents[name] = None if a is None else float(a)*ratio+float(b)
    slit_x, slit_reached = _join(old, tail, "slit_x_m"), _join(old, tail, "slit_reached")
    slit_dispersion = None
    if slit_reached is not None:
        selected = np.flatnonzero(slit_reached)
        if len(selected) >= 2 and np.ptp(bundle.energy_offset_ev[selected]) > 0:
            slit_dispersion = float(np.polyfit(bundle.energy_offset_ev[selected], slit_x[selected]*1e6, 1)[0])
    partitions = (*old.transport_partitions, tail) if old.transport_partitions else (old, tail)
    native_reports = [part.electrostatic_model_report for part in partitions]
    electrostatic = None
    if any(report is not None for report in native_reports):
        if not all(isinstance(report, dict) for report in native_reports):
            raise ValueError("Gun partitions disagree about verified electrostatic execution")
        electrostatic = dict(native_reports[0])
        electrostatic.update(
            transmitted_particle_count=int(np.count_nonzero(bundle.alive)),
            exit_energy_status="verified" if np.any(bundle.alive) else "no_transmitted_particles",
            launch_boundary={"scope": "Unmodified native partition launch-boundary reports",
                             "partitions": [report.get("launch_boundary") for report in native_reports]},
            native_partition_reports=native_reports,
            transport_scope="Append-only native partitions; maximum errors aggregate conservatively, not by refitting energies")
        for key in ("minimum_exit_kinetic_energy_ev", "minimum_expected_exit_kinetic_energy_ev",
                    "maximum_exit_kinetic_energy_ev", "maximum_expected_exit_kinetic_energy_ev",
                    "maximum_exit_energy_error_ev", "surface_mesh_displacement_m"):
            available = [report[key] for report in native_reports if report.get(key) is not None]
            electrostatic[key] = ((min(available) if key.startswith("minimum_") else max(available))
                                   if available else None)
    record = {**metadata, "reused_original_count": old_count, "transported_original_count": count-old_count,
        "old_quadrature_weight": 1./old_count, "new_quadrature_weight": 1./count,
        "old_measure_ratio_in_new_result": ratio,
        "normalization": "New result uses 1/N for every original ID; earlier result and native partitions retain their historical weights. Physical loss masks are unchanged.",
        "lineage": "Warm append-only reuse of independently adaptive native Gun partitions; not a claim of bitwise equality to a cold whole-population adaptive run",
        "native_partition_counts": [len(part.exit_bundle.ray_id) for part in partitions]}
    reference = {name: np.concatenate((old.emission_reference[name], tail.emission_reference[name]))
                 for name in ("ray_id", "position_m", "direction", "normal")}
    return replace(old, **{name: _join(old, tail, name, axis=1)
                          for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")},
        exit_bundle=bundle, blocked_z_mm=_join(old, tail, "blocked_z_mm"),
        blocked_key=tuple(old.blocked_key)+tuple(tail.blocked_key), **currents,
        output_energy_fwhm_ev=_weighted_fwhm_from_standard_deviation(bundle.energy_offset_ev[bundle.alive], bundle.weight[bundle.alive]),
        slit_x_m=slit_x, slit_reached=slit_reached, slit_dispersion_um_per_ev=slit_dispersion,
        equal_time_history=None, plane_arrivals=tuple(arrivals), emission_reference=reference,
        emission_phase=phase, incremental_source=record, transport_partitions=partitions,
        electrostatic_model_report=electrostatic)


def extend_source_trace(state, previous, metadata, *, cancelled=None):
    """Return a proven appended Gun trace, or None for an incompatible seed."""
    if previous is None or metadata is None or metadata.get("schema") != SCHEMA:
        return None
    record = getattr(previous, "incremental_source", None)
    required = ("schema", "implementation", "source_parameters_digest", "gun_dependency_digest",
                "sampling", "emission_epoch_s", "reference_current_a")
    if (not isinstance(record, dict) or any(record.get(key) != metadata.get(key) for key in required)
            or record.get("original_count") != len(previous.exit_bundle.ray_id)
            or not 0 < record["original_count"] < metadata["original_count"]):
        return None
    from temsim.physics.simulation import _validate_gun_flight_times
    if not _validate_gun_flight_times(previous):
        return None
    gun = state.electron_gun
    emitted = gun.emit()
    phase = capture_emission_phase(gun, emitted)
    prove_emission_prefix(previous.emission_phase, phase, record["original_count"])
    from temsim.optics.electron_gun.tracing import _trace_feg_to_exit
    from temsim.physics.grounded_particle_step import gun_step_workers
    with gun_step_workers():
        tail = _trace_feg_to_exit(gun, cancelled=cancelled,
                                 emitted_bundle=_slice_bundle(emitted, record["original_count"]))
    return merge_gun_traces(previous, tail, phase, metadata)


def merge_population_checkpoints(previous, tail, old_plan, current_plan):
    """Retain old numerical states under a proven count-only field capture.

    New unified runtime metadata names the new input population. Its actual
    field and finite boundary dependencies must equal the old consumed ones
    at every original swept box; no generic restart guard is weakened.
    """
    from temsim.physics.core import PropagationCheckpoints
    if not np.array_equal(previous.z_mm, tail.z_mm):
        raise ValueError("Population checkpoints must have identical executed planes")
    arrays = {name: np.concatenate((getattr(previous, name), getattr(tail, name)), axis=1)
              for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")}
    a, b = previous.unified_state, tail.unified_state
    internal = None
    if a is not None or b is not None:
        from temsim.physics.unified_particle_restart import validate_unified_checkpoint, UnifiedParticleCheckpointState
        from temsim.physics.unified_spatial_dependencies import spatial_runtime_identity
        validate_unified_checkpoint(previous, old_plan)
        validate_unified_checkpoint(tail, current_plan)
        if (a.implementation_digest != b.implementation_digest
                or a.emission_time_s != b.emission_time_s):
            raise ValueError("Population checkpoints have different solver or emission epochs")
        runtime = current_plan.unified_runtime
        for row, box in enumerate(a.swept_bounds_m):
            if a.spatial_dependency_identities[row] != spatial_runtime_identity(runtime, box):
                raise ValueError("Old population consumed changed fields or physical boundaries")
        boxes = np.stack((np.minimum(a.swept_bounds_m[:, 0], b.swept_bounds_m[:, 0]),
                          np.maximum(a.swept_bounds_m[:, 1], b.swept_bounds_m[:, 1])), axis=1)
        internal = UnifiedParticleCheckpointState(
            reference_energy_ev=np.r_[a.reference_energy_ev, b.reference_energy_ev],
            reference_phi_v=np.r_[a.reference_phi_v, b.reference_phi_v],
            induction_work_ev=np.concatenate((a.induction_work_ev, b.induction_work_ev), axis=1),
            active=np.concatenate((a.active, b.active), axis=1),
            blocked_z_mm=np.r_[a.blocked_z_mm, b.blocked_z_mm],
            blocked_key=(*a.blocked_key, *b.blocked_key), runtime_digest=runtime.digest,
            implementation_digest=a.implementation_digest, emission_time_s=a.emission_time_s,
            swept_bounds_m=boxes,
            spatial_dependency_identities=tuple(spatial_runtime_identity(runtime, box) for box in boxes))
    return PropagationCheckpoints(previous.z_mm, **arrays, unified_state=internal)


def extend_column_segment(state, previous, plan, *, name, start, stop, initial,
                          energy, weights, identity, survival, events, boundary,
                          initial_kicks, save_z_mm, dependency_context,
                          kinetic0, preserve_existing_mesh, dependency):
    """Append one static section only when old and new executed meshes agree."""
    from temsim.physics.particle_sections import (
        ParticleSectionSegment, _digest_arrays, _prefix_matches, _trace_segment,
    )
    from temsim.calculation_manifest import solver_source_identity
    if not isinstance(previous, ParticleSectionSegment):
        return None
    n = len(previous.branch.alive)
    if not 0 < n < len(weights) or state.vacuum_map.enabled:
        return None
    old_weights = np.asarray(previous.branch.ray_weight)
    alive, blocked, keys = survival
    prefix_dependency = _digest_arrays(*(value[:n] for value in initial), energy[:n],
        kinetic0[:n], old_weights, alive[:n], blocked[:n], *(value[:n] for value in identity),
        np.asarray(keys[:n], dtype=str), np.asarray(str(dependency_context)))
    if (previous.initial_dependency_signature != prefix_dependency
            or not np.array_equal(previous.plan.z_mm, plan.z_mm)
            or not np.array_equal(previous.checkpoints.z_mm, plan.z_mm[plan.checkpoint_index])
            or not _prefix_matches(previous.plan, plan, len(plan.z_mm)-1, len(plan.z_mm)-1,
                checkpoints=previous.checkpoints, checkpoint_row=len(previous.checkpoints.z_mm)-1,
                implementation_digest=solver_source_identity())):
        return None
    # Only this appended source slice enters the force integrator. The complete
    # population determines auxiliary checkpoint spacing, exactly as requested.
    tail, _, _ = _trace_segment(state, name, start, stop,
        tuple(value[n:] for value in initial), energy[n:], weights[n:],
        tuple(value[n:] for value in identity), (alive[n:], blocked[n:], keys[n:]),
        events, None, boundary, initial_kicks=initial_kicks, save_z_mm=save_z_mm,
        dependency_context=dependency_context, initial_kinetic_energy_ev=kinetic0[n:],
        preserve_existing_mesh=preserve_existing_mesh, checkpoint_ray_count=len(weights))
    if not np.array_equal(previous.branch.z, tail.branch.z):
        raise ValueError("Appended transport changed the verified observation mesh")
    branch = replace(tail.branch, **{
        key: np.concatenate((getattr(previous.branch, key), getattr(tail.branch, key)), axis=1)
        for key in ("x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev")},
        **{key: np.concatenate((getattr(previous.branch, key), getattr(tail.branch, key)))
           for key in ("alive", "blocked_z", "energy_offset_ev", "source_ray_id", "source_azimuth_rad")},
        ray_weight=np.asarray(weights).copy(),
        blocked_key=list(previous.branch.blocked_key)+list(tail.branch.blocked_key))
    cp = merge_population_checkpoints(previous.checkpoints, tail.checkpoints, previous.plan, tail.plan)
    return ParticleSectionSegment(name, branch, tail.plan, cp, dependency)
