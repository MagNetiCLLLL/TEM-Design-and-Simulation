"""Exact executed-particle handoff at the specimen entrance, without a new source."""
from __future__ import annotations

from hashlib import sha256
import math

import numpy as np

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.gun_wave_transport import specimen_entrance_z_mm
from temsim import input_io


def _check(cancelled):
    if cancelled():
        raise InterruptedError("Particle-to-wave handoff cancelled")


def _array_digest(*arrays):
    digest = sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_frequency_shift(state, phase, z_mm):
    """Mechanical ray momenta become canonical frequencies in the fixed gauge."""
    from temsim.simulation_modes import is_unified
    if not is_unified(state):
        return None, {"canonical_vector_potential": "Unavailable in the legacy particle model; downstream mixed field propagation is not admitted"}
    from scipy.constants import e, h
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    runtime = get_unified_field_runtime(state)
    xyz = np.column_stack((phase["x_m"], phase["y_m"], np.full(len(phase["x_m"]), z_mm*.001)))
    snapshot = runtime.snapshot
    if type(snapshot) is DynamicPotentialSnapshot:
        time = float(runtime.metadata["emission_time_s"])+phase["flight_time_s"]
        vector = snapshot.sample(xyz, time_s=time)["A_t_m"]
    else:
        vector = snapshot.sample(xyz)["A_t_m"]
    shift = (-e/h*1e-10)*np.asarray(vector, float)[:, :2]
    return shift, {"canonical_vector_potential": "Captured common-potential gauge; p_can=p_mechanical-e*A for electrons",
        "canonical_shift_units": "cycles per angstrom", "canonical_snapshot_digest": snapshot.numerical_digest,
        "maximum_canonical_frequency_shift_inv_angstrom": float(np.max(np.linalg.norm(shift, axis=1)))}


@input_io.using_state_inputs
def _continue(state, checkpoints, row, target, weights, alive, stops, labels, cancelled):
    """Continue a selected physical population without expanding restart arrays."""
    from dataclasses import replace
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    from temsim.physics.instrument_magnetic import active_column_events, events_overlapping_interval
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.simulation_modes import is_unified
    start = float(checkpoints.z_mm[row])
    selected = np.flatnonzero(alive & (weights > 0.))
    if not len(selected):
        raise ValueError("No original positive-current particles reach the upstream handoff checkpoint")
    fields = {name: np.asarray(getattr(checkpoints, name))[row, selected] for name in
              ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")}
    if not all(np.isfinite(values).all() for values in fields.values()):
        raise ValueError("Exact continuation requires finite retained phase space, energies and clocks")
    events = events_overlapping_interval(state, active_column_events(state), start, target)
    planes = tuple(float(item.z_mm) for item in (*state.apertures, *state.recording_planes)
                   if start <= float(item.z_mm) <= target)
    plan = build_propagation_plan(state, start, target, events,
        checkpoint_z_mm=(target,), save_z_mm=(target, *planes))
    plan = replace(plan, checkpoint_index=plan.save_index)
    state._tuning_cancelled = lambda: (_check(cancelled) or False)
    output = execute_propagation_plan(state, plan, fields["x_m"], fields["tx_rad"],
        fields["y_m"], fields["ty_rad"], initial_kinetic_energy_ev=fields["kinetic_energy_ev"],
        initial_time_s=fields["flight_time_s"], include_initial_plane_kicks=False,
        defer_nonfinite_until_clipping=True, return_flight_times=True)
    z, x, _, y, _, _, checkpoint = output
    live = np.ones(len(selected), bool)
    blocked, keys = stops[selected].copy(), [labels[i] for i in selected]
    if is_unified(state):
        from temsim.physics.unified_field_particle import merge_unified_stops
        live, blocked, keys = merge_unified_stops(z, live, blocked, keys)
    else:
        from temsim.physics.aperture_clipping import clip_segment
        from temsim.physics.column_wall import clip_column_wall
        live, blocked, keys = clip_segment(state, z, x, y, live, blocked, keys)
        live, blocked, keys = clip_column_wall(state, z, x, y, live, blocked, keys)
    live, blocked, keys = clip_recording_planes(state, z, x, y, live, blocked, keys)
    _check(cancelled)
    return checkpoint, live, selected


def particle_source_from_result(result, *, cancelled=lambda: False, progress_callback=None):
    """Consume original float64 particle checkpoints at the physical upper face.

    Display interpolation is never a wave input. Missing exact history is
    continued from a retained upstream solver checkpoint using the original
    energies, clocks, identities, weights, fields and physical interception.
    No gun emission is regenerated and no surviving current is restored.
    """
    from temsim.physics.particle_wave_source import ParticleWaveSource
    _check(cancelled)
    original = getattr(result, "state_snapshot", None)
    if original is None:
        raise ValueError("Run a particle calculation first; its captured instrument is required")
    from temsim.calculation_cache import calculation_signatures
    expected = getattr(result, "signatures", {}).get("incident")
    if not expected or calculation_signatures(original)["incident"] != expected:
        raise ValueError("The incident particle result is not bound to these captured source and optics inputs; recalculate particles")
    inputs = capture_instrument_snapshot(original)
    state = inputs.restore()
    if getattr(result, "external_inputs", None) is not None:
        from temsim.calculation_manifest import assert_external_input_inventory_unchanged
        assert_external_input_inventory_unchanged(state, result.external_inputs)
    simulation = result.simulation
    branch = getattr(simulation, "incident", None)
    checkpoints = getattr(simulation, "incident_checkpoints", None)
    gun = getattr(simulation, "gun_trace", None)
    if branch is None or checkpoints is None or gun is None:
        raise ValueError("Mixed coherence requires retained incident solver checkpoints and the original gun ledger")
    target = specimen_entrance_z_mm(state)
    z = np.asarray(checkpoints.z_mm, dtype=float)
    weights, ids = np.asarray(branch.ray_weight, float), np.asarray(branch.source_ray_id)
    stops, alive = np.asarray(branch.blocked_z, float), np.asarray(branch.alive, bool)
    count = len(weights)
    if (z.ndim != 1 or not len(z) or np.any(np.diff(z) <= 0.)
            or ids.shape != (count,) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != count or stops.shape != (count,) or alive.shape != (count,)
            or np.any(~np.isfinite(weights)) or np.any(weights < 0.)):
        raise ValueError("Incident checkpoint columns must retain unique source identities and absolute weights")
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        values = getattr(checkpoints, name, None)
        if values is None or np.shape(values) != (len(z), count):
            raise ValueError("Mixed coherence requires aligned float64 positions, slopes, energies and clocks")
        if np.asarray(values).dtype != np.dtype("float64"):
            raise ValueError("Mixed coherence requires original float64 solver checkpoints")
    emitted = gun.exit_bundle
    emitted_ids = np.asarray(emitted.ray_id)
    emitted_weights = np.asarray(emitted.weight, float)
    reference = getattr(gun, "emission_reference", None)
    current = float(gun.emitted_current_a)
    if (reference is None or "ray_id" not in reference
            or not np.array_equal(emitted_ids, reference["ray_id"])
            or emitted_ids.ndim != 1 or emitted_weights.shape != emitted_ids.shape
            or len(np.unique(emitted_ids)) != len(emitted_ids)
            or not np.isfinite(emitted_weights).all() or np.any(emitted_weights < 0.)
            or not math.isfinite(current) or current <= 0.):
        raise ValueError("Mixed coherence requires the executed original emission identities and positive reference current")
    positive = int(np.count_nonzero(emitted_weights > 0.))
    if positive < 1000:
        raise ValueError(f"Mixed coherence requires at least 1000 original positive-current particles; found {positive}")
    ordered = np.argsort(emitted_ids)
    indices = np.searchsorted(emitted_ids[ordered], ids)
    if np.any(indices >= len(emitted_ids)):
        raise ValueError("Incident identities do not belong to the original emission")
    indices = ordered[indices]
    if not np.array_equal(emitted_ids[indices], ids) or not np.array_equal(emitted_weights[indices], weights):
        raise ValueError("Incident identities or weights differ from the original emitted particles")
    _check(cancelled)
    # Only floating-point round-trip equivalence is permitted, not a physical
    # plane tolerance that might merge the upper face with a nearby waist.
    exact = np.flatnonzero(abs(z-target) <= 4.*abs(np.spacing(target)))
    if len(exact):
        row = int(exact[0])
        keep = ((alive & ~np.isfinite(stops)) | (stops > target)) & (weights > 0.)
        source_checkpoint, method = checkpoints, "exact retained incident solver checkpoint"
    else:
        if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
            raise ValueError("An exact handoff is required when stochastic residual-medium transport is enabled")
        upstream = np.flatnonzero(z < target)
        if not len(upstream):
            raise ValueError("No executed upstream checkpoint can reach the specimen entrance")
        index = int(upstream[-1])
        at_start = ((alive & ~np.isfinite(stops)) | (stops > z[index])) & (weights > 0.)
        replay_stops, replay_keys = stops.copy(), list(branch.blocked_key)
        replay_stops[at_start] = np.nan
        for column in np.flatnonzero(at_start):
            replay_keys[int(column)] = ""
        if progress_callback:
            progress_callback(0, 1, "Continuing original particles to the exact specimen entrance")
        from temsim.cpu_resources import numerical_job
        with numerical_job(1, cancelled=cancelled):
            source_checkpoint, live, selection = _continue(state, checkpoints, index,
                target, weights, at_start, replay_stops, replay_keys, cancelled)
        exact = np.flatnonzero(abs(source_checkpoint.z_mm-target) <= 4.*abs(np.spacing(target)))
        if len(exact) != 1:
            raise ValueError("Particle continuation did not retain the exact specimen entrance")
        row, keep = int(exact[0]), live & (weights[selection] > 0.)
        ids, weights = ids[selection], weights[selection]
        method = "exact continuation from original upstream solver checkpoint"
    if not np.any(keep):
        raise ValueError("No original positive-current particles reach the specimen entrance")
    _check(cancelled)
    if capture_instrument_snapshot(original).digest != inputs.digest:
        raise ValueError("Captured particle inputs changed during the exact handoff")
    phase = {name: np.asarray(getattr(source_checkpoint, name))[row, keep] for name in
             ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")}
    canonical_shift, gauge = _canonical_frequency_shift(state, phase, target)
    if progress_callback:
        progress_callback(1, 1, "Exact particle-to-wave handoff ready")
    return ParticleWaveSource(**phase, ray_id=ids[keep], weight=weights[keep], z_mm=target,
        reference_current_a=current, original_positive_count=positive, input_digest=inputs.digest,
        canonical_frequency_shift_inv_angstrom=canonical_shift,
        provenance={"handoff": method, "interpolated": False,
            **gauge,
            "original_emitted_particles": len(emitted_ids), "original_positive_particles": positive,
            "surviving_positive_particles": int(keep.sum()),
            "original_emission_digest": _array_digest(emitted_ids, emitted_weights),
            "particle_calculation_signatures": dict(getattr(result, "signatures", {})),
            "clipped_particles_restored": False, "source_regenerated": False,
            "source_reference": "Original emitted particle current; no downstream renormalization"})
