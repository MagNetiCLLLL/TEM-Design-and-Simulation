"""Bounded execution slices of one original unified particle population."""
import math
import weakref

import numpy as np


def _selected_checkpoint(checkpoints, selection, row):
    from temsim.physics.core import PropagationCheckpoints
    from temsim.physics.unified_particle_restart import UnifiedParticleCheckpointState

    old = checkpoints.unified_state
    rows = slice(row, row+1)
    internal = UnifiedParticleCheckpointState(
        old.reference_energy_ev[selection], old.reference_phi_v[selection],
        old.induction_work_ev[rows, selection], old.active[rows, selection],
        old.blocked_z_mm[selection], old.blocked_key[selection],
        old.runtime_digest, old.implementation_digest, old.emission_time_s,
        old.swept_bounds_m[rows], old.spatial_dependency_identities[rows])
    return PropagationCheckpoints(checkpoints.z_mm[rows],
        *(getattr(checkpoints, name)[rows, selection] for name in
          ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")),
        unified_state=internal)


def execute_unified_batches(state, plan, phase, energy_offset_ev, *, start_index,
        include_initial_plane_kicks, defer_nonfinite_until_clipping, initial_time_s,
        return_flight_times, initial_kinetic_energy_ev, energy_output, unified_resume,
        history_policy):
    from temsim.physics import unified_field_particle as particle
    from temsim.physics.backend_execution import record_backend
    from temsim.physics.core import PropagationCheckpoints, _frozen_array
    from temsim.physics.optical_tuning import check_tuning_cancelled
    from temsim.physics.transport_progress import transport_progress_segment
    from temsim.physics.unified_particle_restart import UnifiedParticleCheckpointState
    from temsim.physics.unified_spatial_dependencies import spatial_runtime_identity
    from temsim.physics.wave_execution import check_available_memory

    check_tuning_cancelled(state)
    count = len(phase[0])
    budget = particle.MAX_UNIFIED_HISTORY_BYTES
    fixed = particle.unified_history_memory_bytes(state, plan, 0,
        start_index=start_index, history_policy=history_policy)
    per_particle = particle.unified_history_memory_bytes(state, plan, 1,
        start_index=start_index, history_policy=history_policy)-fixed
    capacity = int((budget-fixed)//per_particle)
    if capacity < 1:
        raise ValueError("One unified particle exceeds the bounded execution history budget")
    capacity = min(count, capacity)
    batch_count = math.ceil(count/capacity)
    retained = particle._history_indices(state, plan, start_index=start_index,
                                          history_policy=history_policy)
    checkpoint_indices = np.asarray(plan.checkpoint_index)
    checkpoint_indices = checkpoint_indices[checkpoint_indices >= start_index]
    # Published history is the caller's result. Reserve its exact full source
    # width plus a bounded execution slice before allocating either one.
    assembled_bytes = ((56 if energy_output is not None else 48)*len(retained)
                       + 160*len(checkpoint_indices)+64)*count+fixed
    peak_batch_bytes = particle.unified_history_memory_bytes(state, plan, capacity,
        start_index=start_index, history_policy=history_policy)
    check_available_memory(assembled_bytes+peak_batch_bytes)
    history = np.empty((6, len(retained), count), dtype=np.float64)
    reference_energy, reference_phi, stops = (np.empty(count) for _ in range(3))
    labels = np.empty(count, dtype=object)
    induction = np.empty((len(checkpoint_indices), count))
    activity = np.empty(induction.shape, dtype=bool)
    boxes = None
    output_z = np.asarray(plan.z_mm[retained]).copy()

    def selected(values, selection):
        return None if values is None else np.broadcast_to(np.asarray(values), (count,))[selection]

    for batch_index, lower in enumerate(range(0, count, capacity)):
        check_tuning_cancelled(state)
        selection = slice(lower, min(count, lower+capacity))
        resume = None if unified_resume is None else (
            _selected_checkpoint(unified_resume[0], selection, unified_resume[1]), 0)
        energies = []
        with transport_progress_segment(batch_index, batch_count, label="Unified source batches"):
            result = particle.execute_unified_plan(state, plan,
                *(np.asarray(value)[selection] for value in phase),
                selected(energy_offset_ev, selection), start_index=start_index,
                include_initial_plane_kicks=include_initial_plane_kicks,
                defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
                initial_time_s=selected(initial_time_s, selection), return_flight_times=True,
                initial_kinetic_energy_ev=selected(initial_kinetic_energy_ev, selection),
                energy_output=energies, unified_resume=resume, history_policy=history_policy)
        if not np.array_equal(result[0], output_z):
            raise ValueError("Unified source batches retained different exact observation planes")
        for index, values in enumerate((*result[1:6], energies[0])):
            history[index, :, selection] = values
        checkpoint = result[-1]
        internal = checkpoint.unified_state
        reference_energy[selection] = internal.reference_energy_ev
        reference_phi[selection] = internal.reference_phi_v
        stops[selection], labels[selection] = internal.blocked_z_mm, internal.blocked_key
        induction[:, selection], activity[:, selection] = internal.induction_work_ev, internal.active
        if boxes is None:
            boxes = internal.swept_bounds_m.copy()
            implementation, epoch = internal.implementation_digest, internal.emission_time_s
        else:
            if implementation != internal.implementation_digest or epoch != internal.emission_time_s:
                raise ValueError("Unified source batches executed different implementations or emission epochs")
            np.minimum(boxes[:, 0], internal.swept_bounds_m[:, 0], out=boxes[:, 0])
            np.maximum(boxes[:, 1], internal.swept_bounds_m[:, 1], out=boxes[:, 1])
        del result, checkpoint, internal, energies, resume, values
    check_tuning_cancelled(state)
    runtime = plan.unified_runtime
    spatial_ids = tuple(reversed([spatial_runtime_identity(runtime, box) for box in boxes[::-1]]))
    internal = UnifiedParticleCheckpointState(reference_energy, reference_phi, induction, activity,
        stops, tuple(labels), runtime.digest, implementation, epoch, boxes, spatial_ids)
    cp_rows = np.searchsorted(retained, checkpoint_indices)
    checkpoints = PropagationCheckpoints(plan.z_mm[checkpoint_indices],
        *(values[cp_rows] for values in history), unified_state=internal)
    particle._STOPS[id(output_z)] = (
        weakref.ref(output_z, lambda ref, key=id(output_z): particle._STOPS.pop(key, None)),
        stops.copy(), labels.copy(), runtime.digest)
    if energy_output is not None:
        energy_output.append(_frozen_array(history[5]))
    record_backend("column", getattr(state, "acceleration_backend", "Auto"), "CPU",
        reason="Shared-potential Lorentz RK4 with bounded original-source execution slices",
        retried=False, history_policy=history_policy, requested_history_policy="auto",
        integration_nodes=len(plan.z_mm)-start_index, retained_history_nodes=len(retained),
        history_memory_bytes=peak_batch_bytes, history_budget_bytes=budget,
        assembled_history_bytes=assembled_bytes, source_batch_count=batch_count,
        maximum_source_batch_particles=capacity, source_particles=count,
        retained_history_maximum_gap_mm=float(np.max(np.diff(output_z), initial=0.)),
        history_scope="Exact retained integration nodes; plot-node waist markers remain sampling dependent",
        display_history_step_mm=(float(getattr(state, "history_step_mm", 2.))
                                 if history_policy == "sampled" else None))
    result = (output_z, *history[:4])
    return (*result, history[4], checkpoints) if return_flight_times else (*result, checkpoints)
