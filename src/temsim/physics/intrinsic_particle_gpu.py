"""CUDA column spans joined by device-executed lens aberration events.

Regular aligned spans retain the compiled, per-electron CUDA integration loop.
Only an aberration event is a host checkpoint: all event operations run on the
GPU and the post-event float64 phase space becomes the next span's entrance.
This does not replace material collisions or add a CPU physics fallback.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from temsim.physics.compute_backend import BACKEND_CUDA

_STAGES = (0, 1, 2, 3, 4, 18)
_ACTIONS = (6, 7, 8, 14, 15)


def _slice_inputs(inputs, start, end, coordinates, save, checkpoints, *,
                  suppress_start=False, suppress_end=False):
    """Own action buffers; caller-owned plans must never be changed in place."""
    result = list(inputs)
    for index in _STAGES:
        result[index] = np.ascontiguousarray(inputs[index][2*start:2*end+1])
    for index in _ACTIONS:
        result[index] = np.array(inputs[index][start:end+1], dtype=np.float64,
                                 order="C", copy=True)
        if suppress_start:
            result[index][0] = 0.
        if suppress_end:
            result[index][-1] = 0.
    result[9] = np.ascontiguousarray(inputs[9][start:end])
    for index, value in zip(range(10, 14), coordinates):
        result[index] = np.ascontiguousarray(value, dtype=np.float64)
    result[16] = np.ascontiguousarray(save, dtype=np.int64)
    result[17] = np.ascontiguousarray(checkpoints, dtype=np.int64)
    for index in (19, 20):
        result[index] = np.ascontiguousarray(inputs[index][3*start:3*end])
    return tuple(result)


def _validated_inputs(inputs, z_mm, events):
    from temsim.physics.ray_device_cache import _validated_inputs as validate_abi
    values = tuple(np.ascontiguousarray(value) for value in inputs)
    validate_abi(values)
    if len(values) != 22:
        raise ValueError("Segmented CUDA transport takes 22 inputs; pass time separately")
    z = np.asarray(z_mm, dtype=np.float64)
    if (z.ndim != 1 or not z.size or np.any(~np.isfinite(z))
            or np.any(np.diff(z) <= 0.) or values[9].size != z.size-1):
        raise ValueError("Segmented CUDA transport requires increasing finite Z nodes")
    if any(values[index].shape != (2*z.size-1,) for index in _STAGES):
        raise ValueError("Segmented CUDA stage arrays must include every node and midpoint")
    if any(values[index].shape != z.shape for index in _ACTIONS):
        raise ValueError("Segmented CUDA actions must have one value per Z node")
    count = values[10].size
    if any(values[index].shape != (count,) for index in (5, 10, 11, 12, 13)):
        raise ValueError("Segmented CUDA particle arrays must have identical populations")
    for index in (16, 17):
        selected = values[index]
        if (not np.issubdtype(selected.dtype, np.integer)
                or np.any(selected < 0) or np.any(selected >= z.size)
                or np.any(np.diff(selected) <= 0)):
            raise ValueError("Saved/checkpoint indices must be unique increasing node indices")
    for node in events:
        if isinstance(node, bool) or not isinstance(node, (int, np.integer)) or not 0 <= node < z.size:
            raise ValueError("Aberration event index is outside the supplied Z nodes")
    return values, z


def intrinsic_gpu_transport(
    inputs: Sequence[np.ndarray], *, z_mm: np.ndarray, electric_field=None,
    initial_kinetic_energy_ev, initial_time_s=None, mapped_fields=(),
    posed_spherical_kicks: Mapping[int, Sequence[object]] | None = None,
    defer_nonfinite_until_clipping: bool = False,
    cancel_check: Callable[[], None] | None = None,
    inverse_speed=None,
):
    """Return the existing 8/10/12-array column transport ABI on real CUDA.

    ``inputs`` uses the 22-array magnetic RK4 ABI. Positions are metres,
    slopes radians, ``z_mm`` millimetres and energies eV. Event dictionary keys
    index these Z nodes. At each node native thin-lens/rotation/deflection/Cs
    actions precede the ordered legacy posed-Cs and intrinsic aberration events.
    Saved values and checkpoints are always *after* all node actions.

    General positioned fields use the device reference solver for the entire
    span, avoiding a host transfer at each integration step. Aligned spans use
    the faster compiled CUDA solver. This function never retries on the CPU;
    the caller owns the existing explicit backend policy and failure reporting.
    """
    events = {node: tuple(items) for node, items in (posed_spherical_kicks or {}).items()
              if items}
    values, z = _validated_inputs(inputs, z_mm, events)
    from temsim.physics.particle_gpu_reference import gpu_reference_transport
    common = dict(z_mm=z, electric_field=electric_field,
                  initial_kinetic_energy_ev=initial_kinetic_energy_ev,
                  initial_time_s=initial_time_s, mapped_fields=tuple(mapped_fields),
                  posed_spherical_kicks=events,
                  defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
                  cancel_check=cancel_check, inverse_speed=inverse_speed)
    # Arbitrary positioned fields cannot be removed from a regular span. Keep
    # their arrays resident in the GPU reference integrator instead.
    if mapped_fields:
        return gpu_reference_transport(values, **common)
    if electric_field is not None:
        from temsim.physics.electrostatic_column_transport import _compiled_electric_data
        if _compiled_electric_data(electric_field) is None:
            return gpu_reference_transport(values, **common)

    count = values[10].size
    energy = np.broadcast_to(np.asarray(initial_kinetic_energy_ev, dtype=np.float64),
                             (count,)).copy()
    time = None if initial_time_s is None else np.array(initial_time_s, dtype=np.float64, copy=True)
    if time is not None and time.shape != (count,):
        raise ValueError("Initial time must contain one value per electron")
    electric = electric_field is not None
    timed = time is not None or electric
    if timed and time is None:
        time = np.full(count, np.nan)
    if not electric and timed:
        if inverse_speed is None:
            from temsim.physics.electrostatic_column_transport import M, C
            momentum = 1./values[5]
            inverse_speed = np.sqrt(M*M+(momentum/C)**2)/momentum
        inverse_speed = np.ascontiguousarray(inverse_speed, dtype=np.float64)
    else:
        inverse_speed = None

    save_nodes, checkpoint_nodes = values[16:18]
    history_dtype = np.float64 if electric else np.float32
    saved = [np.empty((len(save_nodes), count), dtype=history_dtype) for _ in range(4)]
    checked = [np.empty((len(checkpoint_nodes), count), dtype=np.float64) for _ in range(4)]
    output = saved + checked
    if timed:
        output += [np.empty(saved[0].shape, dtype=np.float64), np.empty_like(checked[0])]
    if electric:
        output += [np.empty_like(saved[0]), np.empty_like(checked[0])]
    coordinates = tuple(np.array(a, dtype=np.float64, copy=True) for a in values[10:14])

    def cancel():
        if cancel_check is not None:
            cancel_check()

    def merge(result, start, local_save, local_checkpoints):
        """Copy owned results and recover an exact outgoing checkpoint."""
        nonlocal coordinates, time, energy
        for nodes, selected, out_indices in (
            (save_nodes, local_save, (0, 1, 2, 3, 8, 10)),
            (checkpoint_nodes, local_checkpoints, (4, 5, 6, 7, 9, 11)),
        ):
            global_nodes = np.asarray(selected)+start
            positions = np.searchsorted(nodes, global_nodes)
            valid = positions < len(nodes)
            valid[valid] &= nodes[positions[valid]] == global_nodes[valid]
            for index in out_indices:
                if index < len(output):
                    output[index][positions[valid]] = result[index][valid]
        coordinates = tuple(np.array(result[index][-1], dtype=np.float64, copy=True)
                            for index in range(4, 8))
        if timed:
            time = np.array(result[9][-1], dtype=np.float64, copy=True)
        if electric:
            energy = np.array(result[11][-1], dtype=np.float64, copy=True)

    def event(node):
        cancel()
        indices = np.array([0], dtype=np.int64)
        one = _slice_inputs(values, node, node, coordinates, indices, indices)
        result = gpu_reference_transport(one, z_mm=z[node:node+1],
            electric_field=electric_field, initial_kinetic_energy_ev=energy,
            initial_time_s=time, mapped_fields=(),
            posed_spherical_kicks={0: events[node]},
            defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
            cancel_check=cancel_check, inverse_speed=inverse_speed)
        merge(result, node, indices, indices)

    def span(start, end, *, suppress_start=False, suppress_end=False):
        cancel()
        local_save = save_nodes[(save_nodes >= start) & (save_nodes <= end)]-start
        requested = checkpoint_nodes[(checkpoint_nodes >= start) & (checkpoint_nodes <= end)]-start
        # Every segment must return a float64 endpoint even when it is not a
        # requested diagnostic plane. Float32 display history is never resumed.
        local_checkpoints = np.unique(np.append(requested, end-start)).astype(np.int64)
        section = _slice_inputs(values, start, end, coordinates, local_save, local_checkpoints,
                                suppress_start=suppress_start, suppress_end=suppress_end)
        if electric:
            from temsim.physics.electrostatic_column_transport import electrostatic_column_rk4
            result, _, _ = electrostatic_column_rk4(section, z_mm=z[start:end+1],
                electric_field=electric_field, initial_kinetic_energy_ev=energy,
                initial_time_s=time, backend=BACKEND_CUDA, policy="require_gpu",
                defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
                cancel_check=cancel_check)
        else:
            from temsim.physics.ray_integrator import cuda_rk4
            result = cuda_rk4(*section, initial_time_s=time, inverse_speed=inverse_speed)
        merge(result, start, local_save, local_checkpoints)

    start, suppress_start = 0, False
    for node in sorted(events):
        if node > start:
            span(start, node, suppress_start=suppress_start, suppress_end=True)
        event(node)
        start, suppress_start = node, True
    if start < len(z)-1 or not events:
        span(start, len(z)-1, suppress_start=suppress_start)
    cancel()
    return tuple(output)
