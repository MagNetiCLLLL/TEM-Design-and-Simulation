"""Device reference for native, placed post-gun optics and thin screens.

Coordinates and momentum/energy arrays stay on CUDA throughout propagation.
The immutable optical plan and its loop/geometry controls remain host data.
This is a correctness path for placed native fields; the aligned column uses
the existing batched CUDA integrator. No backend fallback happens here: the
caller owns policy, error receipts and retry from its unchanged input.
"""
from __future__ import annotations

import numpy as np

from .compute_backend import GPUExecutionError, cupy_module


def native_gpu_fields_supported(fields):
    """Only audited native providers can execute array physics on CUDA."""
    from .lens_field_provider import FrozenAnalyticField
    from .posed_column_fields import FrozenPosedMultipole
    return all(type(field) in (FrozenAnalyticField, FrozenPosedMultipole) for field in fields)


def gpu_reference_transport(
    inputs, *, z_mm, electric_field=None, initial_kinetic_energy_ev=None,
    initial_time_s=None, mapped_fields=(), posed_spherical_kicks=None,
    defer_nonfinite_until_clipping=False, cancel_check=None,
    optical_reference_invariant_ev=None, inverse_speed=None,
):
    """Return 8 magnetic, 10 timed magnetic, or 12 electric host arrays.

    ``inputs`` has the existing ray-integrator ABI (22 arrays), XY in metres,
    mechanical slopes in radians, and electric entrance energies in eV.
    Inputs are never modified, including if cancellation or allocation fails.
    A one-node/zero-interval request still applies that node's actions exactly
    once before recording it, just like both established CPU integrators.
    """
    if not native_gpu_fields_supported(mapped_fields):
        raise GPUExecutionError("unsupported_stage", "Imported magnetic maps need the reference CPU transport")
    from .electrostatic_column_transport import _compiled_electric_data, _vector_reference, REST_EV, M, C
    electric_data = None if electric_field is None else _compiled_electric_data(electric_field)
    if electric_field is not None and electric_data is None:
        raise GPUExecutionError("unsupported_stage", "This electric provider is not an audited CUDA column field")
    if cancel_check is not None:
        cancel_check()
    cp = cupy_module()
    # Copy only state-bearing arrays. Coefficients and node indices are frozen
    # host controls; each operation involving ray values is CuPy arithmetic.
    device_inputs = list(inputs)
    for index in (5, 10, 11, 12, 13):
        device_inputs[index] = cp.array(inputs[index], dtype=cp.float64, copy=True)
    count = len(device_inputs[10])
    times = None if initial_time_s is None else cp.array(initial_time_s, dtype=cp.float64, copy=True)
    if electric_field is None:
        from .vector_field_transport import vector_map_rk4
        speed_inverse = None
        if times is not None:
            if inverse_speed is not None:
                speed_inverse = cp.asarray(inverse_speed, dtype=cp.float64)
            elif initial_kinetic_energy_ev is not None:
                ratio = cp.asarray(initial_kinetic_energy_ev, dtype=cp.float64)/REST_EV
                speed = C*cp.sqrt(ratio/(ratio+1.))*cp.sqrt((ratio+2.)/(ratio+1.))
                speed_inverse = 1./speed
            else:
                momentum = 1./device_inputs[5]
                speed_inverse = cp.sqrt(momentum*momentum+(M*C)**2)/(C*momentum)
        values = vector_map_rk4(*device_inputs, z_mm=np.asarray(z_mm),
            mapped_fields=mapped_fields, posed_spherical_kicks=posed_spherical_kicks,
            initial_time_s=times, inverse_speed=speed_inverse,
            defer_nonfinite_until_clipping=defer_nonfinite_until_clipping,
            cancel_check=cancel_check, xp=cp)
    else:
        if initial_kinetic_energy_ev is None:
            raise ValueError("Electric column transport requires entrance kinetic energies")
        energy = cp.broadcast_to(cp.asarray(initial_kinetic_energy_ev, dtype=cp.float64), (count,)).copy()
        finite_phase = cp.all(cp.isfinite(cp.stack(device_inputs[10:14])), axis=0)
        invalid = ~cp.isfinite(energy) | (energy <= 0.)
        if cp.any(invalid & (finite_phase | (not bool(defer_nonfinite_until_clipping)))):
            raise ValueError("Column entrance kinetic energies must be positive and finite")
        if times is None:
            times = cp.full(count, cp.nan)
        grid = tuple(cp.asarray(value) for value in electric_data)
        history, checkpoints = _vector_reference(device_inputs,
            np.asarray(z_mm)*1e-3, electric_field, energy,
            optical_reference_invariant_ev, times, mapped_fields, None,
            defer_nonfinite_until_clipping, cancel_check, posed_spherical_kicks,
            xp=cp, electric_data=grid)
        values = (*history[:4], *checkpoints[:4], history[4], checkpoints[4],
                  history[5], checkpoints[5])
    if cancel_check is not None:
        cancel_check()
    # Copies synchronize real device execution, surfacing CUDA failures to the
    # policy owner. No partially calculated host result or input is published.
    return tuple(cp.asnumpy(value) for value in values)
