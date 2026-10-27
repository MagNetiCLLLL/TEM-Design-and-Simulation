"""Optional bounded device execution of the unchanged carrier residual.

Field sampling, geometry, the canonical chart, and checkpoints remain on the
host. Only the complete admitted exponential is moved to a private device
scope. Every Taylor-substep boundary and spectral check remains active.
"""
import sys
import traceback

import numpy as np

from temsim.physics.wave_device import (
    check_device_memory, device_scope, gpu_retry_reason, normalise_backend, release_device_memory,
)
from temsim.physics.compute_backend import gpu_failure_category


# Planning estimate after releasing superseded transfers and first-action
# transforms. Actual CUDA allocation remains bounded by device_scope's private
# pool, including FFT work buffers; an unusual FFT plan may still fail that
# hard limit and must follow the existing resource-failure policy. The host
# field/coefficient workspace and Chebyshev planner retain their larger bounds.
RESIDUAL_DEVICE_BYTES_PER_CELL = 448
# Chebyshev explicitly adds its recurrence scratch and retained group outputs
# to this kernel base. Its actual private CUDA pool remains the hard bound.
CHEBYSHEV_DEVICE_BASE_BYTES_PER_CELL = 384
RESIDUAL_HOST_BYTES_PER_CELL = 512
RESIDUAL_HOST_PREPARATION_RESERVE = 64*1024**2


def residual_host_working_bytes(shape):
    """Full host coefficient/Taylor stage, including cold provider scratch.

    The direct exponential executor below only owns its already prepared
    coefficients and recurrence. Carrier callers reserve this additional
    fixed preparation allowance before generating any field coefficients.
    Chebyshev's independent, larger recurrence plan remains authoritative.
    """
    import math
    return RESIDUAL_HOST_BYTES_PER_CELL*math.prod(shape)+RESIDUAL_HOST_PREPARATION_RESERVE


def execute_residual_exponential(amplitude, velocity, scalar, mechanical,
        frequency, defect, beta, *, phase, scalar_splitting, kinetic_order,
        quartic_scalar_splitting, higher_order_scalar_splitting, count, degree,
        k_distance, boundary_limit, envelope_limit, cancelled, requested="CPU",
        maximum_device_working_bytes=24*1024**3, acceleration_enabled=True,
        exponential_method="taylor", generator_norm_bound=None, series_error_budget=None,
        maximum_host_working_bytes=72*1024**3, maximum_chebyshev_group=8):
    """Return host samples and evidence, never relaxing a physical gate.

    A permitted resource fallback reruns from the ORIGINAL host input. Model,
    numerical-admission, cancellation, and unknown device errors propagate.
    """
    from temsim.physics.potential_wave_carrier import (
        _apply_residual, _check_boundary, _check_envelope_sampling,
    )
    policy = normalise_backend(requested)
    if exponential_method not in ("taylor", "chebyshev"):
        raise ValueError("exponential_method must be 'taylor' or 'chebyshev'")
    from temsim.physics.wave_grid import WaveGridBudgetError
    if type(maximum_host_working_bytes) is not int or maximum_host_working_bytes <= 0:
        raise ValueError("maximum_host_working_bytes must be an explicit positive integer")
    if RESIDUAL_HOST_BYTES_PER_CELL*amplitude.size > maximum_host_working_bytes:
        raise WaveGridBudgetError("Residual host workspace exceeds its declared working-memory budget")
    if exponential_method == "chebyshev":
        from temsim.physics.potential_wave_chebyshev import (
            ChebyshevPlanUnavailable, plan_chebyshev_exponential,
        )
        if type(maximum_chebyshev_group) is not int or not 1 <= maximum_chebyshev_group <= 32:
            raise ValueError("maximum_chebyshev_group must be an explicit integer from 1 to 32")

    def run(selected):
        with device_scope(selected, maximum_working_bytes=maximum_device_working_bytes,
                work_items=amplitude.size*count*degree,
                required_bytes=RESIDUAL_DEVICE_BYTES_PER_CELL*amplitude.size,
                acceleration_enabled=acceleration_enabled,
                workspace_shape=amplitude.shape) as device:
            xp = device.xp
            plan, method_fallback = None, None
            if exponential_method == "chebyshev" and count:
                allowed = maximum_host_working_bytes
                base_bytes_per_cell = 640
                if device.backend == "cupy":
                    # Host coefficients were admitted independently above.
                    # Device recurrence buffers use the device's own budget;
                    # a CPU retry enters this block anew with the CPU base.
                    allowed = device.maximum_working_bytes
                    base_bytes_per_cell = CHEBYSHEV_DEVICE_BASE_BYTES_PER_CELL
                group_limit = maximum_chebyshev_group
                while group_limit:
                    try:
                        candidate = plan_chebyshev_exponential(k_distance, generator_norm_bound,
                            count, degree, series_error_budget, cells=amplitude.size,
                            maximum_working_bytes=allowed, maximum_group=group_limit,
                            base_bytes_per_cell=base_bytes_per_cell, cancelled=cancelled)
                    except ChebyshevPlanUnavailable as error:
                        method_fallback = str(error)
                        break
                    try:
                        # Admission precedes all group-output allocation. A
                        # smaller group or the original Taylor kernel can fit
                        # without changing Require GPU into a CPU request.
                        check_device_memory(candidate.working_bytes)
                    except Exception as error:
                        if gpu_failure_category(error, stage="unified_carrier_residual") != "out_of_memory":
                            raise
                        method_fallback = str(error)
                        group_limit = max(group.count for group in candidate.groups)-1
                        continue
                    plan = candidate
                    break
            values = [xp.asarray(value) for value in
                      (amplitude, velocity, scalar, mechanical, frequency, defect, phase)]
            current, flow, multiplication, momentum, symbols, electric, real_phase = values
            # Locals now own the required arrays. The transfer list would pin
            # the obsolete entrance, real phase and scalar for the entire
            # recurrence after their role has already finished.
            del values
            maximum_boundary = _check_boundary(current, boundary_limit)
            maximum_tail = _check_envelope_sampling(current, envelope_limit)
            if scalar_splitting:
                scalar_phase = xp.exp(-1j*real_phase)
                current = current*scalar_phase
                maximum_tail = max(maximum_tail, _check_envelope_sampling(current, envelope_limit))
                differential_scalar = xp.zeros_like(multiplication)
            else:
                differential_scalar = multiplication
            del multiplication, real_phase

            def apply(value):
                return _apply_residual(value, flow, differential_scalar, momentum, symbols,
                    kinetic_order, electric, beta, quartic_scalar_splitting=quartic_scalar_splitting,
                    higher_order_scalar_splitting=higher_order_scalar_splitting)

            if plan is None:
                for _ in range(count):
                    if cancelled():
                        raise InterruptedError("Carrier potential propagation cancelled")
                    term, total = current, current.copy()
                    for number in range(1, degree+1):
                        if cancelled():
                            raise InterruptedError("Carrier potential propagation cancelled")
                        term = (-1j*k_distance/(count*number))*apply(term)
                        total += term
                    current = total
                    maximum_boundary = max(maximum_boundary, _check_boundary(current, boundary_limit))
                    maximum_tail = max(maximum_tail, _check_envelope_sampling(current, envelope_limit))
            else:
                for group in plan.groups:
                    if cancelled():
                        raise InterruptedError("Carrier potential propagation cancelled")
                    coefficients = group.coefficients
                    outputs = [current*row[0] for row in coefficients]
                    previous, this = current, apply(current)
                    this /= plan.norm_bound
                    scratch = xp.empty_like(current)
                    for index, row in enumerate(coefficients):
                        if cancelled():
                            raise InterruptedError("Carrier potential propagation cancelled")
                        xp.multiply(this, row[1], out=scratch)
                        outputs[index] += scratch
                    for number in range(2, group.degree+1):
                        if cancelled():
                            raise InterruptedError("Carrier potential propagation cancelled")
                        following = apply(this)
                        following /= plan.norm_bound
                        following *= 2
                        following -= previous
                        for index, row in enumerate(coefficients):
                            if cancelled():
                                raise InterruptedError("Carrier potential propagation cancelled")
                            xp.multiply(following, row[number], out=scratch)
                            outputs[index] += scratch
                        previous, this = this, following
                    # No intermediate state is published. All original Taylor
                    # times, including those within this group, must pass.
                    for output in outputs:
                        if cancelled():
                            raise InterruptedError("Carrier potential propagation cancelled")
                        maximum_boundary = max(maximum_boundary, _check_boundary(output, boundary_limit))
                        maximum_tail = max(maximum_tail, _check_envelope_sampling(output, envelope_limit))
                    current = outputs[-1]
                    del outputs, previous, this, scratch
                    if group.degree >= 2:
                        del following
            if scalar_splitting:
                current = current*scalar_phase
                maximum_tail = max(maximum_tail, _check_envelope_sampling(current, envelope_limit))
            result = current if xp is np else xp.asnumpy(current)
            evidence = {**device.evidence(), "requested": policy,
                        "scope": "complete residual exponential; host field and canonical propagation",
                        "maximum_boundary_flux": maximum_boundary,
                        "maximum_envelope_spectral_tail": maximum_tail,
                        "requested_exponential_method": exponential_method,
                        "exponential_method": "taylor" if plan is None else "chebyshev",
                        "old_taylor_planned_applications": count*degree,
                        "actual_operator_applications": count*degree if plan is None else plan.applications,
                        "original_flux_tolerance_unchanged": True,
                        "device_planning_bytes_per_cell": RESIDUAL_DEVICE_BYTES_PER_CELL,
                        "device_limit_enforcement": "private allocation pool including FFT work buffers"}
            if plan is not None:
                evidence["chebyshev_certificate"] = plan.to_metadata()
            if method_fallback is not None:
                evidence["exponential_method_fallback"] = method_fallback
        return result, evidence

    try:
        return run(policy)
    except (InterruptedError, KeyboardInterrupt) as error:
        traceback.clear_frames(error.__traceback__)
        raise
    except Exception as error:
        # Completed private GPU frames must not keep live arrays through an
        # exception traceback. Stack locations and the exception are retained.
        traceback.clear_frames(error.__traceback__)
        if policy in ("CPU", "Numba CPU"):
            raise
        if gpu_failure_category(error, stage="unified_carrier_residual") not in {
                "unavailable", "out_of_memory", "context_corrupted"}:
            raise
        reason = gpu_retry_reason(error, policy, stage="unified_carrier_residual")
        result, evidence = run("CPU")
        evidence["fallback_reason"] = reason
        evidence["retry_input"] = "unchanged host residual entrance"
        return result, evidence
    finally:
        # All device locals have expired before releasing their private pool.
        primary = sys.exc_info()[1]
        try:
            release_device_memory()
        except Exception as cleanup:
            if primary is None:
                raise
            primary.add_note(f"Residual device cleanup also failed: {type(cleanup).__name__}: {cleanup}")


__all__ = ("execute_residual_exponential",)
