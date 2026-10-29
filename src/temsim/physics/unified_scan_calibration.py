"""Local scan-drive calibration measured with the shared potential executor.

The probe rays are differential response measurements, never an electron
source or a replacement for the executed beam. DC alignment, finite coil
lengths, material boundaries, and the captured vector-potential gauge remain
in the solve. This does not certify a finite probe waist or an image plane.
"""
from copy import deepcopy

import numpy as np
from scipy.constants import c, e, m_e

from temsim import input_io
from temsim.immutable_json import json_digest, thaw_json
from temsim.optics.shared_deflectors import shared_channel_enabled
from temsim.physics.scan_calibration import (
    held, restore_held, sample_reference_z_mm, validate_descan_shared_peaks, capture_record,
)


MODEL = "shared-potential-finite-coil-response-v1"


def _execution_environment(private):
    """Version the numerical runtime as well as the complete solver sources."""
    from importlib.metadata import PackageNotFoundError, version
    import platform
    import sys
    versions = {}
    for package in ("numpy", "scipy", "numba", "llvmlite"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "machine": platform.machine(), "system": platform.system(), "byteorder": sys.byteorder,
            "float64_dtype": np.dtype("float64").str, "packages": versions,
            "backend": private.acceleration_backend, "acceleration_enabled": private.acceleration_enabled}


def _solve(matrix, target, label):
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"{label} is singular or ill-conditioned; scan drives were not changed")
    result = np.linalg.solve(matrix, target)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} produced a nonfinite drive")
    return result


def _momentum(energy_ev):
    energy = np.asarray(energy_ev)*e
    return np.sqrt(energy*(energy+2*m_e*c*c))/c


def _canonical_values(row, reference_momentum):
    phase = np.asarray(row["xy_slopes"], dtype=float)[:, 0]
    slopes = phase[:, 2:]
    momentum = _momentum(np.asarray(row["kinetic_energy_ev"])[:, 0])
    mechanical = momentum[:, None]*slopes/np.sqrt(1+np.sum(slopes*slopes, axis=1))[:, None]
    vector = np.asarray(row["vector_potential_t_m"])[:, 0, :2]
    # Electron canonical momentum is p_mechanical + q*A, q = -e.
    return (mechanical-e*vector)/reference_momentum


@input_io.using_state_inputs
def measure_scan_responses(state, observation_z_mm, *, perturbation_rad=1e-6,
                           maximum_step_mm=.2, _defer_disk=False):
    """Differentiate independent physical AC/Descan coils about one DC chief.

    All targets use the same chief launched at the upstream AC field support.
    The nominal beam energy defines this *local calibration*; the actual
    source energy and flight histories remain owned by particle/wave runs.
    Dynamic controls are disabled only on a private response-measurement copy.
    """
    from temsim.physics.finite_scan_response import _private_reference
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.physics.unified_field_particle import _positive_step_mm
    from temsim.calculation_manifest import solver_source_identity
    implementation = solver_source_identity()
    targets = tuple(sorted(set(float(z) for z in observation_z_mm)))
    h = float(perturbation_rad)
    requested_step = _positive_step_mm(maximum_step_mm, "Shared-potential response maximum step")
    state_step = _positive_step_mm(state.step_mm, "Shared-potential response particle step")
    # All entry points share the actual integration bound, including callers
    # that leave the response cap at its default. Validate before taking min.
    step = min(requested_step, state_step)
    if not targets or not np.isfinite(targets).all() or not np.isfinite((h, step)).all() or min(h, step) <= 0:
        raise ValueError("Shared-potential scan response needs finite planes and positive numerical steps")
    private, _ = _private_reference(state, "ac_deflector")
    if getattr(state, "_tuning_cancelled", None) is not None:
        # Cancellation is an execution hook, deliberately excluded from the
        # immutable input graph and restored only on this private diagnostic.
        private._tuning_cancelled = state._tuning_cancelled
    coils = tuple(column_dipole_fields(private))
    ac_coils = sorted((row for row in coils if row.key.startswith("ac_deflector:")), key=lambda row: row.key)
    if len(ac_coils) != 2:
        raise ValueError("Shared-potential calibration requires two installed AC coils")
    source = min(row.field_support_mm[0] for row in ac_coils)
    if targets[0] <= source:
        raise ValueError("Scan calibration observations must follow the AC entrance")
    host = str(getattr(private.descan_deflector, "_physical_host_key", private.descan_deflector.key))
    selected = tuple(row for row in coils if row.key.startswith(("ac_deflector:", host+":"))
                     and row.field_support_mm[0] < targets[-1])
    energy = float(state.beam_voltage_kv)*1000
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.optical_tuning import check_tuning_cancelled
    from temsim.physics.unified_scan_cache import cached_scan_response, matching_response_dependencies
    check_tuning_cancelled(private)
    runtime = get_unified_field_runtime(private)
    from temsim.physics.unified_field_particle import build_unified_plan
    # RK nodes include the effective state step, physical faces and inserted
    # observation planes. Equal fields alone do not guarantee equal numerical
    # results when any of these split an integration interval differently.
    target_mm = np.asarray(targets)*.001*1000.
    plan = build_unified_plan(private, float(source*.001)*1000., float(target_mm[-1]),
        checkpoint_z_mm=target_mm, maximum_step_mm=step, _captured_runtime=runtime)
    # These inputs define the differentiation experiment, independently of the
    # source population, raster/FOV and the eventual requested drive solution.
    key = json_digest({"schema": "unified-scan-response-cache-v2", "model": MODEL,
        "implementation": implementation, "source_z_mm": source, "targets_z_mm": targets,
        "execution_environment": _execution_environment(private),
        "energy_ev": energy, "perturbation_rad": h, "maximum_step_mm": step,
        "integration_nodes_mm": np.asarray(plan.z_mm).tolist(),
        "coils": [{"key": coil.key, "lower_m": coil.lower_m, "upper_m": coil.upper_m,
                   "reference_momentum": coil.reference_momentum,
                   "registration": None if coil.registration is None else {
                       "origin_global_m": coil.registration.origin_global_m,
                       "rotation_local_to_global": coil.registration.rotation_local_to_global},
                   "radial_support_m": coil.radial_support_m} for coil in selected]})
    expected = {"z_mm": targets, "coil_keys": tuple(coil.key for coil in selected),
        "model": MODEL, "implementation": implementation, "source_z_mm": source,
        "perturbation_rad": h, "maximum_step_mm": step, "reference_kinetic_energy_ev": energy}
    result = cached_scan_response(key,
        lambda: _trace_scan_responses(private, runtime, targets, source, selected, energy, h, step, implementation),
        lambda: check_tuning_cancelled(private),
        compatible=lambda value: matching_response_dependencies(value, runtime),
        expected=expected, defer_disk=_defer_disk)
    try:
        check_tuning_cancelled(private)
        if solver_source_identity() != implementation:
            raise ValueError("Solver implementation changed during scan calibration; no drive solution was installed")
    except BaseException:
        from temsim.physics.unified_scan_cache import discard_scan_response
        discard_scan_response(result["receipt"]["response_cache"]["key"])
        raise
    return result


def _trace_scan_responses(private, runtime, targets, source, selected, energy, h, step, implementation):
    from temsim.physics.unified_field_particle import trace_unified_response
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.unified_scan_cache import response_dependencies
    trace_index = 0
    trace_count = 1+4*len(selected)
    query_bounds = []
    increments = [None]
    for coil in selected:
        for axis in range(2):
            kick = np.eye(2)[axis]*h
            increments.extend(({coil.key: tuple(kick)}, {coil.key: tuple(-kick)}))
    from temsim.physics.unified_scan_parallel import parallel_scan_responses
    parallel = parallel_scan_responses(private, runtime, targets, source, energy, step, increments)
    execution = ({"mode": "serial", "workers": 1, "trace_count": trace_count,
                  "algorithm": "scalar Lorentz RK4 physical-coil perturbations"}
                 if parallel is None else parallel[1])

    def trace(perturbations=None):
        nonlocal trace_index
        from temsim.physics.transport_progress import transport_progress_segment
        if parallel is None:
            with transport_progress_segment(trace_index, trace_count, label="Calibrating shared-potential scan response"):
                row = trace_unified_response(private, np.zeros((1, 4)), source*1e-3,
                    tuple(z*1e-3 for z in targets), energy_ev=energy,
                    flight_time_s=0., maximum_step_m=step*1e-3,
                    coil_perturbations_rad=perturbations)
        else:
            row = parallel[0][trace_index]
        if not np.asarray(row["alive"]).all() or not np.isfinite(row["xy_slopes"]).all():
            raise ValueError("The scan calibration chief is intercepted; no drive solution was installed")
        receipt = row["receipt"]
        bounds = receipt.get("swept_bounds_m")
        if bounds is None:
            query_bounds.append(None)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                    or np.any(bounds[0] > bounds[1]) or bounds[0, 2] > source*.001
                    or bounds[1, 2] < targets[-1]*.001):
                raise ValueError("Incomplete scan-response executed-query bounds")
            query_bounds.append(bounds)
        trace_index += 1
        return row

    baseline = trace()
    p_ref = _momentum(energy)
    response, canonical, receipts = {}, {}, [thaw_json(baseline["receipt"])]
    for coil in selected:
        values = np.empty((len(targets), 4, 2))
        canon = np.empty((len(targets), 2, 2))
        for axis in range(2):
            kick = np.eye(2)[axis]*h
            plus, minus = trace({coil.key: tuple(kick)}), trace({coil.key: tuple(-kick)})
            values[:, :, axis] = (np.asarray(plus["xy_slopes"])[:, 0]
                                    -np.asarray(minus["xy_slopes"])[:, 0])/(2*h)
            canon[:, :, axis] = (_canonical_values(plus, p_ref)
                                  -_canonical_values(minus, p_ref))/(2*h)
            receipts.extend((thaw_json(plus["receipt"]), thaw_json(minus["receipt"])))
        response[coil.key], canonical[coil.key] = values, canon
    if solver_source_identity() != implementation:
        raise ValueError("Solver implementation changed during scan calibration; no drive solution was installed")
    bounds = (np.stack((np.min([box[0] for box in query_bounds], axis=0),
                        np.max([box[1] for box in query_bounds], axis=0)))
              if query_bounds and all(box is not None for box in query_bounds) else None)
    dependencies = response_dependencies(runtime, bounds)
    return dict(z_mm=targets, phase=response, canonical=canonical,
                reference=np.asarray(baseline["xy_slopes"])[:, 0],
                receipt=dict(model=MODEL, source_z_mm=source,
                    perturbation_rad=h, maximum_step_mm=step,
                    reference_kinetic_energy_ev=energy,
                    implementation=implementation, response_dependencies=dependencies,
                    scope="Local DC chief derivatives; not a source or finite-beam qualification",
                    trace_receipts=receipts, execution=execution))


@input_io.using_state_inputs
def calibrate_unified_scan(state, *, observation_stop_z_mm=None,
                           maximum_step_mm=.2, perturbation_rad=1e-6,
                           force=False, hold=False):
    """Resolve derived drives; explicit force may capture a new held record.

    Ordinary held execution restores its fixed drives and requested FOV scaling.
    Force measures the current shared fields even when a previous record is held.
    Neither operation changes lenses, source inputs or shared-host alignment.
    """
    return _calibrate_unified_scan(state, observation_stop_z_mm=observation_stop_z_mm,
        maximum_step_mm=maximum_step_mm, perturbation_rad=perturbation_rad,
        force=force, hold=hold)


@input_io.using_state_inputs
def recalibrate_unified_descan_only(state, *, maximum_step_mm=.2, perturbation_rad=1e-6):
    """Measure current Descan response while preserving the saved AC drive."""
    if not held(state):
        raise ValueError("Descan-only calibration requires a saved held AC calibration")
    if not all(shared_channel_enabled(part) and part.scan_enabled
               for part in (state.ac_deflector, state.descan_deflector)):
        raise ValueError("Enable AC scan and Descan scan, including their physical hosts, before calibrating Descan")
    if state.descan_deflector.kick_x_mrad or state.descan_deflector.kick_y_mrad:
        raise ValueError("Descan-only calibration requires zero logical Descan static bias; the shared coil coupling would otherwise change that bias")
    return _calibrate_unified_scan(state, maximum_step_mm=maximum_step_mm,
        perturbation_rad=perturbation_rad, keep_ac=True)[2]


def _calibrate_unified_scan(state, *, observation_stop_z_mm=None,
                            maximum_step_mm=.2, perturbation_rad=1e-6,
                            force=False, hold=False, keep_ac=False):
    """One transactional solve for automatic, held capture and Descan-only work."""
    from temsim.physics.scan_geometry import _descan_target, synchronize_scan_raster, DescanCalibrationResult
    from temsim.physics.instrument_magnetic import column_dipole_fields
    ac, descan = state.ac_deflector, state.descan_deflector
    coils = tuple(column_dipole_fields(state))
    active = {key for row in coils if observation_stop_z_mm is None
              or row.field_support_mm[0] < observation_stop_z_mm for key in row.drive_keys}
    consumed = {part.key for part in (ac, descan)
                if shared_channel_enabled(part) and part.scan_enabled and part.key in active}
    # A bounded request still skips coils outside its observed domain. An
    # explicit in-domain capture may prepare AC while its raster is off.
    if observation_stop_z_mm is not None and not consumed and not (force and ac.key in active):
        return None
    if force:
        if not shared_channel_enabled(ac) or ac.key not in active:
            raise ValueError("Enable the physical AC coils and observe downstream before capturing scan calibration")
        # An explicit capture may prepare AC while its raster is switched off.
        consumed.add(ac.key)
    if not consumed:
        return None
    snapshots = [deepcopy({k: v for k, v in vars(part).items() if k != "_physical_host"}) for part in (ac, descan)]
    measured = None
    try:
        from temsim.physics.optical_tuning import check_tuning_cancelled
        check_tuning_cancelled(state)
        if force:
            ac.calibration_mode = "automatic"
        if held(state) and not keep_ac:
            command = restore_held(state, restore_descan=descan.key in consumed)
            synchronize_scan_raster(ac, descan)
            object.__setattr__(ac, "_scan_execution_receipt", {
                "model": MODEL, "status": "explicit_held_drives", "record_identity": json_digest(ac.calibration_record_json),
                "scope": "Restored user-held drives; no claim of a new shared-potential calibration"})
            return command, float(ac.scan_scale_residual), None
        if keep_ac:
            restore_held(state, restore_descan=False)
        sample = sample_reference_z_mm(state)
        target = _descan_target(state) if descan.key in consumed else None
        measured = measure_scan_responses(state, (sample,) if target is None else (sample, target[2]),
            maximum_step_mm=maximum_step_mm, perturbation_rad=perturbation_rad, _defer_disk=True)
        if keep_ac:
            # Do not solve AC against changed optics: its deliberately retained
            # focus/pivot error is part of the requested Descan-only experiment.
            command = np.asarray(ac.scan_command_matrix_mrad)*1e-3
            residual = float(ac.scan_scale_residual)
            upper_map, lower_map = map(np.asarray, ac.coil_kick_matrices())
        else:
            sample_index = measured["z_mm"].index(sample)
            up, low = (measured["phase"][f"ac_deflector:{i}"][sample_index] for i in (0, 1))
            cu, cl = (measured["canonical"][f"ac_deflector:{i}"][sample_index] for i in (0, 1))
            ratio = _solve(cl, -cu, "AC canonical-angle response")
            ac.set_pure_shift_coupling(ratio, float(np.linalg.norm(cu+cl@ratio)/max(np.linalg.norm(cu), 1e-30)))
            object.__setattr__(ac, "_pure_shift_angle_target", "canonical")
            upper_map, lower_map = map(np.asarray, ac.coil_kick_matrices())
            response = up[:2]@upper_map+low[:2]@lower_map
            desired = np.diag((ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm))*.5e-9
            command = _solve(response, desired, "AC specimen-position response")
            residual = float(np.linalg.norm(response@command-desired)/np.linalg.norm(desired))
            ac.set_scan_command_matrix_mrad(command*1e3, residual)
            ac.validate()
        synchronize_scan_raster(ac, descan)
        result = None
        if target is not None:
            key, name, z = target
            index = measured["z_mm"].index(z)
            host = str(getattr(descan, "_physical_host_key", descan.key))
            du, dl = (measured["phase"][f"{host}:{i}"][index] for i in (0, 1))
            incident_response = (measured["phase"]["ac_deflector:0"][index]@upper_map
                                 +measured["phase"]["ac_deflector:1"][index]@lower_map)
            constraint = descan.descan_constraint
            if constraint == "position_and_angle":
                from temsim.physics.descan_phase_space import solve_position_angle_cancellation
                if descan.kick_x_mrad or descan.kick_y_mrad:
                    raise ValueError("Position-and-angle Descan requires zero logical static bias; physical host alignment is retained")
                solved = solve_position_angle_cancellation(du, dl, incident_response@command,
                    separation_mm=descan.lower_z_mm-descan.upper_z_mm, upper_gain=descan.upper_coil_gain)
                ratio, descan_command, error = solved.lower_from_upper, solved.command_matrix_mrad, solved.position_relative_residual
            elif constraint == "position":
                gain = float(descan.upper_coil_gain)
                if gain == 0:
                    raise ValueError("Descan upper coil gain must be nonzero")
                ratio = _solve(dl[:2], incident_response[:2]-gain*du[:2], "Descan position response")/gain
                descan_command = -command*1e3
                error = float(np.linalg.norm(gain*(du[:2]+dl[:2]@ratio)-incident_response[:2])
                              /max(np.linalg.norm(incident_response[:2]), 1e-30))
            else:
                raise ValueError("Unknown Descan constraint")
            descan.set_image_plane_coupling(ratio, error, target_key=key, target_z_mm=z)
            descan.set_scan_command_matrix_mrad(descan_command, error)
            object.__setattr__(descan, "_calibrated_constraint", constraint)
            descan.validate()
            validate_descan_shared_peaks(state)
            result = DescanCalibrationResult(key, name, z, ratio, error, float("nan"),
                "unqualified", constraint=constraint)
        object.__setattr__(ac, "_scan_execution_receipt", {**thaw_json(measured["receipt"]),
            "status": "executed_shared_potential_descan_response" if keep_ac else "executed_shared_potential_response",
            "ac_recalibrated": not keep_ac, "sample_z_mm": sample,
            "descan_target_z_mm": None if target is None else target[2],
            "sample_position_relative_residual": None if keep_ac else residual,
            "commands_identity": json_digest((ac.scan_command_matrix_mrad, descan.scan_command_matrix_mrad))})
        if force or keep_ac:
            capture_record(state, descan_calibrated=result is not None,
                           ac_angle_target=None if keep_ac else "canonical")
            if force:
                ac.calibration_mode = "held" if hold else snapshots[0]["calibration_mode"]
        from temsim.physics.unified_scan_cache import commit_scan_response
        check_tuning_cancelled(state)
        cache_key = measured["receipt"].get("response_cache", {}).get("key")
        if cache_key is not None:
            commit_scan_response(cache_key, lambda: check_tuning_cancelled(state))
        return command*1e3, residual, result
    except Exception:
        if measured is not None:
            from temsim.physics.unified_scan_cache import discard_scan_response
            cache_key = measured.get("receipt", {}).get("response_cache", {}).get("key")
            if cache_key is not None:
                discard_scan_response(cache_key)
        for part, saved in zip((ac, descan), snapshots):
            host = vars(part).get("_physical_host")
            vars(part).clear()
            vars(part).update(saved)
            if host is not None:
                object.__setattr__(part, "_physical_host", host)
        raise
