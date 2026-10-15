"""Read-only finite-pupil residuals of the installed optical model.

This observer never changes a component coefficient or supplies a propagation
operator. It includes the same intrinsic lens actions, placed fields and
correctors as production rays. A truncated gradient fit is diagnostic evidence,
not a claim that every transmitted electron shares one aberration polynomial.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from temsim import input_io
from temsim.optics.aberration_basis import (
    CARTESIAN_NAMES, UNIMPLEMENTED_TERMS, coefficient_fields, predict_displacement_m,
)
from temsim.optics.aberrations import EffectiveAberrationSet
from temsim.optics.aberration_validation import _control_snapshot
from temsim.optics.field_aberrations import fit_wave_gradient, pupil_rays


SCOPE = (
    "Finite-pupil diagnostic of the captured optical model, including intrinsic "
    "lens terms, optical placement and active correctors. Independent training "
    "and holdout rays use laboratory X/Y mechanical angles. This is not the "
    "retained source population or a wave fit. The fitted C1 is retained as a "
    "residual to nominal first-order transport with user lens-intrinsic terms "
    "removed only from that reference. No coefficients or excitations "
    "are written back. Pupil interception, finite-source averaging, energy "
    "spread and specimen scattering are not measured. Step convergence and "
    "the completeness of the aberration basis remain unassessed. This axial "
    "pupil does not measure field-dependent/off-axial coma or certify B-COR correction."
)


@dataclass(frozen=True)
class SystemResidualReport:
    coefficients: EffectiveAberrationSet
    diagnostics: dict


@dataclass(frozen=True)
class SystemResidualReference:
    """Fixed pupil and first-order reference for *diagnostic* hardware comparisons.

    No part of this reference is an applied wavefront or a focusing correction.
    Arrays are backed by immutable bytes so callers cannot change the pupil.
    """
    system: str
    alpha_rad: float
    energy_kev: float
    z0_mm: float
    z1_mm: float
    reference_name: str
    pupil_reference: str
    matrix: np.ndarray
    normalise: np.ndarray
    inputs: np.ndarray
    training: np.ndarray
    holdout: np.ndarray
    linear: np.ndarray
    transfer_condition: float


def _readonly(array):
    a = np.asarray(array, dtype=float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def capture_system_residual_reference(state, system, *, semiangle_mrad=10.,
                                      maximum_step_mm=.025, cancelled=None):
    """Capture the pupil once; reuse it when measuring hardware perturbations."""
    return measure_system_residual(state, system, semiangle_mrad=semiangle_mrad,
        maximum_step_mm=maximum_step_mm, cancelled=cancelled, _reference_only=True)


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RuntimeError("System residual measurement cancelled")


def image_residual_reference(state):
    """Use the installed corrector's output, never its upstream objective image."""
    if not bool(getattr(state, "image_corrector_installed", False)):
        return float(state.objective_image_plane_z_mm), "objective image (sample-referred coefficients)"
    planes = [item for item in getattr(state, "corrector_elements", ())
              if item.key == "image_sad_plane" and bool(getattr(item, "enabled", False))
              and bool(getattr(item, "_layout_installed", True))]
    if len(planes) != 1:
        raise ValueError("Installed image corrector requires one active SAD reference plane")
    target = float(planes[0].z_mm)
    downstream = [float(item.z_mm) for item in (
        *getattr(state, "lenses", ()), *getattr(state, "corrector_elements", ()))
        if str(getattr(item, "key", "")).startswith("image_")
        and item.key != "image_sad_plane" and bool(getattr(item, "enabled", False))
        and bool(getattr(item, "_layout_installed", True))]
    if not np.isfinite(target) or any(not np.isfinite(z) or z >= target for z in downstream):
        raise ValueError("Image corrector SAD reference must be downstream of its active elements")
    return target, "image-corrector SAD (sample-referred coefficients)"


def measure_system_residual(state, system, *, semiangle_mrad=10.0,
                            maximum_step_mm=.025, cancelled=None, reference=None,
                            _reference_only=False):
    """Measure a captured working point without mutating its optical inputs.

    Probe coefficients use angles at the exact sample plane, back-projected by
    the nominal 4-D first-order map. Image coefficients use angles at the sample
    and intercepts at the exact corrector SAD plane (or the objective image
    without a corrector), referred back by J_img.
    Neither construction changes physical lens settings or refocuses the rays.
    """
    _check_cancelled(cancelled)
    if system not in {"probe", "image"}:
        raise ValueError("System must be probe or image")
    alpha, requested_step = float(semiangle_mrad)*1e-3, float(maximum_step_mm)
    if not np.isfinite(alpha) or not 0 < alpha <= .1:
        raise ValueError("Diagnostic pupil semi-angle must be in (0, 100] mrad")
    state_step = float(state.step_mm)
    if not np.isfinite([requested_step, state_step]).all() or min(requested_step, state_step) <= 0:
        raise ValueError("Diagnostic and optical integration steps must be positive")
    work = _control_snapshot(state)
    # Public coefficient dictionaries must not alias the live editable inputs.
    from copy import deepcopy
    for name in ("lens_intrinsic_aberrations", "probe_aberrations", "image_aberrations"):
        if hasattr(state, name):
            setattr(work, name, deepcopy(getattr(state, name)))
    work.step_mm = min(state_step, requested_step)
    work._tuning_cancelled = lambda: bool(cancelled and cancelled())
    return _measure(work, system, alpha, cancelled, reference, _reference_only)


@input_io.using_state_inputs
def _measure(work, system, alpha, cancelled, reference=None, reference_only=False):
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    from temsim.physics.first_order import trace_transverse_transfer
    from temsim.physics.instrument_magnetic import active_column_events, events_overlapping_interval
    from temsim.physics.backend_execution import backend_receipts

    started = perf_counter()
    if system == "probe":
        z0, z1 = float(work.electron_gun.exit_plane_z_mm), float(work.sample.z_mm)
        reference_name = "sample/probe"
    else:
        z0 = float(work.sample.z_mm)
        z1, reference_name = image_residual_reference(work)
    if not np.isfinite([z0, z1]).all() or z1 <= z0:
        raise ValueError("System residual requires an ordered physical reference interval")
    energy_kev = float(work.beam_voltage_kv)
    if not np.isfinite(energy_kev) or energy_kev <= 0:
        raise ValueError("System residual requires positive beam energy")
    events = events_overlapping_interval(work, active_column_events(work), z0, z1)
    # Keep an explicit first-order reference: recapturing it after each
    # stigmator perturbation would absorb exactly the A1 we need to measure.
    def make_reference():
        nominal = _control_snapshot(work)
        nominal.lens_intrinsic_aberrations = {}
        nominal._tuning_cancelled = work._tuning_cancelled
        transfer = trace_transverse_transfer(nominal, z0, z1,
            maximum_step_mm=work.step_mm, events=events)
        _check_cancelled(cancelled)
        matrix = np.asarray(transfer.matrix, float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Nominal transfer must be a finite 4 by 4 map")
        # Scale metre/radian coordinates before testing numerical rank.
        units = np.array([1e-6, 1e-6, 1e-3, 1e-3])
        scaled = matrix*units[None, :]/units[:, None]
        transfer_condition = float(np.linalg.cond(scaled))
        if not np.isfinite(transfer_condition) or transfer_condition > 1e12:
            raise ValueError("Nominal transfer is singular or too ill-conditioned for this pupil")
        training, holdout = pupil_rays(alpha), pupil_rays(alpha, holdout=True)
        pupil = np.vstack((training, holdout))
        phase_reference = np.column_stack((np.zeros_like(pupil), pupil))
        if system == "probe":
            inputs = np.linalg.solve(matrix, phase_reference.T).T
            normalise = np.eye(2)
            pupil_reference = "nominal sample-plane mechanical angles"
        else:
            image = np.asarray(transfer.j_img, float)
            if not np.isfinite(np.linalg.cond(image)) or np.linalg.cond(image) > 1e8:
                raise ValueError("Image reference position map is singular for a sample-referred fit")
            inputs, normalise = phase_reference, np.linalg.inv(image)
            pupil_reference = "sample-plane mechanical angles"
        inputs = np.vstack((np.zeros((1, 4)), inputs))
        linear = (inputs[1:] @ matrix.T)[:, :2] @ normalise.T
        return SystemResidualReference(system, alpha, energy_kev, z0, z1,
            reference_name, pupil_reference, *map(_readonly,
                (matrix, normalise, inputs, training, holdout, linear)), transfer_condition)

    with backend_receipts([]) as receipts:
        if reference is None:
            reference = make_reference()
        elif not isinstance(reference, SystemResidualReference) or (
                reference.system != system or reference.alpha_rad != alpha or
                reference.energy_kev != energy_kev or reference.z0_mm != z0 or
                reference.z1_mm != z1):
            raise ValueError("Fixed system residual reference does not match system, pupil, energy or planes")
        if reference_only:
            return reference
        matrix, normalise, inputs = reference.matrix, reference.normalise, reference.inputs
        training, holdout, linear = reference.training, reference.holdout, reference.linear
        pupil_reference = reference.pupil_reference
        transfer_condition = reference.transfer_condition
        plan = build_propagation_plan(work, z0, z1, events,
            include_spherical_aberration=True, include_hexapole=True,
            save_z_mm=(z1,), checkpoint_z_mm=(z1,), maximum_step_mm=work.step_mm)
        _check_cancelled(cancelled)
        trace = execute_propagation_plan(work, plan, inputs[:, 0], inputs[:, 2],
                                         inputs[:, 1], inputs[:, 3])
        points = trace[-1]
        matches = np.flatnonzero(np.asarray(points.z_mm) == z1)
        if len(matches) != 1:
            raise ValueError("System residual requires an executed exact target checkpoint")
        row = int(matches[0])
        positions = np.column_stack((points.x_m[row], points.y_m[row]))
        if positions.shape != (len(inputs), 2) or not np.isfinite(positions).all():
            raise ValueError("Diagnostic rays left finite field support; reduce pupil or inspect optics")
        actual = (positions[1:]-positions[0]) @ normalise.T
        residual = actual-linear
        slopes = None
        if hasattr(points, "tx_rad") and hasattr(points, "ty_rad"):
            slopes = np.column_stack((points.tx_rad[row], points.ty_rad[row]))
            if slopes.shape != positions.shape or not np.isfinite(slopes).all():
                raise ValueError("Diagnostic ray slopes must be finite at the reference plane")
    _check_cancelled(cancelled)
    coefficients, rms, condition = fit_wave_gradient(training, residual[:len(training)], alpha)
    if not np.isfinite(coefficients).all() or not np.isfinite(condition) or condition > 1e8:
        raise ValueError("System residual gradient fit is numerically ill-conditioned")
    error = predict_displacement_m(holdout, coefficients)-residual[len(training):]
    result = EffectiveAberrationSet(reference_plane=reference_name,
        correction_state="system-residual-diagnostic", **coefficient_fields(coefficients),
        source="production optical-model rays; intrinsic terms and correctors retained",
        status="partial_numerical_diagnostic").validate()
    diagnostics = dict(
        scope=SCOPE, system=system, reference_interval_mm=[z0, z1],
        reference_z_mm=z1, coefficient_reference_z_mm=z1 if system == "probe" else z0,
        beam_energy_kev=energy_kev, pupil_reference=pupil_reference,
        coordinate_frame="right-handed laboratory X/Y; mechanical slopes in radians",
        nominal_reference="first-order optics without user lens-intrinsic terms; actual trace retains them",
        fit_semiangle_mrad=alpha*1e3, maximum_step_mm=work.step_mm,
        actual_step_range_mm=[float(np.min(plan.step_m)*1e3), float(np.max(plan.step_m)*1e3)],
        training_ray_count=len(training), holdout_ray_count=len(holdout),
        holdout_independent=True, fit_rms_m=rms, fit_condition_number=condition,
        transfer_condition_number=transfer_condition,
        holdout_rms_m=float(np.sqrt(np.mean(error**2))),
        residual_rms_m=float(np.sqrt(np.mean(residual**2))),
        holdout_max_error_m=float(np.max(np.linalg.norm(error, axis=1))),
        reference_centroid_xy_m=positions[0].tolist(),
        reference_centroid_slopes_rad=None if slopes is None else slopes[0].tolist(),
        fixed_reference_pupil=True,
        cartesian_coefficient_names=CARTESIAN_NAMES,
        cartesian_coefficients_mm=coefficients.tolist(), c1_retained=True,
        unmeasured_coefficients=("Cc",), unimplemented_terms=UNIMPLEMENTED_TERMS,
        interception_status="NOT_ASSESSED; mathematical pupil rays, not detector-accepted electrons",
        convergence_status="NOT_ASSESSED", coefficients_applied=False,
        backend_receipts=list(receipts), elapsed_s=perf_counter()-started,
    )
    return SystemResidualReport(result, diagnostics)
