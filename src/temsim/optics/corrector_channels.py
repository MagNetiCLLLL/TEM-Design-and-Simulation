"""Local, propagation-calibrated controls of the simulator's real hardware.

These are simulator channel assignments, not OEM calibrations. Coarse and fine
use different physical components. No aberration coefficient, compensation
phase, or Cs override is ever written. All cross-coupling remains in the rays
and waves; the reported polynomial is only a finite-pupil diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from temsim.immutable_json import freeze_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.aberration_basis import cartesian_coefficients
from temsim.optics.aberration_validation import _control_snapshot
from temsim.optics.system_residual_aberrations import (
    SystemResidualReport, SystemResidualReference, capture_system_residual_reference,
    measure_system_residual,
)


CHANNELS = ("a1_coarse", "a1_fine", "a2_coarse", "a2_fine")
SCOPE = (
    "Simulator calibration of existing distributed fields, not an OEM mapping. "
    "Normal/skew are laboratory Cartesian aberration components in mm. A fixed "
    "input pupil and nominal transfer are reused for every perturbation. "
    "C1, C3, beam displacement and direction are neither constrained nor removed "
    "from physical propagation. Finite-source, scattering, interception and "
    "integration-step convergence are not qualified by this local response."
)


@dataclass(frozen=True)
class ControlSpec:
    key: str
    field: str
    unit: str
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class ChannelCalibration:
    system: str
    channel: str
    state_digest: str
    control_specs: tuple[ControlSpec, ...]
    base_control_values: tuple[float, ...]
    steps: np.ndarray
    jacobian: np.ndarray
    baseline: SystemResidualReport
    reference: SystemResidualReference
    diagnostics: dict


@dataclass(frozen=True)
class ChannelCandidate:
    calibration: ChannelCalibration
    state_digest: str
    control_changes: tuple
    before: SystemResidualReport
    after: SystemResidualReport
    requested_delta_xy_mm: tuple[float, float]
    achieved_delta_xy_mm: tuple[float, float]
    diagnostics: dict
    resulting_state_digest: str


def _readonly(value):
    a = np.asarray(value, float)
    return np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape)


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise RuntimeError("Corrector channel calculation cancelled")


def _state_digest(state):
    return capture_instrument_snapshot(state).physical_digest


def _clone_working_point(state):
    # The lightweight ray-only _control_snapshot does not retain all aliases
    # in the complete instrument graph. A candidate must predict the exact
    # editable graph identity, including its existing alias relationships.
    return capture_instrument_snapshot(state).restore()


def _components(state):
    result = {}
    for name in ("corrector_elements", "deflectors", "stigmators"):
        for component in getattr(state, name, ()):
            key = str(component.key)
            if key in result and result[key] is not component:
                raise ValueError(f"Ambiguous physical component key: {key}")
            result[key] = component
    return result


def channel_control_specs(state, system, channel):
    """Describe independent physical drives, with finite calibration steps."""
    if system not in {"probe", "image"} or channel not in CHANNELS:
        raise ValueError("Select probe/image and an A1/A2 coarse/fine channel")
    components = _components(state)
    if channel == "a1_fine":
        keys = ["condenser_stigmator" if system == "probe" else "objective_stigmator"]
    elif channel == "a1_coarse":
        keys = [f"{system}_dph{i}_deflector" for i in (1, 2)]
    elif channel == "a2_coarse":
        keys = [f"{system}_hp{i}_hexapole" for i in (1, 2)]
        third = f"{system}_hp3_hexapole"
        if third in components and getattr(components[third], "_layout_installed", True):
            keys.append(third)
    else:
        keys = (["probe_hpc_hexapole", "probe_hpol_hexapole"] if system == "probe"
                else ["image_hpol_hexapole"])
    specs = []
    for key in keys:
        item = components.get(key)
        if item is None or not bool(getattr(item, "enabled", False)) or not bool(
                getattr(item, "_layout_installed", True)):
            raise ValueError(f"Channel requires installed, enabled hardware: {key}")
        if channel == "a1_fine":
            entries = (("strength_x_percent", "percent", 100., .01),
                       ("strength_y_percent", "percent", 100., .01))
        elif channel == "a1_coarse":
            limit = float(item.maximum_kick_mrad)
            entries = tuple((f"kick_{axis}_mrad", "mrad", limit, min(.001, limit/100))
                            for axis in ("x", "y"))
        else:
            limit = float(item.maximum_strength_m3)
            step = min(max(100., abs(float(item.strength_m3))*.001), limit*.001)
            entries = tuple((f"hexapole_{axis}_m3", "m^-3", limit, step)
                            for axis in ("normal", "skew"))
        for field, unit, limit, step in entries:
            if not np.isfinite([limit, step]).all() or min(limit, step) <= 0:
                raise ValueError(f"Invalid physical drive limit: {key}")
            specs.append(ControlSpec(key, field, unit, -limit, limit, step))
    return tuple(specs)


def _control_values(state, specs):
    components = _components(state)
    values = []
    for spec in specs:
        item = components[spec.key]
        if spec.field.startswith("hexapole_"):
            angle = 3*float(item.orientation_rad)
            value = float(item.strength_m3)*(np.cos(angle) if "normal" in spec.field else np.sin(angle))
        else:
            value = float(getattr(item, spec.field))
        values.append(value)
    return np.asarray(values)


def _set_controls(state, specs, values):
    components, hexapoles = _components(state), {}
    if len(specs) != len(values):
        raise ValueError("Physical control vector has the wrong size")
    for spec, value in zip(specs, values):
        value = float(value)
        if not np.isfinite(value) or not spec.minimum <= value <= spec.maximum:
            raise ValueError(f"Physical drive exceeds range: {spec.key}.{spec.field}")
        if spec.field.startswith("hexapole_"):
            hexapoles.setdefault(spec.key, {})[spec.field] = value
        else:
            setattr(components[spec.key], spec.field, value)
    for key, axes in hexapoles.items():
        item = components[key]
        normal, skew = axes["hexapole_normal_m3"], axes["hexapole_skew_m3"]
        old_strength, old_angle = float(item.strength_m3), float(item.orientation_rad)
        if (normal == old_strength*np.cos(3*old_angle) and
                skew == old_strength*np.sin(3*old_angle)):
            # A Cartesian roundtrip must not rewrite an uninvolved physical
            # drive or canonicalise its existing signed representation.
            continue
        strength = float(np.hypot(normal, skew))
        if strength > float(item.maximum_strength_m3):
            raise ValueError(f"Combined normal/skew field exceeds range: {key}")
        if strength:
            # h exp(i*3*theta) has multiple equivalent signed descriptions.
            # Keep an established signed branch and choose its nearest angle;
            # at zero excitation either sign is valid, so use the nearest of
            # both branches. A small drive change must not jump by 120 degrees.
            sign = -1. if old_strength < 0 else 1.
            angle = float(np.arctan2(sign*skew, sign*normal)/3)
            period = 2*np.pi/3 if old_strength else np.pi/3
            turns = round((old_angle-angle)/period)
            angle += turns*period
            if not old_strength and turns % 2:
                sign *= -1
            item.strength_m3 = sign*strength
            item.orientation_rad = angle
        else:
            item.strength_m3 = 0.
    for key in {spec.key for spec in specs}:
        validate = getattr(components[key], "validate", None)
        if validate:
            validate()


def _target(report, channel):
    values = np.asarray(cartesian_coefficients(report.coefficients), float)
    return values[[1, 2] if channel.startswith("a1") else [5, 6]]


def _measure(state, system, reference, maximum_step_mm, cancelled):
    _check_cancelled(cancelled)
    return measure_system_residual(state, system, semiangle_mrad=reference.alpha_rad*1e3,
        maximum_step_mm=maximum_step_mm, reference=reference, cancelled=cancelled)


def calibrate_channel(state, system, channel, *, semiangle_mrad=.5,
                      maximum_step_mm=.05, cancelled=None):
    """Measure local two-axis response with signed and half-step perturbations.

    Each drive has its own physical unit. Conditioning uses J*steps, so the
    minimum-norm solution minimises fractions of tested physical drive ranges.
    No state mutation is made, including on failure or cancellation.
    """
    started = perf_counter()
    _check_cancelled(cancelled)
    digest = _state_digest(state)
    work = _control_snapshot(state)
    specs = channel_control_specs(work, system, channel)
    base = _control_values(work, specs)
    if not np.isfinite(base).all():
        raise ValueError("Physical control working point must be finite")
    steps = np.asarray([spec.step for spec in specs])
    # A working point at a range limit cannot be calibrated with signed probes.
    for spec, value, step in zip(specs, base, steps):
        if value-step < spec.minimum or value+step > spec.maximum:
            raise ValueError(f"Insufficient range for signed calibration: {spec.key}.{spec.field}")
    reference = capture_system_residual_reference(work, system, semiangle_mrad=semiangle_mrad,
        maximum_step_mm=maximum_step_mm, cancelled=cancelled)
    baseline = _measure(work, system, reference, maximum_step_mm, cancelled)
    full, half, asymmetry = [], [], []
    for index, step in enumerate(steps):
        samples = []
        for fraction in (1., -1., .5, -.5):
            _check_cancelled(cancelled)
            perturbed = _control_snapshot(work)
            values = base.copy()
            values[index] += step*fraction
            _set_controls(perturbed, specs, values)
            samples.append(_target(_measure(perturbed, system, reference,
                                            maximum_step_mm, cancelled), channel))
        plus, minus, half_plus, half_minus = samples
        full.append((plus-minus)/(2*step))
        half.append((half_plus-half_minus)/step)
        scale = max(float(np.linalg.norm(half_plus-half_minus)), 1e-15)
        asymmetry.append(float(np.linalg.norm(half_plus+half_minus-2*_target(baseline, channel))/scale))
    full, jacobian = np.asarray(full).T, np.asarray(half).T
    scaled = jacobian*steps
    singular = np.linalg.svd(scaled, compute_uv=False)
    if (not np.isfinite(scaled).all() or len(singular) != 2 or singular[-1] <= 1e-14
            or singular[0]/singular[-1] > 1e6):
        raise ValueError("Two-axis physical channel response is singular, too weak or ill-conditioned")
    relative_error = float(np.linalg.norm((full-jacobian)*steps)/np.linalg.norm(scaled))
    if relative_error > .1 or max(asymmetry) > .2:
        raise ValueError("Physical response fails signed/half-step linearity; reduce the calibration range or pupil")
    _check_cancelled(cancelled)
    if _state_digest(state) != digest:
        raise ValueError("Working point changed during calibration; calculate the response again")
    diagnostics = freeze_json(dict(scope=SCOPE, condition_number=float(singular[0]/singular[-1]),
        signed_half_step_relative_error=relative_error, signed_asymmetry=asymmetry,
        singular_values_mm=singular.tolist(), maximum_step_mm=float(maximum_step_mm),
        semiangle_mrad=float(semiangle_mrad), trust_radius_in_steps=.5,
        physical_measurement_count=1+4*len(specs), elapsed_s=perf_counter()-started,
        model_calibration=True, oem_calibration=False, coefficients_applied=False))
    return ChannelCalibration(system, channel, digest, specs, tuple(base), _readonly(steps),
                              _readonly(jacobian), baseline, reference, diagnostics)


def _physical_changes(before, after, specs):
    original, changed = _components(before), _components(after)
    changes = []
    pairs = set()
    for spec in specs:
        fields = ("strength_m3", "orientation_rad") if spec.field.startswith("hexapole_") else (spec.field,)
        for field in fields:
            if (spec.key, field) in pairs:
                continue
            pairs.add((spec.key, field))
            old, new = float(getattr(original[spec.key], field)), float(getattr(changed[spec.key], field))
            if old != new:
                changes.append(freeze_json(dict(key=spec.key, field=field, before=old, after=new)))
    return tuple(changes)


def prepare_channel_change(state, calibration, delta_xy_mm, *, cancelled=None):
    """Solve and independently propagate a bounded candidate without applying it."""
    _check_cancelled(cancelled)
    if _state_digest(state) != calibration.state_digest:
        raise ValueError("Channel calibration is stale; calculate the response again")
    delta = np.asarray(delta_xy_mm, float)
    if delta.shape != (2,) or not np.isfinite(delta).all() or not np.any(delta):
        raise ValueError("Requested normal/skew increment must contain two finite values and be nonzero")
    scaled = calibration.jacobian*calibration.steps
    fractions = np.linalg.lstsq(scaled, delta, rcond=None)[0]
    if not np.isfinite(fractions).all() or np.max(np.abs(fractions)) > .5:
        raise ValueError("Requested change exceeds the calibrated local range; request a smaller increment")
    work = _clone_working_point(state)
    _set_controls(work, calibration.control_specs,
        np.asarray(calibration.base_control_values)+calibration.steps*fractions)
    after = _measure(work, calibration.system, calibration.reference,
                     calibration.diagnostics["maximum_step_mm"], cancelled)
    achieved = _target(after, calibration.channel)-_target(calibration.baseline, calibration.channel)
    error = float(np.linalg.norm(achieved-delta))
    tolerance = max(1e-11, .1*float(np.linalg.norm(delta)))
    if not np.isfinite(achieved).all() or error > tolerance:
        raise ValueError("Independent physical candidate does not achieve the requested local response")
    _check_cancelled(cancelled)
    if _state_digest(state) != calibration.state_digest:
        raise ValueError("Working point changed during candidate measurement; calculate the response again")
    changes = _physical_changes(state, work, calibration.control_specs)
    if not changes:
        raise ValueError("Requested change is below physical drive resolution")
    before_values = np.asarray(cartesian_coefficients(calibration.baseline.coefficients))
    after_values = np.asarray(cartesian_coefficients(after.coefficients))
    diagnostics = freeze_json(dict(scope=SCOPE, independent_candidate_measured=True,
        target_error_mm=error, target_tolerance_mm=tolerance, control_fractions=fractions.tolist(),
        all_coefficient_delta_mm=(after_values-before_values).tolist(),
        centroid_before_m=calibration.baseline.diagnostics.get("reference_centroid_xy_m"),
        centroid_after_m=after.diagnostics.get("reference_centroid_xy_m"),
        direction_before_rad=calibration.baseline.diagnostics.get("reference_centroid_slopes_rad"),
        direction_after_rad=after.diagnostics.get("reference_centroid_slopes_rad"),
        coefficients_applied=False, compensation_phase_applied=False))
    return ChannelCandidate(calibration, calibration.state_digest, changes, calibration.baseline,
        after, tuple(delta), tuple(achieved), diagnostics, _state_digest(work))


def apply_channel_change(state, candidate, *, cancelled=None):
    """Atomically apply the already measured physical drives to the same state.

    The caller owns the GUI/state thread. Revalidate a detached complete result
    before any write; rollback all writes on error or cancellation. A successful
    application invalidates this calibration through the physical digest.
    """
    _check_cancelled(cancelled)
    if _state_digest(state) != candidate.state_digest:
        raise ValueError("Channel candidate is stale; calculate the response again")
    allowed = {(c["key"], c["field"]) for c in candidate.control_changes}
    expected_fields = set()
    for spec in candidate.calibration.control_specs:
        for field in (("strength_m3", "orientation_rad") if spec.field.startswith("hexapole_") else (spec.field,)):
            expected_fields.add((spec.key, field))
    if not allowed or not allowed <= expected_fields:
        raise ValueError("Candidate contains changes outside its physical channel")
    detached = _clone_working_point(state)
    copied = _components(detached)
    live = _components(state)
    writes = []
    for change in candidate.control_changes:
        key, field = change["key"], change["field"]
        before, after = float(change["before"]), float(change["after"])
        if not np.isfinite([before, after]).all() or float(getattr(live[key], field)) != before:
            raise ValueError("Candidate does not match the current physical drive")
        setattr(copied[key], field, after)
        writes.append((live[key], field, before, after))
    for spec, value in zip(candidate.calibration.control_specs,
                           _control_values(detached, candidate.calibration.control_specs)):
        if not np.isfinite(value) or not spec.minimum <= value <= spec.maximum:
            raise ValueError("Candidate physical drive is outside its range")
    for key in {key for key, _field in expected_fields}:
        validate = getattr(copied[key], "validate", None)
        if validate:
            validate()
    if _state_digest(detached) != candidate.resulting_state_digest:
        raise ValueError("Candidate physical result failed identity verification")
    _check_cancelled(cancelled)
    if _state_digest(state) != candidate.state_digest:
        raise ValueError("Working point changed before application")
    done = []
    try:
        for item, field, before, after in writes:
            _check_cancelled(cancelled)
            done.append((item, field, before))
            setattr(item, field, after)
        _check_cancelled(cancelled)
        if _state_digest(state) != candidate.resulting_state_digest:
            raise ValueError("Applied physical result failed identity verification")
    except BaseException:
        for item, field, before in reversed(done):
            setattr(item, field, before)
        raise
