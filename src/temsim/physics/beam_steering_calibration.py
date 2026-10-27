"""Selected-model entrance steering about one provenance-bound reference electron.

Positions are laboratory X/Y metres; tx/ty are mechanical slopes dx/dz,
dy/dz, not canonical momenta. Tilt commands are projected-angle increments
in radians. Every evaluation uses the selected column model and physical stops.
Finite-field models retain their configured HP, Cs and posed fields; thin
optics uses the same declared optical events as its particle propagation.
This is an optical reference-electron calibration, not a beam-centroid or
specimen-scattering measurement. Dynamic drives are frozen at the captured
simulation time, exactly as the column's static particle propagation plan.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np

from temsim import input_io
from temsim.calculation_cache import state_model_signature
from temsim.calculation_manifest import solver_source_identity
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields, events_overlapping_interval

DRIVE_FIELDS = ("upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad")
SCHEMA = "finite-entrance-steering-reference-electron-v1"
SCOPE = ("One identified reference electron; full nonlinear optical fields and physical stops; "
         "mechanical slopes in laboratory X/Y. Raster/wobble held at the captured time; "
         "no finite-probe, time-varying-field or specimen-scattering qualification.")
THIN_SCOPE = ("One identified reference electron; Ideal Optics thin optical events and physical stops; "
              "mechanical slopes in laboratory X/Y. Raster/wobble held at the captured time; "
              "no finite-thickness, empirical-aberration, finite-probe, time-varying-field "
              "or specimen-scattering qualification.")


def _scope(state):
    from temsim.simulation_modes import is_thin
    return THIN_SCOPE if is_thin(state) else SCOPE


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Beam steering cancelled; no live drives changed")


def _model_identity(state):
    # State.to_dict() finalizes some assembly-owned enablement. Signature
    # inspection must not apply that normalization to the editable instrument.
    return state_model_signature(decode_instrument(encode_instrument(state)))


def beam_steering_state_identity(state):
    """Physical/source/geometry identity plus numerical and requested backend policy."""
    return json_digest(dict(schema=SCHEMA, model=_model_identity(state),
        implementation=solver_source_identity(), step_mm=float(state.step_mm),
        history_step_mm=float(getattr(state, "history_step_mm", state.step_mm)),
        backend=str(state.acceleration_backend), acceleration_enabled=bool(state.acceleration_enabled)))


@dataclass(frozen=True)
class BeamSteeringReference:
    z_mm: float
    phase_space: tuple[float, float, float, float]
    kinetic_energy_ev: float
    flight_time_s: float
    provenance: str
    input_identity: str
    ray_id: int | None = None
    include_initial_plane_kicks: bool = False

    def __post_init__(self):
        phase = tuple(float(v) for v in self.phase_space)
        if (len(phase) != 4 or not np.isfinite((*phase, self.z_mm, self.kinetic_energy_ev,
                self.flight_time_s)).all() or self.kinetic_energy_ev <= 0 or self.flight_time_s < 0):
            raise ValueError("Reference requires finite XY/slopes, positive actual energy and a nonnegative clock")
        if not str(self.provenance).strip() or not str(self.input_identity).strip():
            raise ValueError("Reference provenance and input identity are required")
        object.__setattr__(self, "phase_space", phase)


@dataclass(frozen=True)
class BeamSteeringCalibration:
    input_identity: str
    target_z_mm: float
    baseline: tuple[float, ...]
    jacobian: tuple[tuple[float, ...], ...]
    condition_number: float
    drives_mrad: tuple[float, ...]
    backend: str
    reference: BeamSteeringReference
    scope: str = SCOPE


@dataclass(frozen=True)
class BeamSteeringSolution:
    input_identity: str
    target_z_mm: float
    baseline: tuple[float, ...]
    target: tuple[float, ...]
    achieved: tuple[float, ...]
    residual: tuple[float, ...]
    initial_drives_mrad: tuple[float, ...]
    drives_mrad: tuple[float, ...]
    condition_number: float
    iterations: int
    backend: str
    reference: BeamSteeringReference
    position_tolerance_m: float
    slope_tolerance: float
    scope: str = SCOPE

    def __post_init__(self):
        for name in ("baseline", "target", "achieved", "residual", "initial_drives_mrad", "drives_mrad"):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) != 4 or not np.isfinite(values).all():
                raise ValueError(f"Steering solution {name} must contain four finite values")
            object.__setattr__(self, name, values)
        if not np.isfinite((self.target_z_mm, self.condition_number, self.position_tolerance_m,
                self.slope_tolerance)).all() or min(self.position_tolerance_m, self.slope_tolerance) <= 0:
            raise ValueError("Steering solution tolerances must be positive and finite")
        if self.condition_number < 1 or self.iterations < 0 or not self.input_identity:
            raise ValueError("Invalid steering solution verification record")
        if self.reference.input_identity != self.input_identity:
            raise ValueError("Steering solution and reference identities differ")
        expected = np.asarray(self.achieved)-np.asarray(self.target)
        if not np.array_equal(expected, np.asarray(self.residual)):
            raise ValueError("Steering residual does not match its actual endpoint and target")


def reference_from_values(state, *, z_mm, phase_space, kinetic_energy_ev,
                          flight_time_s, provenance, ray_id=None,
                          include_initial_plane_kicks=False):
    """Explicit caller-supplied boundary; never infer a chief from a centroid."""
    return BeamSteeringReference(float(z_mm), tuple(phase_space), float(kinetic_energy_ev),
        float(flight_time_s), str(provenance), beam_steering_state_identity(state), ray_id,
        bool(include_initial_plane_kicks))


def _pair_and_support(state):
    if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
        raise ValueError("Optical beam steering cannot replay residual-gas scattering from an entrance checkpoint; use deterministic vacuum transport")
    pair = state.beam_deflector
    if not pair.enabled:
        raise ValueError("Enable the entrance Beam Shift/Tilt deflector before calibration")
    pair.validate()
    fields = [f for f in column_dipole_fields(state) if f.key.startswith(str(pair.key)+":")]
    if len(fields) != 2:
        raise ValueError("Entrance steering requires exactly two finite physical coils")
    return pair, min(f.field_support_mm[0] for f in fields), max(f.field_support_mm[1] for f in fields)


def _validate_plane(state, reference, target):
    _, start, end = _pair_and_support(state)
    if reference.z_mm > start + 1e-10:
        raise ValueError("Reference must be upstream of the complete entrance-coil field support")
    if not math.isfinite(target) or target <= end or target <= reference.z_mm:
        raise ValueError("Steering reference plane must be downstream of both entrance coils")
    if bool(getattr(state.sample, "inserted", False)) and target > float(state.sample.z_mm):
        raise ValueError("Optical steering beyond an inserted specimen requires material transport; choose its plane or an upstream plane")


@input_io.using_state_inputs
def _transport(state, reference, target_z_mm, maximum_step_mm, cancelled=None):
    """Actual float64 endpoint and node-wise clipping; no plotting interpolation."""
    _check_cancel(cancelled)
    events = events_overlapping_interval(state, active_column_events(state), reference.z_mm, target_z_mm)
    plan = build_propagation_plan(state, reference.z_mm, target_z_mm, events,
        include_spherical_aberration=True, include_hexapole=True,
        checkpoint_z_mm=(target_z_mm,), maximum_step_mm=maximum_step_mm)
    # The one-electron node history is small. Use exact float64 node states for
    # stopping, including finite tilted apertures, rather than float32 display data.
    nodes = np.arange(len(plan.z_mm), dtype=np.int64)
    plan = replace(plan, checkpoint_index=nodes)
    x, y, tx, ty = reference.phase_space
    points = execute_propagation_plan(state, plan,
        np.array([x]), np.array([tx]), np.array([y]), np.array([ty]),
        include_initial_plane_kicks=reference.include_initial_plane_kicks,
        initial_kinetic_energy_ev=np.array([reference.kinetic_energy_ev]),
        initial_time_s=np.array([reference.flight_time_s]), return_flight_times=True)[-1]
    _check_cancel(cancelled)
    if not all(np.isfinite(getattr(points, key)).all() for key in ("x_m", "y_m", "tx_rad", "ty_rad")):
        raise ValueError("Reference electron left the finite propagation domain")
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wall import clip_column_wall
    from temsim.physics.recording_clipping import clip_recording_planes
    alive, blocked, keys = clip_segment(state, points.z_mm, points.x_m, points.y_m)
    alive, blocked, keys = clip_column_wall(state, points.z_mm, points.x_m, points.y_m, alive, blocked, keys)
    alive, blocked, keys = clip_recording_planes(state, points.z_mm, points.x_m, points.y_m, alive, blocked, keys)
    if not bool(alive[0]):
        raise ValueError(f"Reference electron is blocked by {keys[0]} at Z {blocked[0]:.9g} mm")
    phase = tuple(float(getattr(points, key)[-1, 0]) for key in ("x_m", "y_m", "tx_rad", "ty_rad"))
    energy = points.kinetic_energy_ev
    if energy is None or points.flight_time_s is None:
        raise ValueError("Transport did not retain actual endpoint energy and flight time")
    endpoint = replace(reference, z_mm=float(target_z_mm), phase_space=phase,
        kinetic_energy_ev=float(energy[-1, 0]), flight_time_s=float(points.flight_time_s[-1, 0]),
        include_initial_plane_kicks=False)
    return endpoint, str(state.active_backend)


def capture_beam_steering_reference(state, *, result=None, gun_trace=None,
        source_state=None, ray_id=None, cancelled=None, maximum_step_mm=.05, target_z_mm=None):
    """Reuse an exact upstream checkpoint, else transport one actual gun electron.

    A supplied result must match both the working point and implementation.
    Bare gun data require their original source_state; they are never relabelled
    as current inputs. No centroid, random replacement, or phase-space zero is used.
    """
    _check_cancel(cancelled)
    identity = beam_steering_state_identity(state)
    private = decode_instrument(encode_instrument(state))
    _, entrance, _ = _pair_and_support(private)
    checkpoint = None
    simulation = None
    target = float(state.sample.z_mm if target_z_mm is None else target_z_mm)
    if result is not None:
        source_state = getattr(result, "state_snapshot", None)
        manifest = getattr(result, "calculation_manifest", None)
        if source_state is None or manifest is None or manifest.solver.source_digest != solver_source_identity():
            raise ValueError("Retained rays lack a matching implementation receipt; calculate current rays first")
        simulation = result.simulation
        gun_trace = simulation.gun_trace
        checkpoint = getattr(simulation, "incident_checkpoints", None)
    if gun_trace is not None:
        if source_state is None or _model_identity(source_state) != _model_identity(state):
            raise ValueError("Retained gun/source conditions do not match current optics; calculate current rays first")
    else:
        from temsim.optics.electron_gun.source import trace_source_to_exit
        private._tuning_cancelled = cancelled
        gun_trace = trace_source_to_exit(private)
    _check_cancel(cancelled)
    bundle = gun_trace.exit_bundle
    valid = np.asarray(bundle.alive, bool) & (np.asarray(bundle.weight) > 0.)
    for key in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev"):
        valid &= np.isfinite(getattr(bundle, key))
    if bundle.flight_time_s is None:
        raise ValueError("Reference gun electrons have no recorded flight times")
    valid &= np.isfinite(bundle.flight_time_s) & (np.asarray(bundle.flight_time_s) >= 0.)
    if simulation is not None:
        # Prefer an actual electron already known to reach the requested plane,
        # rather than whichever gun survivor happened to be stored first.
        blocked = np.asarray(simulation.incident.blocked_z, float)
        valid &= ~np.isfinite(blocked) | (blocked > target)
    row = None
    if checkpoint is not None and checkpoint.kinetic_energy_ev is not None and checkpoint.flight_time_s is not None:
        eligible = np.flatnonzero(np.asarray(checkpoint.z_mm) <= entrance)
        if len(eligible):
            row = int(eligible[-1])
            blocked = np.asarray(simulation.incident.blocked_z, float)
            valid &= ~np.isfinite(blocked) | (blocked > float(checkpoint.z_mm[row]))
            for key in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
                valid &= np.isfinite(getattr(checkpoint, key)[row])
    if ray_id is not None:
        valid &= np.asarray(bundle.ray_id) == int(ray_id)
    chosen = np.flatnonzero(valid)
    if not len(chosen):
        raise ValueError("No identified reference electron reaches the entrance boundary")
    radius_squared = (np.asarray(checkpoint.x_m[row])**2+np.asarray(checkpoint.y_m[row])**2
                      if row is not None else np.asarray(bundle.x_m)**2+np.asarray(bundle.y_m)**2)
    chosen = chosen[np.argsort(radius_squared[chosen], kind="stable")]
    last_error = None
    reference = None
    # A gun-only capture has no downstream survival receipt. Try at most eight
    # existing electrons, in radial order, retaining each electron's own energy
    # and clock. Explicit IDs are never replaced with a more convenient ray.
    for index in chosen[:1 if ray_id is not None else 8]:
        _check_cancel(cancelled)
        identifier = int(bundle.ray_id[index])
        if row is not None:
            phase = tuple(float(getattr(checkpoint, key)[row, index]) for key in ("x_m", "y_m", "tx_rad", "ty_rad"))
            candidate = BeamSteeringReference(float(checkpoint.z_mm[row]), phase,
                float(checkpoint.kinetic_energy_ev[row, index]), float(checkpoint.flight_time_s[row, index]),
                f"Retained exact upstream checkpoint; actual Tip electron id {identifier}; not beam centroid",
                identity, identifier, False)
        else:
            phase = tuple(float(getattr(bundle, key)[index]) for key in ("x_m", "y_m", "tx_rad", "ty_rad"))
            candidate = BeamSteeringReference(float(private.electron_gun.exit_plane_z_mm), phase,
                float(private.electron_gun.nominal_exit_energy_ev+bundle.energy_offset_ev[index]),
                float(bundle.flight_time_s[index]),
                f"Executed Tip to gun exit; actual Tip electron id {identifier}; not beam centroid",
                identity, identifier, True)
        try:
            _validate_plane(private, candidate, target)
            if candidate.z_mm < entrance:
                candidate, _ = _transport(private, candidate, entrance, maximum_step_mm, cancelled)
            if simulation is None:
                _transport(private, candidate, target, maximum_step_mm, cancelled)
            reference = candidate
            break
        except ValueError as exc:
            last_error = exc
    if reference is None:
        raise ValueError(f"No admissible recorded reference electron in the bounded candidate set: {last_error}")
    if identity != beam_steering_state_identity(state):
        raise ValueError("Inputs changed while preparing the steering reference")
    return reference


class BeamSteeringCalibrator:
    def __init__(self, state, reference, *, target_z_mm=None, maximum_step_mm=.05,
                 perturbation_rad=1e-6, maximum_condition=1e8, cancelled=None):
        _check_cancel(cancelled)
        self.input_identity = beam_steering_state_identity(state)
        if reference.input_identity != self.input_identity:
            raise ValueError("Steering reference is stale for these source/optical inputs")
        self._state = decode_instrument(encode_instrument(state))
        self.reference = reference
        self.target_z_mm = float(state.sample.z_mm if target_z_mm is None else target_z_mm)
        _validate_plane(self._state, reference, self.target_z_mm)
        self.maximum_step_mm = float(maximum_step_mm)
        self.perturbation_rad = float(perturbation_rad)
        self.maximum_condition = float(maximum_condition)
        if not np.isfinite((maximum_step_mm, perturbation_rad, maximum_condition)).all() or min(maximum_step_mm, perturbation_rad) <= 0 or maximum_condition <= 1:
            raise ValueError("Steering numerical step and condition limit must be positive and finite")
        self.initial_drives_mrad = tuple(float(getattr(self._state.beam_deflector, key)) for key in DRIVE_FIELDS)
        self._cache = {}
        self._calibration = None
        self._initial_cancelled = cancelled

    def matches(self, state):
        return self.input_identity == beam_steering_state_identity(state)

    def _evaluate(self, drives, cancelled):
        _check_cancel(cancelled)
        drives = tuple(float(v) for v in drives)
        pair = self._state.beam_deflector
        if len(drives) != 4 or not np.isfinite(drives).all() or max(map(abs, drives)) > pair.maximum_kick_mrad:
            raise ValueError("Steering solution exceeds the entrance-deflector hardware range")
        if drives in self._cache:
            return self._cache[drives]
        for key, value in zip(DRIVE_FIELDS, drives):
            setattr(pair, key, value)
        pair.validate()
        endpoint, backend = _transport(self._state, self.reference, self.target_z_mm,
            self.maximum_step_mm, cancelled)
        result = (np.asarray(endpoint.phase_space), backend)
        if len(self._cache) >= 128:
            self._cache.clear()
        self._cache[drives] = result
        return result

    def _jacobian(self, drives, cancelled):
        h = self.perturbation_rad*1e3
        columns = []
        for i in range(4):
            plus, minus = np.array(drives), np.array(drives)
            plus[i] += h
            minus[i] -= h
            columns.append((self._evaluate(plus, cancelled)[0]-self._evaluate(minus, cancelled)[0])/(2*self.perturbation_rad))
        matrix = np.column_stack(columns)
        # Normalize output rows before a condition test: raw metres/radians
        # would otherwise make the result depend on the selected length unit.
        row_norm = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(matrix).all() or np.any(row_norm <= 1e-20):
            raise ValueError("Entrance steering cannot control all four endpoint coordinates")
        condition = float(np.linalg.cond(matrix/row_norm[:, None]))
        if not math.isfinite(condition) or condition > self.maximum_condition:
            raise ValueError(f"Entrance steering response is singular or ill-conditioned ({condition:.6g})")
        return matrix, condition

    def calibrate(self, *, cancelled=None):
        callback = self._initial_cancelled if cancelled is None else cancelled
        self._initial_cancelled = None
        _check_cancel(callback)
        if self._calibration is None:
            baseline, backend = self._evaluate(self.initial_drives_mrad, callback)
            jacobian, condition = self._jacobian(self.initial_drives_mrad, callback)
            self._calibration = BeamSteeringCalibration(self.input_identity, self.target_z_mm,
                tuple(baseline), tuple(tuple(row) for row in jacobian), condition,
                self.initial_drives_mrad, backend, self.reference, _scope(self._state))
        return self._calibration

    def solve(self, *, kind, x, y, cancelled=None, maximum_iterations=8,
              position_tolerance_m=1e-11, slope_tolerance=1e-9):
        _check_cancel(cancelled)
        if kind not in {"tilt", "shift"} or not np.isfinite((x, y)).all():
            raise ValueError("Choose finite tilt radians or shift metres")
        if maximum_iterations < 1 or not np.isfinite((position_tolerance_m, slope_tolerance)).all() or min(position_tolerance_m, slope_tolerance) <= 0:
            raise ValueError("Steering iterations and endpoint tolerances must be positive")
        calibration = self.calibrate(cancelled=cancelled)
        target = np.asarray(calibration.baseline).copy()
        if kind == "shift":
            target[:2] += (x, y)
        else:
            angles = np.arctan(target[2:])+np.array((x, y))
            if np.any(np.abs(angles) >= .5*np.pi):
                raise ValueError("Requested tilt leaves forward propagation")
            target[2:] = np.tan(angles)
        tolerance = np.array((position_tolerance_m, position_tolerance_m, slope_tolerance, slope_tolerance))
        drives = np.asarray(self.initial_drives_mrad).copy()
        phase, backend = self._evaluate(drives, cancelled)
        jacobian, condition = np.asarray(calibration.jacobian), calibration.condition_number
        for iteration in range(maximum_iterations+1):
            residual = phase-target
            if np.all(np.abs(residual) <= tolerance):
                return BeamSteeringSolution(self.input_identity, self.target_z_mm,
                    calibration.baseline, tuple(target), tuple(phase), tuple(residual),
                    self.initial_drives_mrad, tuple(drives), condition, iteration, backend,
                    self.reference, position_tolerance_m, slope_tolerance, calibration.scope)
            if iteration == maximum_iterations:
                break
            if iteration:
                jacobian, condition = self._jacobian(drives, cancelled)
            delta = np.linalg.solve(jacobian/tolerance[:, None], -residual/tolerance)*1e3
            score = float(np.linalg.norm(residual/tolerance))
            accepted = False
            for fraction in (1., .5, .25, .125, .0625, .03125, .015625):
                candidate = drives + fraction*delta
                try:
                    candidate_phase, candidate_backend = self._evaluate(candidate, cancelled)
                except ValueError:
                    continue
                if np.linalg.norm((candidate_phase-target)/tolerance) < score:
                    drives, phase, backend = candidate, candidate_phase, candidate_backend
                    accepted = True
                    break
            if not accepted:
                raise ValueError("No admissible steering step improves the actual endpoint; range, stops or nonlinearity prevent this command")
        raise ValueError("Steering did not meet the actual position and slope tolerances; no live drives changed")


def apply_beam_steering_solution(state, solution):
    """Transactional four-drive commit only, with stale-input and residual checks."""
    if not isinstance(solution, BeamSteeringSolution):
        raise ValueError("Apply requires a complete verified steering solution")
    # Revalidate even a record modified by low-level callers after construction.
    solution.__post_init__()
    if beam_steering_state_identity(state) != solution.input_identity:
        raise ValueError("Steering solution is stale; source or optical settings changed")
    pair = state.beam_deflector
    previous = tuple(float(getattr(pair, key)) for key in DRIVE_FIELDS)
    if previous != solution.initial_drives_mrad:
        raise ValueError("Entrance drives changed after calibration")
    residual = np.asarray(solution.residual)
    tolerance = np.array((solution.position_tolerance_m,)*2+(solution.slope_tolerance,)*2)
    if residual.shape != (4,) or not np.isfinite(residual).all() or np.any(np.abs(residual) > tolerance):
        raise ValueError("Steering endpoint verification did not pass")
    try:
        for key, value in zip(DRIVE_FIELDS, solution.drives_mrad):
            setattr(pair, key, float(value))
        pair.validate()
    except Exception:
        for key, value in zip(DRIVE_FIELDS, previous):
            setattr(pair, key, value)
        raise
