"""Measure a tip-origin incident particle probe at the specimen upper surface.

Z increases downstream. Sample Z is the centre; the entrance is Z-t/2.
Full-precision transport checkpoints, not plotted coordinates, define focus.
The three focus planes are all upstream of (or exactly on) the material.
This module does not calculate diffraction or infer a wave-optical probe size.
"""
from dataclasses import dataclass
import math

import numpy as np

from temsim.optics.direct_alignment import _pre_sample_kick_events
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.beam_statistics import TransverseBeamStatistics, transverse_beam_statistics
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import propagate


@dataclass(frozen=True)
class SurfaceFocusMeasurement:
    surface_z_mm: float
    statistics: TransverseBeamStatistics
    local_waist_offset_nm: float
    variance_second_derivative: float
    upstream_rms_radii_m: tuple[float, float, float]
    step_mm: float
    effective_rays: float | None = None
    source_population_identity: str | None = None
    surviving_source_ids: tuple[int, ...] | None = None
    transmitted_source_weight: float | None = None
    column_prefix_reused: bool = False

    def accepts(self, target_mrad: float, *, angle_relative_tolerance=.01,
                focus_tolerance_nm=1., minimum_current_fraction=0.,
                minimum_rays=16) -> bool:
        """Do not confuse preserved crossover count with preserved current.

        Real pupils may accept far less than 1% of source current. Require positive
        current and resolved statistical support; an optional minimum-current target
        belongs to the experiment, not an implicit source-transmission constraint.
        """
        s=self.statistics
        effective=self.effective_rays if self.effective_rays is not None else s.surviving_rays
        return bool(
            math.isfinite(target_mrad) and target_mrad>0
            and math.isfinite(minimum_current_fraction) and 0<=minimum_current_fraction<=1
            and math.isfinite(self.variance_second_derivative)
            and self.variance_second_derivative>0
            and math.isfinite(self.local_waist_offset_nm)
            and abs(self.local_waist_offset_nm)<=focus_tolerance_nm
            and abs(s.waist_offset_m*1e9)<=focus_tolerance_nm
            and abs(s.convergence_95_mrad/target_mrad-1)<=angle_relative_tolerance
            and s.surviving_rays>=minimum_rays
            and math.isfinite(effective)
            and effective>=minimum_rays*(1-32*np.finfo(float).eps)
            and math.isfinite(s.surviving_fraction) and s.surviving_fraction>0
            and s.surviving_fraction>=minimum_current_fraction)


def surface_z_mm(state) -> float:
    z=float(state.sample.z_mm)
    thickness=float(state.sample.thickness_nm)
    if not math.isfinite(z) or not math.isfinite(thickness) or thickness<0:
        raise ValueError('Surface focus requires finite sample Z and non-negative thickness')
    return z-thickness*.5e-6


def local_waist_from_radii(radii_m, spacing_m: float) -> tuple[float,float]:
    """One-sided variance derivative at the last of three upstream planes.

Returns (estimated local waist offset in nm, variance second derivative).
A positive second derivative distinguishes a local minimum from a maximum.
No field-free extrapolation across the specimen is needed.
"""
    r=np.asarray(radii_m,dtype=float)
    h=float(spacing_m)
    if r.shape!=(3,) or not np.all(np.isfinite(r)) or np.any(r<0) or not math.isfinite(h) or h<=0:
        raise ValueError('Focus derivative requires three finite radii and positive plane spacing')
    v=r*r
    # Difference before combining to reduce subtraction of the common radius.
    d0,d1=v[2]-v[1],v[1]-v[0]
    first=(3*d0-d1)/(2*h)
    second=(d0-d1)/(h*h)
    offset=-first/second*1e9 if second>0 else math.nan
    return float(offset),float(second)


def measure_surface_focus(state, *, step_mm=.05, upstream_spacing_nm=100.) -> SurfaceFocusMeasurement:
    """Execute the physical gun (or reuse its exact cached trace), then focus."""
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.beam_current import effective_source_current_a
    if state.electron_gun.source_representation!='classical_particles':
        raise ValueError('Surface particle focus requires the classical tip-origin source')
    if state.vacuum_map.enabled:
        raise ValueError('Surface focus with active residual-medium transport is not yet qualified')
    if effective_source_current_a(state)<=0:
        raise ValueError('Surface focus requires a nonzero physical source current')
    end=surface_z_mm(state)
    start=float(state.electron_gun.exit_plane_z_mm)
    h=float(upstream_spacing_nm)*1e-6
    if not math.isfinite(step_mm) or step_mm<=0 or not math.isfinite(h) or h<=0 or end-2*h<=start:
        raise ValueError('Focus planes must be downstream of the gun, with positive finite steps')
    planes=(end-2*h,end-h,end)
    from temsim.simulation_modes import is_unified
    if is_unified(state):
        return UnifiedSurfaceFocusEvaluator(state, step_mm=step_mm,
            upstream_spacing_nm=upstream_spacing_nm).measure(state)
    gun=trace_source_to_exit(state)
    e=gun.exit_bundle
    apertures=tuple(a.z_mm for a in state.apertures if start<a.z_mm<end)
    z,x,tx,y,ty,checkpoints=propagate(state,start,end,e.x_m,e.tx_rad,e.y_m,e.ty_rad,
        events=_pre_sample_kick_events(state),energy_offset_ev=e.energy_offset_ev,
        # Save and checkpoint requests both retain exact integration surfaces.
        save_z_mm=apertures+planes,checkpoint_z_mm=planes,return_checkpoints=True,
        maximum_step_mm=step_mm)
    alive=np.asarray(e.alive,bool).copy()
    stops=np.asarray(gun.blocked_z_mm,float).copy()
    keys=list(gun.blocked_key)
    alive,stops,keys=clip_segment(state,z,x,y,alive,stops,keys)
    alive,stops,keys=clip_column_wall(state,z,x,y,alive,stops,keys)
    alive &= np.asarray(e.weight)>0
    rows=[]
    for plane in planes:
        indices=np.flatnonzero(np.abs(checkpoints.z_mm-plane)<1e-10)
        if len(indices)!=1:
            raise ValueError('Missing exact surface-focus checkpoint')
        row=int(indices[0])
        values=(checkpoints.x_m[row],checkpoints.y_m[row],checkpoints.tx_rad[row],checkpoints.ty_rad[row])
        if any(not np.all(np.isfinite(v[alive])) for v in values):
            raise ValueError('Nonfinite current-carrying surface-focus trajectory')
        rows.append(transverse_beam_statistics(*values,alive=alive,weights=e.weight))
    radii=tuple(row.radius_rms_m for row in rows)
    offset,second=local_waist_from_radii(radii,h*1e-3)
    weights=np.asarray(e.weight,float)[alive]
    weights=weights/np.sum(weights)
    effective_rays=float(1/np.sum(weights*weights))
    return SurfaceFocusMeasurement(end,rows[-1],offset,second,radii,float(step_mm),effective_rays)


class UnifiedSurfaceFocusEvaluator:
    """Read-only surface observations with exact source and proved prefix reuse.

    Only the declared round-lens excitation percentages may differ between
    trials. No preset, scan calibration, legacy kick or field cutoff is used.
    Gun reuse uses the production dependency signature; column reuse uses the
    production swept-domain, numerical-grid and full-state checkpoint proof.
    """

    def __init__(self, state, *, controls=(), step_mm=.05, upstream_spacing_nm=100.):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.simulation_modes import is_unified
        if not is_unified(state):
            raise ValueError("Unified surface evaluation requires Finite-Field Optics")
        if state.electron_gun.source_representation != "classical_particles" or state.vacuum_map.enabled:
            raise ValueError("Unified surface evaluation requires classical particles without residual-medium transport")
        self.controls = tuple(controls)
        lenses = {lens.key: lens for lens in state.lenses}
        if (len(self.controls) != len(set(self.controls)) or any(
                key not in lenses or not lenses[key].enabled or lenses[key].z_mm > state.sample.z_mm
                for key in self.controls)):
            raise ValueError("Surface controls must be distinct installed pre-sample lenses")
        self.step_mm, self.spacing_nm = float(step_mm), float(upstream_spacing_nm)
        if not math.isfinite(self.step_mm) or self.step_mm <= 0 or not math.isfinite(self.spacing_nm) or self.spacing_nm <= 0:
            raise ValueError("Surface step and plane spacing must be positive and finite")
        self._snapshot = capture_instrument_snapshot(state)
        self._initial = {key: float(lenses[key].percent) for key in self.controls}
        self._gun = self._gun_signature = self._segment = None
        self._sources = {}

    def at_step(self, step_mm):
        """Start an independent integration mesh, sharing only exact gun traces."""
        other = type(self)(self._snapshot.restore(), controls=self.controls,
                           step_mm=step_mm, upstream_spacing_nm=self.spacing_nm)
        other._sources = self._sources
        return other

    def measure(self, state):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.physics.beam_current import effective_source_current_a
        from temsim.physics.particle_sections import _trace_segment, _events, _digest_arrays, gun_dependency_signature
        from temsim.physics.ray_identity import emitted_source_identity
        from temsim.physics.simulation import _validate_gun_flight_times
        from temsim.optics.electron_gun.source import trace_source_to_exit
        from temsim.physics.optical_tuning import check_tuning_cancelled

        check_tuning_cancelled(state)
        original = capture_instrument_snapshot(state)
        candidate = original.restore()
        if callable(getattr(state, "_tuning_cancelled", None)):
            candidate._tuning_cancelled = state._tuning_cancelled
        lenses = {lens.key: lens for lens in candidate.lenses}
        values = {}
        for key in self.controls:
            value = float(lenses[key].percent)
            if not math.isfinite(value) or not 0 <= value <= lenses[key].max_percent:
                raise ValueError("Surface trial excitation is outside the configured bounds")
            values[key] = value
            lenses[key].percent = self._initial[key]
        if capture_instrument_snapshot(candidate).digest != self._snapshot.digest:
            raise ValueError("Surface trial changed inputs outside the declared excitation controls")
        for key, value in values.items():
            lenses[key].percent = value
        if effective_source_current_a(candidate) <= 0:
            raise ValueError("Surface focus requires a nonzero physical source current")
        candidate.step_mm = min(float(candidate.step_mm), self.step_mm)
        start = float(candidate.electron_gun.exit_plane_z_mm)
        end, spacing = surface_z_mm(candidate), self.spacing_nm * 1e-6
        if not math.isfinite(candidate.step_mm) or candidate.step_mm <= 0 or end - 2 * spacing <= start:
            raise ValueError("Surface observation planes must follow the physical gun exit")
        planes = (end - 2 * spacing, end - spacing, end)
        signature = gun_dependency_signature(candidate)
        if self._gun is None or not signature or signature != self._gun_signature:
            self._gun = self._sources.get(signature) if signature else None
            if self._gun is None:
                self._gun = trace_source_to_exit(candidate)
            if not _validate_gun_flight_times(self._gun):
                self._gun = None
                raise ValueError("Unified surface focus requires the executed gun flight-time history")
            if signature:
                # Trial fields can change the exact gun dependency identity.
                # Bound retained gun history instead of accumulating every fit.
                self._sources.clear()
                self._sources[signature] = self._gun
            self._gun_signature = signature
            self._segment = None
        gun, emitted = self._gun, self._gun.exit_bundle
        identity = emitted_source_identity(gun)
        initial = tuple(np.asarray(getattr(emitted, name)) for name in
                        ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
        offsets, weights = np.asarray(emitted.energy_offset_ev), np.asarray(emitted.weight)
        kinetic = float(candidate.electron_gun.nominal_exit_energy_ev) + offsets
        survival = (np.asarray(emitted.alive, bool), np.asarray(gun.blocked_z_mm), list(gun.blocked_key))
        segment, _resume, reused = _trace_segment(candidate, "incident", start, end,
            initial, offsets, weights, identity, survival, _events(candidate), self._segment, math.inf,
            initial_kicks=True, save_z_mm=planes, initial_kinetic_energy_ev=kinetic)
        self._segment = segment
        alive = np.asarray(segment.branch.alive, bool) & (weights > 0)
        checkpoints = segment.checkpoints
        rows = []
        for plane in planes:
            indices = np.flatnonzero(np.abs(checkpoints.z_mm - plane) < 1e-10)
            if len(indices) != 1:
                raise ValueError("Missing exact unified surface-focus checkpoint")
            row = int(indices[0])
            values = tuple(getattr(checkpoints, key)[row] for key in ("x_m", "y_m", "tx_rad", "ty_rad"))
            if any(not np.all(np.isfinite(value[alive])) for value in values):
                raise ValueError("Nonfinite transmitted unified surface-focus trajectory")
            rows.append(transverse_beam_statistics(*values, alive=alive, weights=weights))
        radii = tuple(row.radius_rms_m for row in rows)
        offset, second = local_waist_from_radii(radii, spacing * 1e-3)
        transmitted = weights[alive]
        effective = float(transmitted.sum()**2 / (transmitted @ transmitted)) if np.any(transmitted) else 0.
        population = _digest_arrays(*initial, kinetic, weights, survival[0], survival[1], identity[0])
        known = np.asarray(identity[0])
        surviving_ids = (tuple(map(int, known[alive])) if np.all(known >= 0)
                         and len(np.unique(known)) == len(known) else None)
        if capture_instrument_snapshot(state).digest != original.digest:
            raise ValueError("Live inputs changed during unified surface measurement")
        return SurfaceFocusMeasurement(end, rows[-1], offset, second, radii, candidate.step_mm, effective,
            population, surviving_ids, float(transmitted.sum()), bool(reused))


def unified_surface_acceptance(measurement, baseline, target_mrad=30., *,
                               maximum_rms_radius_nm=.2, maximum_diameter95_nm=.7):
    """Return explicit rejection reasons, including source and current identity.

    This gate is separate from the solver objective. A covariance waist alone
    cannot qualify an aberrated probe, and removing aberrated rays is not focus.
    """
    if (not math.isfinite(maximum_rms_radius_nm) or maximum_rms_radius_nm <= 0
            or not math.isfinite(maximum_diameter95_nm) or maximum_diameter95_nm <= 0):
        raise ValueError("Surface probe size limits must be positive and finite")
    reasons = []
    if not measurement.accepts(target_mrad):
        reasons.append("Focus, convergence angle or resolved-current gate failed")
    stats = measurement.statistics
    if not math.isfinite(stats.radius_rms_m) or stats.radius_rms_m * 1e9 >= maximum_rms_radius_nm:
        reasons.append("RMS probe radius exceeds the declared limit")
    if not math.isfinite(stats.radius_95_m) or stats.radius_95_m * 2e9 >= maximum_diameter95_nm:
        reasons.append("95-percent current diameter exceeds the declared limit")
    if not baseline.source_population_identity or measurement.source_population_identity != baseline.source_population_identity:
        reasons.append("Executed source population changed or is unidentified")
    if (baseline.surviving_source_ids is None or measurement.surviving_source_ids is None
            or not set(baseline.surviving_source_ids).issubset(measurement.surviving_source_ids)):
        reasons.append("Previously transmitted source rays were removed or are unidentified")
    old, new = baseline.transmitted_source_weight, measurement.transmitted_source_weight
    if (old is None or new is None or not math.isfinite(old) or not math.isfinite(new)
            or old <= 0 or new < old):
        reasons.append("Original transmitted source weight was not retained")
    return tuple(reasons)


def refine_unified_surface_focus(state, target_mrad=30., *, controls=None,
                                 maximum_evaluations=30, maximum_rms_radius_nm=.2,
                                 maximum_diameter95_nm=.7,
                                 maximum_step_rms_difference_nm=.005,
                                 maximum_step_waist_difference_nm=.1,
                                 maximum_step_diameter95_difference_nm=.02,
                                 maximum_step_alpha_relative_difference=.001, callback=None):
    """Explicit detached baseline calibration; ordinary lens edits never call it.

    Objective-only fitting requires alpha95 already within one percent. An
    explicit (C3, Objective) selection may fit both convergence and focus.
    Every measured trial counts toward the budget, including numerical-Jacobian
    evaluations and the independent 0.05/0.025 mm acceptance measurements.
    Failure never widens tolerances, moves pupils, changes source sampling,
    adjusts hexapoles or supplies a candidate for application.
    Passing the declared two-step differences is not an asymptotic convergence
    proof or a particle-sampling qualification.
    """
    from scipy.optimize import least_squares
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.optical_tuning import check_tuning_cancelled
    allowed = (("objective_lens",), ("condenser_lens_3", "objective_lens"))
    keys = allowed[0] if controls is None else tuple(controls)
    if keys not in allowed:
        raise ValueError("Explicit focus controls must be Objective or (C3, Objective)")
    step_limits = dict(radius_rms_nm=maximum_step_rms_difference_nm,
        local_waist_nm=maximum_step_waist_difference_nm,
        covariance_waist_nm=maximum_step_waist_difference_nm,
        diameter95_nm=maximum_step_diameter95_difference_nm,
        alpha95_relative=maximum_step_alpha_relative_difference)
    if (not math.isfinite(target_mrad) or target_mrad <= 0
            or type(maximum_evaluations) is not int or maximum_evaluations < 3
            or not math.isfinite(maximum_rms_radius_nm) or maximum_rms_radius_nm <= 0
            or not math.isfinite(maximum_diameter95_nm) or maximum_diameter95_nm <= 0
            or any(not math.isfinite(value) or value <= 0 for value in step_limits.values())):
        raise ValueError("Positive focus limits and an integer budget of at least three are required")
    original = capture_instrument_snapshot(state)
    candidate = original.restore()
    if callable(getattr(state, "_tuning_cancelled", None)):
        candidate._tuning_cancelled = state._tuning_cancelled
    evaluator = UnifiedSurfaceFocusEvaluator(candidate, controls=keys, step_mm=.05)
    fine_evaluator = evaluator.at_step(.025)
    lenses = {lens.key: lens for lens in candidate.lenses}
    initial = np.array([lenses[key].percent for key in keys], float)
    upper = np.array([lenses[key].max_percent for key in keys], float)
    if (not np.all(np.isfinite(initial)) or not np.all(np.isfinite(upper))
            or np.any(upper <= 0) or np.any(initial < 0) or np.any(initial > upper)):
        raise ValueError("Initial focus controls must lie inside finite physical limits")
    measurements, history = {}, []
    limits = dict(maximum_rms_radius_nm=maximum_rms_radius_nm,
                  maximum_diameter95_nm=maximum_diameter95_nm)

    class EvaluationBudgetReached(Exception):
        pass

    def measure(values, *, fine=False, fitting=False):
        check_tuning_cancelled(candidate)
        identity = (fine, *map(float, values))
        if identity not in measurements:
            # Conservatively reserve a final coarse measurement and both the
            # original and proposed fine bundles, even if some are cache hits.
            limit = maximum_evaluations - 3 if fitting else maximum_evaluations
            if len(history) >= limit:
                raise EvaluationBudgetReached("Declared actual-measurement budget reached")
            for key, value in zip(keys, values):
                lenses[key].percent = float(value)
            measured = (fine_evaluator if fine else evaluator).measure(candidate)
            measurements[identity] = measured
            entry = dict(evaluation=len(history)+1, strengths=dict(zip(keys, map(float, values))),
                step_mm=measured.step_mm, radius_rms_nm=measured.statistics.radius_rms_m*1e9,
                diameter95_nm=measured.statistics.radius_95_m*2e9,
                alpha95_mrad=measured.statistics.convergence_95_mrad,
                local_waist_offset_nm=measured.local_waist_offset_nm,
                covariance_waist_offset_nm=measured.statistics.waist_offset_m*1e9,
                surviving_rays=measured.statistics.surviving_rays,
                transmitted_source_weight=measured.transmitted_source_weight,
                column_prefix_reused=measured.column_prefix_reused)
            history.append(entry)
            if callback is not None:
                callback({**entry, "strengths": dict(entry["strengths"])})
        return measurements[identity]

    baseline = measure(initial)
    evidence = dict(status="FAIL", applied=False, controls=keys,
        input_identity=original.digest, target_alpha95_mrad=float(target_mrad),
        thresholds=limits, step_consistency_limits=step_limits,
        particle_count=int(state.electron_gun.emitter.ray_count),
        sampling_convergence="NOT_TESTED", topology_qualification="NOT_TESTED",
        qualification_scope="Fixed supplied source; surface focus and two integration steps only",
        baseline=baseline, history=history)
    if (keys == allowed[0] and
            (not math.isfinite(baseline.statistics.convergence_95_mrad) or
             abs(baseline.statistics.convergence_95_mrad/target_mrad-1) > .01)):
        evidence.update(reason="Objective-only calibration requires alpha95 within one percent; explicitly select (C3, Objective) otherwise",
                        evaluations=len(history), strengths=dict(zip(keys, map(float, initial))))
        return None, evidence

    def residual(values):
        measured = measure(values, fitting=True)
        stats = measured.statistics
        # Current/source preservation is checked again at final acceptance.
        # Keep it outside the optimizer's ability to trade it against focus.
        retained = unified_surface_acceptance(measured, baseline, target_mrad, **limits)
        if any("source" in reason.lower() for reason in retained):
            return np.full(len(keys), 1e6)
        focus = stats.waist_offset_m/1e-6
        result = [focus] if len(keys) == 1 else [math.log(stats.convergence_95_mrad/target_mrad), focus]
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite unified surface residual")
        return np.asarray(result)

    values = initial.copy()
    initial_pass = not unified_surface_acceptance(baseline, baseline, target_mrad, **limits)
    optimizer_success, message = initial_pass, "Initial coarse-step probe satisfies the declared gates"
    if not initial_pass:
        try:
            fit = least_squares(residual, initial, bounds=(np.zeros(len(keys)), upper),
                max_nfev=maximum_evaluations, diff_step=1e-6, x_scale="jac",
                ftol=1e-11, xtol=1e-11, gtol=1e-11)
            values = fit.x
            optimizer_success, message = bool(fit.success), str(fit.message)
        except EvaluationBudgetReached as exc:
            message = str(exc)
    reference = measure(values)
    fine_baseline = measure(initial, fine=True)
    refined = measure(values, fine=True)
    reasons = list(unified_surface_acceptance(reference, baseline, target_mrad, **limits))
    reasons += ["Fine step: "+reason for reason in unified_surface_acceptance(refined, fine_baseline, target_mrad, **limits)]
    reasons += ["Across steps: "+reason for reason in unified_surface_acceptance(refined, baseline, target_mrad, **limits)
                if "source" in reason.lower()]
    if reference.step_mm != .05 or refined.step_mm != .025:
        reasons.append("Independent 0.05/0.025 mm integration steps were not executed")
    step_difference = dict(
        radius_rms_nm=(refined.statistics.radius_rms_m-reference.statistics.radius_rms_m)*1e9,
        diameter95_nm=(refined.statistics.radius_95_m-reference.statistics.radius_95_m)*2e9,
        alpha95_relative=(refined.statistics.convergence_95_mrad-reference.statistics.convergence_95_mrad)/target_mrad,
        local_waist_nm=refined.local_waist_offset_nm-reference.local_waist_offset_nm,
        covariance_waist_nm=(refined.statistics.waist_offset_m-reference.statistics.waist_offset_m)*1e9)
    for key, difference in step_difference.items():
        if not math.isfinite(difference) or abs(difference) >= step_limits[key]:
            reasons.append("Integration-step difference exceeds the declared limit: "+key)
    if not optimizer_success:
        reasons.append("Optimizer did not complete within the declared budget")
    if capture_instrument_snapshot(state).digest != original.digest:
        raise ValueError("Live inputs changed during detached baseline refinement")
    for key, value in zip(keys, values):
        lenses[key].percent = float(value)
    passed = not reasons
    evidence.update(status="PASS_AT_FIXED_PARTICLE_BUDGET" if passed else "FAIL",
        reasons=tuple(dict.fromkeys(reasons)), strengths=dict(zip(keys, map(float, values))),
        evaluations=len(history), optimizer_success=optimizer_success, optimizer_message=message,
        reference=reference, refined=refined, fine_baseline=fine_baseline,
        step_difference=step_difference)
    return candidate if passed else None, evidence


def refine_surface_focus(state, target_mrad=30., *, initial=None, maximum_evaluations=50):
    """Return a detached, step-validated C3/objective candidate and its evidence.

No source, C1/C2, aperture, geometry or input state is changed. Acceptance here
is at the supplied particle budget, NOT a certification of sampling convergence.
"""
    from scipy.optimize import least_squares
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.simulation_modes import is_unified
    if is_unified(state):
        raise ValueError("Finite-Field Optics requires refine_unified_surface_focus with source and probe-size acceptance")
    keys=('condenser_lens_3','objective_lens')
    if not math.isfinite(target_mrad) or target_mrad<=0 or maximum_evaluations<1:
        raise ValueError('Focus target and evaluation budget must be positive')
    candidate=capture_instrument_snapshot(state).restore()
    lenses={lens.key:lens for lens in candidate.lenses}
    if any(key not in lenses or not lenses[key].enabled for key in keys):
        raise ValueError('Surface focus requires enabled C3 and objective lenses')
    upper=np.array([lenses[key].max_percent for key in keys],float)
    initial=np.array([lenses[key].percent for key in keys] if initial is None else initial,float)
    if initial.shape!=(2,) or not np.all(np.isfinite(initial)) or np.any(initial<0) or np.any(initial>upper):
        raise ValueError('Initial C3/objective settings must be inside their physical limits')
    measurements={}

    def set_controls(values):
        for key,value in zip(keys,values):
            lenses[key].percent=float(value)

    def measure(values):
        identity=tuple(float(v) for v in values)
        if identity not in measurements:
            set_controls(values)
            measurements[identity]=measure_surface_focus(candidate,step_mm=.025)
        return measurements[identity]

    def residual(values):
        s=measure(values).statistics
        return np.array([math.log(s.convergence_95_mrad/target_mrad),s.waist_offset_m/1e-6])

    fit=least_squares(residual,initial,bounds=(np.zeros(2),upper),
        max_nfev=int(maximum_evaluations),diff_step=1e-6,x_scale='jac',
        ftol=1e-11,xtol=1e-11,gtol=1e-11)
    refined=measure(fit.x)
    set_controls(fit.x)
    reference=measure_surface_focus(candidate,step_mm=.05)
    passed=bool(fit.success and reference.accepts(target_mrad) and refined.accepts(target_mrad))
    evidence=dict(status='PASS_AT_FIXED_PARTICLE_BUDGET' if passed else 'FAIL',
        sampling_convergence='NOT_TESTED',target_alpha95_mrad=float(target_mrad),
        probe_qualification='NOT_QUALIFIED: spot size, crossover topology and sampling require separate acceptance',
        particle_count=int(candidate.electron_gun.emitter.ray_count),
        strengths=dict(zip(keys,map(float,fit.x))),evaluations=len(measurements),
        optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
        reference=reference,refined=refined)
    return candidate if passed else None,evidence
