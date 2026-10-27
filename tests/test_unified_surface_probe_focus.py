"""Bounded surface-route contracts; no actual gun, full column or wave solve."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.surface_probe_focus import UnifiedSurfaceFocusEvaluator, measure_surface_focus
from temsim.physics.beam_statistics import transverse_beam_statistics


def fixture(monkeypatch):
    from temsim.optics.electron_gun import source
    from temsim.optics import surface_probe_focus as focus
    from temsim.physics import particle_sections, simulation, unified_scan_calibration
    state = default_state()
    state.simulation_mode = "unified"
    state.step_mm = .1
    n = 33
    angle = np.arange(n) * 2 * np.pi / 32
    x, y = 1e-10 * np.cos(angle), 1e-10 * np.sin(angle)
    tx, ty = -.03 * np.sin(angle), .03 * np.cos(angle)
    weights = np.r_[np.full(32, 1 / 32), 0.]
    alive = np.ones(n, bool)
    alive[0] = False
    times = np.linspace(2e-9, 3e-9, n)
    times[0] = np.nan
    emitted = SimpleNamespace(x_m=x, y_m=y, tx_rad=tx, ty_rad=ty, alive=alive,
        flight_time_s=times, energy_offset_ev=np.linspace(-.15, .15, n), weight=weights, ray_id=np.arange(n))
    gun = SimpleNamespace(exit_bundle=emitted, blocked_z_mm=np.r_[10., np.full(n - 1, np.nan)],
        blocked_key=["gun_stop"] + [""] * (n - 1),
        emission_reference={"ray_id":np.arange(n), "position_m":np.c_[x, y, np.zeros(n)]})
    calls = {"gun": [], "segments": []}
    monkeypatch.setattr(source, "trace_source_to_exit", lambda candidate: calls["gun"].append(candidate) or gun)
    monkeypatch.setattr(particle_sections, "gun_dependency_signature", lambda candidate: "exact-gun-dependencies")
    monkeypatch.setattr(simulation, "_validate_gun_flight_times", lambda trace: trace is gun)
    monkeypatch.setattr(particle_sections, "_events", lambda candidate: ())
    def forbidden(*args, **kwargs):
        raise AssertionError("Surface observation must not recalibrate scans or reapply legacy mechanics")
    monkeypatch.setattr(focus, "clip_segment", forbidden)
    monkeypatch.setattr(focus, "clip_column_wall", forbidden)
    monkeypatch.setattr(focus, "propagate", forbidden)
    monkeypatch.setattr(unified_scan_calibration, "calibrate_unified_scan", forbidden)
    def segment(candidate, name, start, end, initial, offsets, weight, identity, survival,
                events, previous, boundary, **kwargs):
        assert candidate is not state
        assert candidate.step_mm <= .05
        assert end == pytest.approx(state.sample.upper_surface_z_mm, abs=1e-12)
        np.testing.assert_array_equal(initial[4], times)
        np.testing.assert_array_equal(kwargs["initial_kinetic_energy_ev"],
            state.electron_gun.nominal_exit_energy_ev + emitted.energy_offset_ev)
        np.testing.assert_array_equal(weight, weights)
        np.testing.assert_array_equal(survival[0], alive)
        assert survival[2][0] == "gun_stop"
        planes = np.asarray(kwargs["save_z_mm"])
        dz = (planes - end)[:, None] * 1e-3
        cp = SimpleNamespace(z_mm=planes, x_m=x + dz * tx, y_m=y + dz * ty,
                             tx_rad=np.broadcast_to(tx, (3, n)), ty_rad=np.broadcast_to(ty, (3, n)))
        final_alive = alive.copy()
        final_alive[1] = False  # Executed unified mechanical interception.
        result = SimpleNamespace(checkpoints=cp, branch=SimpleNamespace(alive=final_alive))
        earlier = [call for call in calls["segments"] if call["step"] == candidate.step_mm]
        assert previous is (earlier[-1]["result"] if earlier else None)
        calls["segments"].append(dict(result=result, previous=previous, step=candidate.step_mm))
        return result, start, previous is not None
    monkeypatch.setattr(particle_sections, "_trace_segment", segment)
    return state, gun, calls


def test_unified_surface_passes_executed_clock_energy_and_stop_receipts(monkeypatch):
    state, gun, calls = fixture(monkeypatch)
    before = capture_instrument_snapshot(state).digest
    result = measure_surface_focus(state)
    emitted = gun.exit_bundle
    alive = emitted.alive.copy()
    alive[1] = False
    alive &= emitted.weight > 0
    expected = transverse_beam_statistics(emitted.x_m, emitted.y_m, emitted.tx_rad, emitted.ty_rad,
                                          alive=alive, weights=emitted.weight)
    assert result.statistics == expected
    assert result.surviving_source_ids == tuple(range(2, 32))
    assert result.transmitted_source_weight == pytest.approx(30 / 32)
    assert result.step_mm == .05 and result.source_population_identity
    assert len(calls["gun"]) == len(calls["segments"]) == 1
    assert capture_instrument_snapshot(state).digest == before


def test_detached_fit_reuses_exact_gun_and_requests_proved_column_prefix(monkeypatch):
    state, _gun, calls = fixture(monkeypatch)
    evaluator = UnifiedSurfaceFocusEvaluator(state, controls=("objective_lens",))
    first = evaluator.measure(state)
    candidate = capture_instrument_snapshot(state).restore()
    candidate.objective_lens.percent += .0001
    second = evaluator.measure(candidate)
    assert len(calls["gun"]) == 1 and len(calls["segments"]) == 2
    assert not first.column_prefix_reused and second.column_prefix_reused
    assert first.source_population_identity == second.source_population_identity


def test_step_refinement_reuses_the_exact_gun_but_never_the_coarser_mesh(monkeypatch):
    state, _gun, calls = fixture(monkeypatch)
    coarse = UnifiedSurfaceFocusEvaluator(state, controls=("objective_lens",))
    fine = coarse.at_step(.025)
    reference, refined = coarse.measure(state), fine.measure(state)
    assert len(calls["gun"]) == 1
    assert all(call["previous"] is None for call in calls["segments"])
    assert reference.source_population_identity == refined.source_population_identity
    assert (reference.step_mm, refined.step_mm) == (.05, .025)


@pytest.mark.parametrize("change", ["source", "aperture", "mode", "unlisted_lens"])
def test_unlisted_physical_changes_reject_before_source_execution(monkeypatch, change):
    state, _gun, calls = fixture(monkeypatch)
    evaluator = UnifiedSurfaceFocusEvaluator(state, controls=("objective_lens",))
    candidate = capture_instrument_snapshot(state).restore()
    if change == "source":
        candidate.electron_gun.emitter.energy_spread_fwhm_ev *= 2
    elif change == "aperture":
        candidate.condenser_aperture_2.diameter_mm *= .9
    elif change == "mode":
        candidate.simulation_mode = "ideal"
    else:
        next(lens for lens in candidate.lenses if lens.key == "condenser_lens_3").percent += .1
    with pytest.raises(ValueError, match="outside the declared"):
        evaluator.measure(candidate)
    assert not calls["gun"] and not calls["segments"]


def test_missing_executed_source_clock_cannot_be_replaced_with_zero(monkeypatch):
    state, _gun, calls = fixture(monkeypatch)
    from temsim.physics import simulation
    monkeypatch.setattr(simulation, "_validate_gun_flight_times", lambda trace: False)
    with pytest.raises(ValueError, match="executed gun flight-time"):
        measure_surface_focus(state)
    assert len(calls["gun"]) == 1 and not calls["segments"]


@pytest.mark.parametrize("step", [.05, .025])
def test_each_requested_refinement_step_is_observed_without_mutating_state(monkeypatch, step):
    state, _gun, calls = fixture(monkeypatch)
    before = state.step_mm
    result = measure_surface_focus(state, step_mm=step)
    assert result.step_mm == step and calls["segments"][0]["step"] == step
    assert state.step_mm == before


def qualified_measurement(**changes):
    from temsim.optics.surface_probe_focus import SurfaceFocusMeasurement
    angles = np.arange(32)*2*np.pi/32
    stats = transverse_beam_statistics(1e-10*np.cos(angles), 1e-10*np.sin(angles),
        -.03*np.sin(angles), .03*np.cos(angles), weights=np.ones(32)/32)
    stats = replace(stats, convergence_95_rad=.03)
    result = SurfaceFocusMeasurement(1599.2, stats, 0., .001, (1e-10,)*3, .05,
        32., "executed-source", tuple(range(32)), 1.)
    return replace(result, **changes)


@pytest.mark.parametrize("defect", ["rms", "diameter", "pruning", "weight", "source", "unidentified"])
def test_qualification_rejects_narrow_waist_from_wrong_source_or_broad_aberrations(defect):
    from temsim.optics.surface_probe_focus import unified_surface_acceptance
    baseline = qualified_measurement()
    changed = baseline
    if defect == "rms":
        changed = replace(changed, statistics=replace(changed.statistics, radius_rms_m=83e-9))
    elif defect == "diameter":
        changed = replace(changed, statistics=replace(changed.statistics, radius_95_m=.5e-9))
    elif defect == "pruning":
        changed = replace(changed, surviving_source_ids=tuple(range(1, 32)))
    elif defect == "weight":
        changed = replace(changed, transmitted_source_weight=.99)
    elif defect == "source":
        changed = replace(changed, source_population_identity="different-source")
    else:
        changed = replace(changed, surviving_source_ids=None)
    assert not unified_surface_acceptance(baseline, baseline)
    assert unified_surface_acceptance(changed, baseline)


def mocked_refinement(monkeypatch, *, wrong_angle=False, fine_defect=False, broad=False):
    from temsim.optics import surface_probe_focus as module
    state = default_state()
    state.simulation_mode = "unified"
    state.step_mm = .05
    original = {lens.key: lens.percent for lens in state.lenses}
    calls = []
    class Evaluator:
        def __init__(self, captured, *, controls, step_mm):
            self.controls, self.step_mm = controls, step_mm
        def at_step(self, step_mm):
            return type(self)(state, controls=self.controls, step_mm=step_mm)
        def measure(self, candidate):
            values = {lens.key: lens.percent for lens in candidate.lenses}
            offset = values["objective_lens"] - original["objective_lens"] - .01
            alpha = .03 if not wrong_angle else .02 + .1*(values["condenser_lens_3"]-original["condenser_lens_3"])
            result = qualified_measurement(step_mm=self.step_mm)
            stats = replace(result.statistics, waist_offset_m=offset*1e-6, convergence_95_rad=alpha,
                radius_rms_m=83e-9 if broad else .3e-9 if fine_defect and self.step_mm == .025 else 1e-10)
            calls.append((self.step_mm, values))
            return replace(result, statistics=stats, local_waist_offset_nm=offset*1000)
    monkeypatch.setattr(module, "UnifiedSurfaceFocusEvaluator", Evaluator)
    return module, state, original, calls


def test_explicit_objective_refinement_is_detached_step_checked_and_never_changes_other_controls(monkeypatch):
    module, state, original, calls = mocked_refinement(monkeypatch)
    before = capture_instrument_snapshot(state).digest
    progress = []
    candidate, evidence = module.refine_unified_surface_focus(state, callback=progress.append)
    assert candidate is not None and candidate is not state
    assert evidence["status"] == "PASS_AT_FIXED_PARTICLE_BUDGET" and evidence["applied"] is False
    assert evidence["controls"] == ("objective_lens",)
    assert candidate.objective_lens.percent == pytest.approx(original["objective_lens"]+.01)
    assert {step for step, _ in calls} == {.05, .025}
    assert len(progress) == len(calls) == evidence["evaluations"] <= 30
    assert evidence["sampling_convergence"] == "NOT_TESTED"
    candidate.objective_lens.percent = original["objective_lens"]
    assert capture_instrument_snapshot(candidate).digest == before
    assert capture_instrument_snapshot(state).digest == before


def test_wrong_initial_angle_requires_explicit_second_control(monkeypatch):
    module, state, _original, calls = mocked_refinement(monkeypatch, wrong_angle=True)
    candidate, evidence = module.refine_unified_surface_focus(state)
    assert candidate is None and evidence["status"] == "FAIL"
    assert "explicitly select" in evidence["reason"] and len(calls) == 1
    candidate, evidence = module.refine_unified_surface_focus(state,
        controls=("condenser_lens_3", "objective_lens"))
    assert candidate is not None and evidence["status"] == "PASS_AT_FIXED_PARTICLE_BUDGET"


@pytest.mark.parametrize("defect", ["fine", "broad", "budget"])
def test_solver_failure_never_supplies_candidate_or_widens_declared_limits(monkeypatch, defect):
    module, state, _original, calls = mocked_refinement(monkeypatch,
        fine_defect=defect == "fine", broad=defect == "broad")
    before = capture_instrument_snapshot(state).digest
    limit = 3 if defect == "budget" else 30
    candidate, evidence = module.refine_unified_surface_focus(state, maximum_evaluations=limit)
    assert candidate is None and evidence["status"] == "FAIL"
    assert evidence["thresholds"] == {"maximum_rms_radius_nm":.2, "maximum_diameter95_nm":.7}
    assert evidence["reasons"] and len(calls) <= limit
    assert capture_instrument_snapshot(state).digest == before


def test_focus_controls_cannot_silently_include_hexapoles(monkeypatch):
    module, state, _original, calls = mocked_refinement(monkeypatch)
    with pytest.raises(ValueError, match="Explicit focus controls"):
        module.refine_unified_surface_focus(state, controls=("objective_lens", "hp2"))
    assert not calls


def test_refinement_cancelled_before_first_source_measurement(monkeypatch):
    module, state, _original, calls = mocked_refinement(monkeypatch)
    state._tuning_cancelled = lambda: True
    with pytest.raises(RuntimeError, match="Superseded"):
        module.refine_unified_surface_focus(state)
    assert not calls


def test_legacy_focus_only_acceptance_cannot_qualify_unified_probe(monkeypatch):
    module, state, _original, calls = mocked_refinement(monkeypatch)
    with pytest.raises(ValueError, match="refine_unified_surface_focus"):
        module.refine_surface_focus(state)
    assert not calls


@pytest.mark.parametrize("quantity", ["radius_rms_nm", "diameter95_nm", "alpha95_relative",
                                      "local_waist_nm", "covariance_waist_nm"])
def test_two_individually_acceptable_steps_still_require_numerical_consistency(monkeypatch, quantity):
    module, state, _original, _calls = mocked_refinement(monkeypatch)
    previous = module.UnifiedSurfaceFocusEvaluator.measure
    def shifted(self, candidate):
        result = previous(self, candidate)
        if self.step_mm != .025:
            return result
        stats = result.statistics
        if quantity == "radius_rms_nm":
            stats = replace(stats, radius_rms_m=stats.radius_rms_m+.006e-9)
        elif quantity == "diameter95_nm":
            stats = replace(stats, radius_95_m=stats.radius_95_m+.015e-9)
        elif quantity == "alpha95_relative":
            stats = replace(stats, convergence_95_rad=stats.convergence_95_rad+.00006)
        elif quantity == "covariance_waist_nm":
            stats = replace(stats, waist_offset_m=stats.waist_offset_m+.15e-9)
        else:
            return replace(result, local_waist_offset_nm=result.local_waist_offset_nm+.15)
        return replace(result, statistics=stats)
    monkeypatch.setattr(module.UnifiedSurfaceFocusEvaluator, "measure", shifted)
    candidate, evidence = module.refine_unified_surface_focus(state)
    assert candidate is None and evidence["status"] == "FAIL"
    assert evidence["reference"].accepts(30.) and evidence["refined"].accepts(30.)
    assert any("Integration-step difference" in reason and quantity in reason for reason in evidence["reasons"])


def test_final_uncached_fit_and_both_fine_measurements_fit_reserved_budget(monkeypatch):
    import scipy.optimize
    module, state, _original, calls = mocked_refinement(monkeypatch)
    def fit(fun, initial, **kwargs):
        fun(initial)
        fun(initial+1e-6)
        return SimpleNamespace(x=initial+.01, success=True, message="Controlled uncached final candidate")
    monkeypatch.setattr(scipy.optimize, "least_squares", fit)
    candidate, evidence = module.refine_unified_surface_focus(state, maximum_evaluations=5)
    assert candidate is not None and evidence["status"] == "PASS_AT_FIXED_PARTICLE_BUDGET"
    assert len(calls) == evidence["evaluations"] == 5
