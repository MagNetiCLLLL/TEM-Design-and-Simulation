"""Read-only bounded design execution and rejection, not hardware calibration."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.bounded_optical_design import DesignParameter, optimize_captured_design
from scripts.example_quadratic_field_reduction import run_example


def test_executable_example_recovers_a_declared_optical_target_without_applying_it():
    result = run_example()
    study = result["design"]
    assert study["converged"]
    assert study["best"]["parameters"][0] == pytest.approx(68., abs=1e-5)
    assert study["best"]["squared_residual"] < 1e-16
    assert study["applied"] is False and study["aberration_feedback"] is False
    assert result["live_projector_percent_after_study"] == 50.
    assert len(study["evaluations"]) <= 24
    assert all(30 <= row["parameters"][0] <= 90 for row in study["evaluations"])
    assert np.isfinite(result["uniform_field_position_m"]).all()


def test_native_potential_design_recovers_drives_and_retains_raw_complex_phase(record_testsuite_property):
    import json
    from scripts.example_potential_phase_design import run_example as phase_design, native_stigmator_wave, compact_summary

    result = phase_design()
    study = result["design"]
    assert study["converged"] and not study["applied"] and not study["aberration_feedback"]
    np.testing.assert_allclose(study["best"]["parameters"], [.18, -.11], atol=1e-7, rtol=0.)
    assert study["best"]["squared_residual"] < 1e-14
    assert result["live_drive_percent_after_study"] == [.05, .03]
    assert result["target_snapshot_digest"] != result["initial_snapshot_digest"]
    assert result["initial_complex_l2_difference"] > 1e-8
    assert result["target_probability"] == pytest.approx(.37, abs=2e-12)
    assert all(len(row["residual"]) == 2*33*33 for row in study["evaluations"])

    # Check recovered-parameter consistency on another discretization. This
    # does not measure propagation error: equal drives share that error. The
    # separate refinement test below compares the actual propagated fields.
    state = default_state()
    state.simulation_mode = "ideal"
    component = state.condenser_stigmator
    component.strength_x_percent, component.strength_y_percent = .18, -.11
    target, _ = native_stigmator_wave(state, pixels=65, axial_steps=20)
    component.strength_x_percent, component.strength_y_percent = study["best"]["parameters"]
    actual, _ = native_stigmator_wave(state, pixels=65, axial_steps=20)
    error = float(np.linalg.norm(actual-target))
    assert error < 1e-12
    record_testsuite_property("native_field_phase_design", json.dumps({**compact_summary(result),
        "holdout_scope": "recovered-parameter consistency, not absolute propagation accuracy",
        "holdout_pixels": 65, "holdout_axial_steps": 20, "holdout_complex_l2_error": error}, sort_keys=True))


def test_native_phase_design_has_separate_grid_step_and_drive_derivative_refinement(record_testsuite_property):
    import json
    from scripts.example_potential_phase_design import native_stigmator_wave

    state = default_state()
    state.simulation_mode = "ideal"
    def wave(pixels, steps=20, drive=.18):
        state.condenser_stigmator.strength_x_percent = drive
        state.condenser_stigmator.strength_y_percent = -.11
        return native_stigmator_wave(state, pixels=pixels, axial_steps=steps)[0]

    fields = {n: wave(n) for n in (33, 65, 129)}
    # The common laboratory points coincide exactly. Amplitude includes square
    # root pixel area: half the pitch means a factor of two in two dimensions.
    # This unit conversion is not a fitted flux or phase normalization.
    spatial = [float(np.linalg.norm(fields[n]-2*fields[2*n-1][::2, ::2])) for n in (33, 65)]
    assert 0 < spatial[1] < spatial[0]/3
    assert spatial[1]/np.sqrt(.37) < 1e-4
    assert all(abs(float(np.sum(abs(field)**2))-.37) < 2e-12 for field in fields.values())

    axial_fields = {10: wave(65, 10), 20: fields[65], 40: wave(65, 40)}
    axial = [float(np.linalg.norm(axial_fields[k]-axial_fields[2*k])) for k in (10, 20)]
    assert max(axial) < 1e-10

    derivatives = {n: (wave(n, drive=.1801)-wave(n, drive=.1799))/.0002 for n in (33, 65, 129)}
    sensitivity = [float(np.linalg.norm(derivatives[n]-2*derivatives[2*n-1][::2, ::2])
                         /np.linalg.norm(derivatives[n])) for n in (33, 65)]
    assert 0 < sensitivity[1] < sensitivity[0]/3
    assert sensitivity[1] < 2e-4
    steps = (.0004, .0002, .0001)
    finite_difference = [(wave(65, drive=.18+h)-wave(65, drive=.18-h))/(2*h) for h in steps]
    derivative_errors = [float(np.linalg.norm(a-b)/np.linalg.norm(b))
                         for a, b in zip(finite_difference[:-1], finite_difference[1:])]
    # Roundoff eventually dominates the very weak drive response; do not
    # require fictitious monotone improvement of arbitrarily small steps.
    assert max(derivative_errors) < 1e-4
    record_testsuite_property("native_phase_design_independent_refinement", json.dumps({
        "spatial_pixels": [33, 65, 129], "spatial_complex_l2": spatial,
        "axial_steps": [10, 20, 40], "axial_complex_l2": axial,
        "spatial_drive_derivative_relative_l2": sensitivity,
        "finite_difference_steps_percent": steps,
        "finite_difference_derivative_relative_l2": derivative_errors,
        "phase_alignment": False, "intensity_renormalization": False,
        "scope": "synthetic native-component design; discretization error separate from parameter recovery"
    }, sort_keys=True))


def test_evaluation_budget_counts_all_actual_diagnostics_and_preserves_live_state():
    state = default_state()
    state.projector_lens_p1.percent = 50.
    before = deepcopy(state.to_dict())
    calls = []
    def diagnostic(trial):
        calls.append(trial.projector_lens_p1.percent)
        return np.array([trial.projector_lens_p1.percent-68.])
    result = optimize_captured_design(state,
        [DesignParameter(("projector_lens_p1", "percent"), 30., 90.)], diagnostic, maximum_evaluations=1)
    assert len(calls) == len(result["evaluations"]) == 1
    assert result["converged"] is False and result["applied"] is False
    assert state.to_dict() == before


def test_unrepresentable_design_step_cannot_report_false_convergence():
    state = default_state()
    state.projector_lens_p1.percent = 50.
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="not representable"):
        optimize_captured_design(state,
            [DesignParameter(("projector_lens_p1", "percent"), 30., 90., 1e-300)],
            lambda trial: pytest.fail("Reject an ineffective step before running an expensive diagnostic"))
    assert state.to_dict() == before


def test_design_derivative_uses_the_actual_representable_parameter_increment(monkeypatch):
    from types import SimpleNamespace
    import temsim.physics.bounded_optical_design as design
    state = default_state()
    state.projector_lens_p1.percent = 50.
    def inspect_solver(function, initial, *, jac, **kwargs):
        np.testing.assert_array_equal(jac(np.asarray(initial)), [[1.]])
        return SimpleNamespace(success=True, message="Independent linear derivative check")
    monkeypatch.setattr(design, "least_squares", inspect_solver)
    result = optimize_captured_design(state,
        [DesignParameter(("projector_lens_p1", "percent"), 30., 90., 1.5*np.spacing(50.))],
        lambda trial: np.array([trial.projector_lens_p1.percent-49.]))
    assert len(result["evaluations"]) == 2
    assert state.projector_lens_p1.percent == 50.


def test_mutating_diagnostic_is_rejected_and_cannot_change_live_instrument():
    state = default_state()
    before = deepcopy(state.to_dict())
    parameter = DesignParameter(("projector_lens_p1", "percent"), 0., 100.)
    def diagnostic(trial):
        trial.sample.thickness_nm += 1.
        return np.array([0.])
    with pytest.raises(ValueError, match="mutated"):
        optimize_captured_design(state, [parameter], diagnostic)
    assert state.to_dict() == before


@pytest.mark.parametrize("diagnostic", [lambda _: np.array([np.nan]), lambda _: np.array([]), lambda _: np.array([1j])])
def test_nonfinite_empty_and_complex_residuals_reject(diagnostic):
    state = default_state()
    with pytest.raises(ValueError, match="finite real residual"):
        optimize_captured_design(state, [DesignParameter(("projector_lens_p1", "percent"), 0., 100.)], diagnostic)


def test_expression_paths_and_out_of_range_initial_values_are_not_evaluated():
    with pytest.raises(ValueError, match="attribute paths"):
        DesignParameter(("__dict__",), 0., 1.)
    with pytest.raises(ValueError, match="attribute paths"):
        DesignParameter(("lenses[0]", "percent"), 0., 100.)
    state = default_state()
    state.projector_lens_p1.percent = 50.
    with pytest.raises(ValueError, match="outside"):
        optimize_captured_design(state, [DesignParameter(("projector_lens_p1", "percent"), 60., 90.)],
            lambda _: pytest.fail("No diagnostic for an invalid initial design"))
    with pytest.raises(InterruptedError, match="cancelled"):
        optimize_captured_design(state, [DesignParameter(("projector_lens_p1", "percent"), 0., 100.)],
            lambda _: pytest.fail("Cancelled before execution"), cancelled=lambda: True)
