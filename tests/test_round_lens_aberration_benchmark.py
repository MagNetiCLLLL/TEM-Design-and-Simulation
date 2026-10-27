"""Separate field-order, Lorentz-integration and aberration-fit evidence."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, integrate_adaptive_reference,
    momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s,
)
from temsim.physics.round_lens_aberration_benchmark import (
    measure_round_lens_aberrations, paraxial_focus_reference,
    third_order_reference, trace_full_lorentz,
)


def lens(order=5):
    return HighOrderRoundLensPotential(FrozenAnalyticField("benchmark-round", ((.3, 0., .006),),
        CoordinateRegistration(), (-80., 200.), .006), order, .004)


OPTIONS = dict(energy_ev=300000., z_start_m=-.06, focus_search_end_m=.12, maximum_step_m=.001)


@pytest.fixture(scope="module")
def fitted():
    return measure_round_lens_aberrations(lens(), **OPTIONS, semiangle_rad=.01)


def test_full_forward_lorentz_matches_independent_time_domain_solver():
    field = lens(7)
    initial = np.array([[.0004, .0001, .002, -.001]])
    start, stop = -.04, .05
    phase = RelativisticPhaseSpace(np.r_[initial[0, :2], start],
        momentum_from_kinetic_energy_ev(300000., np.r_[initial[0, 2:], 1.]))

    def at_plane(position, _momentum, _time):
        return position[2]-stop
    at_plane.terminal, at_plane.direction = True, 1
    speed = np.linalg.norm(velocity_from_momentum_m_per_s(phase.momentum_kg_m_per_s))
    reference = integrate_adaptive_reference(phase, 1.5*(stop-start)/speed, field,
        stop_event=at_plane, relative_tolerance=2e-12, position_tolerance_m=1e-14,
        momentum_tolerance_kg_m_per_s=1e-34, maximum_step_s=.0005/speed)
    assert reference.terminated_by_event
    final = reference.final_state
    expected = np.r_[final.position_m[:2], final.momentum_kg_m_per_s[:2]/final.momentum_kg_m_per_s[2]]
    result = trace_full_lorentz(field, initial, energy_ev=300000., z_start_m=start,
                              z_end_m=stop, maximum_step_m=.0005)["final_xy_slopes"][0]
    np.testing.assert_allclose(result[:2], expected[:2], rtol=2e-9, atol=2e-13)
    np.testing.assert_allclose(result[2:], expected[2:], rtol=2e-9, atol=2e-12)
    # A static magnetic field does no work in the independent time-domain run.
    momenta = np.linalg.norm(reference.momentum_kg_m_per_s, axis=-1)
    assert np.max(abs(momenta/momenta[0]-1)) < 2e-11


def test_fixed_step_lorentz_integration_converges_independently_of_field_order():
    field = lens(7)
    initial = np.array([[.0008, .0001, .002, -.001], [-.0004, .0002, 0., 0.]])
    options = dict(energy_ev=300000., z_start_m=-.06, z_end_m=.04)
    reference = trace_full_lorentz(field, initial, **options, maximum_step_m=.00025)["final_xy_slopes"]
    errors = []
    for step in (.002, .001, .0005):
        actual = trace_full_lorentz(field, initial, **options, maximum_step_m=step, method="RK4")["final_xy_slopes"]
        errors.append((np.linalg.norm(actual[:, :2]-reference[:, :2]),
                       np.linalg.norm(actual[:, 2:]-reference[:, 2:])))
    errors = np.array(errors)
    assert np.all(errors[:-1] > 12*errors[1:])
    assert errors[-1, 0] < 1e-9 and errors[-1, 1] < 2e-8


def test_field_order_converges_with_fixed_high_accuracy_dynamics():
    initial = np.array([[.0008, .0001, .002, -.001], [-.0004, .0002, 0., 0.]])
    options = dict(energy_ev=300000., z_start_m=-.06, z_end_m=.04, maximum_step_m=.0005)
    reference = trace_full_lorentz(lens(9), initial, **options)["final_xy_slopes"]
    errors = []
    for order in (1, 3, 5, 7):
        actual = trace_full_lorentz(lens(order), initial, **options)["final_xy_slopes"]
        errors.append((np.linalg.norm(actual[:, :2]-reference[:, :2]),
                       np.linalg.norm(actual[:, 2:]-reference[:, 2:])))
    errors = np.array(errors)
    assert np.all(errors[:-1] > 50*errors[1:])
    assert errors[-1, 0] < 1e-12 and errors[-1, 1] < 2e-12


def test_paraxial_focus_is_independent_of_high_order_radial_terms():
    references = [paraxial_focus_reference(lens(order), **OPTIONS) for order in (1, 3, 5, 7)]
    assert all(row["image_z_m"] == references[0]["image_z_m"] for row in references)
    focus = references[0]
    position = np.array([[1e-6, .3e-6]])
    slopes = position@focus["initial_slopes_per_position_m1"].T
    result = trace_full_lorentz(lens(7), np.column_stack((position, slopes)), energy_ev=300000.,
        z_start_m=-.06, z_end_m=focus["image_z_m"], maximum_step_m=.0005)["final_xy_slopes"]
    assert np.linalg.norm(result[0, :2]) < 2e-14
    expected_slopes = position@focus["image_slopes_per_initial_position_m1"].T
    np.testing.assert_allclose(result[:, 2:], expected_slopes, rtol=2e-7, atol=1e-13)


def test_natural_cs_cc_match_independent_order_by_order_ray_equations(fitted):
    independent = third_order_reference(lens(), **OPTIONS)
    assert fitted["cs_mm"] > 1 and fitted["cc_mm"] > 1
    assert fitted["cs_mm"] == pytest.approx(independent["cs_mm"], rel=5e-7)
    assert fitted["cc_mm"] == pytest.approx(independent["cc_mm"], rel=5e-8)
    assert np.linalg.norm(independent["linear_focus_residual_per_initial_height"]) < 2e-12
    assert fitted["holdout_rms_m"] < 3e-12 and fitted["holdout_max_error_m"] < 1e-11
    assert fitted["training_rays"] == 192 and fitted["holdout_rays"] == 161
    assert fitted["applied_manual_aberrations"] is False


def test_third_order_requires_radial_vacuum_terms_not_only_exact_kinematics():
    first = third_order_reference(lens(1), **OPTIONS)
    third = third_order_reference(lens(3), **OPTIONS)
    higher = third_order_reference(lens(7), **OPTIONS)
    assert abs(first["cs_mm"]-third["cs_mm"]) > .1*abs(third["cs_mm"])
    assert third["cs_mm"] == higher["cs_mm"]
    assert first["cc_mm"] == pytest.approx(third["cc_mm"], rel=0., abs=5e-11)
    assert third["cc_mm"] == pytest.approx(higher["cc_mm"], rel=0., abs=5e-11)


def test_energy_scan_freezes_field_input_rays_and_reference_plane(fitted):
    scan = fitted["energy_scan"]
    assert scan["fixed_field"] and scan["fixed_initial_rays"] and scan["fixed_image_plane"]
    assert len(fitted["integration"]) == 7
    assert len({row["energy_ev"] for row in fitted["integration"]}) == 7
    assert {row["field_digest"] for row in fitted["integration"]} == {fitted["field_digest"]}
    assert all(row["reference_interval_m"] == fitted["reference_interval_m"] for row in fitted["integration"])
    np.testing.assert_array_equal(fitted["chief_intercept_m"], [0., 0.])
    np.testing.assert_array_equal(scan["wave_c1_energy_derivative_mm"], -np.array(scan["cc_mm"]))
    changes = [row["absolute_change_mm"] for row in scan["comparisons"]]
    assert changes[0] > 3.5*changes[1] and changes[1] < 1e-5


def test_pupil_refinement_retains_natural_coefficients_and_decreases_holdout_error(fitted):
    coarse = measure_round_lens_aberrations(lens(), **OPTIONS, semiangle_rad=.02)
    assert coarse["cs_mm"] == pytest.approx(fitted["cs_mm"], rel=5e-6)
    assert coarse["cc_mm"] == pytest.approx(fitted["cc_mm"], rel=1e-6)
    assert coarse["holdout_rms_m"] > 8*fitted["holdout_rms_m"]


def test_field_controls_are_not_recalibrated_or_mutated():
    field = lens()
    before = field.to_dict()
    reference = third_order_reference(field, **OPTIONS)
    changed = replace(field, native=replace(field.native, terms_t_m=((.25, 0., .006),)))
    different = third_order_reference(changed, **OPTIONS)
    assert field.to_dict() == before
    assert different["image_z_m"] != reference["image_z_m"]
    assert different["cs_mm"] != reference["cs_mm"]
    assert different["cc_mm"] != reference["cc_mm"]


def test_benchmark_rejects_unqualified_pose_bad_budget_and_missing_focus():
    shifted = replace(lens().native, registration=CoordinateRegistration((.001, 0., 0.)))
    with pytest.raises(ValueError, match="centred axis-aligned"):
        third_order_reference(HighOrderRoundLensPotential(shifted), **OPTIONS)
    with pytest.raises(ValueError, match="No first paraxial focus"):
        paraxial_focus_reference(lens(), **{**OPTIONS, "focus_search_end_m": -.02})
    with pytest.raises(ValueError, match="positive"):
        third_order_reference(lens(), **{**OPTIONS, "maximum_step_m": 0.})
    with pytest.raises(ValueError, match="decreasing"):
        measure_round_lens_aberrations(lens(), **OPTIONS, energy_relative_steps=(.001, .002, .0005))
