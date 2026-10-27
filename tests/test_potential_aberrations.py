"""Canonical chart and fixed-family diagnostic tests with independent laws."""
import json
import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e

from temsim.immutable_json import thaw_json
from temsim.physics.potential_aberrations import (
    canonical_coordinates, capture_canonical_pupil, measure_canonical_aberrations,
    mechanical_coordinates, momentum, fit_canonical_intercepts,
)
from temsim.physics.potential_snapshot import StaticPolynomialGauge
from test_potential_ray import uniform_snapshot


def test_immersed_canonical_chart_roundtrip_includes_same_vector_potential():
    snapshot = uniform_snapshot(bz=.2, gauge=StaticPolynomialGauge(gradient_t_m=(.0001, -.0002, 0.)))
    mechanical = np.array([[.0004, -.0001, .03, -.02], [-.0002, .0003, -.04, .01]])
    energy, reference_p = np.array([200000., 300000.]), float(momentum(300000.))
    canonical = canonical_coordinates(snapshot, mechanical, 0., energy, reference_p)
    potential = snapshot.sample(np.c_[mechanical[:, :2], np.zeros(2)])["A_t_m"][:, :2]
    expected = (momentum(energy)[:, None]*mechanical[:, 2:]
        /np.sqrt(1+np.sum(mechanical[:, 2:]**2, axis=1))[:, None]-e*potential)/reference_p
    np.testing.assert_allclose(canonical[:, 2:], expected, atol=1e-16)
    np.testing.assert_allclose(mechanical_coordinates(snapshot, canonical, 0., energy, reference_p),
                               mechanical, rtol=1e-14, atol=1e-17)


def test_vacuum_lagrangian_focus_matches_exact_geometrical_cs_and_canonical_chromatic_response():
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    assert reference["scaled_symplectic_defect"] < 1e-12
    assert reference["relative_transfer_change"] < 1e-10
    assert reference["reference_was_symplectified"] is False
    report = measure_canonical_aberrations(snapshot, reference)
    # q=L*k*(1/sqrt(1-k^2)-1) for the captured paraxial focusing sheet.
    assert report["cs_response_mm"] == pytest.approx(-5., rel=1e-7)
    coefficient = report["coefficients_mm"][0][report["coefficient_names"].index("C5")]
    assert coefficient == pytest.approx(-3.75, rel=.001)
    kinetic = 300000.*e
    logarithmic_momentum_derivative = (kinetic+m_e*c*c)/(kinetic+2*m_e*c*c)
    expected_cc = -10.*logarithmic_momentum_derivative
    assert report["cc_response_mm"][-1] == pytest.approx(expected_cc, rel=2e-7)
    assert report["finite_basis_gradient_qualification"]["status"] == "PASS"
    assert report["chromatic_qualification"]["status"] == "PASS"
    assert report["chief_tilt_defocus_removed"] is False
    assert report["coherent_wave_admission"] is False
    assert max(report["gradient_holdout_rms_m"]) < 1e-16


def test_fixed_mechanical_energy_scan_is_separate_and_not_scalar_family_qualified():
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    canonical = measure_canonical_aberrations(snapshot, reference)
    mechanical = measure_canonical_aberrations(snapshot, reference, input_convention="fixed_mechanical")
    assert mechanical["finite_basis_gradient_qualification"]["status"] == "REJECTED"
    assert mechanical["scalar_family_eligible"] is False
    np.testing.assert_array_equal(mechanical["raw_input_mechanical"][0], mechanical["raw_input_mechanical"][1])
    assert not np.array_equal(canonical["raw_input_mechanical"][0], canonical["raw_input_mechanical"][1])
    np.testing.assert_allclose(mechanical["raw_output_mechanical"][0], mechanical["raw_output_mechanical"][1], atol=1e-18)


def test_reference_is_immutable_tamper_detected_and_gauge_changes_require_a_new_reference():
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    with pytest.raises(TypeError):
        reference["energy_ev"] = 200000.
    changed = thaw_json(reference)
    changed["input_canonical"][1][0] += .001
    with pytest.raises(ValueError, match="identity"):
        measure_canonical_aberrations(snapshot, changed)
    shifted = uniform_snapshot(gauge=StaticPolynomialGauge(gradient_t_m=(1e-5, 0., 0.)))
    with pytest.raises(ValueError, match="gauge"):
        measure_canonical_aberrations(shifted, reference)


@pytest.mark.parametrize("change", [{"energy_ev": True}, {"semiangle_rad": True},
                                   {"transfer_position_step_m": -1.}, {"transfer_momentum_step": np.nan}])
def test_reference_rejects_invalid_physical_controls(change):
    options = dict(energy_ev=300000., maximum_step_m=.001)
    options.update(change)
    with pytest.raises(ValueError):
        capture_canonical_pupil(uniform_snapshot(), (0., .01), **options)


def test_forward_hemisphere_and_electric_scope_are_not_silently_extended():
    with pytest.raises(ValueError, match="hemisphere"):
        mechanical_coordinates(uniform_snapshot(), [[0., 0., 1., 0.]], 0., 300000., momentum(300000.))
    with pytest.raises(ValueError, match="magnetic potentials only"):
        capture_canonical_pupil(uniform_snapshot(potential_gradient=100.), (0., .01),
                               energy_ev=300000., maximum_step_m=.001)


def test_nominal_only_retention_and_intercept_fit_preserve_the_same_raw_output():
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    report = measure_canonical_aberrations(snapshot, reference, energy_relative_steps=(), retain_nominal_trajectory=True)
    assert report["cc_response_mm"] == []
    assert report["chromatic_qualification"]["status"] == "NOT_RUN"
    assert report["nominal_trajectory_xy_slopes"].shape == (2, 354, 4)
    with pytest.raises(ValueError):
        report["nominal_trajectory_xy_slopes"].setflags(write=True)
    fitted = fit_canonical_intercepts(snapshot, reference, report["raw_output_mechanical"][0])
    np.testing.assert_array_equal(fitted["coefficients_mm"], report["coefficients_mm"][0])
    assert fitted["scalar_wavefront_qualification"] == "not granted by an intercept-only fit"


@pytest.mark.parametrize("steps", [(True, False), (".002", ".001", ".0005")])
def test_chromatic_steps_reject_coercible_boolean_or_string_values(steps):
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    with pytest.raises(ValueError, match="real numerical"):
        measure_canonical_aberrations(snapshot, reference, energy_relative_steps=steps)


def test_fewer_than_three_energy_steps_never_qualify_cc():
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    report = measure_canonical_aberrations(snapshot, reference, energy_relative_steps=(.001,))
    assert report["chromatic_qualification"]["status"] == "INSUFFICIENT_LEVELS"


def test_chromatic_derivative_convergence_must_meet_declared_image_budget():
    snapshot = uniform_snapshot()
    reference = capture_canonical_pupil(snapshot, (0., .01), energy_ev=300000., maximum_step_m=.001)
    report = measure_canonical_aberrations(snapshot, reference, energy_relative_steps=(.01, .005, .0025),
                                           gradient_error_budget_m=1e-13)
    qualification = report["chromatic_qualification"]
    expected = np.abs(np.diff(report["cc_response_mm"]))*1e-3*.01*.01
    np.testing.assert_allclose(qualification["successive_cc_image_error_m"], expected)
    assert np.max(np.abs(np.diff(report["cc_response_mm"]))) < qualification["absolute_change_budget_mm"]
    assert report["finite_basis_gradient_qualification"]["status"] == "PASS"
    assert qualification["image_error_budget_m"] == 1e-13
    assert qualification["observable_budget_pass"] is False
    assert qualification["status"] == "REJECTED"


def test_native_round_action_and_cubic_gradient_agree_with_executed_wave_phase(record_testsuite_property):
    """A local no-caustic pupil compares real wave evolution with ray action.

    This is not a focal-plane or whole-lens Cs convention. Its radial action
    quartic term generates a cubic canonical slope in position coordinates.
    The two independent evolutions use the same P_perp=0 entrance sheet,
    source phase, energy and regular azimuthal gauge. No fitted piston, tilt
    or defocus enters the raw phase comparison. The declared budgets are
    1e-3 rad raw phase and 1e-10 normalized canonical momentum (small-angle
    radians); numerical error and finite-packet phase are reported separately.
    """
    from scipy.integrate import solve_ivp
    from scipy.ndimage import map_coordinates
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.potential_ray import trace_potential_rays
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_potential_wave import _snapshot

    energy, start, stop, radius, width = 300000., -.0001, .0001, 20e-6, 12.5e-6
    p, k = float(momentum(energy)), float(momentum(energy))/hbar
    native = FrozenAnalyticField("round-action", ((.3, 0., .001),),
        CoordinateRegistration(), (-10., 10.), .001)
    potentials = {order: HighOrderRoundLensPotential(native, order, .0005, "gaussian_all_z") for order in (5, 7)}
    snapshots = {order: _snapshot(value) for order, value in potentials.items()}
    radii = np.repeat(np.linspace(0., radius, 33), 8)
    azimuth = np.tile(np.arange(8)*np.pi/4, 33)
    entrance = np.column_stack((radii*np.cos(azimuth), radii*np.sin(azimuth), np.zeros((len(radii), 2))))
    mechanical = mechanical_coordinates(snapshots[5], entrance, start, energy, p)
    ray = trace_potential_rays(snapshots[5], mechanical, np.linspace(start, stop, 17), energy_ev=energy,
        maximum_step_m=5e-6, relative_tolerance=1e-12, position_tolerance_m=1e-15, slope_tolerance_rad=1e-13)
    positions = ray.xy_slopes[-1, :, :2]
    rho = np.linalg.norm(positions, axis=1)
    end_radius = float(np.max(rho))
    scaled = rho/end_radius
    ray_phase = k*ray.reduced_action_m[-1]
    canonical = canonical_coordinates(snapshots[5], ray.xy_slopes[-1], stop, energy, p)
    ray_gradient = np.divide(np.sum(positions*canonical[:, 2:], axis=1), rho,
        out=np.zeros_like(rho), where=rho > 0.)
    powers = np.column_stack([scaled**degree for degree in (0, 2, 4, 6)])
    training = np.repeat(np.arange(33) % 2 == 0, 8)
    ray_coefficients = np.linalg.lstsq(powers[training], ray_phase[training], rcond=None)[0]
    assert abs(ray_coefficients[2]) > .01  # A resolved physical quartic action, not just a quadratic test.

    # Independently integrate canonical H4 equations and action, rather than
    # comparing only two samplers of the mechanical Lorentz implementation.
    def h4(z, flat):
        values = flat.reshape(-1, 5)
        points = np.column_stack((values[:, :2], np.full(len(values), z)))
        jet = potentials[5].sample_jet(points)
        a, jac = -e*jet["A_t_m"]/p, -e*jet["jac_A_t"]/p
        u = values[:, 2:4]-a[:, :2]
        squared = np.sum(u*u, axis=1)
        velocity = u*(1+.5*squared)[:, None]
        force = np.einsum("nji,nj->ni", jac[:, :2, :2], velocity)+jac[:, 2, :2]
        action = np.sum(values[:, 2:4]*velocity, axis=1)-.5*squared-.125*squared*squared+a[:, 2]
        return np.column_stack((velocity, force, action)).ravel()
    independent = solve_ivp(h4, (start, stop), np.column_stack((entrance, np.zeros(len(entrance)))).ravel(),
        method="DOP853", rtol=1e-12, atol=1e-16, max_step=5e-6)
    assert independent.success
    h4_output = independent.y[:, -1].reshape(-1, 5)
    # At fixed output coordinates the first-order endpoint contribution is
    # P dot delta_x. It is explicitly recorded, never a fitted phase shift.
    h4_fixed_point_phase = k*(h4_output[:, 4]
        +np.sum(h4_output[:, 2:4]*(positions-h4_output[:, :2]), axis=1))
    h4_lorentz_phase = float(np.max(abs(h4_fixed_point_phase-ray_phase)))
    assert h4_lorentz_phase < 1e-6

    evidence = []
    phases = {}
    for name, n, steps, order, packet_width in (
        ("base", 128, 16, 5, width), ("axial", 128, 32, 5, width),
        ("axial_fine", 128, 64, 5, width), ("grid", 256, 32, 5, width),
        ("field", 128, 32, 7, width), ("finite_packet", 128, 32, 5, width/2)):
        pitch = packet_width*16/n
        axis = (np.arange(n)-n//2)*pitch
        xx, yy = np.meshgrid(axis, axis)
        amplitude = np.exp(-(xx*xx+yy*yy)/(4*packet_width**2)+.37j)
        amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
        wave = PlaneWave(amplitude, np.eye(2)*pitch, np.zeros(2), np.zeros((2, 2)), np.zeros(2))
        output, record = propagate_carrier_potential(snapshots[order], wave,
            np.linspace(start, stop, steps+1), energy_ev=energy, kinetic_order=4,
            reference_quadrature="cf4", scalar_splitting=True, sampling_backend="native_plane",
            grid_numerics=WaveGridNumerics(maximum_pixels=512), kinetic_phase_ledger=KineticPhaseLedger(.01))
        grid_index = np.linalg.solve(output.basis_m, positions.T-output.origin_m[:, None])
        index_yx = np.vstack((grid_index[1]+output.amplitude.shape[0]//2,
                              grid_index[0]+output.amplitude.shape[1]//2))
        sampled = (map_coordinates(output.amplitude.real, index_yx, order=5)
            +1j*map_coordinates(output.amplitude.imag, index_yx, order=5))
        xy = positions-output.origin_m
        carrier = k*(.5*np.einsum("ni,ij,nj->n", xy, output.curvature_m1, xy)+xy@output.tilt_rad)
        envelope_phase = np.angle(sampled*np.exp(-.37j))
        assert np.max(abs(envelope_phase)) < 1.  # No branch unwrapping or ambiguity in this bounded pupil.
        phase = envelope_phase+carrier
        coefficients = np.linalg.lstsq(powers[training], phase[training], rcond=None)[0]
        fitted_gradient = sum(degree*coefficients[column]*scaled**(degree-1)
            for column, degree in enumerate((0, 2, 4, 6)) if degree)/(k*end_radius)
        phase_error = float(np.max(abs(phase-ray_phase)))
        gradient_error = float(np.max(abs(fitted_gradient-ray_gradient)))
        holdout = float(np.max(abs((powers@coefficients-phase)[~training])))
        assert phase_error < 1e-3
        assert gradient_error < 1e-10
        assert holdout < 1e-3
        assert abs(coefficients[2]-ray_coefficients[2]) < .01*abs(ray_coefficients[2])
        assert output.probability == pytest.approx(.37, abs=3e-12)
        assert record["normalization_applied"] is False and record["physical_phase_removed"] is False
        phases[name] = phase
        evidence.append({"name": name, "input_grid": n, "output_shape": list(output.amplitude.shape),
            "axial_steps": steps, "radial_A_order": order, "packet_width_m": packet_width,
            "raw_phase_max_error_rad": phase_error, "canonical_gradient_max_error": gradient_error,
            "phase_fit_holdout_rad": holdout, "phase_coefficients_rad": coefficients.tolist(),
            "axis_phase_departure_rad": float(np.mean(phase[radii == 0.])),
            "flux": output.probability, "kinetic_bound_rad": record["kinetic_remainder_phase_bound"]})
    axial_changes = [float(np.max(abs(phases[b]-phases[a])))
        for a, b in (("base", "axial"), ("axial", "axial_fine"))]
    assert axial_changes[1] < .35*axial_changes[0]
    grid_change = float(np.max(abs(phases["grid"]-phases["axial"])))
    field_change = float(np.max(abs(phases["field"]-phases["axial"])))
    assert max(grid_change, field_change) < 1e-5
    # The finite-width on-axis diffraction phase is kept as a real difference.
    # Doubling the width decreases that phase by the expected semiclassical
    # trend. It is not removed to improve the ray/wave comparison.
    narrow_axis = float(np.mean(phases["finite_packet"][radii == 0.]))
    wide_axis = float(np.mean(phases["axial"][radii == 0.]))
    assert 3.5 < abs(narrow_axis/wide_axis) < 4.5
    record_testsuite_property("local_round_action_wave_phase", json.dumps({
        "scope": "same-entrance local round-lens action/phase and cubic-gradient comparison; not focal Cs or full C3 pilot",
        "energy_ev": energy, "interval_m": [start, stop], "B0_peak_t": .3, "B0_sigma_m": .001,
        "entrance_canonical_momentum": [0., 0.], "entrance_pupil_radius_m": radius,
        "initial_phase_rad": .37, "raw_phase_budget_rad": 1e-3, "canonical_gradient_budget": 1e-10,
        "ray_phase_coefficients_rad": ray_coefficients.tolist(), "ray_fit_condition": float(np.linalg.cond(powers[training])),
        "h4_full_lorentz_fixed_point_phase_difference_rad": h4_lorentz_phase,
        "axial_phase_changes_rad": axial_changes, "spatial_phase_change_rad": grid_change,
        "field_phase_change_rad": field_change, "observations": evidence,
        "piston_tilt_defocus_removed": False, "intensity_renormalized": False}, sort_keys=True))
