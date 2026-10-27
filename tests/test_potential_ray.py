"""Independent static E/B Lorentz, action, loss and numerical checks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.planar_gun_field import PlanarGunField
from temsim.physics.potential_ray import trace_potential_rays
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge
from temsim.physics.relativistic_lorentz import (
    ParticleSpecies, RelativisticPhaseSpace, integrate_adaptive_reference, momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


def uniform_snapshot(*, bz=0., potential_gradient=0., potential_offset=0., gauge=None):
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    rows = []
    if bz:
        r, z = np.array([0., .02]), np.array([-.1, .1])
        potential = AxisymmetricMagneticPotential(r, z, np.broadcast_to(.5*bz*r[:, None], (2, 2)))
        rows.append(PotentialContribution("uniform_B", ("uniform_B",), potential, "magnetic"))
    if potential_gradient or potential_offset:
        r, z = np.array([0., .02]), np.array([-.1, .1])
        native = PlanarGunField({}, r, z, np.broadcast_to(potential_offset+potential_gradient*z, (2, 2)))
        rows.append(PotentialContribution("uniform_E", ("uniform_E",),
                     ElectricPotentialSnapshot.from_provider(native), "electric"))
    return ElectromagneticPotentialSnapshot(scene, tuple(rows), gauge or StaticPolynomialGauge())


def momentum(energy):
    kinetic = energy*e
    return np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c


def run(snapshot, initial, *, z=(0., .01), energy=300000., step=.001, **kwargs):
    return trace_potential_rays(snapshot, initial, z, energy_ev=energy,
                                maximum_step_m=step, **kwargs)


def test_vacuum_drift_keeps_true_centroid_weights_and_geometric_action():
    initial = np.array([[.0001, -.0002, .03, -.02], [-.0003, .0001, -.01, .015]])
    weights = np.array([.2, 1.7])
    result = run(uniform_snapshot(), initial, weights=weights)
    expected = initial.copy()
    expected[:, :2] += .01*initial[:, 2:]
    np.testing.assert_allclose(result.xy_slopes[-1], expected, rtol=0., atol=2e-18)
    exact_action = .01*np.sum(initial[:, 2:]**2, axis=1)/(np.sqrt(1+np.sum(initial[:, 2:]**2, axis=1))+1)
    np.testing.assert_allclose(result.reduced_action_m[-1], exact_action, rtol=2e-14)
    np.testing.assert_array_equal(result.weights, weights)
    assert result.metadata["output_weight"] == pytest.approx(1.9)
    assert result.metadata["normalization_applied"] is False


def test_uniform_electric_acceleration_matches_relativistic_analytic_solution():
    initial = np.array([[.0002, -.0001, .04, -.02], [-.0001, .0002, -.03, .015]])
    energies = np.array([100000., 300000.])
    gradient, distance = 2e6, .01
    result = run(uniform_snapshot(potential_gradient=gradient), initial, energy=energies)
    p0 = momentum(energies)
    pz0 = p0/np.sqrt(1+np.sum(initial[:, 2:]**2, axis=1))
    perpendicular = initial[:, 2:]*pz0[:, None]
    final_p = momentum(energies+gradient*distance)
    pzf = np.sqrt(final_p**2-np.sum(perpendicular**2, axis=1))
    mass_momentum = np.sqrt((m_e*c)**2+np.sum(perpendicular**2, axis=1))
    integral = (np.arcsinh(pzf/mass_momentum)-np.arcsinh(pz0/mass_momentum))*c/(e*gradient)
    expected_position = initial[:, :2]+perpendicular*integral[:, None]
    np.testing.assert_allclose(result.xy_slopes[-1, :, :2], expected_position, rtol=2e-10, atol=2e-13)
    np.testing.assert_allclose(result.xy_slopes[-1, :, 2:], perpendicular/pzf[:, None], rtol=2e-10, atol=2e-12)


def test_uniform_b_helix_matches_analytic_mechanical_slopes_and_positions():
    initial = np.array([[.0002, -.0001, .04, -.02], [-.0001, .0002, -.03, .015]])
    energies = np.array([100000., 300000.])
    bz, distance = .04, .01
    result = run(uniform_snapshot(bz=bz), initial, energy=energies)
    pz = momentum(energies)/np.sqrt(1+np.sum(initial[:, 2:]**2, axis=1))
    omega = e*bz/pz
    initial_slope = initial[:, 2]+1j*initial[:, 3]
    final_slope = initial_slope*np.exp(1j*omega*distance)
    displacement = initial_slope*np.expm1(1j*omega*distance)/(1j*omega)
    np.testing.assert_allclose(result.xy_slopes[-1, :, :2], initial[:, :2]+
                               np.column_stack((displacement.real, displacement.imag)), rtol=2e-10, atol=2e-13)
    np.testing.assert_allclose(result.xy_slopes[-1, :, 2:],
                               np.column_stack((final_slope.real, final_slope.imag)), rtol=2e-10, atol=2e-12)


def test_combined_static_fields_match_independent_time_domain_lorentz():
    snapshot = uniform_snapshot(bz=.04, potential_gradient=1e6)
    initial = np.array([[.0001, -.0002, .03, -.015]])
    energy, stop = 100000., .01
    # Compare equations with identical constants; the older generic module's
    # default electron mass comes from a different CODATA release.
    species = ParticleSpecies(-e, m_e, "electron with matched constants")
    phase = RelativisticPhaseSpace(np.r_[initial[0, :2], 0.],
        momentum_from_kinetic_energy_ev(energy, np.r_[initial[0, 2:], 1.], species))
    magnetic = SimpleNamespace(field_at_global_positions_t=lambda p: snapshot.sample(p)["B_t"])
    electric = SimpleNamespace(field_at_global_positions_v_per_m=lambda p: snapshot.sample(p)["E_v_per_m"])

    def at_plane(position, _p, _t):
        return position[2]-stop
    at_plane.terminal, at_plane.direction = True, 1
    speed = np.linalg.norm(velocity_from_momentum_m_per_s(phase.momentum_kg_m_per_s, species))
    reference = integrate_adaptive_reference(phase, 2*stop/speed, magnetic,
        electric_field=electric, species=species, stop_event=at_plane, relative_tolerance=2e-12,
        position_tolerance_m=1e-14, momentum_tolerance_kg_m_per_s=1e-34, maximum_step_s=.0005/speed)
    assert reference.terminated_by_event
    final = reference.final_state
    # SciPy's time-event root has an absolute tolerance in seconds, so compare
    # at the actual returned plane rather than silently mixing nearby planes.
    assert abs(final.position_m[2]-stop) < 1e-9
    result = run(snapshot, initial, z=(0., float(final.position_m[2])), energy=energy, step=.0005)
    np.testing.assert_allclose(result.xy_slopes[-1, 0, :2], final.position_m[:2], rtol=2e-9, atol=2e-13)
    np.testing.assert_allclose(result.xy_slopes[-1, 0, 2:],
        final.momentum_kg_m_per_s[:2]/final.momentum_kg_m_per_s[2], rtol=2e-9, atol=2e-12)


def test_scalar_reference_shift_does_not_change_work_rays_or_action():
    initial = np.array([[.0001, .0002, .02, -.01]])
    first = run(uniform_snapshot(potential_gradient=1e6), initial)
    changed = run(uniform_snapshot(potential_gradient=1e6, potential_offset=150000.), initial)
    np.testing.assert_allclose(first.xy_slopes, changed.xy_slopes, rtol=2e-13, atol=1e-16)
    np.testing.assert_allclose(first.reduced_action_m, changed.reduced_action_m, rtol=2e-12, atol=1e-16)


def test_gauge_changes_action_by_only_the_declared_endpoint_term():
    initial = np.array([[.0001, -.0002, .03, -.02], [-.0003, .0001, -.01, .015]])
    energies = np.array([100000., 300000.])
    gauge = StaticPolynomialGauge(.3, [1e-6, -2e-6, 3e-7],
                                 [[.003, .001, .002], [.001, -.002, 0.], [.002, 0., .001]])
    baseline = run(uniform_snapshot(bz=.02), initial, energy=energies)
    shifted = run(uniform_snapshot(bz=.02, gauge=gauge), initial, energy=energies)
    np.testing.assert_allclose(shifted.xy_slopes, baseline.xy_slopes, rtol=1e-9, atol=2e-13)
    start = np.column_stack((initial[:, :2], np.zeros(len(initial))))
    end = np.column_stack((baseline.xy_slopes[-1, :, :2], np.full(len(initial), .01)))
    # Avoid subtracting a large irrelevant constant in this independent check.
    no_constant = replace(gauge, constant_t_m2=0.)
    expected = -e*(no_constant.chi_t_m2(end)-no_constant.chi_t_m2(start))/momentum(energies)
    np.testing.assert_allclose(shifted.reduced_action_m[-1]-baseline.reduced_action_m[-1], expected,
                               rtol=2e-10, atol=2e-13)
    assert "not a coherent wave" in shifted.metadata["action_scope"]


def test_offset_circular_stops_retain_lost_weights_and_do_not_recenter_survivors():
    initial = np.array([[0., 0., .01, 0.], [.0002, 0., .01, 0.], [-.0002, 0., 0., 0.]])
    weights = np.array([1., 2., 7.])
    result = run(uniform_snapshot(), initial, z=(0., .005, .01), weights=weights,
        stop_planes=((1, .0001, 0., .0002), (2, 0., 0., .00015)))
    np.testing.assert_array_equal(result.alive, [[True, True, True], [True, True, False], [True, False, False]])
    np.testing.assert_array_equal(result.weights, weights)
    assert result.metadata["input_weight"] == 10. and result.metadata["output_weight"] == 1.
    assert result.metadata["stopped_weight"] == 9.
    assert result.xy_slopes[-1, 0, 0] == pytest.approx(.0001, abs=1e-18)
    assert np.isnan(result.xy_slopes[-1, 1:]).all()
    assert np.isnan(result.reduced_action_m[-1, 1:]).all()
    for value in (result.z_m, result.xy_slopes, result.weights, result.alive, result.reduced_action_m):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_initial_aperture_and_all_lost_population_remain_explicit():
    initial = np.array([[.001, 0., 0., 0.], [.002, 0., 0., 0.]])
    result = run(uniform_snapshot(), initial, weights=[.2, .8], stop_planes=((0, 0., 0., .0005),))
    assert not result.alive.any() and np.isnan(result.xy_slopes).all()
    assert result.metadata["output_weight"] == 0. and result.metadata["stopped_weight"] == 1.
    assert result.metadata["field_evaluations"] == 0


def test_rk4_step_refinement_is_independent_of_the_fixed_field():
    snapshot = uniform_snapshot(bz=.15, potential_gradient=1e6)
    initial = np.array([[.0001, -.0002, .04, -.02]])
    reference = run(snapshot, initial, step=.0001)
    errors = []
    for step in (.002, .001, .0005):
        result = run(snapshot, initial, step=step, method="RK4")
        errors.append((np.linalg.norm(result.xy_slopes[-1, :, :2]-reference.xy_slopes[-1, :, :2]),
            np.linalg.norm(result.xy_slopes[-1, :, 2:]-reference.xy_slopes[-1, :, 2:]),
            np.linalg.norm(result.reduced_action_m[-1]-reference.reduced_action_m[-1])))
    errors = np.array(errors)
    assert np.all(errors[:-1] > 8*errors[1:])
    assert errors[-1, 0] < 1e-10 and errors[-1, 1] < 1e-8


def test_retarding_field_rejects_loss_of_positive_kinetic_energy():
    with pytest.raises(ValueError, match="positive-energy"):
        run(uniform_snapshot(potential_gradient=-1e6), [[0., 0., 0., 0.]], energy=1000.)


@pytest.mark.parametrize("kwargs", [dict(energy_ev=True), dict(energy_ev=1+1j),
    dict(weights=[1+1j]), dict(maximum_step_m=True), dict(relative_tolerance=0.),
    dict(stop_planes=((True, 0., 0., .001),)), dict(stop_planes=((1, 0., 0., -.001),))])
def test_invalid_scalar_and_stop_inputs_fail_without_silent_coercion(kwargs):
    options = dict(energy_ev=300000., maximum_step_m=.001)
    options.update(kwargs)
    with pytest.raises(ValueError):
        trace_potential_rays(uniform_snapshot(), [[0., 0., 0., 0.]], [0., .01], **options)


def test_cancellation_is_not_published_as_a_completed_partial_result():
    with pytest.raises(InterruptedError):
        run(uniform_snapshot(), [[0., 0., 0., 0.]], cancelled=lambda: True)


def test_compact_native_dipole_faces_use_interval_interior_limits():
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.lens_field_provider import CoordinateRegistration
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    energy, by, length = 300000., .005, .002
    p = momentum(energy)
    field = FrozenPosedMultipole("bounded_dipole", "bounded_dipole", CoordinateRegistration(),
        ((1, 0, -by),), "uniform", .002, .001, (1., 3.), .02, p)
    potential = AnalyticPotentialDomain(AnalyticMagneticPotential(field))
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])),
        (PotentialContribution("bounded_dipole", ("bounded_dipole",), potential, "magnetic"),))
    bend = e*by*length/p
    cosine = np.sqrt(1-bend*bend)
    slope = bend/cosine
    position = length*bend/(1+cosine)+.002*slope
    for method in ("RK4", "DOP853"):
        result = run(snapshot, [[0., 0., 0., 0.]], z=(0., .001, .003, .005),
                     step=.001, method=method)
        np.testing.assert_allclose(result.xy_slopes[-1, 0], (position, 0., slope, 0.),
                                   rtol=1e-9, atol=2e-13)


@pytest.mark.parametrize("method", ["RK4", "DOP853"])
def test_opt_in_compiled_native_sampling_preserves_complete_ray_and_action_result(method):
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from test_analytic_potential_domain import _base
    round_field = HighOrderRoundLensPotential(_base().field, radial_order=5, axial_domain="gaussian_all_z")
    quadrupole = AnalyticPotentialDomain(_base("quadrupole"))
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])),
        (PotentialContribution("round", ("round",), round_field, "magnetic"),
         PotentialContribution("quad", ("quad",), quadrupole, "magnetic")))
    initial = [[.0001, -.0002, .001, -.002], [-.0002, .0001, -.001, .002]]
    settings = dict(z=(-.015, -.012, .012, .02), step=.00025, method=method)
    ordinary = run(snapshot, initial, **settings)
    compiled = run(snapshot, initial, **settings, sampling_backend="native_compiled")
    np.testing.assert_allclose(compiled.xy_slopes, ordinary.xy_slopes, rtol=3e-10, atol=2e-14)
    np.testing.assert_allclose(compiled.reduced_action_m, ordinary.reduced_action_m, rtol=3e-10, atol=2e-17)
    np.testing.assert_array_equal(compiled.alive, ordinary.alive)
    np.testing.assert_array_equal(compiled.weights, ordinary.weights)
    assert ordinary.metadata["potential_sampling_backend"] == "snapshot"
    assert compiled.metadata["potential_sampling_backend"] == "numba-native-strict"


def test_compiled_sampling_rejects_unadmitted_fem_instead_of_silent_fallback():
    with pytest.raises(ValueError):
        run(uniform_snapshot(bz=.1), [[0., 0., 0., 0.]], sampling_backend="native_compiled")
