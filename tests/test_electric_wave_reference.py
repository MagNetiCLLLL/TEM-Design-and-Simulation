"""Same scalar field determines energy, flux carrier and the electric H4 terms."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e
from scipy.integrate import quad, solve_ivp

from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory, momentum_velocity
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.planar_gun_field import PlanarGunField
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.potential_wave_carrier import _reference, _coefficients, _apply_residual, propagate_carrier_potential
from temsim.physics.potential_wave_observation import observe_potential_mode, observe_potential_checkpoint
from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_flux import WaveMode, BeamState
from temsim.physics.wave_reference import AxialWaveReference
from test_potential_wave_carrier import _wave


def _electric(*, axial_rise=1e-3, radial_coefficient=0., offset=0., length=1e-4):
    r, z = np.array([0., 1e-3]), np.array([0., length])
    voltage = offset+radial_coefficient*r[:, None]**2+axial_rise*z[None, :]/length
    provider = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, r, z, voltage))
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    return ElectromagneticPotentialSnapshot(scene, (PotentialContribution("E", ("E",), provider, "electric"),))


def _mode():
    return WaveMode(_wave(n=48, width=2e-7, curvature=.1), .27, TIP_REFERENCE, "actual-energy:0", 300.,
                    AxialWaveReference(3e-9, 1e-22))


def test_scalar_offset_does_not_change_energy_and_tiny_tail_action_is_not_subtracted_away():
    snapshot = _electric(axial_rise=1e-7, offset=300000.)
    reference = ElectrostaticWaveReference.capture(snapshot, 0., 300000.)
    native_rise = snapshot.contributions[0].potential.arrays["voltage"][0, 1]-300000.
    assert reference.axis_energy(snapshot, 1e-4) == 300000.+native_rise
    record = reference.axial_integrals(snapshot, 0., 1e-4)
    p, v = momentum_velocity(300000.)
    expected_phase = e*native_rise*1e-4/(2*hbar*v)
    assert record["electric_phase_correction_rad"] == pytest.approx(expected_phase, rel=3e-4)
    assert record["electric_phase_correction_rad"] != 0.
    assert reference == ElectrostaticWaveReference.from_dict(reference.to_dict())
    changed = dict(reference.to_dict(), entrance_phi_v=0.)
    with pytest.raises(ValueError, match="digest"):
        ElectrostaticWaveReference.from_dict(changed)
    with pytest.raises(ValueError, match="original scalar"):
        replace(reference, entrance_phi_v=0.).require(snapshot)


def test_axial_acceleration_action_time_matches_independent_relativistic_quadrature():
    snapshot = _electric(axial_rise=1000.)
    reference = ElectrostaticWaveReference.capture(snapshot, 0., 300000.)
    record = reference.axial_integrals(snapshot, 0., 1e-4)
    rest = m_e*c*c
    def p(z):
        kinetic = (300000.+1e7*z)*e
        return np.sqrt(kinetic*(kinetic+2*rest))/c
    action = quad(p, 0., 1e-4, epsabs=1e-40, epsrel=1e-12)[0]
    flight = quad(lambda z: ((300000.+1e7*z)*e+rest)/(c*c*p(z)), 0., 1e-4,
                  epsabs=1e-30, epsrel=1e-12)[0]
    assert record["longitudinal_action_j_s"] == pytest.approx(action, rel=3e-15)
    assert record["axial_flight_time_s"] == pytest.approx(flight, rel=3e-15)


def test_electric_h4_retains_operator_cross_derivatives_and_is_hermitian():
    snapshot = _electric(axial_rise=.001, radial_coefficient=2e8)
    energy, z = 300000., 3e-5
    momentum, _ = momentum_velocity(energy)
    wavelength = 2*np.pi*hbar/momentum
    wave = _wave(n=24, pitch=1e-7, width=2e-7)
    reference = ElectrostaticWaveReference.capture(snapshot, 0., energy)
    quadratic = _reference(snapshot, wave, z, 0., momentum, electric_reference=reference, chart_energy_ev=energy)
    velocity, scalar, mechanical, frequency, defect, beta = _coefficients(
        snapshot, wave, z, quadratic, wavelength, return_energy_terms=True)
    rng = np.random.default_rng(19)
    a, b = (rng.normal(size=wave.amplitude.shape)+1j*rng.normal(size=wave.amplitude.shape) for _ in range(2))
    def d(value):
        return np.fft.ifft2(np.sum(frequency**2, axis=0)*np.fft.fft2(value))+defect*value
    expected = .5*beta*defect*a+beta**3*d(d(a))/8
    actual = _apply_residual(a, velocity, scalar, mechanical, frequency, 4, defect, beta)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-18)
    apply = lambda value: _apply_residual(value, velocity, scalar, mechanical, frequency, 4, defect, beta)
    assert np.vdot(a, apply(b)) == pytest.approx(np.vdot(apply(a), b), rel=1e-12, abs=1e-15)
    # A multiplication-only d^2 screen omits the two covariant cross terms.
    wrong = .5*beta*defect*a+beta**3*(d(d(a)-defect*a)-defect*(d(a)-defect*a)+defect**2*a)/8
    assert np.linalg.norm(actual-wrong) > 1e-11*np.linalg.norm(actual)


def test_accelerating_mode_preserves_absolute_flux_and_updates_carrier_energy():
    snapshot, mode = _electric(), _mode()
    result, = observe_potential_mode(snapshot, mode, 0., (1e-4,), maximum_step_m=2e-5)
    assert result.mode.energy_kev == (300000.+.001)*.001
    assert result.mode.weight_per_reference_electron == pytest.approx(.27, abs=2e-12)
    assert result.mode.axial_reference.flight_time_s > mode.axial_reference.flight_time_s
    assert result.metadata["electric_axial_integrals"]["electric_phase_correction_rad"] > 0.
    assert result.metadata["electric_wave_history"]["events"]
    assert result.mode.plane.curvature_m1 is not None


@pytest.mark.parametrize("method", ["taylor", "chebyshev"])
def test_phase_plate_consumes_weak_longitudinal_electric_phase_without_manual_reference_correction(method):
    from temsim.physics.wave_plane_observables import mode_phase_samples
    source, flat, mode = _electric(axial_rise=1e-7, offset=300000.), _electric(axial_rise=0., offset=300000.), _mode()
    changed, = observe_potential_mode(source, mode, 0., (1e-4,), maximum_step_m=2e-5,
                                     residual_exponential_method=method)
    baseline, = observe_potential_mode(flat, mode, 0., (1e-4,), maximum_step_m=2e-5,
                                      residual_exponential_method=method)
    wavelength = 2*np.pi*hbar/momentum_velocity(changed.mode.energy_kev*1000.)[0]
    actual = changed.mode.plane.full_amplitude(wavelength)
    original = baseline.mode.plane.full_amplitude(2*np.pi*hbar/momentum_velocity(300000.)[0])
    raw = np.angle(np.vdot(original, actual))
    expected = changed.metadata["electric_axial_integrals"]["electric_phase_correction_rad"]
    assert raw == pytest.approx(expected, rel=5e-4, abs=2e-10)
    assert raw != 0.
    index = tuple(size//2 for size in changed.mode.plane.amplitude.shape)
    displayed = np.angle(np.exp(1j*(mode_phase_samples(changed.mode)[index]-mode_phase_samples(baseline.mode)[index])))
    assert displayed == pytest.approx(expected, rel=5e-4, abs=2e-10)


def test_electric_geometry_resume_retains_energy_and_both_budgets_without_reset():
    snapshot, mode = _electric(), _mode()
    geometry = PotentialGeometrySnapshot(1e-6, 1e-4, ())
    initial = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), .001, 1e-12, {})
    options = dict(maximum_step_m=2e-5, kinetic_phase_ledger=KineticPhaseLedger(.01))
    whole, = observe_potential_geometry_checkpoint(snapshot, geometry, initial, (1e-4,), **options)
    first, = observe_potential_geometry_checkpoint(snapshot, geometry, initial, (4e-5,), **options)
    resumed, = observe_potential_geometry_checkpoint(snapshot, geometry, first, (1e-4,), maximum_step_m=2e-5)
    assert resumed.beam.modes[0].energy_kev == whole.beam.modes[0].energy_kev
    assert resumed.beam.total_weight == pytest.approx(whole.beam.total_weight, abs=1e-12)
    wavelength = 2*np.pi*hbar/momentum_velocity(whole.beam.modes[0].energy_kev*1000.)[0]
    np.testing.assert_allclose(resumed.beam.modes[0].plane.full_amplitude(wavelength),
        whole.beam.modes[0].plane.full_amplitude(wavelength), rtol=2e-9, atol=1e-12)
    assert resumed.record["kinetic_phase_ledger"]["binding"]["energy_ev"] == 300000.
    history = ElectricWaveHistory.from_dict(resumed.record["electric_wave_history"])
    assert history.last_z_m == 1e-4 and history.used_forward_error > 0.
    with pytest.raises(ValueError, match="cannot be reset"):
        observe_potential_geometry_checkpoint(snapshot, geometry, first, (1e-4,), maximum_step_m=2e-5,
            electric_wave_history=ElectricWaveHistory(ElectrostaticWaveReference.capture(snapshot, 4e-5, first.beam.modes[0].energy_kev*1000.)))


def test_unqualified_acceleration_is_rejected_instead_of_assumed_adiabatic():
    snapshot, mode = _electric(axial_rise=1000.), _mode()
    with pytest.raises(ValueError, match="forward branch"):
        observe_potential_mode(snapshot, mode, 0., (1e-4,), maximum_step_m=2e-5)


@pytest.mark.parametrize("rise", [.001, .1])
def test_added_scalar_flux_bound_covers_independent_two_way_klein_gordon_slab(rise):
    # Twenty reduced wavelengths keep the independent two-way ODE resolved;
    # it is intentionally NOT a long-column paraxial reference calculation.
    energy = 300000.
    momentum, _ = momentum_velocity(energy)
    length = 20.*hbar/momentum
    snapshot = _electric(axial_rise=rise, length=length)
    reference = ElectrostaticWaveReference.capture(snapshot, 0., energy)
    end_momentum, _ = momentum_velocity(energy+rise)
    k0, k1 = momentum/hbar, end_momentum/hbar
    def rhs(u, values):
        local_momentum, _ = momentum_velocity(energy+rise*u)
        return [values[1], -(local_momentum*length/hbar)**2*values[0]]
    answer = solve_ivp(rhs, (1., 0.), np.array([1.+0j, 1j*k1*length]),
        method="DOP853", rtol=2e-13, atol=2e-13, max_step=.005)
    assert answer.success
    psi, derivative = answer.y[:, -1]
    incoming = .5*(psi+derivative/(1j*k0*length))
    reflected = .5*(psi-derivative/(1j*k0*length))/incoming
    transmitted_flux = np.sqrt(k1/k0)/incoming
    integrals = reference.axial_integrals(snapshot, 0., length)
    forward = np.exp(1j*(integrals["baseline_longitudinal_action_j_s"]/hbar
                             +integrals["electric_phase_correction_rad"]))
    bound = reference.forward_variation_bound(snapshot, 0., length, 1e-6)
    assert abs(transmitted_flux-forward) < bound
    assert abs(reflected) < bound
    assert abs(transmitted_flux)**2+abs(reflected)**2 == pytest.approx(1., abs=2e-12)


def test_radial_entrance_energy_and_composed_scalar_fields_enter_forward_gate():
    original = _electric(axial_rise=0., radial_coefficient=-3.01e11)
    reference = ElectrostaticWaveReference.capture(original, 0., 300000.)
    with pytest.raises(ValueError, match="positive forward"):
        reference.forward_variation_bound(original, 0., 1e-4, 1e-3)
