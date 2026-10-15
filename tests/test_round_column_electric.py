"""Independent round electric Hamiltonian fixtures, not source qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e
from scipy.integrate import quad, solve_ivp

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.column_wave import _propagate_column
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics
from temsim.physics.round_column_checkpoint import execute_round_prefix
from temsim.physics.wave_grid import WaveGridNumerics, refine_plane_wave
from test_cartesian_round_route import fixture
from test_column_wave_electric import prepared


def momentum_velocity(energy_ev):
    kinetic = energy_ev*e
    momentum = np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
    return momentum, momentum*c*c/(kinetic+m_e*c*c)


def source():
    original = fixture(False)
    pitch, sigma = 25e-9, 100e-9
    axis = (np.arange(64)-32)*pitch
    r2 = axis[:, None]**2+axis[None, :]**2
    plane = PlaneWave(pitch/(np.sqrt(np.pi)*sigma)*np.exp(-r2/(2*sigma*sigma)+.73j),
        np.eye(2)*pitch, np.zeros(2), curvature_m1=np.eye(2)*-600.)
    return replace(original, plane_z_mm=2000., beam=replace(original.beam,
        modes=(replace(original.beam.modes[0], plane=plane),)))


def run_round(monkeypatch, item, initial=None):
    initial = source() if initial is None else initial
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *a, **kw: item)
    state = SimpleNamespace(electron_gun=SimpleNamespace(exit_plane_z_mm=2000.))
    return execute_round_prefix(state, initial, 2001., allow_cartesian=True,
        numerics=RadialColumnNumerics(initial_samples=32768, maximum_samples=131072,
                                     initial_cartesian_pixels=64),
        grid_numerics=WaveGridNumerics(maximum_pixels=512, maximum_working_bytes=256*1024**2))


def analytic_field(plane, gradient, radial):
    """Independent ODE for the continuous 2x2 electrostatic Hamiltonian."""
    p0, _ = momentum_velocity(300000.)
    def rhs(z, values):
        p, velocity = momentum_velocity(300000.+gradient*z)
        generator = np.array(((0., p0/p), (e*2*radial/(velocity*p0), 0.)))
        return (generator@values.reshape(2, 2)).ravel()
    matrix = solve_ivp(rhs, (0., .001), np.eye(2).ravel(), rtol=2e-12, atol=1e-14).y[:, -1].reshape(2, 2)
    lam = float(wavelength_m(300000.))
    precision = -600.+1j*lam/(2*np.pi*(100e-9)**2)
    factor = matrix[0, 0]+matrix[0, 1]*precision
    final_precision = (matrix[1, 0]+matrix[1, 1]*precision)/factor
    x, y = plane.coordinates_m()
    return np.sqrt(.3)*np.exp(.73j)/(np.sqrt(np.pi)*100e-9*factor)*np.exp(
        1j*np.pi/lam*final_precision*(x*x+y*y))


def resolved_comparison_mode(mode):
    # Materializing the known analytic carrier requires more pixels than
    # transporting its envelope. Refine that same Fourier interpolant solely
    # for the complex comparison, without modifying any calculated state.
    if max(mode.plane.amplitude.shape) < 768:
        return replace(mode, plane=refine_plane_wave(mode.plane, (768, 768),
            numerics=WaveGridNumerics(maximum_pixels=768, maximum_working_bytes=256*1024**2)))
    return mode


def test_accelerating_round_field_matches_independent_complex_solution_and_2d(monkeypatch):
    gradient, radial = 2e8, -2e10
    errors = []
    for step in (.02, .01):
        item = prepared(gradient=gradient, radial=radial, step=step)
        initial = source()
        result = run_round(monkeypatch, item, initial)
        assert result.plane_z_mm == 2001.
        output = result.beam.modes[0]
        assert output.energy_kev == pytest.approx(500., abs=1e-8)
        assert output.mode_id == initial.beam.modes[0].mode_id
        assert output.scattering_history == initial.beam.modes[0].scattering_history
        observed = resolved_comparison_mode(output)
        exact = analytic_field(observed.plane, gradient, radial)
        error = np.linalg.norm(observed.weighted_density_amplitude()-exact)*np.sqrt(abs(np.linalg.det(observed.plane.basis_m)))
        errors.append(error)
        assert result.beam.total_weight == pytest.approx(initial.beam.total_weight, rel=1e-8)
        # Explicit continuous quadrature, independent of the midpoint clock.
        old = initial.beam.modes[0].axial_reference
        flight = quad(lambda z: 1/momentum_velocity(300000.+gradient*z)[1], 0., .001, epsabs=1e-24)[0]
        action = quad(lambda z: momentum_velocity(300000.+gradient*z)[0], 0., .001, epsabs=1e-32)[0]
        assert output.axial_reference.flight_time_s-old.flight_time_s == pytest.approx(flight, rel=1e-5)
        assert output.axial_reference.longitudinal_action_j_s-old.longitudinal_action_j_s == pytest.approx(action, rel=1e-5)
        # The actual 2-D operator sees exactly the same prepared electric law.
        cartesian = _propagate_column(SimpleNamespace(), initial, 2001., _prepared=item)
        other = resolved_comparison_mode(cartesian.beam.modes[0])
        two_d_exact = analytic_field(other.plane, gradient, radial)
        other_error = np.linalg.norm(other.weighted_density_amplitude()-two_d_exact)*np.sqrt(abs(np.linalg.det(other.plane.basis_m)))
        assert other_error < 1e-4
        assert output.axial_reference.flight_time_s == pytest.approx(other.axial_reference.flight_time_s, rel=1e-14)
        assert output.axial_reference.longitudinal_action_j_s == pytest.approx(other.axial_reference.longitudinal_action_j_s, rel=1e-14)
        # Different lattices/phase carriers are allowed; neither comparison
        # removes a fitted global phase or normalizes away numerical loss.
        assert error < 1e-4
    assert errors[1] < errors[0]/3.8


def test_cs_uses_local_momentum_in_the_fixed_entrance_phase_convention(monkeypatch):
    item = prepared(gradient=2e8, radial=0., step=.05)
    item[0].cs_kick_m3[-1] = 7e11
    seen = []
    def record(wave, wavelength, strength):
        seen.append((wavelength, strength))
        return wave
    monkeypatch.setattr("temsim.physics.radial_column_wave.radial_cs", record)
    output = run_round(monkeypatch, item)
    p0, _ = momentum_velocity(300000.)
    pend, _ = momentum_velocity(500000.)
    assert seen
    for lam, strength in seen:
        assert lam == pytest.approx(float(wavelength_m(300000.)), rel=1e-14)
        assert strength == pytest.approx(7e11*pend/p0, rel=1e-13)
    row = output.record["round_execution"]["modes"][0]
    assert row["exit_carrier_momentum_ratio"] == pytest.approx(p0/pend, rel=1e-13)


@pytest.mark.parametrize("kind", ("gradient", "anisotropy", "skew"))
def test_nonround_electric_coefficients_decline_without_discarding_small_terms(monkeypatch, kind):
    from temsim.physics.radial_column_wave import _round_column_prefix
    item = prepared(gradient=0., radial=0.)
    item[0].electric_field = None
    count = len(item[0].step_m)
    gradient, hessian = np.zeros((count, 2)), np.zeros((count, 2, 2))
    if kind == "gradient":
        gradient[0, 0] = 1e-20
    elif kind == "anisotropy":
        hessian[0, 0, 0] = 1e-20
    else:
        hessian[0, 0, 1] = hessian[0, 1, 0] = 1e-20
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *a, **kw: item)
    monkeypatch.setattr("temsim.physics.column_wave._electric_coefficients", lambda *_:
        (np.zeros(count+1), np.zeros(count), gradient, hessian))
    z, modes, record = _round_column_prefix(SimpleNamespace(), source(), 2001.)
    assert z == 2000. and modes == ()
    assert "Non-axisymmetric electric" in record["reason"]


def test_unresolved_electric_provider_declines_before_requiring_plan_details(monkeypatch):
    from temsim.physics.radial_column_wave import _round_column_prefix
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column", lambda *a, **kw:
        (SimpleNamespace(electric_field=object()), (), {}, ()))
    z, modes, record = _round_column_prefix(SimpleNamespace(), SimpleNamespace(plane_z_mm=2000.), 2001.)
    assert z == 2000. and modes == ()
    assert "Unresolved electric" in record["reason"]


def test_phase_path_refinement_replays_actual_electric_generator(monkeypatch):
    import temsim.physics.canonical_action as action
    import temsim.physics.column_wave as column
    original_path, original_factor = action.CanonicalPath, column._linear_factor
    seen = []
    class RefinedPending(original_path):
        rejected_once = False
        def append(self, matrix, *args, **kwargs):
            if not self.rejected_once and not np.array_equal(matrix, np.eye(4)):
                self.rejected_once = True
                raise ValueError("Canonical phase path is undersampled; refine axial steps")
            return super().append(matrix, *args, **kwargs)
    def record(path, generator, distance, *args, **kwargs):
        if isinstance(path, RefinedPending):
            seen.append(generator.copy())
        return original_factor(path, generator, distance, *args, **kwargs)
    monkeypatch.setattr(action, "CanonicalPath", RefinedPending)
    monkeypatch.setattr(column, "_linear_factor", record)
    output = run_round(monkeypatch, prepared(gradient=2e8, radial=-2e10, step=.05))
    assert output.plane_z_mm == 2001. and seen
    assert any(np.any(generator[2:, :2]) for generator in seen)
    assert any(not np.allclose(generator[:2, 2:], np.eye(2), rtol=1e-4) for generator in seen)
