"""CPU-only finite-field aperture sampling, phase and bounded failure checks."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_aperture_sampling import ApertureSamplingNumerics, apply_circular_aperture
from temsim.physics.wave_grid import WaveGridBudgetError, WaveGridNumerics, WaveSamplingError


def envelope(xy):
    x, y = xy
    return np.exp(-.25*((x/.9)**2+(y/1.3)**2)+1j*(.13*x+.04*y+.02*x*y))/np.sqrt(2*np.pi*.9*1.3)


def wave_fixture(*, affine=False, count=128):
    basis = np.eye(2)*.2
    if affine:
        angle = .37
        rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        basis = basis@rotation@np.array(((1., .3), (0., .8)))
    wave = PlaneWave(np.zeros((count, count), complex), basis, np.array((.013, -.025)),
                     np.array(((2., .4), (.4, -.7))), np.array((.07, -.02)))
    return replace(wave, amplitude=envelope(wave.coordinates_m())*np.sqrt(abs(np.linalg.det(basis))))


def carrier(wave, xy, wavelength):
    delta = xy-wave.origin_m[:, None, None]
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    action = .5*np.einsum("iyx,ij,jyx->yx", delta, q, delta)+np.einsum("i,iyx->yx", tilt, delta)
    return np.exp(2j*np.pi*action/wavelength)


def run(wave, **kwargs):
    return apply_circular_aperture(wave, radius_m=kwargs.pop("radius_m", .04),
        wavelength_m=.12, grid_numerics=kwargs.pop("grid_numerics", WaveGridNumerics(
            maximum_pixels=2048, maximum_working_bytes=512*1024**2)), **kwargs)


@pytest.mark.parametrize("centre", [(0., 0.), (.09, .08), (.115, -.107)])
@pytest.mark.parametrize("affine", [False, True])
def test_subcell_aperture_preserves_original_field_and_agrees_with_independent_integral(centre, affine):
    wave = wave_fixture(affine=affine)
    original = wave.amplitude.copy()
    result, evidence = run(wave, centre_m=centre)
    xy = result.coordinates_m()
    expected = envelope(xy)*np.sqrt(abs(np.linalg.det(result.basis_m)))*carrier(wave, xy, .12)
    actual = result.amplitude*carrier(result, xy, .12)
    # Ignore only floating-point comparisons exactly on a circular boundary.
    interior = np.sum((xy-np.asarray(centre)[:, None, None])**2, axis=0) < (.04*(1-1e-12))**2
    np.testing.assert_allclose(actual[interior], expected[interior], rtol=2e-11, atol=1e-13)
    assert np.count_nonzero(result.amplitude) > 100
    cx, cy = centre
    def angular(theta, radius):
        xy_point = np.array((cx+radius*np.cos(theta), cy+radius*np.sin(theta)))
        return float(abs(envelope(xy_point))**2)*radius
    reference = quad(lambda radius: quad(lambda theta: angular(theta, radius), 0., 2*np.pi,
                                         epsabs=1e-13)[0], 0., .04, epsabs=1e-13)[0]
    assert result.probability == pytest.approx(reference, rel=.003)
    assert evidence["status"] == "converged"
    assert len(evidence["trials"]) >= 2
    assert evidence["complex_error_norm"] <= evidence["maximum_complex_error_norm"]
    assert evidence["normalized"] is False and evidence["phase_fitted"] is False
    assert result.probability < wave.probability
    np.testing.assert_array_equal(original, wave.amplitude)
    np.testing.assert_array_equal(result.curvature_m1, wave.curvature_m1)
    np.testing.assert_allclose(result.tilt_rad, wave.tilt_rad+wave.curvature_m1@(np.asarray(centre)-wave.origin_m))


def test_noop_does_not_allocate_or_refine_even_with_zero_memory_budget(monkeypatch):
    wave = wave_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("No-op aperture must not allocate a resampling grid")
    monkeypatch.setattr("temsim.physics.wave_aperture_sampling._sample_envelope", forbidden)
    result, evidence = run(wave, radius_m=100., grid_numerics=WaveGridNumerics(
        maximum_working_bytes=1, automatic_refinement=False))
    assert result is wave and evidence["status"] == "exact_noop"
    assert evidence["trials"] == []


def test_budget_failure_occurs_before_sampling_and_leaves_input_unchanged(monkeypatch):
    wave = wave_fixture()
    original = wave.amplitude.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("Budget must be checked before sampling")
    monkeypatch.setattr("temsim.physics.wave_aperture_sampling._sample_envelope", forbidden)
    with pytest.raises(WaveGridBudgetError, match="budget"):
        run(wave, grid_numerics=WaveGridNumerics(maximum_working_bytes=1))
    np.testing.assert_array_equal(original, wave.amplitude)


def test_failure_after_first_trial_does_not_publish_a_coarse_result(monkeypatch):
    import temsim.physics.wave_aperture_sampling as module
    wave = wave_fixture()
    original = wave.amplitude.copy()
    calls = []
    sample = module._sample_envelope
    def capture(*args):
        calls.append(args[1])
        return sample(*args)
    monkeypatch.setattr(module, "_sample_envelope", capture)
    with pytest.raises(WaveGridBudgetError):
        run(wave, grid_numerics=WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=20*1024**2),
            numerics=ApertureSamplingNumerics(maximum_relative_current_change=1e-12,
                                            maximum_complex_change_per_input_norm=1e-12))
    assert calls
    np.testing.assert_array_equal(original, wave.amplitude)


def test_cancelled_between_trials_leaves_input_unchanged():
    wave = wave_fixture()
    original = wave.amplitude.copy()
    count = 0
    def cancelled():
        nonlocal count
        count += 1
        return count >= 5
    with pytest.raises(InterruptedError, match="cancelled"):
        run(wave, cancelled=cancelled)
    np.testing.assert_array_equal(original, wave.amplitude)


def test_collapsed_input_cannot_be_reconstructed_as_a_new_gaussian():
    wave = wave_fixture()
    amplitude = np.zeros_like(wave.amplitude)
    amplitude[64, 64] = 1.
    wave = replace(wave, amplitude=amplitude)
    with pytest.raises(WaveSamplingError, match="undersampled"):
        run(wave)


def test_input_boundary_fails_closed_even_for_partial_domain_apertures():
    wave = wave_fixture()
    with pytest.raises(WaveSamplingError, match="physical-domain edges"):
        run(replace(wave, amplitude=np.ones_like(wave.amplitude)*.001))
    with pytest.raises(WaveSamplingError, match="physical-domain edges"):
        run(replace(wave, amplitude=np.ones_like(wave.amplitude)*.001), centre_m=(12.7, 0.))


def test_large_partial_domain_circle_retains_original_physical_window():
    wave = wave_fixture(affine=True)
    result, evidence = run(wave, radius_m=12.)
    assert evidence["chart"] == "original_physical_window"
    assert evidence["status"] == "converged"
    np.testing.assert_array_equal(result.origin_m, wave.origin_m)
    np.testing.assert_allclose(result.basis_m@np.diag(result.amplitude.shape[::-1]),
                               wave.basis_m@np.diag(wave.amplitude.shape[::-1]))
    assert result.probability == pytest.approx(wave.probability, rel=1e-12)


def test_odd_lattice_retains_exact_complex_envelope():
    wave = wave_fixture(count=127)
    result, evidence = run(wave, centre_m=(.073, -.032))
    xy = result.coordinates_m()
    inside = np.sum((xy-result.origin_m[:, None, None])**2, axis=0) < (.04*(1-1e-12))**2
    expected = envelope(xy)*np.sqrt(abs(np.linalg.det(result.basis_m)))*carrier(wave, xy, .12)
    np.testing.assert_allclose((result.amplitude*carrier(result, xy, .12))[inside], expected[inside],
                               rtol=2e-11, atol=1e-13)
    assert evidence["status"] == "converged"


def test_absolute_probability_scales_without_norm_fitting():
    wave = wave_fixture()
    first, first_record = run(wave)
    scaled, scaled_record = run(replace(wave, amplitude=wave.amplitude*(.27+.08j)))
    np.testing.assert_allclose(scaled.amplitude, first.amplitude*(.27+.08j), rtol=1e-12, atol=1e-15)
    assert scaled.probability == pytest.approx(first.probability*abs(.27+.08j)**2, rel=1e-13)
    assert scaled_record["complex_error_norm"] == pytest.approx(
        first_record["complex_error_norm"]*abs(.27+.08j), rel=1e-12)


def test_complex_convergence_includes_the_hard_edge_not_just_the_incident_interpolant():
    from temsim.physics.wave_grid import refine_plane_wave
    wave = wave_fixture()
    result, evidence = run(wave)
    assert evidence["complex_error_norm"] > 1e-7
    # An independently sampled analytic envelope and mask on a further doubled
    # grid measures the actual transmitted complex field, including its edge.
    shape = tuple(2*n for n in result.amplitude.shape)
    fine = PlaneWave(np.zeros(shape, complex), result.basis_m/2, result.origin_m,
                     result.curvature_m1, result.tilt_rad)
    xy = fine.coordinates_m()
    mask = np.sum((xy-fine.origin_m[:, None, None])**2, axis=0) <= .04**2
    full = envelope(xy)*np.sqrt(abs(np.linalg.det(fine.basis_m)))*carrier(wave, xy, .12)
    expected = np.where(mask, full/carrier(fine, xy, .12), 0j)
    embedded = refine_plane_wave(result, shape)
    difference = np.linalg.norm(embedded.amplitude-expected)
    assert difference <= evidence["maximum_complex_error_norm"]


def test_disabled_refinement_and_zero_error_budget_cannot_accept_unchecked_mask():
    wave = wave_fixture()
    with pytest.raises(WaveSamplingError, match="disabled"):
        run(wave, grid_numerics=WaveGridNumerics(automatic_refinement=False))
    with pytest.raises(WaveSamplingError, match="positive declared L2"):
        run(wave, grid_numerics=WaveGridNumerics(maximum_accumulated_support_l2=0.))


def test_closed_aperture_preserves_geometry_and_removes_all_probability():
    wave = wave_fixture()
    result, evidence = run(wave, radius_m=0.)
    assert result.probability == 0 and evidence["status"] == "closed"
    np.testing.assert_array_equal(result.basis_m, wave.basis_m)
    np.testing.assert_array_equal(result.origin_m, wave.origin_m)
    assert wave.probability > 0


def test_bore_strict_boundary_is_retained_on_the_resolved_chart():
    wave = wave_fixture()
    result, evidence = run(wave, strict_boundary=True)
    xy = result.coordinates_m()
    outside_or_wall = np.sum(xy*xy, axis=0) >= .04**2
    assert np.count_nonzero(result.amplitude[outside_or_wall]) == 0
    assert evidence["strict_boundary"] is True


@pytest.mark.parametrize("kwargs", [{"radius_m": -1.}, {"radius_m": float("nan")},
                                    {"centre_m": (float("inf"), 0.)}])
def test_invalid_physical_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError, match="Circular aperture"):
        run(wave_fixture(), **kwargs)


def test_column_small_aperture_tracks_absolute_flux_and_resets_coordinates_after_regridding():
    from temsim.optics.model import Aperture
    from temsim.physics.column_wave import _clip
    wave = wave_fixture()
    apertures = (Aperture("small", "small", 1., 40.),
                 Aperture("offset", "offset", 1., 40., offset_x_mm=20.))
    output, losses = _clip(wave, 10000., apertures, 1., .27,
        wavelength_m=.12, grid_numerics=WaveGridNumerics(maximum_working_bytes=512*1024**2))
    sampling = [row for row in losses if "sampling" in row]
    assert sampling and sampling[0]["sampling"]["status"] == "converged"
    assert output.amplitude.shape != wave.amplitude.shape
    xy = output.coordinates_m()*1e3
    assert np.count_nonzero(output.amplitude[np.hypot(xy[0]-20., xy[1]) > 40.]) == 0
    assert sum(row["lost_weight"] for row in losses) == pytest.approx(.27*(wave.probability-output.probability), abs=1e-15)


def test_placed_circular_hole_applies_body_rotation_and_translation_once():
    from types import SimpleNamespace
    from temsim.optics.model import Aperture
    from temsim.physics.column_wave import _PlacedAperture, _clip
    angle = .6
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                         (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    registration = SimpleNamespace(rotation_array=rotation, origin_array_m=np.array((.04, -.03, 0.)))
    aperture = Aperture("placed", "placed", 1., 40., offset_x_mm=70., offset_y_mm=-50.)
    placed = _PlacedAperture(aperture, registration, 1.)
    expected_centre = rotation[:2, :2]@np.array((.07, -.05))+registration.origin_array_m[:2]
    wave = wave_fixture(affine=True)
    output, losses = _clip(wave, np.inf, [placed], 1., .37,
        wavelength_m=.12, grid_numerics=WaveGridNumerics(maximum_working_bytes=512*1024**2))
    np.testing.assert_allclose(output.origin_m, expected_centre, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(losses[0]["sampling"]["centre_m"], expected_centre)


def test_round_component_with_overridden_custom_mask_is_not_silently_replaced():
    from temsim.optics.condenser_aperture import ContinuousApertureComponent
    from temsim.physics.column_wave import _circular_opening
    class Slit(ContinuousApertureComponent):
        def transmission_mask(self, x, y):
            return abs(x) < .001
    # Only these fields participate in circular-mask dispatch. No physical
    # source, assembly build or optical calculation is needed for this check.
    slit = object.__new__(Slit)
    slit.radius_mm = 10.
    slit.offset_x_mm = slit.offset_y_mm = 0.
    assert _circular_opening(slit) is None
