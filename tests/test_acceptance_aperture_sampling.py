"""Local receiving-aperture interpolation versus independent full-period FFTs."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from temsim.physics.column_acceptance_window import AcceptanceWindowNumerics, sample_aperture_plane
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridBudgetError, WaveGridNumerics, refine_plane_wave


def source(*, shape=(96, 96), affine=False, origin=(.018, -.013), pitch=.015, sigma=.12):
    basis = np.eye(2)*pitch
    if affine:
        angle = .37
        basis = basis@np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        basis = basis@np.array(((1., .23), (0., .83)))
    wave = PlaneWave(np.ones(shape, complex), basis, np.asarray(origin),
                     np.array(((1.2, .13), (.13, .9))), np.array((.03, -.02)))
    xy = wave.coordinates_m()
    amplitude = np.exp(-np.sum(xy*xy, axis=0)/(2*sigma*sigma)+.17j*xy[0]/sigma)
    amplitude *= (1+.1*xy[1]/sigma)*np.sqrt(abs(np.linalg.det(basis)))
    return replace(wave, amplitude=amplitude)


def full_period_reference(wave, radius, policy, curvature, wavelength=.25):
    """Previous full-grid Fourier refinement, independently interpolated.

    This reference intentionally allocates the full refined period only for
    small fixtures. It does not use the new local evaluator or its patch.
    """
    pitch = radius/policy.aperture_radius_cells
    factor = max(1, math.ceil(np.linalg.norm(wave.basis_m, ord=2)/pitch))
    refined = refine_plane_wave(wave, tuple(factor*n for n in wave.amplitude.shape))
    half = math.ceil(policy.output_halfwidth_radii*policy.aperture_radius_cells)
    axis = (np.arange(2*half)-half)*pitch
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    positions = np.stack((xx, yy))
    inside = xx*xx+yy*yy <= (1.05*radius)**2
    points = positions[:, inside]
    delta = points-wave.origin_m[:, None]
    indices = np.linalg.inv(refined.basis_m)@delta+np.array(refined.amplitude.shape[::-1])[:, None]//2
    values = map_coordinates(refined.amplitude, indices[::-1], order=3, mode="constant", cval=0.)
    action = (.5*np.einsum("in,ij,jn->n", delta, wave.curvature_m1, delta)+wave.tilt_rad@delta
              -.5*np.einsum("in,ij,jn->n", points, curvature, points))
    output = np.zeros(xx.shape, complex)
    output[inside] = values*np.exp(2j*np.pi*action/wavelength)*pitch/np.sqrt(abs(np.linalg.det(refined.basis_m)))
    return PlaneWave(output, np.eye(2)*pitch, np.zeros(2), curvature, np.zeros(2))


def reconstruct(wave, *, radius=.05, policy=None, grid=None, cancelled=lambda: False):
    return sample_aperture_plane(wave, radius_m=radius, wavelength_m=.25, curvature=np.eye(2)*.7,
        numerics=policy or AcceptanceWindowNumerics(aperture_radius_cells=16, output_halfwidth_radii=2.),
        grid_numerics=grid or WaveGridNumerics(compute_backend="CPU", maximum_working_bytes=256*1024**2),
        cancelled=cancelled)


@pytest.mark.parametrize("shape", [(96, 96), (95, 97)])
@pytest.mark.parametrize("affine", [False, True])
def test_local_patch_matches_full_period_refinement_complex_field_and_probability(shape, affine):
    wave = source(shape=shape, affine=affine)
    original = wave.amplitude.copy()
    policy = AcceptanceWindowNumerics(aperture_radius_cells=16, output_halfwidth_radii=2.)
    actual, record = reconstruct(wave, policy=policy)
    expected = full_period_reference(wave, .05, policy, np.eye(2)*.7)
    np.testing.assert_allclose(actual.amplitude, expected.amplitude, rtol=2e-10, atol=2e-14)
    assert actual.probability == pytest.approx(expected.probability, rel=2e-11)
    assert record["input_unmasked_probability"] == wave.probability
    assert record["interpolation_domain"] == "local_finite_fourier_patch"
    assert record["sampling_patch_spline_guard_cells"] == 32
    assert np.prod(record["sampling_patch_shape"]) < np.prod(record["virtual_full_refined_shape"])
    np.testing.assert_array_equal(wave.amplitude, original)
    np.testing.assert_array_equal(actual.curvature_m1, np.eye(2)*.7)
    assert record["physical_mask"] == "pending actual component clip; no aperture taper"
    assert record["phase_fit"] is False


def test_full_complex_carrier_uses_original_origin_tilt_and_curvature():
    wave = source(shape=(127, 129), affine=True, origin=(.031, -.044))
    result, _ = reconstruct(wave)
    xy = result.coordinates_m()
    inside = np.sum(xy*xy, axis=0) <= .05**2
    density = np.exp(-np.sum(xy*xy, axis=0)/(2*.12**2)+.17j*xy[0]/.12)*(1+.1*xy[1]/.12)
    delta = xy-wave.origin_m[:, None, None]
    action = .5*np.einsum("iyx,ij,jyx->yx", delta, wave.curvature_m1, delta)+np.einsum("i,iyx->yx", wave.tilt_rad, delta)
    expected = density*np.exp(2j*np.pi*action/.25)*result.basis_m[0, 0]
    # Compare the analytic carrier only on the physical aperture's interior.
    # This temporary chart's outer 1.05-radius numerical cutoff is not a
    # publishable unmasked wave and need not pass whole-plane phase expansion.
    output_action = .5*np.einsum("iyx,ij,jyx->yx", xy, result.curvature_m1, xy)
    actual = result.amplitude*np.exp(2j*np.pi*output_action/.25)
    np.testing.assert_allclose(actual[inside], expected[inside], rtol=2e-7, atol=2e-10)


def test_negative_nyquist_coefficient_is_preserved_by_local_fourier_evaluation():
    wave = source()
    checkerboard = (-1.)**np.arange(wave.amplitude.shape[1])
    wave = replace(wave, amplitude=wave.amplitude*(1.+1e-4j*checkerboard))
    policy = AcceptanceWindowNumerics(aperture_radius_cells=24, output_halfwidth_radii=2.)
    result, _ = reconstruct(wave, policy=policy)
    expected = full_period_reference(wave, .05, policy, np.eye(2)*.7)
    np.testing.assert_allclose(result.amplitude, expected.amplitude, rtol=2e-10, atol=2e-14)


def test_patch_clipped_at_original_domain_boundary_keeps_same_spline_boundary():
    wave = source(origin=(-.635, 0.))
    policy = AcceptanceWindowNumerics(aperture_radius_cells=16, output_halfwidth_radii=2.)
    result, record = reconstruct(wave, policy=policy)
    expected = full_period_reference(wave, .05, policy, np.eye(2)*.7)
    np.testing.assert_allclose(result.amplitude, expected.amplitude, rtol=2e-9, atol=2e-13)
    assert record["complex_interpolation_relative_change"] < policy.maximum_relative_complex_change*.1


def test_factor_one_retains_existing_unmodified_grid_path():
    wave = source(pitch=.001)
    policy = AcceptanceWindowNumerics(aperture_radius_cells=16, output_halfwidth_radii=2.)
    result, record = reconstruct(wave, radius=.02, policy=policy)
    expected = full_period_reference(wave, .02, policy, np.eye(2)*.7)
    np.testing.assert_array_equal(result.amplitude, expected.amplitude)
    assert record["pre_aperture_refinement_factor"] == 1
    assert record["interpolation_domain"] == "original_input_grid"
    assert record["sampling_patch_shape"] == list(wave.amplitude.shape)
    assert record["sampling_patch_spline_guard_cells"] == 0


def test_large_factor_succeeds_with_small_budget_without_global_refinement(monkeypatch):
    wave = source(shape=(128, 128), pitch=.2, sigma=1.2)
    original = wave.amplitude.copy()
    policy = AcceptanceWindowNumerics(aperture_radius_cells=32, output_halfwidth_radii=2.)
    grid = WaveGridNumerics(compute_backend="CPU", maximum_pixels=512, maximum_working_bytes=16*1024**2)
    def forbidden(*args, **kwargs):
        raise AssertionError("Receiving aperture must not allocate its virtual full refined grid")
    monkeypatch.setattr("temsim.physics.wave_grid.refine_plane_wave", forbidden)
    result, record = reconstruct(wave, radius=.02, policy=policy, grid=grid)
    assert record["pre_aperture_refinement_factor"] >= 300
    assert max(record["virtual_full_refined_shape"]) > 40000
    assert max(record["sampling_patch_shape"]) < 256
    assert result.amplitude.shape == (128, 128)
    assert result.probability > 0
    np.testing.assert_array_equal(wave.amplitude, original)
    assert record["input_unmasked_probability"] == wave.probability


def test_budget_failure_and_mid_reconstruction_cancellation_preserve_input(monkeypatch):
    wave = source()
    original = wave.amplitude.copy()
    with pytest.raises(WaveGridBudgetError):
        reconstruct(wave, grid=WaveGridNumerics(maximum_working_bytes=1))
    count = 0
    def cancelled():
        nonlocal count
        count += 1
        return count >= 4
    with pytest.raises(InterruptedError, match="cancelled"):
        reconstruct(wave, cancelled=cancelled)
    np.testing.assert_array_equal(wave.amplitude, original)


def test_unknown_outside_input_is_rejected_before_local_interpolation(monkeypatch):
    wave = source(origin=(-1., 0.))
    def forbidden(*args, **kwargs):
        raise AssertionError("Aperture outside input domain must not be interpolated")
    monkeypatch.setattr("temsim.physics.wave_aperture_sampling._sample_envelope", forbidden)
    with pytest.raises(ValueError, match="does not cover"):
        reconstruct(wave)
