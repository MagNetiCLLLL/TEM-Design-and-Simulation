"""Analytic independent basis, ray/wave convention and real GPU regression."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import hbar
from scipy.spatial.transform import Rotation

from temsim.optics.aberration_basis import WAVE_TERMS, cartesian_coefficients, polynomial_basis
from temsim.physics.intrinsic_aberrations import (
    FrozenIntrinsicAberrationKick, apply_intrinsic_wave, polynomial_jet,
)
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import _momentum_velocity


P = float(_momentum_velocity(300000.)[0])
WAVELENGTH = 2*np.pi*hbar/P


def screen(values=None, tilt=0., azimuth=0., offset=(0., 0.)):
    rotation = Rotation.from_euler("zy", (azimuth, tilt)).as_matrix()
    return FrozenIntrinsicAberrationKick("test", 0., .01,
        tuple((values or {}).items()), CoordinateRegistration((*offset, .5), tuple(map(tuple, rotation))))


def packet():
    axis = (np.arange(64)-32)*5e-9
    x, y = np.meshgrid(axis, axis)
    amplitude = np.exp(-(x*x+y*y)/(2*(25e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.eye(2)*5e-9, np.array((30e-6, 2e-6)),
                     np.zeros((2, 2)), np.array((.00002, -.00001)))


def apply(wave, kick, **kwargs):
    return apply_intrinsic_wave(wave, WAVELENGTH, kick,
        momentum_kg_m_s=P, reference_momentum_kg_m_s=P, **kwargs)


@pytest.mark.parametrize("term", [term for term in WAVE_TERMS if term.name != "C3"], ids=lambda t: t.name)
def test_each_local_term_matches_independent_angular_wave_basis_and_derivative(term):
    values = {term.field: 2.5}
    if term.azimuth_field:
        values[term.azimuth_field] = 23.
    kick = screen(values)
    _, _, _, terms = kick.geometry()
    all_values = {name: 0. for t in WAVE_TERMS for name in (t.field, t.azimuth_field) if name}
    all_values.update(values)
    coefficients = cartesian_coefficients(SimpleNamespace(**all_values))
    xy = np.array(((.02, .013), (-.03, .021), (.008, -.015)))*kick.focal_m
    action, gradient, hessian = polynomial_jet(terms, *xy.T)
    expected = -polynomial_basis(xy/kick.focal_m)@coefficients*1e-3
    expected_gradient = -(polynomial_basis(xy/kick.focal_m, gradient=True)@coefficients)*1e-3/kick.focal_m
    np.testing.assert_allclose(action, expected, rtol=2e-14, atol=1e-24)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-14, atol=1e-24)
    step = 1e-9
    for axis in range(2):
        high, low = xy.copy(), xy.copy()
        high[:, axis] += step; low[:, axis] -= step
        numerical = (polynomial_jet(terms, *high.T)[1]-polynomial_jet(terms, *low.T)[1])/(2*step)
        np.testing.assert_allclose(hessian[..., axis], numerical, rtol=2e-8, atol=1e-16)


def test_zero_coefficients_are_identity_even_outside_tilt_domain():
    kick, wave = screen(tilt=.2), packet()
    actual, record = apply(wave, kick)
    assert actual is wave and record["split_steps"] == 0
    values = (np.array((0.,)),)*4
    assert all(first is second for first, second in zip(values, kick.apply_canonical(*values, momentum_kg_m_s=P)))


def test_c3_rejected_to_prevent_duplicate_legacy_cs():
    with pytest.raises(ValueError, match="Unknown intrinsic"):
        screen({"c3_mm": 1.})


def test_aligned_ray_kick_equals_wave_phase_gradient_with_offset_and_azimuth():
    values = {"a1_mm": .002, "a1_azimuth_deg": 13., "b2_mm": .3,
              "b2_azimuth_deg": -7., "s3_mm": 2., "s3_azimuth_deg": 35., "c5_mm": 100.}
    kick = screen(values, azimuth=.3, offset=(2e-6, -3e-6))
    wave = packet()
    actual, _ = apply(wave, kick)
    centre, _, _, terms = kick.geometry()
    xy = wave.coordinates_m()
    action, _, _ = polynomial_jet(terms, xy[0]-centre[0], xy[1]-centre[1])
    expected = wave.full_amplitude(WAVELENGTH)*np.exp(2j*np.pi*action/WAVELENGTH)
    np.testing.assert_allclose(actual.full_amplitude(WAVELENGTH), expected, rtol=3e-12, atol=1e-14)
    x, y = wave.origin_m
    _, gradient, _ = polynomial_jet(terms, x-centre[0], y-centre[1])
    ray = kick.apply_canonical(np.array((x, np.nan)), np.array((.002, 0.)),
        np.array((y, 0.)), np.array((-.001, 0.)), momentum_kg_m_s=P)
    assert ray[1][0]-.002 == pytest.approx(gradient[0], rel=1e-12, abs=1e-18)
    assert ray[3][0]+.001 == pytest.approx(gradient[1], rel=1e-12, abs=1e-18)
    np.testing.assert_allclose(actual.tilt_rad-wave.tilt_rad, gradient, rtol=1e-12, atol=1e-18)
    assert np.isnan(ray[0][1])


def test_tilted_action_is_nontrivial_lossless_and_uses_same_energy_reference():
    wave, kick = packet(), screen({"b2_mm": 10., "a2_mm": 2., "a2_azimuth_deg": 25.}, tilt=.001)
    result, record = apply(wave, kick, aligned_bz_t=.01)
    aligned, _ = apply(wave, screen(dict(kick.coefficients)))
    assert np.linalg.norm(result.full_amplitude(WAVELENGTH)-aligned.full_amplitude(WAVELENGTH)) > 1e-5
    assert result.probability == pytest.approx(wave.probability, rel=2e-11)
    assert record["split_error_estimate"] <= 5e-11
    ratio = .8
    changed, _ = apply_intrinsic_wave(replace(wave, tilt_rad=wave.tilt_rad/ratio), WAVELENGTH/ratio, kick,
        momentum_kg_m_s=P, reference_momentum_kg_m_s=P*ratio, aligned_bz_t=.01)
    np.testing.assert_allclose(changed.full_amplitude(WAVELENGTH/ratio), result.full_amplitude(WAVELENGTH), rtol=3e-10, atol=1e-13)
    x, y = (np.array((value,)) for value in wave.origin_m)
    rays = kick.apply_canonical(x, np.array((.002,)), y, np.array((-.001,)), momentum_kg_m_s=P,
        aligned_bz_t=.01)
    changed_rays = kick.apply_canonical(x, np.array((.002,)), y, np.array((-.001,)), momentum_kg_m_s=P,
        reference_momentum_kg_m_s=P*ratio, aligned_bz_t=.01)
    for first, second in zip(rays, changed_rays):
        np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-17)
    assert abs(rays[0][0]-x[0]) > 1e-14


def test_invalid_tilt_and_sampling_fail_without_mutating_wave():
    from temsim.physics.wave_grid import WaveSamplingError
    wave = packet()
    before = wave.amplitude.copy()
    with pytest.raises(ValueError, match="10 mrad"):
        apply(wave, screen({"b2_mm": 1.}, tilt=.02))
    with pytest.raises(ValueError, match="phase remainder"):
        apply(wave, screen({"b2_mm": 10.}, tilt=.001), maximum_obliquity_phase_error_rad=1e-20)
    with pytest.raises(WaveSamplingError):
        apply(wave, screen({"b2_mm": 1e12}))
    with pytest.raises(InterruptedError):
        apply(wave, screen({"b2_mm": 1.}), cancelled=lambda: True)
    np.testing.assert_array_equal(wave.amplitude, before)


def test_real_cuda_intrinsic_action_matches_cpu_complex_field():
    from test_wave_device import _cuda
    cp = _cuda()
    wave, kick = packet(), screen({"b2_mm": 10., "a2_mm": 2., "a2_azimuth_deg": 25.}, tilt=.001)
    cpu, _ = apply(wave, kick, aligned_bz_t=.01)
    gpu, record = apply(replace(wave, amplitude=cp.asarray(wave.amplitude)), kick, aligned_bz_t=.01)
    np.testing.assert_allclose(cp.asnumpy(gpu.amplitude), cpu.amplitude, rtol=3e-10, atol=1e-13)
    np.testing.assert_allclose(gpu.tilt_rad, cpu.tilt_rad, rtol=1e-13)
    assert record["compute_backend"] == "cupy"
    assert gpu.probability == pytest.approx(wave.probability, rel=2e-11)


def test_tilted_wave_centroid_matches_same_hamiltonian_ray_ensemble():
    """Ehrenfest check for a narrow packet, not phase code tested against itself."""
    wave = packet()
    kick = screen({"b2_mm": 10., "a2_mm": 2., "a2_azimuth_deg": 25.}, tilt=.003)
    actual, _ = apply(wave, kick)
    xy = wave.coordinates_m().reshape(2, -1)
    weights = abs(wave.amplitude.ravel())**2
    slopes = np.broadcast_to(wave.tilt_rad[:, None], xy.shape)
    ray = kick.apply_canonical(xy[0], slopes[0], xy[1], slopes[1], momentum_kg_m_s=P)
    expected_xy = np.array((np.dot(ray[0], weights), np.dot(ray[2], weights)))/weights.sum()
    actual_weights = abs(actual.amplitude)**2
    actual_xy = np.sum(actual.coordinates_m()*actual_weights[None], axis=(1, 2))/actual.probability
    np.testing.assert_allclose(actual_xy, expected_xy, rtol=0., atol=2e-17)
    # Momentum from FFT of the envelope plus its stored analytic carrier.
    spectrum = abs(np.fft.fft2(actual.amplitude, norm="ortho"))**2
    frequency_y = np.fft.fftfreq(actual.amplitude.shape[0])
    frequency_x = np.fft.fftfreq(actual.amplitude.shape[1])
    canonical_bins = np.array((np.sum(spectrum*frequency_x[None]), np.sum(spectrum*frequency_y[:, None])))/spectrum.sum()
    actual_theta = (WAVELENGTH*np.linalg.inv(actual.basis_m).T@canonical_bins
        +actual.tilt_rad+actual.curvature_m1@(actual_xy-actual.origin_m))
    expected_theta = np.array((np.dot(ray[1], weights), np.dot(ray[3], weights)))/weights.sum()
    np.testing.assert_allclose(actual_theta, expected_theta, rtol=0., atol=2e-12)


def test_capture_skips_disabled_mapped_and_zero_and_does_not_change_strength(monkeypatch):
    from temsim.physics.intrinsic_aberrations import capture_intrinsic_kicks
    import temsim.physics.lens_field_provider as provider
    import temsim.optics.lens_focal_length as focal
    import temsim.lens_pose as pose
    monkeypatch.setattr(provider, "active_mapped_providers", lambda state: (SimpleNamespace(lens_key="mapped"),))
    monkeypatch.setattr(focal, "focal_length_mm", lambda lens, voltage: 10.)
    monkeypatch.setattr(pose, "lens_pose_registration", lambda state, key: CoordinateRegistration())
    state = SimpleNamespace(beam_voltage_kv=300., lenses=[SimpleNamespace(key=key, enabled=key!="disabled", z_mm=5., excitation=90.)
        for key in ("active", "zero", "mapped", "disabled")],
        lens_intrinsic_aberrations={key: {"b2_mm": 1.} for key in ("active", "mapped", "disabled")})
    actual, = capture_intrinsic_kicks(state, 0., 10.)
    assert actual.lens_key == "active" and actual.z_mm == 5.
    assert all(lens.excitation == 90. for lens in state.lenses)
    assert all(key != "c3_mm" for key, _ in actual.coefficients)
    assert capture_intrinsic_kicks(state, 6., 10.) == ()
