"""Numerical-tail error accounting; no claim of full-column qualification."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_support import compact_wave_support, compact_wave_spectrum


def _wave():
    yy, xx = np.indices((129, 160))
    amplitude = np.exp(-((xx-108.)**2/32.+(yy-39.)**2/50.))
    amplitude = amplitude.astype(complex)*np.exp(.013j*xx+.021j*yy)
    amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
    return PlaneWave(amplitude, np.array(((2e-9, .3e-9), (-.4e-9, 3e-9))),
                     np.array((7e-6, -9e-6)), np.array(((30., 7.), (7., -20.))),
                     np.array((2e-6, -1e-6)))


def _full_on_native_grid(wave, wavelength):
    xy = wave.coordinates_m()-wave.origin_m[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    action = np.einsum("iyx,ij,jyx->yx", xy, curvature, xy)/2+np.einsum("i,iyx->yx", tilt, xy)
    return wave.amplitude*np.exp(2j*np.pi*action/wavelength)


def test_off_axis_affine_crop_preserves_full_carrier_and_accounts_exact_error():
    wave = _wave()
    result, record = compact_wave_support(wave, maximum_loss_fraction=1e-6, wavelength_m=2e-12)
    (y0, y1), (x0, x1) = record["crop_indices_yx"]
    assert all(size >= 32 and size % 2 == 0 for size in result.amplitude.shape)
    assert result.amplitude.size <= .75*wave.amplitude.size
    np.testing.assert_array_equal(result.basis_m, wave.basis_m)
    np.testing.assert_allclose(result.coordinates_m(), wave.coordinates_m()[:, y0:y1, x0:x1], atol=2e-21)
    old_full, new_full = _full_on_native_grid(wave, 2e-12), _full_on_native_grid(result, 2e-12)
    ratio = new_full/old_full[y0:y1, x0:x1]
    np.testing.assert_allclose(ratio.imag, 0., atol=4e-13)
    assert ratio.real.min() >= -1e-14 and ratio.real.max() <= 1.+1e-14
    (cy0, cy1), (cx0, cx1) = record["core_indices_yx_inclusive"]
    np.testing.assert_allclose(ratio[cy0-y0:cy1-y0+1, cx0-x0:cx1-x0+1], 1., atol=4e-13)
    embedded = np.zeros_like(old_full)
    embedded[y0:y1, x0:x1] = new_full
    actual_error = np.linalg.norm(embedded-old_full)/np.linalg.norm(old_full)
    assert record["relative_l2_error"] == pytest.approx(actual_error, rel=2e-8)
    assert wave.probability-result.probability == pytest.approx(record["absolute_probability_loss"], abs=2e-16)
    assert 0 < record["loss_fraction"] <= 1e-6
    assert result.probability < wave.probability
    json.dumps(record, allow_nan=False)


def test_disabled_empty_and_broad_fields_are_not_changed():
    wave = _wave()
    assert compact_wave_support(wave, maximum_loss_fraction=0, wavelength_m=2e-12) == (wave, None)
    for amplitude in (np.zeros((64, 64), complex), np.ones((64, 64), complex)/64):
        field = replace(wave, amplitude=amplitude)
        result, record = compact_wave_support(field, maximum_loss_fraction=1e-6, wavelength_m=2e-12)
        assert result is field and record is None


@pytest.mark.parametrize("tail", [(64, 127), (127, 64), (64, 0), (0, 64)])
@pytest.mark.parametrize("spectral", [False, True])
def test_sub_roundoff_budget_cannot_discard_a_resolved_small_tail(tail, spectral):
    data = np.zeros((128, 128), complex)
    data[64, 64], data[tail] = 1., 1e-11
    if spectral:
        data = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data), norm="ortho"))
    wave = PlaneWave(data, np.eye(2)*1e-9, np.zeros(2))
    budget = 1e-25
    if spectral:
        result, record = compact_wave_spectrum(wave, maximum_loss_fraction=budget)
    else:
        result, record = compact_wave_support(wave, maximum_loss_fraction=budget, wavelength_m=2e-12)
    if record is None:
        assert result is wave
    else:
        assert record["loss_fraction"] <= budget
        assert record["absolute_probability_loss"] <= budget*record["input_probability"]
        assert record["absolute_l2_error"] <= math.sqrt(budget*record["input_probability"])


@pytest.mark.parametrize("fraction", [-1., .11, float("nan"), float("inf"), True])
def test_invalid_loss_budget_is_rejected(fraction):
    with pytest.raises(ValueError, match="loss fraction"):
        compact_wave_support(_wave(), maximum_loss_fraction=fraction, wavelength_m=2e-12)


def test_cupy_compaction_matches_cpu_if_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA is unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA is unavailable")
    wave = _wave()
    expected, expected_record = compact_wave_support(wave, maximum_loss_fraction=1e-6, wavelength_m=2e-12)
    result, record = compact_wave_support(replace(wave, amplitude=cp.asarray(wave.amplitude)),
                                         maximum_loss_fraction=1e-6, wavelength_m=2e-12)
    assert isinstance(result.amplitude, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(result.amplitude), expected.amplitude, rtol=2e-13, atol=1e-16)
    assert record["crop_indices_yx"] == expected_record["crop_indices_yx"]
    assert record["relative_l2_error"] == pytest.approx(expected_record["relative_l2_error"], rel=1e-12)


def _spectral_wave():
    from temsim.physics.wave_grid import refine_plane_wave
    rng = np.random.default_rng(97)
    small = _wave()
    amplitude = rng.normal(size=(32, 48))+1j*rng.normal(size=(32, 48))
    amplitude /= np.linalg.norm(amplitude)
    large = refine_plane_wave(replace(small, amplitude=amplitude), (128, 160))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(large.amplitude), norm="ortho"))
    spectrum[3, 4] = 1e-5+0j
    amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho"))
    return replace(large, amplitude=amplitude)


def test_spectral_projection_inverse_embedding_measures_exact_complex_error():
    from temsim.physics.wave_grid import refine_plane_wave
    wave = _spectral_wave()
    result, record = compact_wave_spectrum(wave, maximum_loss_fraction=1e-8)
    assert result.amplitude.shape == (40, 60)
    for name in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(wave, name), getattr(result, name))
    np.testing.assert_allclose(result.basis_m@np.diag(result.amplitude.shape[::-1]),
                               wave.basis_m@np.diag(wave.amplitude.shape[::-1]), rtol=2e-15)
    embedded = refine_plane_wave(result, wave.amplitude.shape)
    difference = np.linalg.norm(embedded.amplitude-wave.amplitude)
    assert difference == pytest.approx(1e-5, rel=2e-10)
    assert record["absolute_l2_error"] == pytest.approx(difference, rel=2e-10)
    assert record["absolute_probability_loss"] == pytest.approx(1e-10, rel=2e-10)
    assert result.probability == pytest.approx(1., abs=2e-15)
    assert wave.probability > result.probability
    json.dumps(record, allow_nan=False)


def test_spectral_negative_nyquist_sign_survives_projection():
    from temsim.physics.wave_grid import refine_plane_wave
    n = 128
    axis = np.arange(n)-n//2
    amplitude = np.tile(np.exp(-2j*np.pi*16*axis/n), (n, 1))/n
    wave = replace(_wave(), amplitude=amplitude)
    result, record = compact_wave_spectrum(wave, maximum_loss_fraction=1e-8)
    assert result.amplitude.shape == (32, 40)
    embedded = refine_plane_wave(result, (n, n))
    np.testing.assert_allclose(embedded.amplitude, wave.amplitude, atol=1e-16)
    assert record["absolute_probability_loss"] < 1e-27


def test_spectral_disabled_empty_and_broad_fields_are_unchanged():
    wave = _spectral_wave()
    result, record = compact_wave_spectrum(wave, maximum_loss_fraction=0)
    assert result is wave and record is None
    for amplitude in (np.zeros((64, 64), complex), np.pad([[1.+0j]], ((31, 32), (31, 32)))):
        field = replace(wave, amplitude=amplitude)
        result, record = compact_wave_spectrum(field, maximum_loss_fraction=1e-8)
        assert result is field and record is None


def test_cupy_spectral_projection_matches_cpu_if_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA is unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA is unavailable")
    wave = _spectral_wave()
    expected, expected_record = compact_wave_spectrum(wave, maximum_loss_fraction=1e-8)
    result, record = compact_wave_spectrum(replace(wave, amplitude=cp.asarray(wave.amplitude)),
                                          maximum_loss_fraction=1e-8)
    np.testing.assert_allclose(cp.asnumpy(result.amplitude), expected.amplitude, rtol=2e-13, atol=1e-16)
    assert record["crop_indices_yx"] == expected_record["crop_indices_yx"]
    assert record["absolute_probability_loss"] == pytest.approx(expected_record["absolute_probability_loss"], rel=2e-10)
