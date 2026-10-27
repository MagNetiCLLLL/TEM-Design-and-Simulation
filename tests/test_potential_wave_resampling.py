"""Complex-field projection bounds, not whole-column physical qualification."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave_resampling import (
    resample_controlled_spectrum, spectral_resampling_working_bytes,
)
from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError, refine_plane_wave


def _wave(shape=(129, 160)):
    rng = np.random.default_rng(849)
    spectrum = np.zeros(shape, complex)
    cy, cx = np.array(shape)//2
    spectrum[cy-6:cy+7, cx-9:cx+10] = (
        rng.normal(size=(13, 19))+1j*rng.normal(size=(13, 19)))
    spectrum /= np.linalg.norm(spectrum)/math.sqrt(.37)
    spectrum[3, 4] = 2e-5-3e-5j
    amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho"))
    return PlaneWave(amplitude, np.array(((2e-8, .3e-8), (-.2e-8, 3e-8))),
        np.array((7e-6, -8e-6)), np.array(((.2, -.3), (-.3, .4))), np.array((.01, -.02)))


@pytest.mark.parametrize("shape", [(128, 160), (129, 161), (129, 160), (128, 161)])
def test_projection_preserves_retained_complex_coefficients_carriers_and_physical_period(shape):
    wave = _wave(shape)
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-3, absolute_probability_budget=1e-8)
    assert output.amplitude.size < wave.amplitude.size
    for field in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, field), getattr(wave, field))
    np.testing.assert_allclose(output.basis_m@np.diag(output.amplitude.shape[::-1]),
        wave.basis_m@np.diag(shape[::-1]), rtol=2e-15, atol=0.)
    embedded = refine_plane_wave(output, shape)
    original_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave.amplitude), norm="ortho"))
    reconstructed_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(embedded.amplitude), norm="ortho"))
    mask = np.zeros(shape, bool)
    (yl, yu), (xl, xu) = receipt["retained_frequency_intervals_yx"]
    mask[shape[0]//2+yl:shape[0]//2+yu+1, shape[1]//2+xl:shape[1]//2+xu+1] = True
    np.testing.assert_allclose(reconstructed_spectrum[mask], original_spectrum[mask], atol=5e-17, rtol=2e-13)
    np.testing.assert_allclose(reconstructed_spectrum[~mask], 0., atol=3e-17, rtol=0.)
    error = np.linalg.norm(wave.amplitude-embedded.amplitude)
    assert receipt["absolute_l2_error"] == pytest.approx(error, rel=5e-11)
    assert receipt["absolute_probability_loss"] == pytest.approx(13e-10, rel=5e-11)
    assert output.probability < wave.probability
    assert receipt["probability_repair"] is False
    assert receipt["physical_absorption"] is False
    assert not output.amplitude.flags.writeable


@pytest.mark.parametrize("frequency", [(-16, 0), (16, 0), (0, -17), (0, 17), (0, 0)])
def test_signed_frequency_and_even_negative_nyquist_conventions_survive(frequency):
    ny, nx = 128, 160
    y, x = np.meshgrid(np.arange(ny)-ny//2, np.arange(nx)-nx//2, indexing="ij")
    amplitude = np.exp(2j*np.pi*(frequency[0]*y/ny+frequency[1]*x/nx))/math.sqrt(ny*nx)
    wave = replace(_wave(), amplitude=amplitude)
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-5, absolute_probability_budget=1e-9)
    assert receipt is not None
    restored = refine_plane_wave(output, (ny, nx))
    np.testing.assert_allclose(restored.amplitude, wave.amplitude, atol=4e-17, rtol=2e-13)
    for count, interval in zip(output.amplitude.shape, receipt["retained_frequency_intervals_yx"]):
        assert max(abs(interval[0]), abs(interval[1])) < .5*receipt["guard_nyquist_fraction"]*count


def test_uncropped_axis_retains_complete_negative_nyquist_band():
    rng = np.random.default_rng(77)
    spectrum = np.zeros((129, 160), complex)
    spectrum[:, 80] = rng.normal(size=129)+1j*rng.normal(size=129)
    spectrum /= np.linalg.norm(spectrum)
    amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho"))
    wave = replace(_wave(), amplitude=amplitude)
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-4, absolute_probability_budget=1e-8)
    assert output.amplitude.shape == (129, 32)
    assert receipt["retained_frequency_intervals_yx"][0] == [-64, 64]
    np.testing.assert_allclose(refine_plane_wave(output, wave.amplitude.shape).amplitude,
        wave.amplitude, atol=4e-17, rtol=2e-13)


@pytest.mark.parametrize("cap", [1e-14, 1e-25])
def test_both_absolute_error_caps_are_required_without_epsilon_budget(cap):
    wave = _wave()
    for kwargs in (dict(absolute_probability_budget=cap, absolute_l2_budget=1.),
                   dict(absolute_probability_budget=1., absolute_l2_budget=math.sqrt(cap))):
        output, receipt = resample_controlled_spectrum(wave, **kwargs)
        if receipt is None:
            assert output is wave
        else:
            assert receipt["absolute_probability_loss"] <= cap
            assert receipt["absolute_l2_error"] <= math.sqrt(cap)


def test_weak_high_frequency_does_not_reset_or_renormalize_absolute_weight():
    wave = _wave()
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-3, absolute_probability_budget=1e-8)
    assert output.probability == pytest.approx(.37, abs=2e-15)
    assert receipt["input_probability"]-receipt["absolute_probability_loss"] == pytest.approx(
        receipt["output_probability"], abs=2e-15)


def test_zero_budgets_empty_or_broad_spectrum_are_declined():
    wave = _wave()
    for l2, probability in ((0., 1e-8), (1e-3, 0.)):
        assert resample_controlled_spectrum(wave, absolute_l2_budget=l2,
            absolute_probability_budget=probability) == (wave, None)
    for amplitude in (np.zeros((128, 160), complex), np.pad([[1.+0j]], ((63, 64), (79, 80)))):
        candidate = replace(wave, amplitude=amplitude)
        assert resample_controlled_spectrum(candidate, absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8) == (candidate, None)


@pytest.mark.parametrize("name", ["absolute_l2_budget", "absolute_probability_budget", "guard_nyquist_fraction", "minimum_saved_fraction"])
@pytest.mark.parametrize("value", [True, np.bool_(False), -1., np.inf, np.nan])
def test_invalid_numerical_controls_are_rejected(name, value):
    options = dict(absolute_l2_budget=1e-3, absolute_probability_budget=1e-8)
    options[name] = value
    with pytest.raises(ValueError):
        resample_controlled_spectrum(_wave(), **options)


@pytest.mark.parametrize("guard", [0., .8, 1.])
def test_guard_stays_strictly_inside_existing_operator_guard(guard):
    with pytest.raises(ValueError, match="Nyquist guard"):
        resample_controlled_spectrum(_wave(), absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8, guard_nyquist_fraction=guard)


def test_memory_preflight_includes_external_retained_buffers_before_fft(monkeypatch):
    from temsim.physics import potential_wave_resampling as module
    wave = _wave()
    required = spectral_resampling_working_bytes(wave.amplitude.shape)
    monkeypatch.setattr(module, "_inplace_axis_fft", lambda *a, **k: pytest.fail("FFT ran before resource admission"))
    with pytest.raises(WaveGridBudgetError, match="working bytes"):
        resample_controlled_spectrum(wave, absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8,
            numerics=WaveGridNumerics(maximum_working_bytes=required), retained_bytes=1)


def test_cancellation_precedes_fft_allocation(monkeypatch):
    from temsim.physics import potential_wave_resampling as module
    wave = _wave()
    monkeypatch.setattr(module.np, "empty", lambda *a, **k: pytest.fail("Allocation ran after cancellation"))
    with pytest.raises(InterruptedError, match="resampling cancelled"):
        resample_controlled_spectrum(wave, absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8, cancelled=lambda: True)


def test_bounded_axis_fft_calls_never_request_a_full_2d_fft(monkeypatch):
    from scipy import fft
    monkeypatch.setattr(np.fft, "fft2", lambda *a, **k: pytest.fail("Dense FFT was called"))
    monkeypatch.setattr(np.fft, "ifft2", lambda *a, **k: pytest.fail("Dense FFT was called"))
    shape = (257, 320)
    y, x = np.meshgrid(np.arange(shape[0])-shape[0]//2,
        np.arange(shape[1])-shape[1]//2, indexing="ij")
    wave = PlaneWave(np.exp(-((y/35.)**2+(x/45.)**2)).astype(complex), np.eye(2), np.zeros(2))
    original = fft.fft
    calls = []
    def bounded(values, *args, **kwargs):
        calls.append(values.size)
        assert values.size <= max(65536, *shape)
        return original(values, *args, **kwargs)
    monkeypatch.setattr(fft, "fft", bounded)
    result, receipt = resample_controlled_spectrum(wave, absolute_l2_budget=.01,
        absolute_probability_budget=1e-6)
    assert receipt is not None and calls
    assert result.amplitude.size < wave.amplitude.size


def test_pressure_selects_largest_feasible_family_member_without_spending_budget():
    wave = _wave((128, 128))
    visited = []
    def admissible(shape):
        visited.append(shape)
        return shape in ((96, 96), (64, 64))
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=.01, absolute_probability_budget=1e-4,
        admissible_shape=admissible)
    assert output.amplitude.shape == (96, 96)
    assert visited[-1] == (96, 96) and (64, 64) not in visited
    assert all(a[0]*a[1] > b[0]*b[1] for a, b in zip(visited, visited[1:]))
    assert receipt["absolute_probability_loss"] == pytest.approx(1.3e-9, rel=5e-11)
    assert receipt["absolute_probability_loss"] < 1e-4*.001
    compact, _ = resample_controlled_spectrum(wave,
        absolute_l2_budget=.01, absolute_probability_budget=1e-4)
    assert compact.amplitude.size < output.amplitude.size


@pytest.mark.parametrize("shape", [(128, 160), (129, 161), (129, 160), (128, 161)])
def test_pressure_projection_preserves_complex_phase_and_parseval_for_odd_even_inputs(shape):
    wave = _wave(shape)
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-3, absolute_probability_budget=1e-8,
        admissible_shape=lambda candidate: max(candidate) <= 96)
    assert receipt is not None and max(output.amplitude.shape) <= 96
    assert output.amplitude.size <= .95*wave.amplitude.size
    assert all(count % 2 == 0 for count in output.amplitude.shape)
    restored = refine_plane_wave(output, shape)
    loss = np.linalg.norm(wave.amplitude-restored.amplitude)**2
    assert receipt["absolute_probability_loss"] == pytest.approx(loss, rel=5e-11)
    assert receipt["absolute_l2_error"] == pytest.approx(math.sqrt(loss), rel=5e-11)
    assert receipt["output_probability"] == pytest.approx(wave.probability-loss, abs=2e-15)
    np.testing.assert_allclose(output.basis_m@np.diag(output.amplitude.shape[::-1]),
        wave.basis_m@np.diag(shape[::-1]), rtol=2e-15, atol=0.)
    for name in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, name), getattr(wave, name))
    for count, interval in zip(output.amplitude.shape, receipt["retained_frequency_intervals_yx"]):
        assert max(abs(value) for value in interval) < .5*.76*count
    assert receipt["probability_repair"] is False
    assert receipt["physical_absorption"] is False


@pytest.mark.parametrize("reason", ["resource", "probability", "l2"])
def test_pressure_declines_without_mutating_original_when_no_feasible_projection(reason):
    wave = _wave((128, 128))
    original = wave.amplitude.copy()
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-7 if reason == "l2" else 1e-3,
        absolute_probability_budget=1e-14 if reason == "probability" else 1e-8,
        admissible_shape=lambda shape: reason != "resource" and max(shape) <= 96)
    assert output is wave and receipt is None
    np.testing.assert_array_equal(wave.amplitude, original)


def test_pressure_many_resource_candidates_use_only_one_forward_transform(monkeypatch):
    from temsim.physics import potential_wave_resampling as module
    wave = _wave((129, 161))
    fft = module._inplace_axis_fft
    transforms, candidates = [], []
    def transform(value, axis, **kwargs):
        transforms.append((value.shape, axis, kwargs.get("inverse", False)))
        return fft(value, axis, **kwargs)
    def admissible(shape):
        candidates.append(shape)
        return max(shape) <= 64
    monkeypatch.setattr(module, "_inplace_axis_fft", transform)
    output, receipt = resample_controlled_spectrum(wave,
        absolute_l2_budget=1e-3, absolute_probability_budget=1e-8,
        admissible_shape=admissible)
    assert receipt is not None and len(candidates) > 5
    assert [item for item in transforms if not item[2]] == [(wave.amplitude.shape, 1, False),
                                                         (wave.amplitude.shape, 0, False)]
    assert [item for item in transforms if item[2]] == [(output.amplitude.shape, 1, True),
                                                       (output.amplitude.shape, 0, True)]


@pytest.mark.parametrize("value", [None, 1, "yes", np.array([True]), {}])
def test_pressure_invalid_admission_result_is_not_silently_accepted(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        resample_controlled_spectrum(_wave(), absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8, admissible_shape=lambda _: value)


@pytest.mark.parametrize("error", [InterruptedError("External cancellation"),
                                   ValueError("Invalid resource contract")])
def test_pressure_callback_errors_propagate_without_publication(error):
    wave = _wave()
    original = wave.amplitude.copy()
    def reject(_shape):
        raise error
    with pytest.raises(type(error), match=str(error)):
        resample_controlled_spectrum(wave, absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8, admissible_shape=reject)
    np.testing.assert_array_equal(wave.amplitude, original)


def test_pressure_cancellation_between_candidates_prevents_more_checks():
    checked = []
    def reject(shape):
        checked.append(shape)
        return False
    with pytest.raises(InterruptedError, match="resampling cancelled"):
        resample_controlled_spectrum(_wave(), absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8, admissible_shape=reject,
            cancelled=lambda: bool(checked))
    assert len(checked) == 1


def test_pressure_invalid_callback_is_rejected_before_fft(monkeypatch):
    from temsim.physics import potential_wave_resampling as module
    wave = _wave()
    monkeypatch.setattr(module, "_inplace_axis_fft", lambda *a, **k: pytest.fail("Unexpected FFT"))
    with pytest.raises(ValueError, match="callable or None"):
        resample_controlled_spectrum(wave, absolute_l2_budget=1e-3,
            absolute_probability_budget=1e-8, admissible_shape=True)
