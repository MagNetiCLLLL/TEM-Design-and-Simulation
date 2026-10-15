"""Independent full-spectrum reference for bounded phase-carrier moments."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import solve_sylvester

from temsim.physics.canonical_phase import (
    phase_carrier_covariance_blocks, phase_carrier_moments_working_bytes,
)
from temsim.physics.multiplane_wave import PlaneWave


def _wave(shape, *, seed=519):
    rng = np.random.default_rng(seed)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude,
        np.array(((2e-9, .3e-9), (-.2e-9, 1.4e-9))),
        np.array((4e-8, -3e-8)),
        np.array(((350., 80.), (80., -230.))),
        np.array((1e-4, -2e-4)))


def _full_covariance(wave, wavelength):
    """Original full 2-D Fourier derivative and all four moment vectors."""
    amplitude = wave.amplitude
    ny, nx = amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T,
                          np.stack((fx, fy)))
    xy = wave.coordinates_m()-wave.origin_m[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    momentum = (wavelength*np.fft.ifft2(frequency*np.fft.fft2(amplitude), axes=(-2, -1))
        +(np.einsum("ij,jyx->iyx", curvature, xy)+tilt[:, None, None])*amplitude)
    vectors = np.concatenate((xy*amplitude, momentum)).reshape(4, -1)
    norm = np.sum(abs(amplitude)**2)
    mean = np.real(vectors@amplitude.conj().ravel())/norm
    return np.real(vectors.conj()@vectors.T)/norm-np.outer(mean, mean)


@pytest.mark.parametrize("shape", [(2, 5), (31, 34), (32, 33), (65, 47)])
def test_blocks_match_independent_complete_affine_fourier_covariance(shape):
    wave = _wave(shape)
    expected = _full_covariance(wave, 2e-12)
    xx, xp = phase_carrier_covariance_blocks(wave, 2e-12)
    np.testing.assert_allclose(xx, expected[:2, :2], rtol=3e-12, atol=1e-28)
    np.testing.assert_allclose(xp, expected[:2, 2:], rtol=3e-12, atol=1e-25)


def test_single_line_tiles_keep_all_bins_and_leave_input_immutable(monkeypatch):
    import scipy.fft
    import temsim.physics.canonical_phase as module
    wave = _wave((37, 41))
    expected = _full_covariance(wave, 2e-12)
    before = wave.amplitude.copy()
    calls, fft = [], scipy.fft.fft
    def bounded_fft(values, **kwargs):
        calls.append((values.shape, kwargs["axis"]))
        assert values.size <= max(wave.amplitude.shape)
        assert values.dtype == np.complex128
        return fft(values, **kwargs)
    monkeypatch.setattr(module, "_PHASE_MOMENT_TILE_CELLS", 1)
    monkeypatch.setattr(scipy.fft, "fft", bounded_fft)
    monkeypatch.setattr(np.fft, "fft2", lambda *a, **k: pytest.fail("full spectrum allocation"))
    monkeypatch.setattr(PlaneWave, "coordinates_m", lambda self: pytest.fail("full coordinates allocation"))
    xx, xp = phase_carrier_covariance_blocks(wave, 2e-12)
    assert {axis for _, axis in calls} == {0, 1}
    np.testing.assert_allclose(xx, expected[:2, :2], rtol=3e-12, atol=1e-28)
    np.testing.assert_allclose(xp, expected[:2, 2:], rtol=3e-12, atol=1e-25)
    np.testing.assert_array_equal(wave.amplitude, before)
    assert not wave.amplitude.flags.writeable


def test_translation_and_constant_tilt_cancel_without_changing_covariance():
    wave = _wave((31, 38))
    changed = replace(wave, origin_m=np.array((1e8, -2e8)),
                      tilt_rad=np.array((3e4, -4e4)))
    expected = phase_carrier_covariance_blocks(wave, 2e-12)
    actual = phase_carrier_covariance_blocks(changed, 2e-12)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("axis", [0, 1])
def test_even_negative_nyquist_bin_uses_original_fourier_sign(axis):
    shape = (32, 40)
    y, x = np.meshgrid(np.arange(shape[0])-shape[0]//2,
                       np.arange(shape[1])-shape[1]//2, indexing="ij")
    index = y if axis == 0 else x
    amplitude = (1+.2*np.cos(2*np.pi*x/shape[1])+.1j*np.sin(2*np.pi*y/shape[0]))
    amplitude *= np.exp(-1j*np.pi*index)
    wave = replace(_wave(shape), amplitude=amplitude)
    expected = _full_covariance(wave, 2e-12)
    xx, xp = phase_carrier_covariance_blocks(wave, 2e-12)
    np.testing.assert_allclose(xx, expected[:2, :2], rtol=3e-12, atol=1e-28)
    np.testing.assert_allclose(xp, expected[:2, 2:], rtol=3e-12, atol=1e-25)


def test_selected_carrier_matches_full_moments_and_retains_absolute_complex_phase():
    shape = (63, 68)
    y, x = np.meshgrid(np.arange(shape[0])-shape[0]//2,
                       np.arange(shape[1])-shape[1]//2, indexing="ij")
    amplitude = np.exp(-(x*x+y*y)/80)*np.exp(1j*(.67+.001*x*x-.002*x*y))
    wave = replace(_wave(shape), amplitude=amplitude)
    wavelength = 2e-12
    reference = _full_covariance(wave, wavelength)
    expected = solve_sylvester(reference[:2, :2], reference[:2, :2],
                               reference[:2, 2:]+reference[:2, 2:].T)
    xx, xp = phase_carrier_covariance_blocks(wave, wavelength)
    selected = solve_sylvester(xx, xx, xp+xp.T)
    np.testing.assert_allclose(selected, expected, rtol=3e-12, atol=1e-10)
    local = wave.coordinates_m()-wave.origin_m[:, None, None]
    def full_phase(curvature):
        return (np.pi/wavelength*np.einsum("iyx,ij,jyx->yx", local, curvature, local)
                +2*np.pi/wavelength*np.einsum("i,iyx->yx", wave.tilt_rad, local))
    compensation = -np.pi/wavelength*np.einsum(
        "iyx,ij,jyx->yx", local, selected-wave.curvature_m1, local)
    actual = wave.amplitude*np.exp(1j*compensation)*np.exp(1j*full_phase(selected))
    expected_field = wave.amplitude*np.exp(1j*full_phase(wave.curvature_m1))
    np.testing.assert_allclose(actual, expected_field, rtol=2e-14, atol=2e-15)


def test_cancellation_does_not_mutate_input_or_start_unbounded_work(monkeypatch):
    import temsim.physics.canonical_phase as module
    wave = _wave((32, 33))
    before = wave.amplitude.copy()
    monkeypatch.setattr(module, "_PHASE_MOMENT_TILE_CELLS", 1)
    count = 0
    def cancelled():
        nonlocal count
        count += 1
        return count > 4
    with pytest.raises(InterruptedError, match="moments cancelled"):
        phase_carrier_covariance_blocks(wave, 2e-12, cancelled=cancelled)
    np.testing.assert_array_equal(wave.amplitude, before)


def test_invalid_wavelength_and_empty_field_are_rejected():
    wave = _wave((8, 9))
    for wavelength in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match="positive finite wavelength"):
            phase_carrier_covariance_blocks(wave, wavelength)
    with pytest.raises(ValueError, match="empty wave"):
        phase_carrier_covariance_blocks(replace(wave, amplitude=wave.amplitude*0), 2e-12)


def test_large_grid_estimate_reserves_bounded_scratch_without_allocating_grid():
    # 4096² needs no coordinate vectors or two full momentum components.
    assert phase_carrier_moments_working_bytes((4096, 4096)) < 1024**3


def test_cuda_blocks_match_independent_cpu_covariance(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("CUDA is unavailable")
    import temsim.physics.canonical_phase as module
    from temsim.physics.wave_device import device_scope
    source = _wave((31, 34))
    expected = _full_covariance(source, 2e-12)
    monkeypatch.setattr(module, "_PHASE_MOMENT_TILE_CELLS", 79)
    with device_scope("Require GPU", maximum_working_bytes=64*1024**2):
        device = replace(source, amplitude=cp.asarray(source.amplitude))
        before = cp.asnumpy(device.amplitude)
        xx, xp = phase_carrier_covariance_blocks(device, 2e-12)
        assert isinstance(xx, np.ndarray) and isinstance(xp, np.ndarray)
        np.testing.assert_allclose(xx, expected[:2, :2], rtol=3e-11, atol=1e-28)
        np.testing.assert_allclose(xp, expected[:2, 2:], rtol=3e-11, atol=1e-25)
        np.testing.assert_array_equal(cp.asnumpy(device.amplitude), before)
