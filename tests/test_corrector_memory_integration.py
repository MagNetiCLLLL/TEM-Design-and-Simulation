"""CPU tiled corrector and carrier changes preserve the same complex field."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave, reselect_phase_carrier
from temsim.physics.multipole_wave import apply_multipole_phase, multipole_action
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError


def source():
    shape = (96, 100)
    y, x = np.meshgrid(np.arange(shape[0])-shape[0]//2,
                       np.arange(shape[1])-shape[1]//2, indexing="ij")
    basis = np.array(((.5e-9, .07e-9), (-.04e-9, .55e-9)))
    u, v = basis[0, 0]*x+basis[0, 1]*y, basis[1, 0]*x+basis[1, 1]*y
    amplitude = np.exp(-(u*u/(2e-9)**2+v*v/(2.5e-9)**2)/4)
    amplitude = amplitude*(1+.04*x+.03j*y)*np.exp(1j*(.71+.002*x*x-.001*x*y))
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, basis, np.array((4e-8, -3e-8)),
                     np.array(((300., 70.), (70., -200.))), np.array((1e-4, -2e-4)))


def complete_field(wave, wavelength):
    ny, nx = wave.amplitude.shape
    y, x = np.meshgrid(np.arange(ny)-ny//2, np.arange(nx)-nx//2, indexing="ij")
    u, v = wave.basis_m[0, 0]*x+wave.basis_m[0, 1]*y, wave.basis_m[1, 0]*x+wave.basis_m[1, 1]*y
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    phase = np.pi/wavelength*(q[0, 0]*u*u+(q[0, 1]+q[1, 0])*u*v+q[1, 1]*v*v)
    phase += 2*np.pi/wavelength*(tilt[0]*u+tilt[1]*v)
    return wave.amplitude*np.exp(1j*phase), (u+wave.origin_m[0], v+wave.origin_m[1])


def test_gauge_hexapole_and_gauge_compose_without_changing_physical_phase(monkeypatch):
    import temsim.physics.multipole_wave as multipole
    import temsim.physics.canonical_phase as canonical
    initial, wavelength = source(), 2e-12
    before = initial.amplitude.copy()
    full, (x, y) = complete_field(initial, wavelength)
    strengths = dict(normal_m2=2e8, skew_m2=-3e8, spherical_m3=4e15)
    expected = full*np.exp(2j*np.pi/wavelength*multipole_action(x, y, **strengths))
    monkeypatch.setattr(multipole, "_MULTIPOLE_TILE_CELLS", 200)
    monkeypatch.setattr(canonical, "_PHASE_MOMENT_TILE_CELLS", 200)
    monkeypatch.setattr(PlaneWave, "coordinates_m", lambda self: pytest.fail("Full-grid coordinates allocated"))
    monkeypatch.setattr(np.fft, "fft2", lambda *a, **k: pytest.fail("Full-grid spectrum allocated"))
    numerics = WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=32*1024**2)
    first, first_rows = reselect_phase_carrier(initial, wavelength, grid_numerics=numerics)
    changed = apply_multipole_phase(first, wavelength, **strengths)
    result, last_rows = reselect_phase_carrier(changed, wavelength, grid_numerics=numerics)
    assert first_rows and last_rows
    assert result.amplitude.shape == initial.amplitude.shape
    actual, _ = complete_field(result, wavelength)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
    assert result.probability == pytest.approx(initial.probability, rel=2e-14)
    np.testing.assert_array_equal(initial.amplitude, before)
    assert not result.amplitude.flags.writeable


def dense_required_scale(wave, wavelength, strengths):
    _, (x, y) = complete_field(wave, wavelength)
    x0, y0 = wave.origin_m
    u, v = x-x0, y-y0
    residual = multipole_action(u, v, **strengths)
    residual -= strengths.get("spherical_m3", 0.)*(x0*u+y0*v)*(u*u+v*v)
    phase = 2*np.pi/wavelength*residual
    occupied = abs(wave.amplitude) > abs(wave.amplitude).max()*1e-8
    spectrum = abs(np.fft.fft2(wave.amplitude, norm="ortho"))**2
    largest = 0.
    for axis in (0, 1):
        neighbours = occupied[:-1] & occupied[1:] if axis == 0 else occupied[:, :-1] & occupied[:, 1:]
        added = abs(np.diff(phase, axis=axis))[neighbours].max()
        frequency = abs(2*np.pi*np.fft.fftfreq(wave.amplitude.shape[axis]))
        order = np.argsort(frequency)
        tail = np.cumsum(spectrum.sum(axis=1-axis)[order][::-1])[::-1]
        bandwidth = max(frequency[order][tail > spectrum.sum()*1e-12], default=0.)
        largest = max(largest, bandwidth+added)
    return max(1., largest/(.8*np.pi))


@pytest.mark.parametrize("strengths", [dict(normal_m2=5e12), dict(skew_m2=-5e12),
    dict(normal_m2=4e12, skew_m2=-3e12, spherical_m3=2e20)])
def test_one_row_tiles_keep_the_same_dense_sampling_rejection_scale(monkeypatch, strengths):
    import temsim.physics.multipole_wave as multipole
    initial = source()
    expected = dense_required_scale(initial, 2e-12, strengths)
    assert expected > 1.
    monkeypatch.setattr(multipole, "_MULTIPOLE_TILE_CELLS", 1)
    with pytest.raises(WaveSamplingError) as caught:
        apply_multipole_phase(initial, 2e-12, **strengths)
    assert caught.value.required_scale == pytest.approx(expected, rel=3e-13)


def test_optional_carrier_scratch_refusal_preserves_admitted_input_without_moments(monkeypatch):
    import temsim.physics.canonical_phase as canonical
    initial = source()
    # Current input/norm storage fits. Extra optional moments/rephase scratch
    # does not; its refusal must not invalidate the unchanged physical wave.
    numerics = WaveGridNumerics(maximum_working_bytes=256*initial.amplitude.size)
    numerics.check(initial.amplitude.shape)
    monkeypatch.setattr(canonical, "phase_carrier_covariance_blocks",
                        lambda *a, **k: pytest.fail("Optional covariance allocated before its scratch budget check"))
    result, rows = reselect_phase_carrier(initial, 2e-12, grid_numerics=numerics)
    assert result is initial
    assert rows and "not applied" in rows[0]["method"]
    assert str(numerics.maximum_working_bytes) in rows[0]["reason"]


def test_marginal_cancellation_is_observed_between_fft_tiles(monkeypatch):
    from scipy import fft
    import temsim.physics.multipole_wave as multipole
    initial = source()
    before = initial.amplitude.copy()
    monkeypatch.setattr(multipole, "_MULTIPOLE_TILE_CELLS", 200)
    calls, operation = [], fft.fft
    def transformed(*args, **kwargs):
        calls.append(args[0].shape)
        return operation(*args, **kwargs)
    monkeypatch.setattr(fft, "fft", transformed)
    with pytest.raises(InterruptedError):
        multipole._spectral_marginals(initial.amplitude, cancelled=lambda: bool(calls))
    assert len(calls) == 1
    np.testing.assert_array_equal(initial.amplitude, before)
