"""Small numerical operators with NumPy doubles; no finite-source wave trial."""
import math

import numpy as np
import pytest

from temsim.physics.mixed_stem_operators import (
    ScanPacket, evaluate_periodic, exact_kinetic_phase, resident_observe, rotate_wave,
)


class NumpyDevice:
    """Only the array API of the CUDA resident operator is substituted."""
    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        return getattr(np, name)


def test_exact_kinetic_phase_retains_small_phase_without_axial_cancellation():
    wavelength, distance, squared = .02, 1e8, np.array([0., 1e-12])
    phase = exact_kinetic_phase(squared, wavelength, distance)
    # The next non-paraxial series term is below 1e-21 radians here.
    oracle = np.exp(-1j * np.pi * distance * wavelength * squared)
    np.testing.assert_allclose(phase, oracle, rtol=0, atol=1e-18)
    assert phase[0] == 1 and abs(phase[1].imag) > 1e-6


def test_exact_kinetic_phase_matches_known_nonparaxial_branch():
    wavelength, distance = .25, .3
    phase = exact_kinetic_phase(np.array([.75 / wavelength**2]), wavelength, distance)
    oracle = np.exp(2j * np.pi * distance / wavelength * (.5 - 1.))
    np.testing.assert_allclose(phase, [oracle], rtol=0, atol=4e-16)


@pytest.mark.parametrize("squared", [np.array([16.]), np.array([20.]), np.array([np.nan]), np.array([np.inf])])
def test_kinetic_admission_checks_the_entire_represented_lattice(squared):
    with pytest.raises(ValueError, match="forward hemisphere"):
        exact_kinetic_phase(squared, .25, .1)


def test_three_shear_rotation_is_unitary_reversible_and_preserves_input():
    rng = np.random.default_rng(872)
    wave = rng.normal(size=(3, 8, 12)) + 1j * rng.normal(size=(3, 8, 12))
    original = wave.copy()
    rotated = rotate_wave(wave, .31, (.2, .3), xp=np)
    restored = rotate_wave(rotated, -.31, (.2, .3), xp=np)
    np.testing.assert_allclose(np.sum(abs(rotated)**2, axis=(-2, -1)),
        np.sum(abs(wave)**2, axis=(-2, -1)), rtol=2e-15, atol=0)
    np.testing.assert_allclose(restored, wave, rtol=0, atol=4e-15)
    np.testing.assert_array_equal(wave, original)
    assert rotate_wave(wave, 0., (.2, .3), xp=np) is wave


def test_periodic_detector_series_keeps_absolute_probability_and_phase_sign():
    coefficients = np.zeros((2, 3, 3), complex)
    coefficients[0, 1, 1] = .2
    coefficients[0, 1, (0, 2)] = .05
    coefficients[0, 0, 1], coefficients[0, 2, 1] = .02j, -.02j
    coefficients[1, 1, 1] = .03
    positions = np.array([[0., 0.], [.25, .5], [.7, .2]])
    periods = (1., 2.)
    expected = .2 + .1 * np.cos(2*np.pi*positions[:, 0]) + .04 * np.sin(np.pi*positions[:, 1])
    values = evaluate_periodic(coefficients, positions, periods)
    np.testing.assert_allclose(values[0], expected, rtol=0, atol=6e-17)
    np.testing.assert_allclose(values[1], .03, rtol=0, atol=0)
    np.testing.assert_allclose(evaluate_periodic(coefficients*.2, positions, periods), values*.2,
        rtol=0, atol=2e-17)


@pytest.mark.parametrize("value", [.1+.01j, -.01+0j])
def test_periodic_detector_series_rejects_nonphysical_output(value):
    with pytest.raises(ValueError, match="not real and positive"):
        evaluate_periodic(np.array([[[value]]]), np.array([[0., 0.]]), (1., 1.))


def _resident(gx=0, *, cancelled=lambda: False, field=None, diffraction=True):
    packet = ScanPacket(np.ones((1, 1), complex), np.array([.25]),
        np.array([gx]), np.array([0]), .02)
    potential, slices = np.zeros((2, 8, 8)), np.array([.5, .5])
    positions = np.array([[0., 0.], [.3, -.2]])
    originals = [value.copy() for value in (packet.modes, packet.weights, potential, slices, positions)]
    result = resident_observe(NumpyDevice(), packet, potential, slices, (8., 8.), positions,
        {"bf": (0., 15.), "adf": (50., 150.)}, interaction=.1,
        bandwidth_fraction=.5, batch_size=1, field=field, diffraction=diffraction,
        cancelled=cancelled)
    for actual, original in zip((packet.modes, packet.weights, potential, slices, positions), originals):
        np.testing.assert_array_equal(actual, original)
    return result


def test_resident_zero_potential_preserves_absolute_flux_and_diffraction_norm():
    observed, loss, diffraction = _resident()
    # Several forward/inverse FFTs accumulate double-precision roundoff.
    roundoff = 64 * np.finfo(float).eps * .25
    np.testing.assert_allclose(observed, [[.25, .25], [0., 0.]], rtol=0, atol=roundoff)
    np.testing.assert_allclose(loss, 0., rtol=0, atol=roundoff)
    assert diffraction.shape == (8, 8)
    assert diffraction.sum() == pytest.approx(.25, abs=roundoff)


def test_resident_passband_loss_is_recorded_without_restoring_current():
    observed, loss, diffraction = _resident(gx=3)
    np.testing.assert_allclose(observed, 0., rtol=0, atol=1e-30)
    np.testing.assert_allclose(loss, .25, rtol=0, atol=2e-16)
    assert diffraction.sum() == pytest.approx(0., abs=1e-30)


def test_resident_cancellation_prevents_result_publication():
    with pytest.raises(InterruptedError, match="no partial image published"):
        _resident(cancelled=lambda: True)


def test_resident_rejects_an_operator_that_creates_current():
    with pytest.raises(ValueError, match="current conservation"):
        _resident(field=lambda wave, *_: wave * 1.01)


def test_diffraction_common_grid_embeds_complex_exit_at_the_same_physical_pitch():
    gx, gy = np.array([0, 1]), np.array([0, 0])
    coefficients = np.array([[np.sqrt(.7)], [1j*np.sqrt(.3)]])
    packet = ScanPacket(coefficients, np.array([.25]), gx, gy, .02)
    positions = np.array([[0., 0.], [.3, -.2]])
    _, _, patterns = resident_observe(NumpyDevice(), packet,
        np.zeros((2, 8, 12)), np.array([.5, .5]), (12., 8.), positions,
        {"all": (0., 1000.)}, interaction=.1, bandwidth_fraction=2/3,
        batch_size=1, diffraction=True, diffraction_grid_shape=(16, 24),
        diffraction_per_position=True)
    oracle = []
    for x, y in positions:
        spectrum = np.zeros((8, 12), complex)
        phase = np.exp(-2j*np.pi*(gx*x/12+gy*y/8))
        kinetic = exact_kinetic_phase((gx/12)**2+(gy/8)**2, .02, 1.)
        spectrum[gy, gx] = coefficients[:, 0]*phase*kinetic
        exit_wave = np.fft.ifft2(spectrum, norm="ortho")
        extended = np.zeros((16, 24), complex)
        extended[4:12, 6:18] = exit_wave
        oracle.append(np.fft.fftshift(abs(np.fft.fft2(extended, norm="ortho"))**2)*.25)
    np.testing.assert_allclose(patterns, oracle, rtol=2e-14, atol=2e-17)
    np.testing.assert_allclose(patterns.sum(axis=(-2, -1)), .25, rtol=0, atol=4e-16)
    # Padding native reciprocal intensities gives neither the common q pitch
    # nor the physical sinc structure of the embedded finite exit field.
    native = abs(np.fft.fft2(exit_wave, norm="ortho"))**2*.25
    wrong = np.pad(np.fft.fftshift(native), ((4, 4), (6, 6)))
    assert np.sum(abs(patterns[-1]-wrong)) > .1


@pytest.mark.parametrize("shape", [(6, 8), (9, 10)])
def test_diffraction_embedding_rejects_cropped_or_half_cell_shifted_domains(shape):
    packet = ScanPacket(np.ones((1, 1), complex), np.array([.2]), np.array([0]), np.array([0]), .02)
    with pytest.raises(ValueError, match="containing, centre-aligned"):
        resident_observe(NumpyDevice(), packet, np.zeros((1, 8, 8)), np.array([1.]),
            (8., 8.), np.array([[0., 0.]]), {"all": (0., 1000.)},
            interaction=.1, bandwidth_fraction=2/3, batch_size=1,
            diffraction=True, diffraction_grid_shape=shape)
