"""Independent gradient and complex-field checks, not full column acceptance."""
import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.multipole_wave import apply_multipole_phase, multipole_action


def wave():
    axis = np.arange(128)-64
    x, y = np.meshgrid(axis, axis)
    amplitude = np.exp(-(x*x+y*y)/(4*8**2)).astype(complex)
    amplitude /= np.sqrt(np.sum(abs(amplitude)**2))
    return PlaneWave(amplitude, np.array(((2e-9, .2e-9), (-.1e-9, 2e-9))),
                     np.array((4e-8, -3e-8)), np.array(((3e2, 1e2), (1e2, -2e2))), np.array((1e-4, -2e-4)))


@pytest.mark.parametrize("strengths", [dict(normal_m2=2e8), dict(skew_m2=-3e8),
                                      dict(spherical_m3=4e15),
                                      dict(normal_m2=2e8, skew_m2=-3e8, spherical_m3=4e15)])
def test_factored_carriers_equal_the_complete_nonlinear_complex_phase(strengths):
    source = wave()
    wavelength = 2e-12
    result = apply_multipole_phase(source, wavelength, **strengths)
    x, y = source.coordinates_m()
    expected = source.full_amplitude(wavelength)*np.exp(2j*np.pi*multipole_action(x, y, **strengths)/wavelength)
    np.testing.assert_allclose(result.full_amplitude(wavelength), expected, atol=2e-13, rtol=2e-12)
    assert result.probability == pytest.approx(source.probability, abs=1e-14)


def test_phase_gradient_has_the_existing_ray_kick_signs_and_axes():
    x, y, delta = 2e-5, -1e-5, 1e-10
    hn, hs, cs = 2e5, -3e5, 4e12
    kw = dict(normal_m2=hn, skew_m2=hs, spherical_m3=cs)
    gx = (multipole_action(x+delta, y, **kw)-multipole_action(x-delta, y, **kw))/(2*delta)
    gy = (multipole_action(x, y+delta, **kw)-multipole_action(x, y-delta, **kw))/(2*delta)
    expected_x = -hn*(x*x-y*y)-hs*2*x*y-cs*(x*x+y*y)*x
    expected_y = hn*2*x*y-hs*(x*x-y*y)-cs*(x*x+y*y)*y
    np.testing.assert_allclose((gx, gy), (expected_x, expected_y), rtol=1e-9)


def test_nonlinear_phase_cannot_alias_or_change_a_zero_strength_wave():
    source = wave()
    assert apply_multipole_phase(source, 2e-12) is source
    with pytest.raises(ValueError, match="undersampled"):
        apply_multipole_phase(source, 2e-12, spherical_m3=1e26)


def test_small_tiles_preserve_full_phase_without_whole_coordinate_arrays(monkeypatch):
    import temsim.physics.multipole_wave as module
    source = wave()
    wavelength = 2e-12
    strengths = dict(normal_m2=2e8, skew_m2=-3e8, spherical_m3=4e15)
    x, y = source.coordinates_m()
    expected = source.full_amplitude(wavelength)*np.exp(
        2j*np.pi*multipole_action(x, y, **strengths)/wavelength)
    monkeypatch.setattr(module, "_MULTIPOLE_TILE_CELLS", 128*3)
    with monkeypatch.context() as context:
        context.setattr(PlaneWave, "coordinates_m", lambda self: pytest.fail("full coordinate allocation"))
        result = apply_multipole_phase(source, wavelength, **strengths)
    np.testing.assert_allclose(result.full_amplitude(wavelength), expected, atol=2e-13, rtol=2e-12)
    assert result.probability == pytest.approx(source.probability, abs=1e-14)


def test_one_row_tiles_still_detect_vertical_phase_aliasing(monkeypatch):
    import temsim.physics.multipole_wave as module
    from temsim.physics.wave_grid import WaveSamplingError
    monkeypatch.setattr(module, "_MULTIPOLE_TILE_CELLS", 1)
    # All rows are tile boundaries, including the entire occupied centre.
    with pytest.raises(WaveSamplingError, match="undersampled"):
        apply_multipole_phase(wave(), 2e-12, skew_m2=1e26)


@pytest.mark.parametrize("shape", [(7, 9), (8, 10), (31, 32), (32, 31), (64, 96)])
def test_tiled_parseval_marginals_equal_full_two_dimensional_spectrum(monkeypatch, shape):
    import temsim.physics.multipole_wave as module
    random = np.random.default_rng(872)
    amplitude = random.normal(size=shape)+1j*random.normal(size=shape)
    spectrum = abs(np.fft.fft2(amplitude, norm="ortho"))**2
    monkeypatch.setattr(module, "_MULTIPOLE_TILE_CELLS", 17)
    actual = module._spectral_marginals(amplitude)
    for axis in (0, 1):
        np.testing.assert_allclose(actual[axis], spectrum.sum(axis=1-axis), rtol=3e-15, atol=5e-14)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("relative_power", [.5e-12, 2e-12])
def test_parseval_tail_admission_preserves_both_sides_of_threshold(axis, relative_power):
    from temsim.physics.multipole_wave import _spectral_marginals
    shape = (32, 34)
    spectral = np.zeros(shape, complex)
    spectral[0, 0] = 1.
    index = [0, 0]
    index[axis] = shape[axis]//2  # Include the negative Nyquist bin.
    spectral[tuple(index)] = np.sqrt(relative_power)
    amplitude = np.fft.ifft2(spectral, norm="ortho")
    marginal = _spectral_marginals(amplitude)[axis]
    frequencies = abs(2*np.pi*np.fft.fftfreq(shape[axis]))
    order = np.argsort(frequencies)
    tail = np.cumsum(marginal[order][::-1])[::-1]
    support = frequencies[order][tail > marginal.sum()*1e-12]
    assert max(support) == (np.pi if relative_power > 1e-12 else 0.)


def test_multipole_check_does_not_allocate_a_full_fft2(monkeypatch):
    monkeypatch.setattr(np.fft, "fft2", lambda *args, **kwargs: pytest.fail("Full FFT2 allocated"))
    result = apply_multipole_phase(wave(), 2e-12, normal_m2=2e8)
    assert result.probability == pytest.approx(1., abs=1e-14)


def test_empty_spectrum_remains_zero():
    from temsim.physics.multipole_wave import _spectral_marginals
    result = _spectral_marginals(np.zeros((5, 8), complex))
    for marginal in result:
        np.testing.assert_array_equal(marginal, 0.)


@pytest.mark.parametrize("axis", [0, 1])
def test_relaxed_admission_records_actual_tail_without_projecting_or_changing_phase(axis):
    import json
    from temsim.physics.wave_grid import WaveSamplingError
    shape, tail_power = (32, 34), 1e-9
    spectral = np.zeros(shape, complex)
    spectral[0, 0] = 1.
    index = [0, 0]
    index[axis] = shape[axis]//2
    spectral[tuple(index)] = np.sqrt(tail_power)*np.exp(.7j)
    amplitude = np.fft.ifft2(spectral, norm="ortho")
    initial = PlaneWave(amplitude, np.eye(2)*1e-9, np.zeros(2))
    strict_record = {}
    with pytest.raises(WaveSamplingError):
        apply_multipole_phase(initial, 2e-12, normal_m2=2e8, sampling_record=strict_record)
    assert strict_record["admitted"] is False
    assert strict_record["envelope_bandwidth_rad_per_step_yx"][axis] == np.pi
    record = {}
    result = apply_multipole_phase(initial, 2e-12, normal_m2=2e8,
        spectral_tail_fraction=1e-8, sampling_record=record)
    x, y = initial.coordinates_m()
    phase = 2*np.pi*multipole_action(x, y, normal_m2=2e8)/2e-12
    expected = initial.amplitude*np.exp(1j*phase)
    np.testing.assert_allclose(result.amplitude, expected, rtol=2e-14, atol=2e-17)
    assert result.probability == pytest.approx(initial.probability, rel=2e-14)
    # Recover the original negative-Nyquist coefficient after undoing the
    # exact phase; a filtering shortcut would remove this nonzero component.
    restored = np.fft.fft2(result.amplitude*np.exp(-1j*phase), norm="ortho")
    np.testing.assert_allclose(restored, spectral, rtol=3e-11, atol=3e-16)
    assert record["admitted"] is True
    assert record["ignored_probability_fraction_yx"][axis] == pytest.approx(tail_power/(1+tail_power), rel=3e-11)
    assert max(record["ignored_probability_fraction_yx"]) <= record["spectral_tail_fraction_per_axis"]
    assert record["budget_rad_per_step"] == .8*np.pi
    assert record["occupied_relative_amplitude_threshold"] == 1e-8
    assert "no physical probability loss or continuum error bound" in record["scope"]
    json.dumps(record, allow_nan=False)


def test_explicit_strict_default_is_identical_and_relaxation_cannot_hide_operator_increment():
    from temsim.physics.wave_grid import WaveSamplingError
    initial = wave()
    default_record, explicit_record = {}, {}
    default = apply_multipole_phase(initial, 2e-12, normal_m2=2e8, sampling_record=default_record)
    explicit = apply_multipole_phase(initial, 2e-12, normal_m2=2e8,
                                     spectral_tail_fraction=1e-12, sampling_record=explicit_record)
    np.testing.assert_array_equal(default.amplitude, explicit.amplitude)
    assert default_record == explicit_record
    record = {}
    with pytest.raises(WaveSamplingError):
        apply_multipole_phase(initial, 2e-12, spherical_m3=1e26,
                             spectral_tail_fraction=1e-4, sampling_record=record)
    assert max(record["operator_increment_rad_per_step_yx"]) > record["budget_rad_per_step"]
    assert record["admitted"] is False


@pytest.mark.parametrize("fraction", [-1e-12, 1.00001e-4, np.nan, np.inf, True, np.bool_(False)])
def test_invalid_multipole_admission_is_rejected_even_for_zero_strength(fraction):
    with pytest.raises(ValueError, match="Multipole spectral tail fraction"):
        apply_multipole_phase(wave(), 2e-12, spectral_tail_fraction=fraction)


def test_zero_admission_includes_all_nonzero_frequency_bins():
    from temsim.physics.wave_grid import WaveSamplingError
    spectral = np.zeros((32, 32), complex)
    spectral[0, 0], spectral[0, 16] = 1., 1e-8
    initial = PlaneWave(np.fft.ifft2(spectral, norm="ortho"), np.eye(2)*1e-9, np.zeros(2))
    record = {}
    with pytest.raises(WaveSamplingError):
        apply_multipole_phase(initial, 2e-12, normal_m2=2e8,
                             spectral_tail_fraction=0., sampling_record=record)
    assert record["envelope_bandwidth_rad_per_step_yx"][1] == np.pi
    assert record["ignored_probability_fraction_yx"] == [0., 0.]
