"""Full-complex finite-Fourier reference for bounded conservative CPU drifts."""
import numpy as np
import pytest

from temsim.physics.cpu_angular_spectrum import (
    angular_spectrum_working_bytes, conservative_angular_spectrum,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError


def source(shape=(33, 37)):
    rng = np.random.default_rng(918)
    amplitude = rng.standard_normal(shape)+1j*rng.standard_normal(shape)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.array(((1e-9, .17e-9), (-.11e-9, 1.3e-9))),
                     np.array((3e-8, -2e-8)), np.array(((300., -20.), (-20., -100.))), np.array((1e-4, -2e-4)))


def dense_reference(wave, drift, wavelength):
    fy, fx = np.meshgrid(np.fft.fftfreq(wave.amplitude.shape[0]),
                        np.fft.fftfreq(wave.amplitude.shape[1]), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    phase = -np.pi*wavelength*np.einsum("iyx,ij,jyx->yx", frequency, drift, frequency)
    return np.fft.ifft2(np.fft.fft2(wave.amplitude)*np.exp(1j*phase))


@pytest.mark.parametrize("shape", [(32, 34), (33, 37), (32, 37), (127, 131)])
@pytest.mark.parametrize("drift", [np.array(((1.2e-6, .17e-6), (.17e-6, .7e-6))),
                                   np.array(((-.4e-6, .2e-6), (.2e-6, .8e-6)))])
def test_tiled_drift_retains_all_frequencies_phase_and_affine_axes(shape, drift):
    initial = source(shape)
    before = initial.amplitude.copy()
    expected = dense_reference(initial, drift, 2e-12)
    result = conservative_angular_spectrum(initial, drift, 2e-12)
    np.testing.assert_allclose(result, expected, rtol=4e-12, atol=3e-16)
    assert result.dtype == np.complex128
    assert np.sum(abs(result)**2) == pytest.approx(1., rel=3e-14)
    np.testing.assert_array_equal(initial.amplitude, before)
    assert not np.shares_memory(result, initial.amplitude)


def test_negative_nyquist_and_global_phase_are_not_removed_or_split():
    n = 32
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    amplitude = np.exp(1j*(.71-np.pi*x+2*np.pi*3*y/n))/n
    initial = PlaneWave(amplitude, np.eye(2)*1e-9, np.zeros(2))
    drift = np.array(((1e-6, .2e-6), (.2e-6, .5e-6)))
    result = conservative_angular_spectrum(initial, drift, 2e-12)
    frequency = np.array((-.5e9, 3/n*1e9))
    expected = amplitude*np.exp(-1j*np.pi*2e-12*(frequency@drift@frequency))
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=2e-16)


def test_drift_and_inverse_preserve_complete_wave_without_renormalisation():
    initial = source((64, 67))
    drift = np.array(((1.2e-6, .2e-6), (.2e-6, .8e-6)))
    forward = conservative_angular_spectrum(initial, drift, 2e-12)
    intermediate = initial._replace_owned(forward)
    restored = conservative_angular_spectrum(intermediate, -drift, 2e-12)
    np.testing.assert_allclose(restored, initial.amplitude, rtol=2e-12, atol=3e-16)
    assert np.sum(abs(restored)**2) == pytest.approx(initial.probability, rel=3e-14)


def test_conservative_drift_never_allocates_full_frequency_coordinates_or_fft2(monkeypatch):
    import temsim.physics.cpu_angular_spectrum as angular
    import temsim.physics.wave_grid as grid
    initial = source((67, 71))
    drift = np.array(((1.2e-6, .17e-6), (.17e-6, .7e-6)))
    expected = dense_reference(initial, drift, 2e-12)
    monkeypatch.setattr(angular, "_ANGULAR_TILE_CELLS", 71*2)
    monkeypatch.setattr(grid, "_FOURIER_TILE_CELLS", 71*2)
    def forbidden(*args, **kwargs):
        pytest.fail("Conservative drift allocated full 2-D coordinates or FFT")
    monkeypatch.setattr(np, "meshgrid", forbidden)
    monkeypatch.setattr(np.fft, "fft2", forbidden)
    monkeypatch.setattr(np.fft, "ifft2", forbidden)
    result = conservative_angular_spectrum(initial, drift, 2e-12)
    np.testing.assert_allclose(result, expected, rtol=4e-12, atol=3e-16)


def test_budget_includes_external_waves_and_publication_before_allocation(monkeypatch):
    import temsim.physics.cpu_angular_spectrum as angular
    initial = source()
    retained = 4321
    required = angular_spectrum_working_bytes(initial.amplitude.shape)+retained
    monkeypatch.setattr(angular.np, "array", lambda *a, **k: pytest.fail("Private spectrum allocated before budget check"))
    with pytest.raises(WaveMemoryBudgetError):
        conservative_angular_spectrum(initial, np.eye(2)*1e-6, 2e-12,
            grid_numerics=WaveGridNumerics(maximum_working_bytes=required-1), retained_bytes=retained)


def test_cancellation_during_fft_keeps_input_unchanged(monkeypatch):
    import temsim.physics.wave_grid as grid
    initial = source((64, 64))
    before = initial.amplitude.copy()
    monkeypatch.setattr(grid, "_FOURIER_TILE_CELLS", 128)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 4
    with pytest.raises(InterruptedError):
        conservative_angular_spectrum(initial, np.eye(2)*1e-6, 2e-12, cancelled=cancelled)
    np.testing.assert_array_equal(initial.amplitude, before)


def test_zero_drift_reuses_immutable_input_and_invalid_parameters_still_reject():
    initial = source()
    assert conservative_angular_spectrum(initial, np.zeros((2, 2)), 2e-12) is initial.amplitude
    for wavelength in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match="positive finite wavelength"):
            conservative_angular_spectrum(initial, np.eye(2)*1e-6, wavelength)
    with pytest.raises(ValueError, match="finite 2 by 2 matrix"):
        conservative_angular_spectrum(initial, np.ones((3, 3)), 2e-12)


def _drift_map(distance):
    matrix = np.eye(4)
    matrix[:2, 2:] = np.eye(2)*distance
    return matrix


def test_canonical_cpu_routing_fits_below_dense_budget_with_exact_global_phase(monkeypatch):
    """Test the public map, including its publication and affine phase."""
    import temsim.physics.cpu_angular_spectrum as angular
    import temsim.physics.multiplane_wave as canonical
    initial, wavelength = source((512, 512)), 2e-12
    matrix = _drift_map(1e-6)
    offset = np.array((2e-8, -1e-8, 3e-5, -2e-5))
    numerics = WaveGridNumerics(maximum_working_bytes=48*1024**2, compute_backend="CPU")
    with pytest.raises(WaveMemoryBudgetError):
        numerics.check(initial.amplitude.shape)
    # The reference substitutes only the dense finite Fourier calculation;
    # metadata, affine action, lift and immutable publication remain public.
    with monkeypatch.context() as dense:
        dense.setattr(angular, "conservative_angular_spectrum",
                      lambda wave, drift, wavelength, **kwargs: dense_reference(wave, drift, wavelength))
        expected = canonical.propagate_plane_wave(initial, matrix, offset, wavelength,
                                                  affine_action_m=2e-12)
    def forbidden(*args, **kwargs):
        pytest.fail("Conservative CPU map used a dense fallback")
    monkeypatch.setattr(canonical, "_sampled_angular_spectrum", forbidden)
    monkeypatch.setattr(canonical, "_propagate_fresnel_chart", forbidden)
    rows = []
    result = canonical.propagate_plane_wave(initial, matrix, offset, wavelength,
        affine_action_m=2e-12, grid_numerics=numerics, sampling_records=rows)
    np.testing.assert_allclose(result.amplitude, expected.amplitude, rtol=2e-11, atol=2e-16)
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result, key), getattr(expected, key))
    assert result.probability == pytest.approx(initial.probability, rel=3e-14)
    assert not result.amplitude.flags.writeable


@pytest.mark.parametrize("route", ["nonconservative", "Collins"])
def test_dense_fallthrough_keeps_generic_budget_before_allocation(monkeypatch, route):
    from dataclasses import replace
    import temsim.physics.multiplane_wave as canonical
    initial = replace(source((512, 512)), curvature_m1=None)
    matrix = (_drift_map(.01) if route == "nonconservative" else
              np.block([[np.zeros((2, 2)), np.eye(2)*1e-3],
                        [-np.eye(2)*1e3, np.zeros((2, 2))]]))
    numerics = WaveGridNumerics(maximum_working_bytes=48*1024**2)
    numerics.check_working_memory(initial.amplitude.shape, working_bytes=32*initial.amplitude.size)
    def forbidden(*args, **kwargs):
        pytest.fail("Dense fallback allocated before its conservative budget check")
    monkeypatch.setattr(canonical, "_sampled_angular_spectrum", forbidden)
    monkeypatch.setattr(canonical, "_propagate_fresnel_chart", forbidden)
    rows = []
    with pytest.raises(WaveMemoryBudgetError):
        canonical.propagate_plane_wave(initial, matrix, np.zeros(4), 2e-12,
                                       grid_numerics=numerics, sampling_records=rows)
    assert rows == []
    with pytest.raises(InterruptedError):
        canonical.propagate_plane_wave(initial, matrix, np.zeros(4), 2e-12,
            grid_numerics=numerics, cancelled=lambda: True, sampling_records=rows)
    assert rows == []


def test_public_cancellation_during_cpu_fft_does_not_publish_partial_rows(monkeypatch):
    import temsim.physics.wave_grid as grid
    from temsim.physics.multiplane_wave import propagate_plane_wave
    initial, rows = source((64, 64)), [{"prior": "completed"}]
    before = initial.amplitude.copy()
    monkeypatch.setattr(grid, "_FOURIER_TILE_CELLS", 128)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 5
    with pytest.raises(InterruptedError):
        propagate_plane_wave(initial, _drift_map(1e-6), np.zeros(4), 2e-12,
            grid_numerics=WaveGridNumerics(), cancelled=cancelled, sampling_records=rows)
    np.testing.assert_array_equal(initial.amplitude, before)
    assert rows == [{"prior": "completed"}]


def test_column_cpu_load_and_drift_succeed_below_old_dense_preflight(monkeypatch):
    """An analytic column fixture exercises the actual CPU backend boundary."""
    from dataclasses import replace
    from test_column_wave_transport import _quadratic_fixture
    from test_wave_detector_readout import checkpoint
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.physics.column_wave import _propagate_column
    from temsim.physics.wave_reference import AxialWaveReference
    import temsim.physics.multiplane_wave as canonical
    initial = replace(source((512, 512)), basis_m=np.eye(2)*20e-9,
                      origin_m=np.zeros(2), curvature_m1=None, tilt_rad=None)
    captured = checkpoint()
    mode = replace(captured.beam.modes[0], plane=initial, axial_reference=AxialWaveReference(0., 0.))
    captured = replace(captured, beam=replace(captured.beam, modes=(mode,)))
    prepared = _quadratic_fixture(2)
    plan = prepared[0]
    plan.z_mm = 2000.+np.arange(3)*.001
    plan.step_m = np.diff(plan.z_mm)*1e-3
    wavelength = float(wavelength_m(mode.energy_kev*1000))
    expected = dense_reference(initial, np.eye(2)*float(plan.step_m.sum()), wavelength)
    numerics = WaveGridNumerics(maximum_working_bytes=48*1024**2, compute_backend="CPU")
    with pytest.raises(WaveMemoryBudgetError):
        numerics.check(initial.amplitude.shape)
    def forbidden(*args, **kwargs):
        pytest.fail("Conservative CPU column invoked a dense fallback")
    monkeypatch.setattr(canonical, "_sampled_angular_spectrum", forbidden)
    monkeypatch.setattr(canonical, "_propagate_fresnel_chart", forbidden)
    output = _propagate_column(default_state(), captured, plan.z_mm[-1],
        grid_numerics=numerics, _prepared=prepared)
    actual = output.beam.modes[0]
    np.testing.assert_allclose(actual.plane.amplitude, expected, rtol=3e-11, atol=3e-16)
    assert actual.weight_per_reference_electron == pytest.approx(mode.weight_per_reference_electron, rel=3e-14)
    assert actual.energy_kev == mode.energy_kev
    assert actual.axial_reference.flight_time_s > 0.
    assert actual.axial_reference.longitudinal_action_j_s > 0.


def test_dense_column_mask_keeps_budget_before_coordinates_but_clear_bore_needs_none(monkeypatch):
    from temsim.physics.column_wave import _clip
    initial = source((512, 512))
    numerics = WaveGridNumerics(maximum_working_bytes=48*1024**2)
    monkeypatch.setattr(PlaneWave, "coordinates_m",
                        lambda self: pytest.fail("Dense mask coordinates allocated before budget check"))
    unchanged, rows = _clip(initial, 1., (), 42., .6,
        wavelength_m=2e-12, grid_numerics=numerics)
    assert unchanged is initial and rows == []
    # 200 nm is hundreds of cells (no small-aperture path), but cuts the grid.
    with pytest.raises(WaveMemoryBudgetError):
        _clip(initial, .0002, (), 42., .6, wavelength_m=2e-12, grid_numerics=numerics)


def test_empty_column_segment_checks_publication_budget_before_normalized_copy(monkeypatch):
    from dataclasses import replace
    from test_column_wave_transport import _quadratic_fixture
    from test_wave_detector_readout import checkpoint
    from temsim.optics.column import default_state
    import temsim.physics.column_wave as column
    initial = source((512, 512))
    captured = checkpoint()
    mode = replace(captured.beam.modes[0], plane=initial)
    captured = replace(captured, beam=replace(captured.beam, modes=(mode,)))
    numerics = WaveGridNumerics(maximum_working_bytes=16*1024**2, compute_backend="CPU")
    numerics.check_working_memory(initial.amplitude.shape, working_bytes=32*initial.amplitude.size,
                                 retained_bytes=initial.amplitude.nbytes)
    # No propagation occurs. Admission must still include the publication
    # copies, rather than treating its cheaper input check as permission.
    monkeypatch.setattr(column, "replace", lambda *a, **k: pytest.fail("Checkpoint copy preceded publication guard"))
    with pytest.raises(WaveMemoryBudgetError):
        column._propagate_column(default_state(), captured, 2000.,
            grid_numerics=numerics, _prepared=_quadratic_fixture(0))
