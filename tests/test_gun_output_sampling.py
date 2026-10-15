"""Canonical gun continuation fixtures, not a substitute physical tip solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import GunWaveNumerics, _execute_gun_segments, _mask_plane
from temsim.physics.wave_grid import WaveSamplingError


def empty_gun():
    return SimpleNamespace(bore_components=(),
        dpa_aperture=SimpleNamespace(z_mm=-1., key="unused_dpa"),
        c1_aperture=SimpleNamespace(z_mm=-1., key="unused_c1"))


def canonical_fixture():
    # A genuine coherent tail prevents cropped-CZT replacement of the full
    # Fourier period. The independent finite sum below retains its phase.
    count, pitch, energy = 64, 1e-9, 300_000.
    axis = np.arange(count)-count//2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    amplitude = np.exp(-(x*x+y*y)/(4*3.5**2))*(1+1e-4*np.exp(.3j+2j*np.pi*.47*x))
    amplitude /= np.linalg.norm(amplitude)
    wave = PlaneWave(amplitude, np.eye(2)*pitch, np.zeros(2))
    distance = .001
    matrix = np.block([[np.zeros((2, 2)), distance*np.eye(2)],
                       [-np.eye(2)/distance, np.zeros((2, 2))]])
    return wave, ((1., matrix, np.zeros(4), {}),), energy


def test_gun_continuation_resolves_collins_output_and_keeps_absolute_complex_phase():
    wave, segments, energy = canonical_fixture()
    output, losses, sampling = _execute_gun_segments(empty_gun(), wave, segments, energy, .4,
        numerics=GunWaveNumerics(maximum_checkpoint_bytes=128*1024**2))
    assert output.amplitude.shape == (512, 512)
    assert losses == []
    assert len(sampling) == 1
    assert sampling[0]["method"].startswith("zero-padded spatial Collins")
    assert sampling[0]["z_mm"] == 1.
    assert output.probability == pytest.approx(wave.probability, rel=3e-14)
    index = np.arange(512)-256
    incoming = np.arange(64)-32
    fourier = np.exp(-2j*np.pi*np.outer(index, incoming)/512)
    expected = -1j*(fourier@wave.amplitude@fourier.T)/512
    np.testing.assert_allclose(output.full_amplitude(float(wavelength_m(energy))), expected,
                               rtol=1e-9, atol=2e-13)
    natural_basis = float(wavelength_m(energy))*segments[0][1][:2, 2:]@np.linalg.inv(wave.basis_m).T/64
    np.testing.assert_allclose(output.basis_m*512, natural_basis*64, rtol=1e-13)


@pytest.mark.parametrize("budget,retained", [(2*1024**2, 0), (128*1024**2, 96*1024**2)])
def test_gun_continuation_respects_scratch_and_previous_modes_budget(budget, retained):
    wave, segments, energy = canonical_fixture()
    original = wave.amplitude.copy()
    with pytest.raises(ValueError, match="Collins output refinement budget exceeded"):
        _execute_gun_segments(empty_gun(), wave, segments, energy, .4,
            numerics=GunWaveNumerics(maximum_checkpoint_bytes=budget), retained_bytes=retained)
    np.testing.assert_array_equal(wave.amplitude, original)


def test_gun_continuation_checks_cancellation_before_propagation():
    wave, segments, energy = canonical_fixture()
    with pytest.raises(InterruptedError):
        _execute_gun_segments(empty_gun(), wave, segments, energy, .4,
                             numerics=GunWaveNumerics(), cancelled=lambda: True)


@pytest.mark.parametrize("owner", ["gun", "column", "bore"])
def test_subcell_gun_openings_fail_before_clipping(owner):
    wave, _, _ = canonical_fixture()
    gun = empty_gun()
    aperture = SimpleNamespace(z_mm=1., key="tiny", radius_mm=1e-8,
        enabled=True, kind="continuous_aperture", interaction_kind="hard_edge_circular_stop",
        transmission_mask=lambda *args: pytest.fail("A sub-cell opening was applied"))
    column = ()
    if owner == "gun":
        gun.dpa_aperture = aperture
    elif owner == "column":
        column = (aperture,)
    else:
        gun.bore_components = (SimpleNamespace(key="tiny", mechanical_center_from_tip_mm=1.,
            mechanical_length_mm=.1, mechanical_clear_bore_diameter_mm=2e-8),)
    original = wave.amplitude.copy()
    with pytest.raises(WaveSamplingError, match="Gun opening tiny.*undersampled"):
        _mask_plane(gun, wave, 1., .4, column)
    np.testing.assert_array_equal(wave.amplitude, original)


def test_closed_gun_aperture_has_zero_transmission_without_false_sampling_error():
    wave, _, _ = canonical_fixture()
    gun = empty_gun()
    gun.dpa_aperture = SimpleNamespace(z_mm=1., key="closed", radius_mm=0.,
        enabled=True, kind="continuous_aperture", interaction_kind="hard_edge_circular_stop")
    result, rows = _mask_plane(gun, wave, 1., .4)
    assert result.probability == 0.
    assert rows[0]["lost_probability"] == pytest.approx(.4*wave.probability)


def test_retracted_optional_aperture_does_not_impose_its_unused_small_radius():
    wave, _, _ = canonical_fixture()
    gun = empty_gun()
    gun.dpa_aperture = SimpleNamespace(z_mm=1., key="optional", radius_mm=1e-8,
        enabled=False, kind="continuous_aperture", interaction_kind="hard_edge_circular_stop",
        transmission_mask=lambda x, y: np.ones_like(x, dtype=bool))
    result, _ = _mask_plane(gun, wave, 1., .4)
    np.testing.assert_array_equal(result.amplitude, wave.amplitude)


def test_gun_publication_includes_live_auxiliary_arrays_and_immutable_copies(monkeypatch):
    import temsim.physics.tip_gun_wave as gun_wave
    from temsim.physics.wave_grid import WaveMemoryBudgetError
    arrays = {"near_tip_m0_spectral_amplitude": np.arange(12, dtype=complex),
              "near_tip_m0_spectral_derivative": np.arange(24, dtype=complex)}
    beam = SimpleNamespace(modes=(SimpleNamespace(plane=SimpleNamespace(amplitude=np.zeros((8, 8), complex))),))
    retained = 321
    required = retained+beam.modes[0].plane.amplitude.nbytes+2*sum(a.nbytes for a in arrays.values())+max(a.nbytes for a in arrays.values())
    monkeypatch.setattr(gun_wave, "TipGunCheckpoint",
                        lambda *args: pytest.fail("Gun checkpoint copied auxiliary arrays before its budget check"))
    with pytest.raises(WaveMemoryBudgetError, match="Gun checkpoint publication"):
        gun_wave._publish_gun_checkpoint(beam, 450., 1e-9, {}, arrays,
            maximum_working_bytes=required-1, retained_bytes=retained)
    sentinel = object()
    monkeypatch.setattr(gun_wave, "TipGunCheckpoint", lambda *args: sentinel)
    assert gun_wave._publish_gun_checkpoint(beam, 450., 1e-9, {}, arrays,
        maximum_working_bytes=required, retained_bytes=retained) is sentinel


def test_gun_publication_checks_cancellation_and_available_memory_before_copy(monkeypatch):
    import temsim.physics.tip_gun_wave as gun_wave
    beam = SimpleNamespace(modes=())
    arrays = {"near_tip_m0_spectral_amplitude": np.arange(4, dtype=complex)}
    monkeypatch.setattr(gun_wave, "TipGunCheckpoint",
                        lambda *args: pytest.fail("Unavailable publication memory was allocated"))
    with pytest.raises(InterruptedError, match="publication cancelled"):
        gun_wave._publish_gun_checkpoint(beam, 450., 1e-9, {}, arrays,
            maximum_working_bytes=4096, cancelled=lambda: True)
    def unavailable(additional):
        assert additional == 2*arrays["near_tip_m0_spectral_amplitude"].nbytes
        raise MemoryError("available-memory fixture")
    monkeypatch.setattr("temsim.physics.wave_execution.check_available_memory", unavailable)
    with pytest.raises(MemoryError, match="available-memory fixture"):
        gun_wave._publish_gun_checkpoint(beam, 450., 1e-9, {}, arrays, maximum_working_bytes=4096)
