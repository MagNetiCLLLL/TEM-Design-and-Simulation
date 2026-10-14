"""Analytical phase comparisons and readout guards, not optics qualification."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE, wavelength_m
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wavefront_comparison import compare_wavefronts


def mode(*, shape=(64, 64), basis=None, origin=None, phase=0., key="source:0",
         curvature=None, tilt=None):
    basis = np.eye(2)*2e-9 if basis is None else np.asarray(basis)
    origin = np.zeros(2) if origin is None else np.asarray(origin)
    yy, xx = np.meshgrid(np.arange(shape[0])-shape[0]//2,
                         np.arange(shape[1])-shape[1]//2, indexing="ij")
    xy = origin[:, None, None]+np.einsum("ij,jyx->iyx", basis, np.stack((xx, yy)))
    amplitude = np.exp(-np.sum(xy**2, axis=0)/(2*(15e-9)**2)).astype(complex)
    amplitude *= np.exp(1j*(phase(xy) if callable(phase) else phase))
    amplitude /= np.linalg.norm(amplitude)
    return WaveMode(PlaneWave(amplitude, basis, origin, curvature, tilt), .3,
                    TIP_REFERENCE, key, 300.)


def checkpoint(*modes, z=1000., record=None):
    return TipGunCheckpoint(BeamState(modes, TIP_REFERENCE), z, 1e-9,
                            {"schema": "test-fixture"} if record is None else record)


def compare(actual, ideal, **kwargs):
    return compare_wavefronts(checkpoint(actual), checkpoint(ideal), "source:0", **kwargs)


def test_identical_focused_wave_retains_nonzero_curvature_and_immutable_output():
    focused = mode(curvature=np.array(((2000., 300.), (300., 1500.))), tilt=np.array((2e-5, -1e-5)))
    before = focused.plane.amplitude.copy()
    result = compare(focused, focused)
    assert np.ptp(result.actual_phase_rad[result.valid_mask]) > .1
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], 0., atol=1e-15)
    np.testing.assert_array_equal(result.actual_phase_rad, result.ideal_phase_rad)
    np.testing.assert_array_equal(focused.plane.amplitude, before)
    assert not result.metadata["longitudinal_action_included"]
    assert not result.metadata["time_phase_included"]
    assert not result.metadata["physical_weight_renormalized"]
    for name in ("actual_phase_rad", "ideal_phase_rad", "difference_rad", "valid_mask", "basis_m", "origin_m"):
        array = getattr(result, name)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.metadata["mode_id"] = "changed"


def test_quartic_aberration_difference_does_not_remove_ideal_quadratic_focus():
    quartic = lambda xy: .4*(np.sum(xy**2, axis=0)/(30e-9)**2)**2
    q = np.eye(2)*2300.
    actual, ideal = mode(phase=quartic, curvature=q), mode(curvature=q)
    result = compare(actual, ideal)
    expected = np.angle(np.exp(1j*quartic(actual.plane.coordinates_m())))
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], expected[result.valid_mask], atol=2e-14)
    assert np.ptp(result.ideal_phase_rad[result.valid_mask]) > .1


@pytest.mark.parametrize("piston", [.37, 2.9, -2.9])
def test_common_source_phase_is_retained_by_default_and_optional_piston_is_only_constant(piston):
    actual, ideal = mode(phase=piston), mode()
    raw = compare(actual, ideal)
    np.testing.assert_allclose(raw.difference_rad[raw.valid_mask], piston, atol=1e-14)
    aligned = compare(actual, ideal, remove_piston=True)
    assert aligned.metadata["piston_phase_rad"] == pytest.approx(piston, abs=1e-14)
    np.testing.assert_allclose(aligned.difference_rad[aligned.valid_mask], 0., atol=1e-14)
    np.testing.assert_array_equal(aligned.actual_phase_rad, raw.actual_phase_rad)
    np.testing.assert_array_equal(aligned.ideal_phase_rad, raw.ideal_phase_rad)


def test_wrapped_phase_difference_uses_complex_product_across_branch_cut():
    result = compare(mode(phase=np.pi-.05), mode(phase=-np.pi+.05))
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], -.1, atol=1e-14)


def test_equivalent_envelope_and_analytic_carrier_representations_have_zero_difference():
    q, tilt = np.eye(2)*1000., np.array((1e-5, -2e-5))
    carried = mode(curvature=q, tilt=tilt)
    xy = carried.plane.coordinates_m()
    phase = 2*np.pi/float(wavelength_m(300000.))*(
        .5*np.einsum("iyx,ij,jyx->yx", xy, q, xy)+np.einsum("i,iyx->yx", tilt, xy))
    explicit = replace(carried, plane=replace(carried.plane,
        amplitude=carried.plane.amplitude*np.exp(1j*phase), curvature_m1=None, tilt_rad=None))
    result = compare(explicit, carried)
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], 0., atol=1e-14)


@pytest.mark.parametrize("shape", [(64, 64), (65, 67)])
def test_affine_resampling_evaluates_full_carrier_at_common_laboratory_points(shape):
    q = np.array(((1600., 200.), (200., 2100.)))
    tilt = np.array((1e-5, -2e-5))
    actual = mode(shape=shape, basis=((1.6e-9, .2e-9), (-.15e-9, 1.7e-9)),
                  curvature=q, tilt=tilt)
    offset = np.array((3.2e-9, -2.4e-9))
    # Algebraic origin change: the ideal full field still has the same global
    # quadratic/linear phase, including its constant term at the new origin.
    constant = 2*np.pi/float(wavelength_m(300000.))*(offset@q@offset/2+tilt@offset)
    ideal = mode(shape=(96, 96), basis=np.eye(2)*1.5e-9, origin=offset, phase=constant,
                 curvature=q, tilt=tilt+q@offset)
    result = compare(actual, ideal)
    assert result.metadata["sampling"].startswith("complex-envelope")
    assert result.metadata["valid_pixel_count"] > 1000
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], 0., atol=2e-13)
    np.testing.assert_array_equal(result.basis_m, actual.plane.basis_m)
    np.testing.assert_array_equal(result.origin_m, actual.plane.origin_m)


def test_different_grid_masks_uncovered_or_unresolved_samples_without_extrapolation():
    actual = mode(shape=(64, 64))
    ideal = mode(shape=(24, 24), basis=np.eye(2)*1.5e-9)
    result = compare(actual, ideal)
    assert 0 < result.metadata["valid_fraction"] < .3
    for array in (result.actual_phase_rad, result.ideal_phase_rad, result.difference_rad):
        assert np.isnan(array[~result.valid_mask]).all()
    assert result.metadata["source_edge_guard_cells"] == 3


def test_unresolved_high_frequency_complex_interpolation_is_masked():
    actual = mode(shape=(40, 40), basis=np.eye(2)*1.7e-9)
    ideal = mode(phase=lambda xy: 2*np.pi*.42*xy[0]/2e-9)
    try:
        result = compare(actual, ideal)
    except ValueError as error:
        assert "no common valid support" in str(error)
    else:
        assert result.metadata["interpolation_rejected_pixel_count"] > 100
        assert result.metadata["maximum_interpolation_relative_error"] > result.metadata["interpolation_tolerance"]


def test_intensity_threshold_and_zero_nodes_use_common_mask_without_changing_weights():
    original = mode()
    amplitude = original.plane.amplitude.copy()
    amplitude[32, 32] = 0
    amplitude /= np.linalg.norm(amplitude)
    actual = replace(original, plane=replace(original.plane, amplitude=amplitude))
    result = compare(actual, original)
    assert not result.valid_mask[32, 32]
    assert not result.valid_mask[0, 0]
    assert result.metadata["relative_intensity_threshold"] == 1e-6
    assert result.metadata["actual_weight"] == result.metadata["ideal_weight"] == .3


def test_nonoverlapping_and_zero_weight_modes_have_no_invented_phase():
    actual = mode()
    far = replace(actual, plane=replace(actual.plane, origin_m=np.array((1e-3, 0.))))
    with pytest.raises(ValueError, match="no common valid support"):
        compare(actual, far)
    with pytest.raises(ValueError, match="zero weight"):
        compare(actual, replace(actual, weight_per_reference_electron=0.))


def test_zero_complex_overlap_rejects_only_optional_piston():
    ideal = mode()
    amplitude = np.full((64, 64), 1/64, dtype=complex)
    ideal = replace(ideal, plane=replace(ideal.plane, amplitude=amplitude))
    amplitude[:, 32:] *= -1
    actual = replace(ideal, plane=replace(ideal.plane, amplitude=amplitude))
    raw = compare(actual, ideal)
    assert raw.valid_mask.all()
    assert raw.metadata["relative_complex_overlap"] < 1e-14
    with pytest.raises(ValueError, match="piston is undefined"):
        compare(actual, ideal, remove_piston=True)


def test_mode_reordering_does_not_compare_or_sum_unrelated_modes():
    first, second = mode(phase=.2), mode(key="source:1", phase=-1.7)
    actual = checkpoint(first, second)
    ideal = checkpoint(replace(second, plane=replace(second.plane, amplitude=-second.plane.amplitude)), first)
    result = compare_wavefronts(actual, ideal, "source:0")
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], 0., atol=1e-15)
    with pytest.raises(ValueError, match="matching mode_id"):
        compare_wavefronts(actual, checkpoint(second), "source:0")


@pytest.mark.parametrize("field,value", [("energy_kev", 200.), ("scattering_history", ({"channel": "changed"},))])
def test_same_mode_label_cannot_hide_different_energy_or_scattering_history(field, value):
    original = mode()
    with pytest.raises(ValueError, match="match mode_id, energy"):
        compare(original, replace(original, **{field: value}))


def test_exact_z_and_current_reference_are_required():
    original = checkpoint(mode())
    with pytest.raises(ValueError, match="exactly the same"):
        compare_wavefronts(original, replace(original, plane_z_mm=1000.+1e-10), "source:0")
    with pytest.raises(ValueError, match="reference current"):
        compare_wavefronts(original, replace(original, reference_current_a=2e-9), "source:0")


def test_memory_preflight_precedes_disk_loading_and_missing_mode_does_not_read(tmp_path, monkeypatch):
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore, _StoredModes
    saved = checkpoint(mode())
    store = ExecutedWaveStore(str(tmp_path), "fixture", 10*1024**2)
    key = store.key("checkpoint")
    store.put(key, saved)
    stored = store.get(key)
    monkeypatch.setattr(_StoredModes, "_array", lambda *_: pytest.fail("No arrays before memory/mode preflight"))
    with pytest.raises(MemoryError, match="before loading"):
        compare_wavefronts(stored, stored, "source:0", maximum_working_bytes=1)
    with pytest.raises(ValueError, match="matching mode_id"):
        compare_wavefronts(stored, stored, "absent")


def test_chunking_and_cancellation_leave_inputs_unchanged():
    original = mode(phase=.2)
    other = mode()
    full = compare(original, other)
    minimum = 50*64**2 + 32*(64**2+64**2) + 384*64
    chunked = compare(original, other, maximum_working_bytes=minimum)
    assert chunked.metadata["row_chunk_size"] == 1
    for field in ("actual_phase_rad", "ideal_phase_rad", "difference_rad", "valid_mask"):
        np.testing.assert_array_equal(getattr(full, field), getattr(chunked, field))
    before = original.plane.amplitude.copy()
    calls = []
    def cancelled():
        calls.append(None)
        return len(calls) > 5
    with pytest.raises(InterruptedError, match="cancelled"):
        compare(original, other, cancelled=cancelled, maximum_working_bytes=minimum)
    np.testing.assert_array_equal(original.plane.amplitude, before)


def test_stored_comparison_verifies_only_the_selected_mode(tmp_path, monkeypatch):
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore, _StoredModes
    saved = checkpoint(mode(), mode(key="unrelated"))
    store = ExecutedWaveStore(str(tmp_path), "fixture", 10*1024**2)
    key = store.key("selected-only")
    store.put(key, saved)
    stored = store.get(key)
    read = _StoredModes._array
    calls = []
    def selected_array(self, item):
        assert item["file"].startswith("m0_"), "Unselected mode must stay on disk"
        calls.append(item["file"])
        return read(self, item)
    monkeypatch.setattr(_StoredModes, "_array", selected_array)
    result = compare_wavefronts(stored, stored, "source:0")
    assert calls
    assert result.metadata["actual_digest"] == result.metadata["ideal_digest"]
    np.testing.assert_allclose(result.difference_rad[result.valid_mask], 0., atol=1e-15)


@pytest.mark.parametrize("options", [
    {"maximum_working_bytes": True}, {"remove_piston": "false"},
    {"relative_intensity_threshold": -1}, {"relative_intensity_threshold": np.nan},
    {"interpolation_tolerance": 0}, {"interpolation_tolerance": True},
])
def test_invalid_controls_fail_before_comparison(options):
    with pytest.raises(ValueError):
        compare(mode(), mode(), **options)
