"""Pure-array mixed-state mathematics and immutable boundary ownership."""
from dataclasses import replace
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import particle_wave_source as source_model
from temsim.physics.particle_wave_source import (
    MixedSourceNumerics, ParticleWaveSource, build_particle_wave_checkpoint,
)
from temsim.physics.wave_checkpoint import WaveCheckpoint


def source(count=160):
    """Mathematical phase samples, never a physical emission calculation."""
    rng = np.random.default_rng(618)
    phase = rng.normal(size=(4, count))
    return ParticleWaveSource(phase[0]*3.5e-11, phase[1]*3.5e-11,
        phase[2]*.01, phase[3]*.01, np.arange(count)+14, np.full(count, .2/count),
        300000.+rng.normal(0., .12, count), 8e-9+rng.uniform(0., 1e-12, count),
        1599.2, 1e-5, 1000, "a"*64, {"origin": "Mathematical test fixture"})


class Arrays:
    """NumPy facade for exact small algebra tests, not a CUDA substitute."""
    def __getattr__(self, name):
        return np.asarray if name == "asnumpy" else getattr(np, name)


def test_source_owns_full_precision_arrays_and_original_current_reference():
    original = source()
    values = original.x_m.copy()
    provenance = {"input": ["kept"]}
    captured = replace(original, x_m=values, provenance=provenance)
    digest = captured.digest
    values[:] = 0.
    provenance["input"].append("external edit")
    assert captured.digest == digest
    assert captured.x_m.dtype == np.float64 and captured.ray_id.dtype == np.int64
    assert captured.provenance["input"] == ("kept",)
    assert captured.source_fraction == pytest.approx(.2)
    assert captured.mean_energy_ev == np.average(captured.kinetic_energy_ev, weights=captured.weight)
    with pytest.raises(ValueError):
        captured.x_m.setflags(write=True)
    assert replace(original, flight_time_s=original.flight_time_s+1e-12).digest != original.digest


@pytest.mark.parametrize("change", [
    {"original_positive_count": 999}, {"input_digest": "invalid"},
    {"reference_current_a": 0.}, {"z_mm": np.nan},
    {"ray_id": np.zeros(160, dtype=int)}, {"weight": np.full(160, 1./80)},
    {"flight_time_s": np.full(160, np.nan)}, {"kinetic_energy_ev": np.zeros(160)},
])
def test_source_rejects_missing_ancestry_current_and_actual_energy_time(change):
    with pytest.raises(ValueError):
        replace(source(), **change)


@pytest.mark.parametrize("change", [
    {"pixels": 63}, {"width_nm": 0.}, {"maximum_modes": 0},
    {"maximum_mode_loss_fraction": 2e-5}, {"maximum_working_bytes": 0},
    {"source_image_rms_pm": -1.}, {"spatial_quadrature_order": 1},
])
def test_numerics_preserve_sampling_and_loss_limits(change):
    with pytest.raises(ValueError):
        MixedSourceNumerics(**change).validate()


def test_covariance_matched_lift_retains_measured_residual_second_moments():
    original = source()
    phase = {name: getattr(original, name) for name in source_model._PHASE_FIELDS}
    centers, constant, kernel, phase, report = source_model.lifted_centers(phase, .019687, original.weight)
    weight = original.weight/original.source_fraction
    centered = centers-np.average(centers, weights=weight, axis=0)
    covariance = (centered.T*weight) @ centered
    np.testing.assert_allclose(covariance+report["minimum_uncertainty_packet_covariance"],
        report["deaberrated_geometric_covariance"], rtol=0, atol=2e-15)
    assert np.isfinite(constant) and np.isfinite(kernel).all()
    assert report["phase_fit"]["held_out_particles"] == len(original.weight)//5


def test_density_compression_preserves_absolute_weights_without_refilling_omitted_trace():
    cp = Arrays()
    factor = np.diag(np.sqrt([.12, .06, .0199999, .0000001])).astype(complex)
    modes, weights, report = source_model._compress_factor(cp, factor, .2, 2e-7,
        MixedSourceNumerics(maximum_modes=3), lambda: False)
    np.testing.assert_allclose(np.sum(abs(modes)**2, axis=0), 1., rtol=0, atol=2e-15)
    assert report["retained_modes"] == 3
    assert weights.sum() < .2 and .2-weights.sum() <= 2e-7
    with pytest.raises(ValueError, match="mode-count"):
        source_model._compress_factor(cp, factor, .2, 2e-7,
            MixedSourceNumerics(maximum_modes=2), lambda: False)


def test_local_gaussian_translation_keeps_angular_diagonal_and_current(monkeypatch):
    monkeypatch.setattr(source_model, "_admit_workspace", lambda *args, **kwargs: 0)
    reciprocal = np.column_stack((np.linspace(-.5, .5, 5), np.zeros(5)))
    modes = np.ones((5, 1), complex)/np.sqrt(5)
    weights = np.array([.2])
    output, retained, report = source_model._translation_channel(Arrays(), modes, weights, reciprocal,
        MixedSourceNumerics(), .2, lambda: False)
    before = (abs(modes)**2) @ weights
    after = (abs(output)**2) @ retained
    assert np.sum(abs(after-before)) <= .2e-5+1e-14
    assert 0 <= .2-retained.sum() < .2e-5
    assert report["assumed_translation_rms_per_axis_pm"] == 10.


def test_cuda_admission_precedes_any_lift_and_cancellation_publishes_nothing(monkeypatch):
    from temsim.physics import compute_backend
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("No usable CUDA")

    monkeypatch.setattr(compute_backend, "require_coherent_cuda", unavailable)
    monkeypatch.setattr(source_model, "lifted_centers", lambda *args: pytest.fail("Unadmitted lift"))
    with pytest.raises(RuntimeError, match="usable CUDA"):
        build_particle_wave_checkpoint(source())
    assert calls == [1]
    with pytest.raises(InterruptedError, match="cancelled"):
        build_particle_wave_checkpoint(source(), cancelled=lambda: True)
    assert calls == [1]


def test_energy_span_is_not_replaced_and_workspace_is_bounded(monkeypatch):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *args: "Require GPU")
    original = source()
    with pytest.raises(ValueError, match="2 eV"):
        build_particle_wave_checkpoint(replace(original, kinetic_energy_ev=np.linspace(299998., 300002., 160)))
    with pytest.raises(MemoryError, match="working limit"):
        source_model._admit_workspace(100, 100, MixedSourceNumerics(maximum_working_bytes=1))


def test_generic_checkpoint_is_existing_archive_class_identity():
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    assert TipGunCheckpoint is WaveCheckpoint


def test_canonical_shift_is_owned_hashed_and_never_overwrites_mechanical_slopes():
    original = source()
    shift = np.full((len(original.ray_id), 2), .0002)
    captured = replace(original, canonical_frequency_shift_inv_angstrom=shift)
    digest = captured.digest
    shift[:] = 0.
    assert captured.digest == digest and captured.digest != original.digest
    np.testing.assert_array_equal(captured.canonical_frequency_shift_inv_angstrom, .0002)
    np.testing.assert_array_equal(captured.tx_rad, original.tx_rad)
    np.testing.assert_array_equal(captured.ty_rad, original.ty_rad)
    with pytest.raises(ValueError):
        captured.canonical_frequency_shift_inv_angstrom.setflags(write=True)
    assert replace(original, canonical_frequency_shift_inv_angstrom=np.zeros((160, 2))).digest != original.digest
    for invalid in (np.zeros((160, 3)), np.full((160, 2), np.nan)):
        with pytest.raises(ValueError, match="Canonical frequency"):
            replace(original, canonical_frequency_shift_inv_angstrom=invalid)


def test_electron_vector_potential_shift_has_cycles_per_angstrom_units(monkeypatch):
    from scipy.constants import e, h
    original = source()
    vector_potential_t_m = np.tile([1e-9, -2e-9], (160, 1))
    shift = -e*vector_potential_t_m/h*1e-10
    assert shift[0, 0] == pytest.approx(-.00002417989242084918, rel=1e-14)
    phase = {name: getattr(original, name) for name in source_model._PHASE_FIELDS}
    phase[source_model._CANONICAL_SHIFT] = shift
    seen = []

    def fit(position, frequency, *args):
        seen.append(frequency.copy())
        return np.zeros_like(position), lambda q: np.zeros(len(q)), {}

    monkeypatch.setattr(source_model, "angular_phase_model", fit)
    wavelength_a = .019687
    source_model.lifted_centers(phase, wavelength_a, original.weight)
    slopes = np.column_stack((original.tx_rad, original.ty_rad))
    mechanical = slopes/np.sqrt(1.+np.sum(slopes**2, axis=1))[:, None]/wavelength_a
    np.testing.assert_array_equal(seen[0], mechanical+shift)
    np.testing.assert_array_equal(phase["tx_rad"], original.tx_rad)


def test_builder_passes_only_selected_canonical_rows_to_lift_before_any_device_work(monkeypatch):
    from temsim.physics import compute_backend
    original = source()
    weights = original.weight.copy()
    weights[:7] = 0.
    shift = np.arange(320, dtype=float).reshape(160, 2)*1e-8
    captured = replace(original, weight=weights, canonical_frequency_shift_inv_angstrom=shift)
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *args: "Require GPU")
    seen = []

    def stop(phase, *args):
        seen.append(phase)
        raise InterruptedError("Mathematical observation before device execution")

    monkeypatch.setattr(source_model, "lifted_centers", stop)
    with pytest.raises(InterruptedError, match="Mathematical observation"):
        build_particle_wave_checkpoint(captured)
    np.testing.assert_array_equal(seen[0][source_model._CANONICAL_SHIFT], shift[weights > 0])
    np.testing.assert_array_equal(seen[0]["tx_rad"], original.tx_rad[weights > 0])


def test_mocked_builder_preserves_canonical_ancestry_and_explicit_record(monkeypatch):
    """Small NumPy-backed software integration; not a physical CUDA trial."""
    from temsim.physics import compute_backend, wave_device, backend_execution
    original = source()
    shift = np.tile([-.0002, .0004], (160, 1))
    captured = replace(original, canonical_frequency_shift_inv_angstrom=shift)
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *args: "Require GPU")
    monkeypatch.setattr(source_model, "_admit_workspace", lambda *args, **kwargs: 0)
    monkeypatch.setattr(backend_execution, "record_backend", lambda *args, **kwargs: None)

    @contextmanager
    def arrays(*args, **kwargs):
        yield SimpleNamespace(xp=Arrays(), evidence=lambda: {"scope": "Mocked pure-array integration"})

    monkeypatch.setattr(wave_device, "device_scope", arrays)
    checkpoint = build_particle_wave_checkpoint(captured,
        MixedSourceNumerics(pixels=256, width_nm=2., source_image_rms_pm=0.,
                            frequency_support_inv_angstrom=3.))
    record = checkpoint.record["canonical_frequency_initialization"]
    assert record["executed"] and record["frequency_units"] == "cycles/angstrom"
    assert record["mechanical_trajectory_slopes_retained"]
    np.testing.assert_array_equal(checkpoint.auxiliary_arrays[source_model._CANONICAL_SHIFT], shift)
    np.testing.assert_array_equal(checkpoint.auxiliary_arrays["tx_rad"], original.tx_rad)
    eigenvalues = checkpoint.auxiliary_arrays["retained_density_eigenvalues"]
    np.testing.assert_array_equal(eigenvalues, checkpoint.record["retained_density_eigenvalues_fraction_of_emitted"])
    assert not checkpoint.record["propagation_basis"]["components_orthogonal"]
    assert checkpoint.record["density_eigenvalue_statistics"]["purity"] == pytest.approx(
        np.sum((eigenvalues/eigenvalues.sum())**2), abs=1e-15)


def test_unitary_density_rebasis_preserves_density_trace_and_linear_observations():
    rng = np.random.default_rng(7281)
    vectors, _ = np.linalg.qr(rng.normal(size=(31, 7))+1j*rng.normal(size=(31, 7)))
    weights = np.array([.31, .08, .02, .004, .0003, .00001, .0000001])
    output, actual_weights, report = source_model._unitary_density_rebasis(Arrays(), vectors, weights, lambda: False)
    before_factor = vectors*np.sqrt(weights)[None, :]
    after_factor = output*np.sqrt(actual_weights)[None, :]
    np.testing.assert_allclose(after_factor @ after_factor.conj().T,
                               before_factor @ before_factor.conj().T, rtol=1e-13, atol=2e-17)
    assert actual_weights.sum() == pytest.approx(weights.sum(), rel=5e-15)
    np.testing.assert_allclose(np.sum(abs(output)**2, axis=0), 1., rtol=0, atol=5e-15)
    np.testing.assert_allclose(actual_weights, weights.sum()/len(weights), rtol=0, atol=5e-17)
    operator = (rng.normal(size=(19, 31))+1j*rng.normal(size=(19, 31)))/10.
    before, after = operator @ before_factor, operator @ after_factor
    np.testing.assert_allclose(after @ after.conj().T, before @ before.conj().T, rtol=1e-13, atol=3e-17)
    before_intensity, after_intensity = (np.sum(abs(values)**2, axis=1) for values in (before, after))
    np.testing.assert_allclose(after_intensity, before_intensity, rtol=2e-14, atol=2e-17)
    axis = np.linspace(-1., 1., 19)
    for power in (0, 1, 2):
        assert after_intensity @ axis**power == pytest.approx(before_intensity @ axis**power, rel=1e-13, abs=1e-16)
    assert not np.isclose(np.sum((weights/weights.sum())**2), np.sum((actual_weights/actual_weights.sum())**2))
    assert not report["components_orthogonal"] and report["component_count_unchanged"]
    assert report["factor_roundtrip_relative_l2_error"] < 5e-13
