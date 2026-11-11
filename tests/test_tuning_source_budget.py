"""Preview source budgets preserve original current and structured populations."""
from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest

from temsim.gui.calculation_request import CapturedCalculationRequest, apply_request_numerics
from temsim.optics.column import default_state
from temsim.optics.electron_gun.emitter import (
    ColdFieldEmitter, EmissionCDFQuadrature, EmissionQuadrature, EmissionSampling,
)
from temsim.optics.electron_gun.tip_coherence import DRIVEN_GAUSSIAN_SCHELL, TipCoherence
from temsim.physics.optical_tuning import (
    DEFAULT_TUNING_SOURCE_PARTICLES, TUNING_PROFILES,
    prepare_tuning_snapshot, resolve_tuning_ray_count,
)


def source_state(emitter):
    return SimpleNamespace(
        electron_gun=SimpleNamespace(emitter=emitter), simulation_mode="unified",
        sample=SimpleNamespace(wave_enabled=True, stem_wave_enabled=True),
    )


def explicit_source(positive=1024, zero=33, *, role="quadrature"):
    total = positive + zero
    coordinates = np.tile(np.linspace(.01, .99, total)[:, None], (1, 6))
    weights = np.zeros(total)
    if positive:
        weights[:positive] = np.arange(1., positive + 1.)
        weights /= weights.sum()
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=total)
    emitter.cdf_quadrature = EmissionCDFQuadrature.from_arrays(
        coordinates, weights, signs=np.where(np.arange(total) % 2, -1, 1),
        ray_ids=np.arange(10000, 10000 + total), role=role,
    )
    return emitter


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
@pytest.mark.parametrize("coherent_source", (False, True))
def test_ordinary_preview_counts_positive_current_separately_from_probes(quality, coherent_source):
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=49)
    if coherent_source:
        emitter.coherence = TipCoherence(
            boundary_model=DRIVEN_GAUSSIAN_SCHELL, transverse_coherence_length_nm=20.,
        )
    state = source_state(emitter)
    profile = TUNING_PROFILES[quality]
    apply_request_numerics(state, quality, profile.rays, profile.step_mm)
    total = emitter.ray_count
    assert resolve_tuning_ray_count(state, quality, total) == total
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    first = emitter.emit()
    assert np.count_nonzero(first.weight > 0) == DEFAULT_TUNING_SOURCE_PARTICLES
    assert np.unique(first.ray_id[first.weight > 0]).size == DEFAULT_TUNING_SOURCE_PARTICLES
    assert np.count_nonzero(first.weight == 0) == (33 if quality == "Medium" else 0)
    assert first.weight.sum() == pytest.approx(1.)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    second = emitter.emit()
    assert emitter.ray_count == total
    for field in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id"):
        np.testing.assert_array_equal(getattr(second, field), getattr(first, field))


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
@pytest.mark.parametrize("positive,zero", ((1000, 24), (1024, 33)))
def test_explicit_cdf_keeps_every_node_weight_and_identity(quality, positive, zero):
    emitter = explicit_source(positive, zero)
    emitter.validate()
    original = emitter.emit()
    table, digest = emitter.cdf_quadrature, emitter.cdf_quadrature.digest
    state = source_state(emitter)
    profile = TUNING_PROFILES[quality]
    apply_request_numerics(state, quality, profile.rays, profile.step_mm)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    emitter.validate()
    actual = emitter.emit()
    assert emitter.ray_count == positive + zero
    assert emitter.cdf_quadrature is table
    assert emitter.cdf_quadrature.digest == digest
    assert emitter._tuning_boundary_probes == emitter._tuning_surface_probes == 0
    assert np.count_nonzero(actual.weight > 0) == positive
    for field in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
@pytest.mark.parametrize("positive,zero,role", ((999, 33, "quadrature"), (0, 1024, "boundary-probes")))
def test_insufficient_positive_cdf_nodes_reject_before_preparation(quality, positive, zero, role):
    emitter = explicit_source(positive, zero, role=role)
    emitter.validate()
    state = source_state(emitter)
    with pytest.raises(ValueError, match="at least 1000 original positive-current"):
        prepare_tuning_snapshot(state, quality, particle_signals=True)
    assert emitter.ray_count == positive + zero
    assert not hasattr(emitter, "_tuning_boundary_probes")
    assert not hasattr(state, "_particle_tuning")


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
def test_fixed_product_at_the_minimum_is_preserved(quality):
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=1000)
    emitter.quadrature = EmissionQuadrature(10, 10, 10)
    state = source_state(emitter)
    profile = TUNING_PROFILES[quality]
    apply_request_numerics(state, quality, profile.rays, profile.step_mm)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    bundle = emitter.emit()
    assert emitter.ray_count == emitter.quadrature.total == 1000
    assert np.count_nonzero(bundle.weight > 0) == 1000
    np.testing.assert_array_equal(bundle.weight, np.full(1000, .001))


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
def test_small_product_requires_an_explicit_source_refinement(quality):
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=729)
    emitter.quadrature = EmissionQuadrature(9, 9, 9)
    state = source_state(emitter)
    with pytest.raises(ValueError, match="selected quadrature has 729"):
        prepare_tuning_snapshot(state, quality, particle_signals=True)
    assert emitter.ray_count == emitter.quadrature.total == 729


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
def test_sobol_preview_uses_complete_population_without_support_probes(quality):
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=16)
    emitter.sampling = EmissionSampling(seed=42)
    state = source_state(emitter)
    profile = TUNING_PROFILES[quality]
    apply_request_numerics(state, quality, profile.rays, profile.step_mm)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    emitter.validate()
    bundle = emitter.emit()
    assert emitter.ray_count == 1024
    assert emitter._tuning_boundary_probes == emitter._tuning_surface_probes == 0
    assert np.count_nonzero(bundle.weight > 0) == 1024
    assert resolve_tuning_ray_count(state, quality, 1500) == 2048


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
def test_worker_preparation_enforces_budget_without_a_request_wrapper(quality):
    emitter = ColdFieldEmitter(0., 1., 1., ray_count=49)
    state = source_state(emitter)
    prepare_tuning_snapshot(state, quality, particle_signals=True)
    assert np.count_nonzero(emitter.emit().weight > 0) == 1024


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
def test_captured_gui_request_preserves_the_complete_cdf_and_live_source(quality):
    state = default_state()
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    emitter = state.electron_gun.emitter
    emitter.cdf_quadrature = explicit_source().cdf_quadrature
    emitter.ray_count = emitter.cdf_quadrature.total
    original = emitter.cdf_quadrature
    before = state.to_dict()
    profile = TUNING_PROFILES[quality]
    captured = CapturedCalculationRequest.capture(state, quality, profile.rays, profile.step_mm)
    prepared = captured.prepare(Event())
    prepare_tuning_snapshot(prepared.snapshot, quality, particle_signals=True)
    actual = prepared.snapshot.electron_gun.emitter
    assert captured.ray_count == prepared.ray_count == original.total
    assert actual.cdf_quadrature == original
    actual.validate()
    np.testing.assert_array_equal(actual.emit().weight, original.weights)
    assert state.to_dict() == before
