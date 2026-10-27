"""Transactional phase-chart preview using the complete physical H8 operator."""
from dataclasses import replace

import numpy as np
import pytest

import temsim.physics.potential_wave_carrier as carrier
import temsim.physics.potential_segment_resampling as preparation
from temsim.physics.potential_representation_estimates import (
    RepresentationEstimateLedger, require_combined_support_allowance,
)
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import WaveGridNumerics
from test_preview_segment_carrier import _case, _common_comparison
from test_potential_wave_carrier import WAVELENGTH


def _phase_case(*, steps=1, magnetic=False):
    snapshot, wave, nodes, options = _case(steps=steps, magnetic=magnetic)
    xy = wave.coordinates_m()-wave.origin_m[:, None, None]
    wave = replace(wave, amplitude=wave.amplitude*np.exp(
        1j*np.pi/WAVELENGTH*.5*np.sum(xy*xy, axis=0)))
    ledger = NumericalSupportLedger(.02, 1e-6)
    history = RepresentationEstimateLedger.for_support(ledger)
    options.update(support_ledger=ledger, representation_estimates=history,
        support_projection_policy="preview_phase_segment_and_retry",
        grid_numerics=WaveGridNumerics(maximum_pixels=256,
            maximum_working_bytes=256*1024**2))
    return snapshot, wave, nodes, options


@pytest.mark.parametrize("steps,magnetic", [(1, False), (2, True)])
def test_actual_phase_preview_preserves_full_physical_phase_and_absolute_weight(steps, magnetic):
    snapshot, wave, nodes, options = _phase_case(steps=steps, magnetic=magnetic)
    before = wave.amplitude.tobytes()
    baseline_options = {**options, "support_projection_policy": "weighted_segment_and_retry",
                        "segment_start_m": None, "representation_estimates": None}
    baseline, _ = carrier.propagate_carrier_potential(snapshot, wave, nodes, **baseline_options)
    result, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    history = RepresentationEstimateLedger.from_dict(record["representation_estimates"])
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert history.events and ledger.events
    require_combined_support_allowance(ledger, history)
    assert record["high_accuracy_eligible"] is False
    assert record["physical_phase_removed"] is False
    assert record["normalization_applied"] is False
    assert "not a certified" in record["representation_estimate_scope"]
    assert wave.amplitude.tobytes() == before
    assert options["support_ledger"].events == ()
    assert options["representation_estimates"].events == ()
    assert not np.allclose(result.curvature_m1, baseline.curvature_m1)
    coarse = _common_comparison(baseline, result, 256)
    fine = _common_comparison(baseline, result, 512)
    assert fine["raw_complex_l2"] < 1e-3
    assert abs(coarse["raw_complex_l2"]-fine["raw_complex_l2"]) < 1e-7
    assert fine["reference_probability"] == pytest.approx(baseline.probability, abs=2e-8)
    assert fine["candidate_probability"] == pytest.approx(result.probability, abs=2e-8)
    assert all(event["mode_weight"] == .27 for event in ledger.events)


def test_phase_preparation_occurs_only_at_the_declared_segment_first_residual(monkeypatch):
    snapshot, wave, nodes, options = _phase_case(steps=3)
    original, calls = preparation.prepare_phase_segment_resampling, []

    def inspect(*args, **kwargs):
        calls.append(kwargs["z_m"])
        return original(*args, **kwargs)

    monkeypatch.setattr(preparation, "prepare_phase_segment_resampling", inspect)
    _, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    assert calls == [nodes[0]]
    assert len(record["residual_steps"]) == 6


def test_rejected_phase_trial_restores_original_wave_and_both_histories(monkeypatch):
    snapshot, wave, nodes, options = _phase_case()
    baseline, _ = carrier.propagate_carrier_potential(snapshot, wave, nodes,
        **{**options, "support_projection_policy": "weighted_segment_and_retry",
           "segment_start_m": None, "representation_estimates": None})
    original, seen = carrier._residual, []

    def residual(snapshot, candidate, *args, **kwargs):
        seen.append(candidate.curvature_m1.copy())
        if len(seen) == 1:
            raise carrier.OccupiedCarrierBoundaryError("Private phase-chart candidate rejected")
        return original(snapshot, candidate, *args, **kwargs)

    monkeypatch.setattr(carrier, "_residual", residual)
    result, record = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    assert not np.array_equal(seen[0], wave.curvature_m1)
    assert np.array_equal(seen[1], wave.curvature_m1)
    assert result.amplitude.tobytes() == baseline.amplitude.tobytes()
    assert record["support_ledger"] == options["support_ledger"].to_dict()
    assert record["representation_estimates"] == options["representation_estimates"].to_dict()


@pytest.mark.parametrize("failure", ["domain", "cancel", "canonical"])
def test_failed_phase_segment_cannot_publish_or_spend_external_histories(monkeypatch, failure):
    snapshot, wave, nodes, options = _phase_case()
    original, cancelled, progress = carrier._residual, [False], []

    def residual(*args, **kwargs):
        if failure == "domain":
            raise ValueError("Unqualified physical domain")
        output = original(*args, **kwargs)
        if failure == "cancel":
            cancelled[0] = True
        return output

    monkeypatch.setattr(carrier, "_residual", residual)
    if failure == "canonical":
        def reject(*_args, **_kwargs):
            raise ValueError("Physical canonical failure")
        monkeypatch.setattr(carrier, "propagate_plane_wave", reject)
    with pytest.raises(InterruptedError if failure == "cancel" else ValueError):
        carrier.propagate_carrier_potential(snapshot, wave, nodes,
            cancelled=lambda: cancelled[0], progress_callback=progress.append, **options)
    assert options["support_ledger"].events == ()
    assert options["representation_estimates"].events == ()
    assert progress == []


def test_representation_history_is_rejected_under_the_old_projection_policy():
    snapshot, wave, nodes, options = _phase_case()
    with pytest.raises(ValueError, match="explicit preview phase policy"):
        carrier.propagate_carrier_potential(snapshot, wave, nodes,
            **{**options, "support_projection_policy": "preview_segment_and_retry"})


def test_real_gpu_phase_preview_matches_cpu_complex_field_and_both_histories():
    from temsim.physics.compute_backend import cupy_capability
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    snapshot, wave, nodes, options = _phase_case(steps=2, magnetic=True)
    options["grid_numerics"] = replace(options["grid_numerics"],
        maximum_device_working_bytes=256*1024**2)
    cpu, reference = carrier.propagate_carrier_potential(snapshot, wave, nodes, **options)
    gpu, result = carrier.propagate_carrier_potential(snapshot, wave, nodes,
        **{**options, "residual_compute_backend": "Require GPU"})
    assert reference["residual_backends"] == ["numpy"]
    assert result["residual_backends"] == ["cupy"]
    assert result["representation_estimates"]["events"]
    assert result["representation_estimates"] == reference["representation_estimates"]
    assert result["support_ledger"] == reference["support_ledger"]
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(gpu, name), getattr(cpu, name))
    assert gpu.amplitude.shape == cpu.amplitude.shape
    assert np.linalg.norm(gpu.amplitude-cpu.amplitude) < 2e-13
    assert gpu.probability == pytest.approx(cpu.probability, rel=2e-13)
    assert not result["normalization_applied"]
    assert not result["physical_phase_removed"]
    assert options["representation_estimates"].events == ()
    assert options["support_ledger"].events == ()
