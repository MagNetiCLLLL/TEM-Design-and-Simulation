"""Explicit nominal source preview; dispatch fixtures do not qualify a real gun."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.optics.electron_gun.tip_coherence import TipWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, _execute_tip_wave, simulate_tip_wave
from test_preview_segment_resampling import _request_preview


def _nominal_request(tmp_path):
    request = _request_preview(tmp_path, source_nominal_energy_preview=True)
    return replace(request, stop="gun_exit", source=replace(request.source, energy_samples=1))


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_nominal_source_option_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="source_nominal_energy_preview"):
        replace(TipWaveRequest(), source_nominal_energy_preview=value).validate()


@pytest.mark.parametrize("change", ["preview", "physical", "samples"])
def test_nominal_source_requires_explicit_physical_preview_and_one_energy(tmp_path, change):
    request = _nominal_request(tmp_path)
    if change == "preview":
        request = replace(request, column_exploratory_preview=False)
    elif change == "physical":
        request = replace(request, column_model="ideal_first_order")
    else:
        request = replace(request, source=replace(request.source, energy_samples=9))
    with pytest.raises(ValueError):
        request.validate()


@pytest.mark.parametrize("mode", ["ideal", "analytical"])
def test_nominal_source_rejects_unsupported_executor_before_source_work(tmp_path, mode):
    state = SimpleNamespace(simulation_mode=mode)
    with pytest.raises(ValueError, match="only in unified"):
        _execute_tip_wave(state, _nominal_request(tmp_path), None, "unused", use_cache=False,
            cancelled=lambda: False, progress_callback=None)


def test_joint_surface_nominal_source_is_rejected_without_silent_fallback(tmp_path):
    state = SimpleNamespace(simulation_mode="unified", electron_gun=SimpleNamespace(
        emitter=SimpleNamespace(surface_model=SimpleNamespace(coherence=object()))))
    with pytest.raises(ValueError, match="not implemented for the joint surface/gun"):
        _execute_tip_wave(state, _nominal_request(tmp_path), None, "unused", use_cache=False,
            cancelled=lambda: False, progress_callback=None)


def test_public_gun_dispatch_and_cache_identity_bind_only_active_nominal_option(tmp_path, monkeypatch):
    import temsim.physics.tip_wave_pipeline as pipeline
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from test_tip_wave_pipeline import _quiet_observation_state
    state = _quiet_observation_state()
    state.simulation_mode = "unified"
    state.electron_gun.emitter.emission_energy_ev = 2.
    state.electron_gun.emitter.energy_spread_fwhm_ev = .3
    request = _nominal_request(tmp_path)
    request = replace(request, wave_grid=replace(request.wave_grid, compute_backend="Require GPU"))
    # This public-entry fixture stops at the substituted source; it does not
    # execute a GPU solve. CUDA admission itself has separate policy tests.
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(True, "Substituted nominal source dispatch"))
    before = capture_instrument_snapshot(state).digest
    dependencies, calls = [], []
    def store(_directory, dependency, _maximum):
        dependencies.append(dependency)
        return SimpleNamespace(dependency=dependency)
    def source(gun, **kwargs):
        assert gun.emitter.emission_energy_ev == 2.
        assert gun.emitter.energy_spread_fwhm_ev == .3
        calls.append(kwargs)
        raise RuntimeError("controlled source dispatch reached")
    monkeypatch.setattr(pipeline, "ExecutedWaveStore", store)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", source)
    monkeypatch.setattr("temsim.physics.unified_field_runtime.unified_runtime_preflight", lambda *_: {"issues": []})
    monkeypatch.setattr("temsim.physics.scan_preparation.prepare_scan_drives", lambda *_a, **_kw: None)
    for enabled in (False, True, False):
        with pytest.raises(RuntimeError, match="controlled source dispatch"):
            simulate_tip_wave(state, replace(request, source_nominal_energy_preview=enabled), use_cache=False)
    assert "nominal_energy_preview" not in calls[0]
    assert calls[1]["nominal_energy_preview"] is True
    assert calls[1]["source_numerics"].energy_samples == 1
    assert "nominal_energy_preview" not in calls[2]
    assert dependencies[0] == dependencies[2] != dependencies[1]
    expected = json_digest({"instrument": before, "source": asdict(request.source),
        "gun": asdict(request.gun), "surface": asdict(request.surface), "radial_gun": asdict(request.radial_gun)})
    assert dependencies[0] == expected
    assert capture_instrument_snapshot(state).digest == before


def test_nominal_profile_is_explicit_and_cannot_reuse_non_nominal_executor(tmp_path):
    from temsim.optics.column import default_state
    from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
    from test_unified_wave_runtime import _runtime
    state = default_state()
    state.simulation_mode = "unified"
    request = _nominal_request(tmp_path)
    runtime = _runtime(state, 1., 1.000001)
    old = UnifiedPotentialWaveAdapter(state, replace(request, source_nominal_energy_preview=False), runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    assert "source_nominal_energy_preview" not in old.numerical_profile
    assert new.numerical_profile["source_nominal_energy_preview"] is True
    assert new.numerical_profile["accuracy_qualified"] is False
    assert "spread remains unchanged" in new.numerical_profile["source_energy_approximation"]
    assert "configured FWHM not integrated" in new.numerical_profile["scope"]
    assert old.identity != new.identity
    assert old.snapshot.numerical_digest == new.snapshot.numerical_digest
    assert old.column_grid == new.column_grid


def test_nominal_source_gui_preflight_uses_declared_energy_and_keeps_spread(tmp_path):
    from temsim.optics.column import default_state
    from temsim.physics.coherent_inputs import wave_input_summary
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, DRIVEN_GAUSSIAN_SCHELL
    state = default_state()
    state.electron_gun.emitter.coherence = TipCoherence(boundary_model=DRIVEN_GAUSSIAN_SCHELL)
    state.electron_gun.emitter.emission_energy_ev = 2.
    state.electron_gun.emitter.energy_spread_fwhm_ev = .3
    state.simulation_mode = "unified"
    request = _nominal_request(tmp_path)
    before = state.electron_gun.to_dict()
    summary = wave_input_summary(state, request)
    assert summary["status"] == "SOURCE_READY"
    assert summary["nominal_emission_energy_ev"] == 2.
    assert summary["configured_energy_spread_fwhm_ev"] == .3
    assert not summary["configured_energy_spread_integrated"]
    assert not summary["high_accuracy_eligible"]
    assert state.electron_gun.to_dict() == before
    original = wave_input_summary(state, replace(request, source_nominal_energy_preview=False))
    assert original["status"] == "SOURCE_UNSUPPORTED"
    assert "discard the configured tip energy spread" in original["reason"]
    state.simulation_mode = "ideal"
    assert "only in unified" in wave_input_summary(state, request)["reason"]

