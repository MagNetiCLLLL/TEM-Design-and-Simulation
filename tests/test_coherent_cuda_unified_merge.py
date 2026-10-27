"""Merge contracts for CUDA admission and unified preview; no coherent solve."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import compute_backend as backend
from temsim.physics import tip_wave_pipeline as pipeline


def preview_request():
    original = pipeline.TipWaveRequest()
    return replace(original, stop="plane", observation_z_mm=100., tip_time_s=.25,
        source=replace(original.source, energy_samples=1),
        wave_grid=replace(original.wave_grid, compute_backend="Prefer GPU"),
        column_kinetic_order=6, column_support_projection=True,
        column_weighted_resampling=True, column_exploratory_preview=True,
        source_nominal_energy_preview=True, column_preview_segment_resampling=True,
        column_preview_phase_reselection=True, column_preview_pixels=512,
        column_preview_support_probability=.01, column_preview_support_l2=.1)


def test_admission_preserves_preview_and_rechecks_strict_capture(monkeypatch):
    probes = []
    monkeypatch.setattr(backend, "coherent_cuda_capability", lambda:
        probes.append(True) or backend.BackendCapability(True, "mock CUDA"))
    original = preview_request()
    admitted = pipeline.require_tip_wave_cuda(original)
    assert replace(admitted, wave_grid=original.wave_grid) == original
    assert admitted.wave_grid == replace(original.wave_grid, compute_backend="Require GPU")
    assert pipeline.require_tip_wave_cuda(admitted) is admitted
    assert probes == [True, True]
    assert original.wave_grid.compute_backend == "Prefer GPU"


def test_unified_adapter_keeps_strict_residual_and_material_policy(monkeypatch):
    from temsim.physics import unified_wave_runtime as unified
    monkeypatch.setattr(backend, "coherent_cuda_capability", lambda:
        backend.BackendCapability(True, "mock CUDA"))
    snapshot = SimpleNamespace(numerical_digest="mock snapshot", contributions=())
    monkeypatch.setattr(unified, "_static_wave_snapshot", lambda _: (snapshot, ()))
    state = SimpleNamespace(simulation_mode="unified", simulation_time_s=.25)
    runtime = SimpleNamespace(metadata={}, require_state=lambda _: None,
        digest="mock runtime", start_z_m=0., stop_z_m=1.)
    admitted = pipeline.require_tip_wave_cuda(preview_request())
    adapter = unified.UnifiedPotentialWaveAdapter(state, admitted, runtime=runtime)
    assert adapter.request is admitted
    assert adapter.requested_backend == "Require GPU"
    assert adapter._options(100., 200.)["residual_compute_backend"] == "Require GPU"
    assert adapter.material_grid.compute_backend == "Require GPU"
    # Host chart preparation has a separate explicit policy, not a retry.
    assert adapter.column_grid.compute_backend == "CPU"
    assert adapter.numerical_profile["phase_preparation_backend"] == "CPU"
    assert adapter.numerical_profile["source_nominal_energy_preview"] is True
    assert adapter.numerical_profile["effective_column_maximum_pixels"] == 512
    assert adapter.numerical_profile["accuracy_qualified"] is False


def test_strict_unified_residual_never_retries_on_cpu(monkeypatch):
    from temsim.physics import potential_wave_device as device
    attempted = []
    def unavailable(policy, **kwargs):
        attempted.append(policy)
        raise backend.GPUExecutionError("out_of_memory", "mock device budget")
    monkeypatch.setattr(device, "device_scope", unavailable)
    monkeypatch.setattr(device, "release_device_memory", lambda: None)
    values = np.zeros((2, 2))
    with pytest.raises(backend.GPUExecutionError, match="out_of_memory"):
        device.execute_residual_exponential(values, values, values, values,
            values, values, 1., phase=values, scalar_splitting=False,
            kinetic_order=6, quartic_scalar_splitting=False,
            higher_order_scalar_splitting=False, count=1, degree=1,
            k_distance=1., boundary_limit=1., envelope_limit=1.,
            cancelled=lambda: False, requested="Require GPU")
    assert attempted == ["Require GPU"]


def gui_owner(request):
    messages, progress = [], []
    owner = SimpleNamespace(_closed=False, _settings=lambda: object(),
        state_provider=None, _state=SimpleNamespace(simulation_mode="unified", simulation_time_s=.25),
        _make_request=lambda *_: request, result=object(),
        _show_source_check=lambda *args: None,
        status=SimpleNamespace(setStyleSheet=lambda *_: None, setText=messages.append),
        progress_detail=SimpleNamespace(emit=progress.append))
    return owner, messages, progress


def test_gui_admission_precedes_preparation_and_settings_remain_inspectable(monkeypatch):
    from temsim.gui import coherent_beam as gui
    monkeypatch.setattr(backend, "coherent_cuda_capability", lambda:
        backend.BackendCapability(False, "mock missing CUDA"))
    captures = []
    monkeypatch.setattr(gui, "prepare_coherent_state", lambda state, settings:
        captures.append(state) or state)
    monkeypatch.setattr(gui, "wave_input_summary", lambda *_: {"status": "SOURCE_READY"})
    request = preview_request()
    owner, messages, progress = gui_owner(request)
    previous = owner.result
    with pytest.raises(backend.GPUExecutionError, match="mock missing CUDA"):
        gui.CoherentBeamPage.capture_calculation_inputs(owner)
    assert not captures
    assert progress[-1].status == "failed"
    assert owner.result is previous
    assert "Previous result remains displayed" in messages[-1]
    state, captured_request, _ = gui.CoherentBeamPage.capture_calculation_inputs(owner, for_execution=False)
    assert captures == [state]
    assert captured_request is request
    assert captured_request.source_nominal_energy_preview
    assert captured_request.tip_time_s == .25


def test_captured_gui_rejection_preserves_session_and_reports_terminal_progress(monkeypatch):
    from temsim.gui import coherent_beam as gui
    monkeypatch.setattr(backend, "coherent_cuda_capability", lambda:
        backend.BackendCapability(False, "mock missing CUDA"))
    request = preview_request()
    owner, messages, progress = gui_owner(request)
    previous, session = owner.result, object()
    owner._observation_session = session
    owner.state_set = SimpleNamespace(cancel=lambda: pytest.fail("Rejected work must retain the session"))
    gui.CoherentBeamPage.start_captured_calculation(owner, owner._state, request, {})
    assert owner._observation_session is session
    assert owner.result is previous
    assert progress[-1].status == "failed"
    assert "Previous result retained" in messages[-1]
