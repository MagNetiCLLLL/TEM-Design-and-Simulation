"""GUI numerical preparation and cache contracts; no physical propagation."""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures, state_model_signature
from temsim.calculation_manifest import capture_calculation_manifest
from temsim.calculation_workflow import workflow_signatures
from temsim.gui import calculation_controller as controllers
from temsim.gui.calculation_request import (
    CapturedCalculationRequest, request_numerical_preparation, resolve_request_step_mm,
)
from temsim.optics.column import default_state
from temsim.simulation_pipeline import CalculationResult


def state_for_request():
    state = default_state()
    state.step_mm = state.history_step_mm = 1.
    state.electron_gun.emitter.ray_count = 2000
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    return state


@pytest.mark.parametrize("step", (1., .12345678901234566, .05, .025, 1e-12, 5e-324))
@pytest.mark.parametrize("quality", ("Preview", "Medium", "High accuracy"))
def test_declared_step_policy_preserves_finer_values_and_preview(quality, step):
    expected = min(step, .05) if quality == "High accuracy" else step
    assert resolve_request_step_mm(quality, step).hex() == expected.hex()
    receipt = request_numerical_preparation(quality, step)
    assert receipt["requested_step_mm"].hex() == step.hex()
    assert receipt["applied_step_mm"].hex() == expected.hex()
    assert receipt["changed"] == (step != expected)


@pytest.mark.parametrize("step", (0., -1., float("nan"), float("inf")))
def test_invalid_resolution_is_rejected(step):
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_request_step_mm("High accuracy", step)


@pytest.mark.parametrize("workflow", ("full", "rays", "stem"))
def test_background_preparation_binds_effective_step_everywhere(monkeypatch, workflow):
    state = state_for_request()
    original = state.to_dict()
    estimates = []
    monkeypatch.setattr(controllers, "estimate_calculation_memory_bytes",
        lambda snapshot, quality, count, step, **kwargs: estimates.append((snapshot.step_mm, step)) or 123)
    captured = CapturedCalculationRequest.capture(state, "High accuracy", 2000, 1., workflow=workflow)
    try:
        prepared = captured.prepare(Event())
        assert captured.step_mm == 1.
        assert prepared.step_mm == prepared.snapshot.step_mm == .05
        assert estimates == [(.05, .05)]
        assert prepared.memory_estimate == 123
        assert prepared.calculation_manifest.state_payload["step_mm"] == .05
        assert prepared.calculation_manifest.instrument_snapshot.restore().step_mm == .05
        assert prepared.calculation_manifest.calculation_signatures == prepared.request_signatures
        assert prepared.request_signatures == workflow_signatures(calculation_signatures(prepared.snapshot), workflow)
        assert prepared.model_signature == state_model_signature(state)
        assert prepared.numerical_preparation["requested_step_mm"] == 1.
        assert prepared.numerical_preparation["applied_step_mm"] == .05
        assert state.to_dict() == original
    finally:
        captured._input_assets.close()


@pytest.mark.parametrize("section_request,workflow", ((None, "rays"), (None, "stem"),
    ({"target_z_mm": 1100., "component_keys": ()}, "full")))
def test_sync_and_background_dispatch_use_same_resolution(monkeypatch, qapp, section_request, workflow):
    state = state_for_request()
    original = state.to_dict()
    estimates, workers = [], []
    monkeypatch.setattr(controllers, "estimate_calculation_memory_bytes",
        lambda snapshot, quality, count, step, **kwargs: estimates.append(step) or 0)
    owner = controllers.CalculationController(persistent_cache_enabled=False)
    monkeypatch.setattr(owner.pool, "start", workers.append)
    owner.submit(state, "High accuracy", 2000, 1., workflow=workflow, section_request=section_request)
    sync = workers.pop()
    owner.invalidate_pending(include_explicit=True)
    owner.submit_background(state, "High accuracy", 2000, 1., workflow=workflow, section_request=section_request)
    preparation = workers.pop()
    preparation.run()
    background = workers.pop()
    assert sync.state.step_mm == background.state.step_mm == .05
    assert sync.request_signatures == background.request_signatures
    assert sync.calculation_manifest.calculation_signatures == background.calculation_manifest.calculation_signatures
    assert sync.numerical_preparation == background.numerical_preparation
    assert all(value == .05 for value in estimates)
    assert state.to_dict() == original
    owner.invalidate_pending(include_explicit=True)


def test_high_accuracy_wrapper_changes_only_new_gui_manifest():
    state = state_for_request()
    old = capture_calculation_manifest(state, ray_count=2000, step_mm=1.)
    new = controllers.CalculationController.build_calculation_manifest(state, 2000, 1.)
    assert old.instrument_snapshot.restore().step_mm == 1.
    assert new.instrument_snapshot.restore().step_mm == .05
    assert old.calculation_signatures["incident"] != new.calculation_signatures["incident"]
    assert state.step_mm == 1.


def test_stem_cannot_continue_old_coarse_incident_checkpoint(monkeypatch):
    from temsim.simulation_workflow import require_incident_result
    state = state_for_request()
    snapshot = controllers.CalculationController._calculation_snapshot(state, "High accuracy", 2000, 1.)
    cp = SimpleNamespace(z_mm=np.array([state.sample.z_mm]), **{
        name: np.zeros((1, 2), dtype=np.float64) for name in
        ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")})
    old = SimpleNamespace(signatures=calculation_signatures(state), simulation=SimpleNamespace(
        incident_checkpoints=cp, gun_trace=object(), incident_plan=object(),
        incident=SimpleNamespace(z=cp.z_mm, x=np.zeros((1, 2)))))
    with pytest.raises(ValueError, match="numerical settings"):
        require_incident_result(snapshot, old, calculation_signatures(snapshot))


def test_completed_result_and_cache_hit_retain_current_request_receipt(qtbot, monkeypatch):
    state = state_for_request()
    workers, results = [], []
    owner = controllers.CalculationController(persistent_cache_enabled=False)
    monkeypatch.setattr(owner.pool, "start", workers.append)
    monkeypatch.setattr(controllers, "estimate_calculation_memory_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(controllers, "calculate", lambda snapshot, **kwargs: CalculationResult(
        simulation=None, energy_filter=None, state_snapshot=snapshot,
        signatures=workflow_signatures(calculation_signatures(snapshot), "rays"), workflow="rays"))
    owner.result_ready.connect(lambda quality, result, duration: results.append(result))
    owner.submit(state, "High accuracy", 2000, 1., workflow="rays")
    workers.pop().run()
    assert len(results) == 1
    first = results[0]
    first_receipt = dict(first.performance["numerical_preparation"])
    assert first_receipt["requested_step_mm"] == 1. and first_receipt["applied_step_mm"] == .05
    owner.submit(state, "High accuracy", 2000, .2, workflow="rays")
    qtbot.waitUntil(lambda: len(results) == 2)
    assert not workers and results[-1].cache_hit
    assert results[-1].performance["numerical_preparation"]["requested_step_mm"] == .2
    assert first.performance["numerical_preparation"] == first_receipt
    assert results[-1].state_snapshot.step_mm == .05
    owner.invalidate_pending(include_explicit=True)


def test_coarse_historical_result_is_not_an_exact_high_accuracy_cache_hit(qapp, monkeypatch):
    state = state_for_request()
    owner = controllers.CalculationController(persistent_cache_enabled=False)
    workers = []
    monkeypatch.setattr(owner.pool, "start", workers.append)
    monkeypatch.setattr(controllers, "estimate_calculation_memory_bytes", lambda *a, **k: 0)
    coarse = CalculationResult(simulation=None, energy_filter=None, state_snapshot=state,
        signatures=workflow_signatures(calculation_signatures(state), "rays"), workflow="rays")
    owner._cache_result(coarse)
    owner.submit(state, "High accuracy", 2000, 1., workflow="rays")
    assert len(workers) == 1
    assert workers[0].state.step_mm == .05
    assert workers[0].request_signatures["request"] != coarse.signatures["request"]
    assert coarse.state_snapshot.step_mm == state.step_mm == 1.
    owner.invalidate_pending(include_explicit=True)


def test_reuse_description_resolves_actual_high_accuracy_step(qapp, monkeypatch):
    state = state_for_request()
    owner = controllers.CalculationController(persistent_cache_enabled=False)
    observed = []
    original = controllers.calculation_signatures_for_request
    def capture(state, **kwargs):
        observed.append(kwargs["step_mm"])
        return original(state, **kwargs)
    monkeypatch.setattr(controllers, "calculation_signatures_for_request", capture)
    owner.describe_high_accuracy_reuse(state, 2000, 1.)
    assert observed == [.05]
    assert state.step_mm == 1.
