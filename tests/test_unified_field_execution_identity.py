"""Source-free execution-policy and device identity checks for particle reuse."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from temsim.physics import unified_field_runtime as module
from temsim.physics.unified_spatial_dependencies import spatial_runtime_identity
from test_unified_field_runtime import parked_state


def fake_cuda_sampler(monkeypatch):
    device = {"ordinal": 0}
    calls = []
    def prepare(snapshot, *, execution_backend):
        calls.append(execution_backend)
        return SimpleNamespace(execution_identity={
            "schema": "synthetic-field-execution-only", "device": device["ordinal"],
            "kernel": "unit-test-kernel", "backend": execution_backend,
        })
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential",
                        SimpleNamespace(prepare_cuda_coil_potential=prepare))
    return device, calls


def test_disabled_acceleration_does_not_prepare_a_cuda_field(monkeypatch):
    state = parked_state()
    state.acceleration_backend = "Require GPU"
    state.acceleration_enabled = False
    _, calls = fake_cuda_sampler(monkeypatch)
    identity = module.field_execution_identity(state, object())
    assert identity["policy"] == "CPU" and not calls


@pytest.mark.parametrize("policy", ["Auto", "CPU", "Numba CPU"])
def test_explicit_cpu_and_auto_keep_existing_field_dispatch_without_cuda(monkeypatch, policy):
    state = parked_state()
    state.acceleration_backend = policy
    _, calls = fake_cuda_sampler(monkeypatch)
    assert module.field_execution_identity(state, object())["sampler"] == "retained-providers"
    assert not calls


def test_device_switch_invalidates_runtime_and_executed_spatial_prefix(monkeypatch):
    from temsim.physics.unified_scan_cache import response_dependencies, matching_response_dependencies
    state = parked_state()
    state.acceleration_backend = "Prefer GPU"
    device, _ = fake_cuda_sampler(monkeypatch)
    first = module.get_unified_field_runtime(state)
    assert module.get_unified_field_runtime(state) is first
    bounds = np.array([[-1e-9, -1e-9, .7], [1e-9, 1e-9, .700001]])
    old_spatial = spatial_runtime_identity(first, bounds)
    old_responses = [response_dependencies(first, region) for region in (None, bounds)]
    device["ordinal"] = 1
    with pytest.raises(ValueError, match="backend or CUDA device changed"):
        first.require_state(state)
    second = module.get_unified_field_runtime(state)
    assert first.source_identity == second.source_identity
    assert first.snapshot.numerical_digest == second.snapshot.numerical_digest
    assert first.digest != second.digest
    assert spatial_runtime_identity(second, bounds) != old_spatial
    for region, old in zip((None, bounds), old_responses):
        assert response_dependencies(second, region) != old
        assert not matching_response_dependencies({"receipt": {"response_dependencies": old}}, second)


def test_cuda_execution_failure_is_not_relabelled_as_cpu_identity(monkeypatch):
    state = parked_state()
    state.acceleration_backend = "Prefer GPU"
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic CUDA execution failure")
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential",
                        SimpleNamespace(prepare_cuda_coil_potential=fail))
    with pytest.raises(RuntimeError, match="Synthetic CUDA execution failure"):
        module.field_execution_identity(state, object())


def test_dynamic_capture_keeps_dynamic_providers_outside_static_cuda_sampler(monkeypatch):
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    state = parked_state()
    state.acceleration_backend = "Prefer GPU"
    snapshot = object.__new__(DynamicPotentialSnapshot)
    static = object()
    object.__setattr__(snapshot, "static_snapshot", static)
    seen = []
    def prepare(value, *, execution_backend):
        seen.append(value)
        return SimpleNamespace(execution_identity={"scope": "Synthetic static field adapter"})
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential",
                        SimpleNamespace(prepare_cuda_coil_potential=prepare))
    identity = module.field_execution_identity(state, snapshot)
    assert seen == [static]
    assert "original interval guards" in identity["dynamic_sources"]
