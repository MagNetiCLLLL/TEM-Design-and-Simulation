"""Mocked acquisition ownership checks; these do not qualify a physical probe."""
from copy import deepcopy
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from temsim import mixed_stem_workflow as workflow


def state():
    return SimpleNamespace(
        ac_deflector=SimpleNamespace(scan_pixels_x=4, scan_lines=3, scan_pixel_size_nm=.02),
        sample=SimpleNamespace(scan_origin_x_nm=.1, scan_origin_y_nm=-.2,
                               stem_wave_enabled=False),
        electron_gun=SimpleNamespace(emitter=SimpleNamespace(ray_count=16384)),
        step_mm=.025, history_step_mm=2., simulation_mode="unified")


def test_selected_rectangular_raster_and_origin():
    positions = workflow.specimen_scan_positions(state())
    assert positions.shape == (3, 4, 2)
    np.testing.assert_allclose(positions[0, :, 0], [.07, .09, .11, .13])
    np.testing.assert_allclose(positions[:, 0, 1], [-.22, -.2, -.18])


@pytest.mark.parametrize("attribute,value", [("scan_pixels_x", 1), ("scan_lines", 4097),
    ("scan_pixel_size_nm", np.nan), ("scan_pixel_size_nm", 0)])
def test_invalid_raster_is_rejected(attribute, value):
    inputs = state()
    setattr(inputs.ac_deflector, attribute, value)
    with pytest.raises(ValueError, match="raster"):
        workflow.specimen_scan_positions(inputs)


@pytest.fixture
def mocked(monkeypatch):
    from temsim.physics import compute_backend, mixed_wave_particles
    from temsim import simulation_pipeline, instrument_snapshot, simulation_modes
    from temsim.specimen import source
    calls = []
    input_state = state()
    digest = "a"*64
    captured = deepcopy(input_state)
    inputs = SimpleNamespace(digest=digest, restore=lambda: captured)
    monkeypatch.setattr(instrument_snapshot, "capture_instrument_snapshot", lambda _state: inputs)
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *_: calls.append("cuda"))
    monkeypatch.setattr(simulation_modes, "require_coherent_mode", lambda *_: None)
    monkeypatch.setattr(source, "validate_sample_source", lambda *_: None)
    monkeypatch.setattr(source, "specimen_is_vacuum", lambda *_: False)
    request = SimpleNamespace(validate=lambda: request)
    backend = SimpleNamespace(MixedStemScanRequest=lambda: request)
    backend.preflight_mixed_stem_material = lambda *_: calls.append("material")
    def scan(state, source, request, **kwargs):
        calls.append(("scan", state, kwargs["positions_nm"].copy()))
        return "mock_scan"
    backend.simulate_mixed_stem_scan = scan
    monkeypatch.setitem(sys.modules, "temsim.physics.mixed_stem_scan", backend)
    particles = SimpleNamespace(state_snapshot=captured)
    def calculate(value, **kwargs):
        calls.append(("particles", value, kwargs["workflow"]))
        assert value is not input_state
        assert value.electron_gun.emitter.ray_count == 16384
        assert value.step_mm == .025 and value.history_step_mm == 2.
        return particles
    monkeypatch.setattr(simulation_pipeline, "calculate", calculate)
    monkeypatch.setattr(mixed_wave_particles, "particle_source_from_result",
                        lambda value, **kwargs: calls.append(("handoff", value)) or "mock_source")
    return SimpleNamespace(calls=calls, inputs=input_state, captured=captured,
                           particles=particles, request=request)


def test_acquisition_retains_selected_particles_and_inputs(mocked):
    result = workflow.calculate_mixed_stem(mocked.inputs)
    assert result.scan == "mock_scan" and result.particle_result is mocked.particles
    assert mocked.calls[0] == "cuda"
    assert mocked.calls[1] == "material"
    assert mocked.calls[2][0] == "particles" and mocked.calls[2][2] == "rays"
    assert not hasattr(mocked.inputs, "_tuning_cancelled")


def test_exact_existing_result_is_reused(mocked):
    result = workflow.calculate_mixed_stem(mocked.inputs, existing_result=mocked.particles)
    assert result.particle_result is mocked.particles
    assert not any(call[0] == "particles" for call in mocked.calls if isinstance(call, tuple))
    assert any(call[0] == "handoff" for call in mocked.calls if isinstance(call, tuple))


def test_cuda_failure_prevents_particle_preparation(mocked, monkeypatch):
    from temsim.physics import compute_backend
    def unavailable(*args):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", unavailable)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        workflow.calculate_mixed_stem(mocked.inputs)
    assert not mocked.calls


def test_unsupported_material_is_rejected_before_classical_transport(mocked, monkeypatch):
    def unsupported(*_):
        mocked.calls.append("material")
        raise ValueError("Unsupported finite Fourier reference")
    monkeypatch.setattr(sys.modules["temsim.physics.mixed_stem_scan"],
                        "preflight_mixed_stem_material", unsupported)
    with pytest.raises(ValueError, match="Unsupported finite Fourier"):
        workflow.calculate_mixed_stem(mocked.inputs)
    assert mocked.calls == ["cuda", "material"]


def test_small_selected_population_is_rejected(mocked):
    mocked.captured.electron_gun.emitter.ray_count = 999
    with pytest.raises(ValueError, match="1000 original"):
        workflow.calculate_mixed_stem(mocked.inputs)
    assert mocked.calls == ["cuda"]


def test_cancelled_request_never_prepares_or_publishes(mocked):
    with pytest.raises(InterruptedError, match="cancelled"):
        workflow.calculate_mixed_stem(mocked.inputs, cancelled=lambda: True)
    assert not mocked.calls


def test_cancel_after_exact_handoff_never_starts_scan(mocked, monkeypatch):
    from temsim.physics import mixed_wave_particles
    cancelled = [False]
    def handoff(*args, **kwargs):
        cancelled[0] = True
        return "mock_source"
    monkeypatch.setattr(mixed_wave_particles, "particle_source_from_result", handoff)
    with pytest.raises(InterruptedError, match="cancelled"):
        workflow.calculate_mixed_stem(mocked.inputs, cancelled=lambda: cancelled[0])
    assert not any(call[0] == "scan" for call in mocked.calls if isinstance(call, tuple))


def test_new_history_reduces_wave_stage_headroom_without_changing_request(mocked, monkeypatch):
    from temsim.physics.host_memory_budget import host_memory_budget, active_host_memory_budget_bytes
    mocked.particles.history = np.ones(4096, dtype=np.float64)
    cap = 256*1024
    observed = []
    def scan(*args, **kwargs):
        observed.append(active_host_memory_budget_bytes())
        return "mock_scan"
    monkeypatch.setattr(sys.modules["temsim.physics.mixed_stem_scan"], "simulate_mixed_stem_scan", scan)
    with host_memory_budget(cap):
        workflow.calculate_mixed_stem(mocked.inputs)
        assert active_host_memory_budget_bytes() == cap
    assert 0 < observed[0] <= cap-mocked.particles.history.nbytes


def test_existing_history_is_not_counted_twice_in_wave_stage(mocked, monkeypatch):
    from temsim.physics.host_memory_budget import host_memory_budget, active_host_memory_budget_bytes
    mocked.particles.history = np.ones(4096, dtype=np.float64)
    cap = 256*1024
    observed = []
    def scan(*args, **kwargs):
        observed.append(active_host_memory_budget_bytes())
        return "mock_scan"
    monkeypatch.setattr(sys.modules["temsim.physics.mixed_stem_scan"], "simulate_mixed_stem_scan", scan)
    with host_memory_budget(cap):
        workflow.calculate_mixed_stem(mocked.inputs, existing_result=mocked.particles)
    assert cap-1024 < observed[0] <= cap
