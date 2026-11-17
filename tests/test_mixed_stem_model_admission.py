"""Model-policy regression with synthetic boundaries; no propagation."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import mixed_stem_workflow as workflow
from temsim.physics import compute_backend, mixed_stem_scan as scan
from temsim.physics.mixed_stem_admission import require_local_mixed_stem_mode
from temsim.physics.particle_wave_source_io import ValidatedParticleBoundary
from test_particle_wave_source_io import boundary


def coil_state():
    return SimpleNamespace(simulation_mode="unified", round_lens_field_model="copper_coil",
        sample=SimpleNamespace(z_mm=1500., thickness_nm=10.),
        ac_deflector=SimpleNamespace(scan_enabled=True, wobble_enabled=False))


def static_runtime(monkeypatch, *, a0=(0., 0., 0.), gradient=None, hessian=None):
    from temsim.physics import unified_field_runtime
    calls = []
    def sample(points):
        calls.append(("sample", points.copy()))
        return {"A_t_m": np.broadcast_to(np.asarray(a0, dtype=float), points.shape)}
    runtime = SimpleNamespace(snapshot=SimpleNamespace(sample=sample,
        gauge=SimpleNamespace(gradient_t_m=np.zeros(3) if gradient is None else np.asarray(gradient),
            hessian_t=np.zeros((3, 3)) if hessian is None else np.asarray(hessian))))
    def capture(state, start, stop):
        calls.append(("capture", state, start, stop))
        return runtime
    monkeypatch.setattr(unified_field_runtime, "capture_unified_field_runtime", capture)
    return calls


def dynamic_runtime(monkeypatch):
    from temsim.physics import unified_field_runtime
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    # Type-only synthetic guard fixture; no field, source or wave execution.
    snapshot = object.__new__(DynamicPotentialSnapshot)
    monkeypatch.setattr(unified_field_runtime, "capture_unified_field_runtime",
                        lambda *_: SimpleNamespace(snapshot=snapshot))


@pytest.mark.parametrize("model", ["ideal_thin", "ideal", "custom"])
def test_direct_scan_rejects_unsupported_model_before_inputs_or_wave_lift(monkeypatch, model):
    state = SimpleNamespace(simulation_mode=model, round_lens_field_model="gaussian")
    monkeypatch.setattr(scan, "require_coherent_cuda", lambda *_a, **_k: "Require GPU")
    monkeypatch.setattr(scan, "capture_instrument_snapshot", lambda *_:
        pytest.fail("Unsupported source reached instrument preparation"))
    with pytest.raises(ValueError, match="Finite-Field Optics|unavailable in Ideal Optics"):
        scan.simulate_mixed_stem_scan(state, object(), checkpoint=object())


def test_direct_static_current_scan_passes_model_admission_but_still_requires_executed_source(monkeypatch):
    state = coil_state()
    static_runtime(monkeypatch)
    monkeypatch.setattr(scan, "require_coherent_cuda", lambda *_a, **_k: "Require GPU")
    monkeypatch.setattr(scan, "capture_instrument_snapshot", lambda *_:
        pytest.fail("An invalid source reached input capture"))
    with pytest.raises(TypeError, match="executed immutable ParticleWaveSource"):
        scan.simulate_mixed_stem_scan(state, object(), checkpoint=object())


def test_boundary_scan_rejects_dynamic_current_field_before_scan_coordinates(boundary, monkeypatch):
    state = coil_state()
    dynamic_runtime(monkeypatch)
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *_a, **_k: "Require GPU")
    monkeypatch.setattr(ValidatedParticleBoundary, "restore_inputs_for_wave", lambda self: state)
    monkeypatch.setattr(workflow, "specimen_scan_positions", lambda *_:
        pytest.fail("Dynamic boundary reached scan preparation"))
    with pytest.raises(ValueError, match="stationary virtual-raster inputs"):
        workflow.calculate_mixed_stem_boundary(boundary)
    assert state.ac_deflector.scan_enabled


def test_current_field_preflight_precedes_new_particle_preparation(monkeypatch):
    from temsim import instrument_snapshot, simulation_pipeline
    state = coil_state()
    dynamic_runtime(monkeypatch)
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *_a, **_k: "Require GPU")
    monkeypatch.setattr(instrument_snapshot, "capture_instrument_snapshot", lambda *_:
        pytest.fail("Dynamic source reached input preparation"))
    monkeypatch.setattr(simulation_pipeline, "calculate", lambda *_a, **_k:
        pytest.fail("Dynamic source reached particle transport"))
    with pytest.raises(ValueError, match="no drive was disabled or frozen automatically"):
        workflow.calculate_mixed_stem(state)
    assert state.ac_deflector.scan_enabled


@pytest.mark.parametrize("axis_a", [0., 1e-20])
def test_static_current_admission_samples_complete_runtime_at_actual_upper_surface(monkeypatch, axis_a):
    state = coil_state()
    calls = static_runtime(monkeypatch, a0=(axis_a, 0., 0.))
    require_local_mixed_stem_mode(state)
    assert calls[0][0] == "capture" and calls[0][1] is state
    start = (state.sample.z_mm-state.sample.thickness_nm*.5e-6)*1e-3
    assert calls[0][2:] == (start, start+state.sample.thickness_nm*1e-9)
    np.testing.assert_array_equal(calls[1][1], [[0., 0., start]])
    assert state.ac_deflector.scan_enabled  # A proved static field is not inferred from this flag.


@pytest.mark.parametrize("a0,gradient,hessian", [
    ((1.01e-20, 0., 0.), None, None),
    ((0., 0., 0.), (0., 1e-30, 0.), None),
    ((0., 0., 0.), None, np.diag([0., 0., 1e-30])),
])
def test_current_admission_retains_native_zero_axis_gauge_gate(monkeypatch, a0, gradient, hessian):
    static_runtime(monkeypatch, a0=a0, gradient=gradient, hessian=hessian)
    with pytest.raises(ValueError, match="native zero-axis magnetic gauge"):
        require_local_mixed_stem_mode(coil_state())


def test_local_current_admission_does_not_unlock_general_coherent_propagation(monkeypatch):
    from temsim.simulation_modes import require_coherent_mode
    state = coil_state()
    static_runtime(monkeypatch)
    require_local_mixed_stem_mode(state)
    with pytest.raises(ValueError, match="[Cc]oherent propagation"):
        require_coherent_mode(state)


@pytest.mark.parametrize("model", ["gaussian", "copper_coil"])
def test_completed_cpu_particle_policy_does_not_select_new_cuda_wave_backend(boundary, monkeypatch, model):
    state = coil_state()
    state.round_lens_field_model = model
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    if model == "copper_coil":
        static_runtime(monkeypatch)
    calls = []
    monkeypatch.setattr(compute_backend, "require_coherent_cuda",
        lambda *a, **k: calls.append(("cuda", a, k)) or "Require GPU")
    monkeypatch.setattr(ValidatedParticleBoundary, "restore_inputs_for_wave", lambda self: state)
    monkeypatch.setattr(workflow, "specimen_scan_positions", lambda *_: np.zeros((1, 1, 2)))
    monkeypatch.setattr(scan, "simulate_mixed_stem_scan", lambda *a, **k:
        calls.append(("scan", a, k)) or "synthetic_cuda_scan")
    result = workflow.calculate_mixed_stem_boundary(boundary)
    assert calls[0] == ("cuda", ("Require GPU",), {})
    assert result.scan == "synthetic_cuda_scan"
    assert result.state_snapshot is state
    assert state.acceleration_backend == "Numba CPU"
    assert calls[1][1][1] is boundary.source
