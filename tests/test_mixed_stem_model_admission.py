"""Model-policy regression with synthetic boundaries; no propagation."""
from types import SimpleNamespace

import pytest

from temsim import mixed_stem_workflow as workflow
from temsim.physics import compute_backend, mixed_stem_scan as scan
from temsim.physics.particle_wave_source_io import ValidatedParticleBoundary
from test_particle_wave_source_io import boundary


@pytest.mark.parametrize("model", ["copper_coil", "ideal_thin"])
def test_direct_scan_rejects_unsupported_model_before_inputs_or_wave_lift(monkeypatch, model):
    state = SimpleNamespace(simulation_mode="ideal_thin" if model == "ideal_thin" else "unified",
                            round_lens_field_model=model if model == "copper_coil" else "gaussian")
    monkeypatch.setattr(scan, "require_coherent_cuda", lambda *_a, **_k: "Require GPU")
    monkeypatch.setattr(scan, "capture_instrument_snapshot", lambda *_:
        pytest.fail("Unsupported source reached instrument preparation"))
    with pytest.raises(ValueError, match="classical particles only|unavailable in Ideal Optics"):
        scan.simulate_mixed_stem_scan(state, object(), checkpoint=object())


def test_boundary_scan_rejects_copper_model_before_scan_coordinates(boundary, monkeypatch):
    state = SimpleNamespace(simulation_mode="unified", round_lens_field_model="copper_coil")
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *_a, **_k: "Require GPU")
    monkeypatch.setattr(ValidatedParticleBoundary, "restore_inputs_for_wave", lambda self: state)
    monkeypatch.setattr(workflow, "specimen_scan_positions", lambda *_:
        pytest.fail("Unsupported boundary reached scan preparation"))
    with pytest.raises(ValueError, match="classical particles only"):
        workflow.calculate_mixed_stem_boundary(boundary)


def test_completed_cpu_particle_policy_does_not_select_new_cuda_wave_backend(boundary, monkeypatch):
    import numpy as np
    state = SimpleNamespace(simulation_mode="unified", round_lens_field_model="gaussian",
                            acceleration_enabled=True, acceleration_backend="Numba CPU")
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
