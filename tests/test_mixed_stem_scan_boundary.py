"""Explicit old classical-boundary to new wave-version handoff, no propagation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.mixed_stem_scan import _source_input_handoff
from test_particle_wave_source_io import boundary


def test_explicit_checked_boundary_retains_particle_identity_and_records_new_wave_version(boundary):
    current = InstrumentSnapshot(boundary.inputs.graph, boundary.inputs.external_inputs, "c"*64)
    record = _source_input_handoff(current, boundary.source, boundary)
    assert record["particle_input_digest"] == boundary.inputs.digest == boundary.source.input_digest
    assert record["wave_input_digest"] == current.digest
    assert record["particle_input_digest"] != record["wave_input_digest"]
    assert record["particle_producer_implementation"] == "a"*64
    assert record["wave_implementation"] == "c"*64
    assert record["original_particle_identity_preserved"] is True


def test_cross_version_source_cannot_bypass_explicit_boundary_ownership(boundary):
    current = InstrumentSnapshot(boundary.inputs.graph, boundary.inputs.external_inputs, "c"*64)
    with pytest.raises(ValueError, match="inputs changed after particle transport"):
        _source_input_handoff(current, boundary.source, None)
    with pytest.raises(TypeError, match="verified ValidatedParticleBoundary"):
        _source_input_handoff(current, boundary.source, SimpleNamespace(source=boundary.source, inputs=boundary.inputs))
    changed = replace(boundary.source, reference_current_a=boundary.source.reference_current_a*2)
    with pytest.raises(ValueError, match="does not own"):
        _source_input_handoff(current, changed, boundary)


def test_public_imported_boundary_wave_still_rejects_unavailable_cuda_first(boundary, monkeypatch):
    from temsim.physics import mixed_stem_scan as scan
    from temsim.physics.compute_backend import GPUExecutionError
    monkeypatch.setattr(scan, "require_coherent_cuda", lambda *_a, **_k:
        (_ for _ in ()).throw(GPUExecutionError("unavailable", "Synthetic unavailable CUDA")))
    monkeypatch.setattr(scan, "capture_instrument_snapshot", lambda *_:
        pytest.fail("Unavailable CUDA reached imported particle inputs"))
    with pytest.raises(GPUExecutionError, match="Synthetic unavailable CUDA"):
        scan.simulate_mixed_stem_scan(object(), boundary.source, validated_boundary=boundary)


@pytest.mark.parametrize("change", ["graph", "external"])
def test_source_reuse_cannot_change_captured_optics_or_external_inventory(boundary, change):
    graph = thaw_json(boundary.inputs.graph)
    external = thaw_json(boundary.inputs.external_inputs)
    if change == "graph":
        graph["source_current_setting"] = "edited"
    else:
        external = [{"role": "new optics file", "path": "changed.toml", "sha256": "d"*64, "content_hex": None}]
    current = InstrumentSnapshot(graph, tuple(external), "c"*64)
    with pytest.raises(ValueError, match="captured physical inputs"):
        _source_input_handoff(current, boundary.source, boundary)
