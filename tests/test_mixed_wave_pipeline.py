"""Mixed boundary ownership, exact planes, cancellation and physical receivers."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics import mixed_wave_pipeline as pipeline
from temsim.physics.gun_wave_transport import specimen_entrance_z_mm
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.particle_wave_source import ParticleWaveSource
from temsim.physics.wave_checkpoint import EMITTED_ELECTRON_REFERENCE, WaveCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode


@pytest.fixture
def boundary(monkeypatch):
    state = default_state()
    state.simulation_mode = "unified"
    state.acceleration_backend = "CPU"
    upper = specimen_entrance_z_mm(state)
    count = 1024
    zeros = np.zeros(count)
    source = ParticleWaveSource(zeros, zeros, zeros, zeros, np.arange(count),
        np.full(count, .5/count), np.full(count, 300000.), np.full(count, 8e-9),
        upper, 1e-5, count, capture_instrument_snapshot(state).digest)
    mode = WaveMode(PlaneWave(np.ones((8, 8), complex)/8, np.eye(2)*1e-10,
        np.zeros(2)), .5, EMITTED_ELECTRON_REFERENCE, "mixed:0", 300.)
    checkpoint = WaveCheckpoint(BeamState((mode,), EMITTED_ELECTRON_REFERENCE), upper, 1e-5,
        {"schema": "particle-mixed-boundary-v1"})
    built = []
    def build(value, numerics, **kwargs):
        built.append(value)
        return checkpoint
    monkeypatch.setattr(pipeline, "require_coherent_cuda", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "build_particle_wave_checkpoint", build)
    return state, source, checkpoint, built


def test_one_accepted_boundary_is_reused_without_adding_amplitudes_or_rerunning_particles(boundary):
    state, source, checkpoint, built = boundary
    supplied = []
    def supply():
        supplied.append(True)
        return source
    session = pipeline.MixedWaveObservationSession(state, source=supply)
    first = session.observe(source.z_mm)
    cached = session.observe(source.z_mm)
    assert len(supplied) == len(built) == 1
    assert first.checkpoint is cached.checkpoint is checkpoint
    assert not first.propagation_cache_hit and cached.propagation_cache_hit
    assert cached.checkpoint.transmitted_current_a == pytest.approx(5e-6)
    assert state.acceleration_backend == session.state.acceleration_backend == "CPU"


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -1.])
def test_invalid_or_upstream_plane_never_constructs_a_wave(boundary, invalid):
    state, source, _, built = boundary
    session = pipeline.MixedWaveObservationSession(state, source=source)
    with pytest.raises(ValueError):
        session.observe(invalid)
    assert not built and session.entrance is None


def test_source_from_different_optics_is_rejected_before_wave_construction(boundary):
    state, source, _, built = boundary
    source = replace(source, input_digest="0"*64)
    with pytest.raises(ValueError, match="different captured"):
        pipeline.MixedWaveObservationSession(state, source=source).observe(source.z_mm)
    assert not built


def test_source_cannot_be_relabelled_at_an_arbitrary_downstream_plane(boundary):
    state, source, _, built = boundary
    source = replace(source, z_mm=source.z_mm+1e-6)
    with pytest.raises(ValueError, match="exact specimen upper"):
        pipeline.MixedWaveObservationSession(state, source=source).observe(source.z_mm)
    assert not built


def test_cancellation_during_lift_does_not_publish_or_cache_partial_fields(boundary, monkeypatch):
    state, source, checkpoint, _ = boundary
    cancelled = [False]
    def build(*args, **kwargs):
        cancelled[0] = True
        return checkpoint
    monkeypatch.setattr(pipeline, "build_particle_wave_checkpoint", build)
    session = pipeline.MixedWaveObservationSession(state, source=source)
    with pytest.raises(InterruptedError):
        session.observe(source.z_mm, cancelled=lambda: cancelled[0])
    assert session.entrance is None and not session._completed


def test_session_rejects_concurrent_observations_and_closed_owner(boundary):
    state, source, _, _ = boundary
    session = pipeline.MixedWaveObservationSession(state, source=source)
    session._lock.acquire()
    with pytest.raises(RuntimeError, match="Only one"):
        session.observe(source.z_mm)
    session._lock.release()
    session.close()
    with pytest.raises(ValueError, match="closed"):
        session.observe(source.z_mm)


def test_explicit_cpu_selection_is_rejected_before_particle_supplier(boundary, monkeypatch):
    state, source, _, _ = boundary
    def rejected(*args, **kwargs):
        raise RuntimeError("CUDA required")
    monkeypatch.setattr(pipeline, "require_coherent_cuda", rejected)
    supplied = []
    with pytest.raises(RuntimeError, match="CUDA required"):
        pipeline.simulate_mixed_wave(state, lambda: supplied.append(True),
            replace(pipeline.MixedWaveRequest(), observation_z_mm=source.z_mm))
    assert not supplied


def _fake_transport(monkeypatch, boundary, *, vacuum):
    from temsim.physics import unified_field_runtime, unified_wave_runtime
    state, source, checkpoint, _ = boundary
    calls = []
    class Adapter:
        def __init__(self, *args, **kwargs):
            pass
        def column(self, current, z, **kwargs):
            calls.append(("column", z, current.beam.total_weight))
            return replace(current, plane_z_mm=z)
        def specimen(self, current, *, observation_z_mm=None, **kwargs):
            bottom = state.sample.z_mm+state.sample.thickness_nm*.5e-6
            stop = bottom if observation_z_mm is None else observation_z_mm
            calls.append(("specimen", stop, current.beam.total_weight))
            return replace(current, plane_z_mm=stop)
    monkeypatch.setattr(unified_field_runtime, "capture_unified_field_runtime",
                        lambda *args, **kwargs: SimpleNamespace(snapshot=object()))
    monkeypatch.setattr(unified_wave_runtime, "UnifiedPotentialWaveAdapter", Adapter)
    session = pipeline.MixedWaveObservationSession(state, source=source)
    monkeypatch.setattr(session, "_is_vacuum", lambda: vacuum)
    monkeypatch.setattr(session, "_store", lambda *args: None)
    return state, source, session, calls


def test_interior_material_observation_uses_original_sample_without_moving_faces(boundary, monkeypatch):
    state, source, session, calls = _fake_transport(monkeypatch, boundary, vacuum=False)
    z = float(state.sample.z_mm)
    result = session.observe(z)
    assert calls == [("specimen", z, .5)]
    assert result.checkpoint.plane_z_mm == z
    assert specimen_entrance_z_mm(session.state) == source.z_mm


def test_downstream_observation_does_not_treat_inserted_receivers_as_transparent(boundary, monkeypatch):
    from temsim.detector import wave_readout
    state, source, _, _ = boundary
    for detector in state.recording_planes:
        detector.inserted = False
    state.camera.inserted = True
    state.camera.z_mm = source.z_mm+1e-6
    source = replace(source, input_digest=capture_instrument_snapshot(state).digest)
    updated = (state, source, boundary[2], boundary[3])
    _, _, session, calls = _fake_transport(monkeypatch, updated, vacuum=True)
    def absorb(current, detector):
        calls.append(("absorb", detector.z_mm, current.beam.total_weight))
        mode = current.beam.modes[0]
        return replace(current, beam=BeamState((replace(mode, weight_per_reference_electron=.25),),
                                               EMITTED_ELECTRON_REFERENCE))
    monkeypatch.setattr(wave_readout, "_apply_recording_stop", absorb)
    target = source.z_mm+2e-6
    result = session.observe(target)
    assert [row[0] for row in calls] == ["column", "absorb", "column"]
    assert calls[-1][-1] == result.checkpoint.beam.total_weight == .25
    assert result.checkpoint.record["tip_coherent_executed"] is False


def test_thin_particle_model_cannot_silently_become_a_field_wave(boundary):
    state, source, _, _ = boundary
    state.simulation_mode = "ideal_thin"
    with pytest.raises(ValueError, match="Ideal Optics"):
        pipeline.MixedWaveObservationSession(state, source=source)
