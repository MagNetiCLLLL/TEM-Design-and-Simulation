"""Request-scoped checkpoint I/O and single-pass display/moments regressions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import os
from pathlib import Path
import weakref

import numpy as np
import pytest

from temsim.gui.coherent_beam import _intensity_preview, _wave_read_session_bytes, _wave_resource_claim
from temsim.immutable_json import thaw_json
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.beam_comparison import _WavePlaneMoments, wave_plane_moments
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore, wave_read_session
from temsim.physics.wave_flux import BeamState, WaveMode
import temsim.physics.wave_checkpoint_store as store_module


@pytest.fixture
def stored(tmp_path):
    store = ExecutedWaveStore(tmp_path, "request-scoped-checkpoint-fixture", 1 << 24)
    modes = []
    yy, xx = np.indices((32, 40))
    for index in range(2):
        amplitude = np.exp(-((xx-18-index*3)**2+(yy-14+index*2)**2)/35.)*np.exp(1j*(xx*.2-yy*.1))
        amplitude /= np.sqrt(np.sum(abs(amplitude)**2))
        plane = PlaneWave(amplitude, np.array(((2., .2), (-.1, 3.)))*1e-9,
                          np.array((index*3e-9, -index*4e-9)))
        modes.append(WaveMode(plane, (.25, .55)[index], TIP_REFERENCE, f"mode-{index}", 300.-index))
    original = TipGunCheckpoint(BeamState(tuple(modes), TIP_REFERENCE), 1605., 1e-8, {})
    key = store.key("fixed-observation")
    return store, key, store.put(key, original)


def _amplitude_path(checkpoint, index=0):
    modes = checkpoint.beam.modes
    return modes.root/modes.rows[index]["arrays"]["amplitude"]["file"]


def _count_hashes(monkeypatch):
    calls = []
    original = store_module._hash_file
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(store_module, "_hash_file", counted)
    return calls


def test_verified_snapshot_reused_within_one_request_and_released(stored, monkeypatch):
    store, key, checkpoint = stored
    calls = _count_hashes(monkeypatch)
    with wave_read_session(1 << 20) as session:
        mode = checkpoint.beam.modes[0]
        weak = weakref.ref(mode.plane.amplitude)
        assert store.get(key).beam.modes[0] is mode
        geometry = next(checkpoint.beam.modes.geometries())
        assert geometry[1] is mode.plane.basis_m
        assert len(calls) == 3
        assert 0 < session.resident_bytes <= session.maximum_bytes
        with pytest.raises(ValueError):
            mode.plane.amplitude.setflags(write=True)
        del mode, geometry
    gc.collect()
    assert session.resident_bytes == 0 and weak() is None
    checkpoint.beam.modes[0]
    assert len(calls) == 6


def test_same_size_timestamp_external_edit_is_checked_on_next_request(stored):
    _, _, checkpoint = stored
    path = _amplitude_path(checkpoint)
    stat = path.stat()
    with wave_read_session(1 << 20):
        mode = checkpoint.beam.modes[0]
        saved = mode.plane.amplitude.copy()
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            previous = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes((previous[0] ^ 1,)))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        # A completed snapshot cannot be mutated by a later external writer.
        assert checkpoint.beam.modes[0] is mode
        np.testing.assert_array_equal(mode.plane.amplitude, saved)
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    with wave_read_session(1 << 20), pytest.raises(ValueError, match="checksum/path"):
        checkpoint.beam.modes[0]


def test_new_request_still_checks_symlink_path(stored, monkeypatch):
    _, _, checkpoint = stored
    with wave_read_session(1 << 20):
        checkpoint.beam.modes[0]
    path = _amplitude_path(checkpoint)
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == path or original(self))
    with wave_read_session(1 << 20), pytest.raises(ValueError, match="checksum/path"):
        checkpoint.beam.modes[0]


def test_oversized_mode_keeps_streaming_and_cancellation_checks_cache_hits(stored, monkeypatch):
    _, _, checkpoint = stored
    calls = _count_hashes(monkeypatch)
    with wave_read_session(16) as session:
        assert checkpoint.beam.modes[0] is not checkpoint.beam.modes[0]
        assert session.peak_bytes == 0 and len(calls) == 6
    cancel = [False]
    with wave_read_session(1 << 20, cancelled=lambda: cancel[0]) as session:
        checkpoint.beam.modes[0]
        cancel[0] = True
        with pytest.raises(InterruptedError, match="read cancelled"):
            checkpoint.beam.modes[0]
    assert session.resident_bytes == 0


def test_eviction_and_exception_do_not_exceed_budget_or_leave_modes_alive(stored):
    _, _, checkpoint = stored
    size = checkpoint.beam.modes[0].plane.amplitude.nbytes+48
    with pytest.raises(RuntimeError, match="request failed"):
        with wave_read_session(size) as session:
            first = checkpoint.beam.modes[0]
            weak = weakref.ref(first.plane.amplitude)
            del first
            checkpoint.beam.modes[1]
            assert weak() is None
            assert session.peak_bytes == size
            raise RuntimeError("request failed")
    assert session.resident_bytes == 0


def test_read_sessions_are_isolated_across_threads_and_nested_requests(stored):
    _, _, checkpoint = stored
    with wave_read_session(1 << 20):
        original = checkpoint.beam.modes[0]
        with wave_read_session(1 << 20):
            assert checkpoint.beam.modes[0] is not original
        assert checkpoint.beam.modes[0] is original
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: checkpoint.beam.modes[0]).result() is not original


@pytest.mark.parametrize("budget", (0, 1 << 20))
def test_streamed_preview_and_moments_match_separate_reads_without_retaining_modes(stored, monkeypatch, budget):
    _, _, checkpoint = stored
    expected_preview = _intensity_preview(checkpoint, bins=32)
    expected_moments = wave_plane_moments(checkpoint)
    calls = _count_hashes(monkeypatch)
    with wave_read_session(budget) as session:
        preview = _intensity_preview(checkpoint, bins=32, include_moments=True)
        actual = preview.wave_moments.finish(checkpoint)
        assert session.resident_bytes <= budget
    assert len([path for path in calls if "amplitude" in path.name]) == 2
    np.testing.assert_array_equal(preview.density, expected_preview.density)
    np.testing.assert_array_equal(preview.bounds_um, expected_preview.bounds_um)
    assert preview.probability == expected_preview.probability
    assert thaw_json(actual) == thaw_json(expected_moments)
    assert preview.retained_bytes == expected_preview.retained_bytes


def test_streamed_moments_reject_incomplete_or_different_checkpoint(stored):
    _, _, checkpoint = stored
    moments = _WavePlaneMoments(checkpoint)
    moments.add(checkpoint.beam.modes[0])
    with pytest.raises(ValueError, match="more than once"):
        moments.add(checkpoint.beam.modes[0])
    with pytest.raises(ValueError, match="every mode"):
        moments.finish(checkpoint)
    moments.add(checkpoint.beam.modes[1])
    with pytest.raises(ValueError, match="same checkpoint"):
        moments.finish(replace(checkpoint, plane_z_mm=1606.))
    assert thaw_json(moments.finish(checkpoint)) == thaw_json(wave_plane_moments(checkpoint))


@pytest.mark.parametrize("budget", (-1, 1.5, True))
def test_read_session_budget_validation(budget):
    with pytest.raises(ValueError, match="budget"):
        with wave_read_session(budget):
            pass


def test_gui_read_session_budget_is_bounded_and_reserved():
    from types import SimpleNamespace
    request = TipWaveRequest()
    state = SimpleNamespace(electron_gun=SimpleNamespace(emitter=SimpleNamespace(surface_model=None)))
    assert _wave_read_session_bytes(request) == 128*1024**2
    low = replace(request, maximum_readout_bytes=1024,
                  execution=replace(request.execution, maximum_ram_cache_bytes=512))
    assert _wave_read_session_bytes(low) == 128
    lower = replace(low, maximum_readout_bytes=128)
    assert _wave_read_session_bytes(lower) == 32
    assert _wave_resource_claim(state, low).working_bytes-_wave_resource_claim(state, lower).working_bytes == 96


@pytest.mark.parametrize("cancelled", (False, True))
def test_paired_worker_uses_single_stream_for_preview_and_comparison_and_releases(stored, qapp, monkeypatch, cancelled):
    """Only expensive propagation is substituted; stored I/O/readouts are real."""
    from threading import Event
    from types import SimpleNamespace
    import temsim.gui.coherent_beam as gui
    import temsim.gui.beam_plane_data as plane_data
    import temsim.instrument_snapshot as snapshots
    _, _, checkpoint = stored
    calls = _count_hashes(monkeypatch)
    state = SimpleNamespace(electron_gun=SimpleNamespace(emitter=SimpleNamespace(surface_model=None)))
    context = SimpleNamespace(instrument_identity="readout-fixture", token="paired-readout",
        verify_particle=lambda: None, verify_wave=lambda _: None, specimen_active=False,
        particle_result=object())
    monkeypatch.setattr(snapshots, "capture_instrument_snapshot",
                        lambda _: SimpleNamespace(digest="readout-fixture"))
    def solve(*_args, **_kwargs):
        for mode in checkpoint.beam.modes:
            del mode
        return SimpleNamespace(checkpoint=checkpoint)
    monkeypatch.setattr(gui, "simulate_tip_wave", solve)
    monkeypatch.setattr(gui, "_plane_magnetic_gauge", lambda *_: None)
    plane = SimpleNamespace(z_mm=checkpoint.plane_z_mm,
        x_m=np.array((-1., 1.))*1e-9, y_m=np.array((0., 0.)),
        source_fraction=np.array((.4, .4)), source_current_pa=10000.,
        weights_valid=True, coordinate_frame="column", kinetic_energy_ev=np.full(2, 300000.),
        ray_count=2, provenance="Incident")
    monkeypatch.setattr(plane_data, "sample_beam_plane", lambda *_: plane)
    # Zero retention tests the same path used by modes larger than the cap.
    request = TipWaveRequest(stop="plane", observation_z_mm=checkpoint.plane_z_mm,
                             maximum_readout_bytes=1)
    event = Event()
    if cancelled:
        event.set()
    worker = gui._WaveWorker(1, 1, state, request, event, pair_context=context)
    solved, failed, finished = [], [], []
    worker.signals.solved.connect(lambda *args: solved.append(args))
    worker.signals.failed.connect(lambda *args: failed.append(args))
    worker.signals.finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert not failed and len(finished) == 1
    assert store_module._READ_SESSION.get() is None
    if cancelled:
        assert not solved and not calls
    else:
        assert len(solved) == 1
        preview = solved[0][-1]
        assert preview.wave_moments is None
        assert preview.comparison_error is None
        assert preview.comparison["wave"]["source_fraction"] == pytest.approx(.8)
        assert len([path for path in calls if "amplitude" in path.name]) == 4
