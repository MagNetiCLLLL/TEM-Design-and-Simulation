"""Offline GUI routing/cancellation fixtures; no optical qualification claims."""
from collections import OrderedDict
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.coherent_beam import CoherentBeamPage, CoherentIntensityView, _intensity_preview
from temsim.gui.electron_beam_observation import ElectronBeamObservation
from temsim.gui.job_coordinator import ResourceClaim
from temsim.gui.wavefront_comparison import WavefrontComparison
from temsim.physics.tip_wave_pipeline import TipWaveRequest, TipWaveResult
from test_wave_beam_analysis import checkpoint


def actual_result(z=1500., *, identity="captured", reverse=False, single=False):
    cp = checkpoint()
    modes = cp.beam.modes[:1] if single else tuple(reversed(cp.beam.modes)) if reverse else cp.beam.modes
    cp = replace(cp, beam=replace(cp.beam, modes=modes), plane_z_mm=z)
    request = replace(TipWaveRequest(), stop="plane", observation_z_mm=z,
                      maximum_readout_bytes=16*1024**2)
    return TipWaveResult(cp, None, request, False, identity)


@pytest.fixture
def backend(monkeypatch):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability",
        lambda: compute_backend.BackendCapability(True, "Substituted reference execution"))
    from temsim.physics import tip_wave_pipeline, wavefront_comparison as numerical
    from temsim.gui import coherent_beam
    propagation, readouts = [], []

    def reference(state, actual, **kwargs):
        propagation.append((state, actual, kwargs))
        return replace(actual, instrument_digest="ideal:"+actual.instrument_digest)

    def compare(actual, ideal, mode_id, *, remove_piston, **kwargs):
        readouts.append((actual, ideal, mode_id, remove_piston))
        mode = next(row for row in actual.beam.modes if row.mode_id == mode_id)
        values = np.full(mode.plane.amplitude.shape, .25)
        values[0, 0] = np.nan
        return numerical.WavefrontComparison(values, values*.5,
            values*0 if remove_piston else values*.5, np.isfinite(values),
            mode.plane.basis_m, mode.plane.origin_m,
            dict(plane_z_mm=actual.plane_z_mm, mode_id=mode_id,
                 piston_removed=remove_piston, piston_phase_rad=.25 if remove_piston else 0.,
                 valid_fraction=np.isfinite(values).mean()))

    monkeypatch.setattr(tip_wave_pipeline, "simulate_ideal_reference", reference)
    monkeypatch.setattr(numerical, "compare_wavefronts", compare)
    monkeypatch.setattr(coherent_beam, "_wave_resource_claim", lambda *_a: ResourceClaim(32*1024**2))
    return SimpleNamespace(propagation=propagation, readouts=readouts)


@pytest.fixture
def view(qtbot, backend):
    captured = object()
    widget = WavefrontComparison(lambda _actual: captured)
    widget._test_capture = captured
    widget.resize(1400, 700)
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    assert widget.shutdown(10000)


def ready(qtbot, view):
    qtbot.waitUntil(lambda: view._worker is None and view.data is not None, timeout=15000)


def test_reference_is_explicit_and_three_views_share_native_lattice(qtbot, view, backend):
    actual = actual_result(single=True)
    view.set_observation(actual)
    assert not backend.propagation and view.data is None
    assert not view.mode.isEnabled() and not view.remove_piston.isChecked()
    view.compute.click()
    ready(qtbot, view)
    assert backend.propagation[0][0] is view._test_capture
    assert backend.propagation[0][1] is actual
    assert len(backend.propagation) == len(backend.readouts) == 1
    assert all(image.getLevels().tolist() == [-np.pi, np.pi] for image in view.images)
    assert view.images[0].transform() == view.images[1].transform() == view.images[2].transform()
    assert np.isnan(view.images[2].image[0, 0])
    assert "Common source phase retained" in view.status.text()


def test_mode_and_constant_offset_changes_reuse_completed_reference(qtbot, view, backend):
    view.set_observation(actual_result())
    view.compute.click()
    ready(qtbot, view)
    view.mode.setCurrentIndex(1)
    ready(qtbot, view)
    assert backend.readouts[-1][2] == "fixture:two"
    view.remove_piston.setChecked(True)
    ready(qtbot, view)
    assert backend.readouts[-1][3] is True
    assert len(backend.propagation) == 1
    assert "offset removed from difference only" in view.status.text()
    assert "0.25 rad" in view.status.text()
    assert "Difference is Actual − Ideal − constant offset" in view.status.text()


def test_failed_new_mode_readout_does_not_display_previous_mode(qtbot, view, backend, monkeypatch):
    from temsim.physics import wavefront_comparison as numerical
    view.set_observation(actual_result())
    view.compute.click()
    ready(qtbot, view)

    def fail(*_args, **_kwargs):
        raise ValueError("No common valid pixels for selected mode")

    monkeypatch.setattr(numerical, "compare_wavefronts", fail)
    view.mode.setCurrentIndex(1)
    assert view.data is None
    assert all(image.image is None for image in view.images)
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    assert "No common valid pixels" in view.status.text()
    assert view.data is None and len(backend.propagation) == 1


def test_exact_z_preserves_mode_id_despite_reordering_and_reuses_cached_plane(qtbot, view, backend):
    first = actual_result()
    view.set_observation(first)
    view.mode.setCurrentIndex(1)
    view.compute.click()
    ready(qtbot, view)
    second = actual_result(1501., reverse=True)
    view.set_observation(second)
    assert view.mode.currentData() == "fixture:two" and view.mode.currentIndex() == 0
    assert view.data is None and len(backend.propagation) == 1
    view.compute.click()
    ready(qtbot, view)
    assert view.data.metadata["plane_z_mm"] == 1501.
    view.set_observation(first)
    ready(qtbot, view)
    assert len(backend.propagation) == 2
    assert view.mode.currentData() == "fixture:two"


def test_changed_input_identity_does_not_reuse_reference(qtbot, view, backend):
    first = actual_result()
    view.set_observation(first)
    view.compute.click()
    ready(qtbot, view)
    view.set_observation(replace(first, instrument_digest="different-captured-inputs"))
    assert view.ideal is None and view.data is None
    view.compute.click()
    ready(qtbot, view)
    assert len(backend.propagation) == 2


def test_wrong_z_reference_is_rejected_without_readout(qtbot, view, backend, monkeypatch):
    from temsim.physics import tip_wave_pipeline
    monkeypatch.setattr(tip_wave_pipeline, "simulate_ideal_reference",
        lambda _state, actual, **_kwargs: replace(actual,
            checkpoint=replace(actual.checkpoint, plane_z_mm=1502.)))
    view.set_observation(actual_result())
    view.compute.click()
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    assert view.data is None and not backend.readouts
    assert "different observation plane" in view.status.text()


def test_cancel_and_late_result_cannot_replace_new_context(qtbot, view, backend, monkeypatch):
    from temsim.physics import tip_wave_pipeline
    started, release = Event(), Event()

    def delayed(_state, actual, **_kwargs):
        started.set()
        release.wait(5)
        return actual

    monkeypatch.setattr(tip_wave_pipeline, "simulate_ideal_reference", delayed)
    first = actual_result()
    view.set_observation(first)
    view.compute.click()
    qtbot.waitUntil(started.is_set, timeout=10000)
    generation, key = view._generation, view._key
    worker = view._worker
    view.set_observation(actual_result(1501.))
    assert worker.event.is_set()
    release.set()
    view._reference_ready(generation, key, first)
    assert view.pool.waitForDone(10000)
    assert view.ideal is None and view.data is None and not backend.readouts


def test_capture_provider_rejection_never_propagates(view, backend):
    view.set_observation(actual_result())
    view.state_provider = lambda _: None
    view.compute.click()
    assert not backend.propagation and "unavailable" in view.status.text()


def test_page_context_provider_uses_owned_current_or_saved_capture_only():
    actual = actual_result()
    captured = object()
    states = SimpleNamespace(admitted_ids=set(), row_keys={}, runs={})
    page = SimpleNamespace(_closed=False, _target_z_mm=1500., _stale=False,
        _captured=captured, _cache=OrderedDict(((1500., (actual, None)),)), state_set=states,
        state_provider=lambda: pytest.fail("Never read live instrument inputs"))
    assert CoherentBeamPage._reference_state_for_result(page, actual) is captured
    page._cache.clear()
    saved_capture = object()
    states.admitted_ids = {"saved"}
    states.row_keys = {"saved": "run"}
    states.runs = {"run": SimpleNamespace(state=saved_capture, instrument_identity="captured",
        cache={1500.: (actual, None)})}
    assert CoherentBeamPage._reference_state_for_result(page, actual) is saved_capture
    states.admitted_ids.clear()
    with pytest.raises(ValueError, match="admitted captured inputs"):
        CoherentBeamPage._reference_state_for_result(page, actual)
    page._target_z_mm = 1501.
    with pytest.raises(ValueError, match="exact-Z"):
        CoherentBeamPage._reference_state_for_result(page, actual)


def test_observation_overlay_computes_only_selected_member(qtbot, backend):
    from temsim.physics.coherent_state_set import combine_intensity_previews
    first, second = actual_result(identity="first"), actual_result(identity="second")
    preview = _intensity_preview(first.checkpoint, bins=32)
    overlay, combined = combine_intensity_previews(((first, preview, 1.), (second, preview, 2.)))
    view = ElectronBeamObservation(CoherentIntensityView())
    qtbot.addWidget(view)
    captures = {"first": object(), "second": object()}
    view.reference_state_provider = lambda result: captures[result.instrument_digest]
    view.set_observation(overlay, combined, 0)
    view.mode.setCurrentIndex(view.mode.findData("wavefront_comparison"))
    view.member.setCurrentIndex(1)
    comparison = view.wavefront_comparison
    assert not backend.propagation
    comparison.compute.click()
    ready(qtbot, comparison)
    assert backend.propagation[0][0] is captures["second"]
    assert backend.propagation[0][1] is second
    view.invalidate_reference_context("Inputs changed")
    assert comparison.data is None and not comparison.compute.isEnabled()
    assert view.shutdown(10000)


def test_existing_phase_observation_retains_mode_id_across_z(qtbot):
    view = ElectronBeamObservation(CoherentIntensityView())
    qtbot.addWidget(view)
    first = actual_result()
    view.set_observation(first, _intensity_preview(first.checkpoint, bins=32), 0)
    view.mode.setCurrentIndex(view.mode.findData("wave_phase"))
    qtbot.waitUntil(lambda: not view.diagnostics.analysis.wave.busy, timeout=15000)
    view.diagnostics.analysis.colour_combo.setCurrentIndex(2)
    qtbot.waitUntil(lambda: not view.diagnostics.analysis.wave.busy, timeout=15000)
    second = actual_result(1501., reverse=True)
    view.set_observation(second, _intensity_preview(second.checkpoint, bins=32), 0)
    qtbot.waitUntil(lambda: not view.diagnostics.analysis.wave.busy, timeout=15000)
    assert view.diagnostics.analysis.colour_combo.currentData() == 0
    assert second.checkpoint.beam.modes[0].mode_id == "fixture:two"
    assert view.shutdown(10000)
