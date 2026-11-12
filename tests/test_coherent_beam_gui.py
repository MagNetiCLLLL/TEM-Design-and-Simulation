"""Mixed-wave GUI lifecycle with substituted physics, never a physical trial."""
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer, Qt

import temsim.gui.coherent_beam as module
from temsim.gui.coherent_beam import CoherentBeamPage, _Preview
from temsim.optics.column import default_state
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.wave_plane_observables import ColumnPlaneMagneticGauge


def instrument():
    state = default_state()
    # Establish the synthetic zero-thickness plane before explicitly placing
    # it; a thickness edit now correctly retains the previous upper surface.
    state.sample.thickness_nm = 0.
    state.sample.z_mm = 100.
    return state


def result(z_mm):
    return SimpleNamespace(checkpoint=SimpleNamespace(plane_z_mm=z_mm,
        reference_current_a=10e-9, record={}), instrument_digest="GUI-only substituted propagation")


def preview():
    return _Preview(np.ones((4, 4))/16, np.array(((-.5, .5), (-.5, .5))),
                    1., 1, 160, ((4, 4),))


class _GuiObservationSession:
    def __init__(self, state, request, source):
        self.state, self.request, self.source = state, request, source
        self.observed_z = []

    def observe(self, z_mm, **kwargs):
        self.observed_z.append(z_mm)
        return module.simulate_mixed_wave(self.state, self.source,
            replace(self.request, observation_z_mm=z_mm), **kwargs)


@pytest.fixture
def panel(qtbot, monkeypatch):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(True, "Mocked GUI execution only"))
    def capture(state):
        if hasattr(state, "to_dict"):
            return capture_instrument_snapshot(state)
        return SimpleNamespace(restore=lambda: deepcopy(state))
    monkeypatch.setattr(module, "capture_instrument_snapshot", capture)
    monkeypatch.setattr(module, "MixedWaveObservationSession", _GuiObservationSession)
    monkeypatch.setattr(module, "simulate_mixed_wave", lambda _state, _source, request, **_kwargs:
                        result(request.observation_z_mm))
    monkeypatch.setattr(module, "_intensity_preview", lambda *_args, **_kwargs: preview())
    monkeypatch.setattr(module, "_plane_magnetic_gauge", lambda _state, z:
        ColumnPlaneMagneticGauge(z, .125, ()))
    view = CoherentBeamPage()
    qtbot.addWidget(view)
    state = instrument()
    accepted = SimpleNamespace(state_snapshot=capture_instrument_snapshot(state).restore(), simulation=SimpleNamespace(
        incident=SimpleNamespace(x=np.array([[1e-9, 2e-9]]), ray_weight=np.array([.3, .2]))))
    view.state_provider = lambda: state
    view.particle_result_provider = lambda: accepted
    view.accepted = accepted
    view.handoffs = []
    def handoff(particles, **kwargs):
        assert not kwargs["cancelled"]()
        source = SimpleNamespace(particle_result=particles,
            x_m=np.asarray(particles.simulation.incident.x[-1]).copy(),
            weight=np.asarray(particles.simulation.incident.ray_weight).copy())
        view.handoffs.append((source, QThread.currentThread() == view.thread()))
        return source
    monkeypatch.setattr(module, "particle_source_from_result", handoff)
    view.set_state(state)
    view.show()
    yield view
    assert view.shutdown()


def calculate(panel, qtbot):
    panel.calculate()
    qtbot.waitUntil(lambda: panel.result is not None or panel._worker is None)
    assert panel.result is not None, panel.status.text()
    qtbot.waitUntil(lambda: panel._worker is None)


def test_page_has_only_near_specimen_source_controls(panel):
    from PySide6.QtWidgets import QPushButton
    assert not hasattr(panel, "tip_energy")
    assert not hasattr(panel, "gun_near_tip_end")
    assert not hasattr(panel, "state_list")
    assert not hasattr(panel, "state_set")
    assert panel.findChild(QPushButton, "applyTipParameters") is None
    assert panel.observation.mode.findData("wavefront_comparison") == -1
    assert "source_image_rms_pm" in panel._source_controls


def test_opening_and_browsing_do_not_start_first_mixed_calculation(panel, qtbot):
    panel.set_selected_z(101.)
    qtbot.wait(100)
    assert panel.handoffs == [] and panel.result is None and panel._worker is None


def test_actual_arrays_from_accepted_snapshot_are_handed_off_in_worker(panel, qtbot, monkeypatch):
    called = []
    def solve(state, source, request, **kwargs):
        called.append((state, source, request))
        return result(request.observation_z_mm)
    monkeypatch.setattr(module, "simulate_mixed_wave", solve)
    panel._state.electron_gun.emitter.energy_spread_fwhm_ev = .8
    calculate(panel, qtbot)
    state, source, request = called[0]
    np.testing.assert_array_equal(source.x_m, panel.accepted.simulation.incident.x[-1])
    np.testing.assert_array_equal(source.weight, [.3, .2])
    assert source.particle_result is panel.accepted
    assert not panel.handoffs[0][1]
    assert state.electron_gun.emitter.energy_spread_fwhm_ev != .8
    assert request.wave_grid.compute_backend == "Require GPU"
    assert panel._state.electron_gun.emitter.energy_spread_fwhm_ev == .8


def test_particle_cpu_policy_is_preserved_while_wave_requests_cuda(panel, qtbot):
    panel.accepted.state_snapshot.acceleration_backend = "CPU"
    panel.accepted.state_snapshot.acceleration_enabled = False
    calculate(panel, qtbot)
    assert panel._captured.acceleration_backend == "CPU"
    assert not panel._captured.acceleration_enabled
    assert panel._request.wave_grid.compute_backend == "Require GPU"


def test_missing_particle_snapshot_never_starts_handoff(panel):
    panel.particle_result_provider = lambda: None
    panel.calculate()
    assert panel._worker is None and panel.handoffs == []
    assert "Calculate particles first" in panel.status.text()


def test_upstream_plane_is_unavailable_without_handoff_or_tip_fallback(panel):
    panel.set_selected_z(99.)
    panel.calculate()
    assert panel._worker is None and panel.handoffs == []
    assert "specimen entrance" in panel.status.text()


def test_cuda_failure_precedes_handoff_and_worker(panel, monkeypatch):
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(False, "Mocked CUDA unavailable"))
    panel.calculate()
    assert panel.handoffs == [] and panel._worker is None
    assert "CUDA" in panel.status.text()


def test_numerical_change_retires_session_then_new_calculation_captures_it(panel, qtbot):
    calculate(panel, qtbot)
    previous = panel.result
    control = panel._source_controls["source_image_rms_pm"]
    control.setValue(12.)
    assert panel._stale and panel._observation_session is None
    assert panel.result is previous
    panel.calculate()
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel._request.source.source_image_rms_pm == 12.
    assert len(panel.handoffs) == 2


def test_particle_republication_invalidates_existing_wave_and_its_cache(panel, qtbot):
    calculate(panel, qtbot)
    previous = panel.result
    panel.particle_result_published(panel.accepted)
    assert panel._stale and panel._observation_session is None and not panel._cache
    assert panel.result is previous


def test_exact_z_observations_reuse_the_same_particle_boundary(panel, qtbot):
    calculate(panel, qtbot)
    observer = panel._observation_session
    panel.set_selected_z(101.123456789123)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 101.123456789123)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel._observation_session is observer and len(panel.handoffs) == 1


def test_intensity_view_skips_duplicate_upload_and_preserves_zoom_for_same_domain(qtbot, monkeypatch):
    view = module.CoherentIntensityView()
    qtbot.addWidget(view)
    view.show()
    first, data = result(100.), preview()
    view.set_observation(first, data, 0)
    uploads = []
    original = view.image.setImage
    monkeypatch.setattr(view.image, "setImage", lambda *args, **kwargs:
                        (uploads.append(1), original(*args, **kwargs))[1])
    view.plot.setRange(xRange=(-.1, .1), yRange=(-.1, .1), padding=0.)
    bounds = view.plot.getViewBox().viewRange()
    view.set_observation(first, data, 0)
    assert uploads == []
    view.set_observation(result(101.), preview(), 0)
    assert uploads == [1]
    np.testing.assert_allclose(view.plot.getViewBox().viewRange(), bounds)


@pytest.mark.parametrize("scale", (0, 1))
def test_intensity_view_keeps_physical_viewport_when_z_changes_wave_domain(qtbot, scale):
    view = module.CoherentIntensityView()
    qtbot.addWidget(view)
    view.resize(600, 500)
    view.show()
    view.set_observation(result(100.), preview(), scale)
    qtbot.wait(20)
    view.plot.setRange(xRange=(-.2, .3), yRange=(-.1, .4), padding=0.)
    fixed_range = np.array(view.plot.getViewBox().viewRange())
    # A later Z may have a larger, shifted or smaller physical wave grid.
    for z_mm, bounds in ((101., ((-20., 30.), (-10., 40.))),
                         (102., ((-.02, .03), (-.01, .04)))):
        data = replace(preview(), bounds_um=np.array(bounds))
        view.set_observation(result(z_mm), data, scale)
        qtbot.wait(20)
        np.testing.assert_allclose(view.plot.getViewBox().viewRange(), fixed_range,
                                   rtol=0., atol=1e-12)
        rect = view.image.mapRectToParent(view.image.boundingRect())
        np.testing.assert_allclose((rect.left(), rect.right(), rect.top(), rect.bottom()),
                                   np.asarray(bounds).ravel())
        assert f"Exact Z {z_mm:.9g} mm" in view.plot.getPlotItem().titleLabel.text


@pytest.mark.parametrize("scale", (0, 1))
def test_coherent_screen_fits_narrow_ray_sidebar_without_clipping(qtbot, scale):
    view = module.CoherentIntensityView()
    qtbot.addWidget(view)
    view.resize(332, 520)
    view.show()
    view.set_observation(result(1605.193957), preview(), scale)
    qtbot.wait(40)
    viewport = view.plot.viewport().rect()
    screen = view.plot.getViewBox().sceneBoundingRect()
    assert screen.right() <= viewport.right()+2
    assert screen.left() >= viewport.left()
    ranges = np.asarray(view.plot.getViewBox().viewRange())
    assert np.all(ranges[:, 0] <= -.5) and np.all(ranges[:, 1] >= .5)


def test_streamed_intensity_preview_reads_complex_modes_only_once():
    def mode(weight, shift, shape):
        return SimpleNamespace(weight_per_reference_electron=weight,
            plane=SimpleNamespace(amplitude=np.full(shape, 1j/np.sqrt(np.prod(shape))),
                basis_m=np.eye(2)*1e-6, origin_m=np.array((shift, 0.))*1e-6,
                curvature_m1=np.zeros((2, 2)), tilt_rad=np.zeros(2)))

    original = (mode(.3, 0., (4, 4)), mode(.7, 1., (8, 4)))

    class StoredModes:
        reads = 0
        geometry_reads = 0

        def geometries(self):
            for value in original:
                self.geometry_reads += 1
                yield value.plane.amplitude.shape, value.plane.basis_m, value.plane.origin_m

        def __iter__(self):
            for value in original:
                self.reads += 1
                yield value

    modes = StoredModes()
    streamed = module._intensity_preview(SimpleNamespace(beam=SimpleNamespace(
        modes=modes, content_identity="executed-test-modes")), bins=8)
    resident = module._intensity_preview(SimpleNamespace(beam=SimpleNamespace(modes=original)), bins=8)
    assert modes.geometry_reads == 2 and modes.reads == 2
    assert streamed.mode_count == 2 and streamed.probability == pytest.approx(1.)
    assert streamed.grid_shapes == ((4, 4), (8, 4))
    assert resident.grid_shapes == streamed.grid_shapes
    np.testing.assert_array_equal(streamed.density, resident.density)
    np.testing.assert_array_equal(streamed.bounds_um, resident.bounds_um)
    assert streamed.retained_bytes == streamed.density.nbytes+streamed.bounds_um.nbytes
    assert resident.retained_bytes > streamed.retained_bytes


def test_log_display_preserves_density_and_current_session(panel, qtbot):
    calculate(panel, qtbot)
    result_before, preview_before = panel.result, panel.preview
    saved = panel.preview.density.copy()
    generation = panel._generation
    panel.intensity_scale.setCurrentIndex(1)
    assert panel.result is result_before and panel.preview is preview_before
    np.testing.assert_array_equal(panel.preview.density, saved)
    assert panel._generation == generation and panel._session_ready
    assert panel._worker is None
    assert np.all(np.isfinite(panel.image.image))
    panel.intensity_scale.setCurrentIndex(0)
    np.testing.assert_array_equal(panel.image.image, saved)


def test_follow_checkbox_prevents_linked_queries_but_manual_z_works(panel, qtbot):
    calculate(panel, qtbot)
    panel.follow_ray.setChecked(False)
    panel.set_selected_z(102.)
    assert panel._target_z_mm == 100.
    panel.target_z.setValue(103.)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 103.)


def test_exact_cache_and_duplicate_z_do_not_recalculate(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    calls = []
    monkeypatch.setattr(module, "simulate_mixed_wave", lambda _state, _source, request, **_kwargs:
        (calls.append(request.observation_z_mm), result(request.observation_z_mm))[1])
    first = 100.00000000012345
    panel.set_selected_z(first)
    panel.set_selected_z(first)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == first)
    panel.set_selected_z(100.)
    assert panel.result.checkpoint.plane_z_mm == 100.
    assert not panel.timer.isActive()
    qtbot.wait(220)
    assert calls == [first]
    panel.set_selected_z(np.nextafter(first, np.inf))
    assert panel.timer.isActive()  # Adjacent linked floats are different exact planes.


def test_late_success_failure_and_progress_cannot_overwrite_current_session(panel, qtbot):
    calculate(panel, qtbot)
    old_generation, old_session = panel._generation, panel._session
    panel.mark_inputs_stale()
    status = panel.status.text()
    old = panel.result
    panel._solved(old_generation, old_session, result(100.), preview())
    panel._failed(old_generation, "Late failure")
    panel._progress(old_generation, "Late progress")
    assert panel.status.text() == status
    assert panel.result is old
    assert panel._cache == {}


def test_cancel_sets_event_and_keeps_complete_result(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    old = panel.result
    started, released = Event(), Event()
    events = []

    def slow(_state, _source, request, *, cancelled, **_kwargs):
        events.append(cancelled)
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_mixed_wave", slow)
    panel.set_selected_z(106.)
    qtbot.waitUntil(started.is_set)
    panel.cancel()
    assert events[0]()
    assert panel.result is old
    assert "cancelled" in panel.status.text()
    released.set()
    assert panel.pool.waitForDone(2000)
    assert panel.result is old


def test_current_failure_keeps_previous_image_and_allows_retry(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    old = panel.result
    monkeypatch.setattr(module, "simulate_mixed_wave", lambda *_args, **_kwargs:
        (_ for _ in ()).throw(ValueError("Unsupported field fixture")))
    panel.set_selected_z(105.)
    qtbot.waitUntil(lambda: "Unsupported field fixture" in panel.status.text())
    assert panel.result is old
    assert "Previous complete plane" in panel.status.text()
    monkeypatch.setattr(module, "simulate_mixed_wave", lambda _state, _source, request, **_kwargs:
                        result(request.observation_z_mm))
    panel.set_selected_z(106.)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 106.)


def test_active_plane_finishes_and_only_latest_pending_plane_is_queued(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    assert panel.plane_positions.text() == "Requested Z 100 mm | Calculating Z — | Displayed Z 100 mm"
    started, released = Event(), Event()
    latest_started, latest_released = Event(), Event()
    calls = []

    def solve(state, _source, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        if request.observation_z_mm == 105.:
            started.set()
            released.wait(2.)
        elif request.observation_z_mm == 125.:
            latest_started.set()
            latest_released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_mixed_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    for z_mm in range(106, 126):
        panel.set_selected_z(float(z_mm))
    qtbot.wait(230)
    assert [z for z, _ in calls] == [105.]
    assert not calls[0][1]()
    assert not any(job.owner is panel.pool for job in panel.pool.coordinator.queue)
    assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z 105 mm | Displayed Z 100 mm"
    panel._progress(panel._generation, "Phase representation at 102 mm (10/56)")
    assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z 105 mm | Displayed Z 100 mm"
    try:
        released.set()
        qtbot.waitUntil(latest_started.is_set)
        assert panel.result.checkpoint.plane_z_mm == 105.
        assert "105" in panel.plot.getPlotItem().titleLabel.text
        assert "125" in panel.status.text()
        assert "105" in panel.status.text()
        assert [z for z, _ in calls] == [105., 125.]
        assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z 125 mm | Displayed Z 105 mm"
    finally:
        released.set()
        latest_released.set()
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 125.)
    assert [z for z, _ in calls] == [105., 125.]
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z — | Displayed Z 125 mm"


def test_z_motion_during_first_calculation_keeps_authorized_session(panel, qtbot, monkeypatch):
    started, released = Event(), Event()
    calls = []

    def solve(_state, _source, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        if len(calls) == 1:
            started.set()
            released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_mixed_wave", solve)
    panel.calculate()
    qtbot.waitUntil(started.is_set)
    try:
        panel.set_selected_z(101.)
        panel.set_selected_z(102.)
        assert not calls[0][1]()
        assert not panel._stale
        assert panel.result is None
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel.result is not None
                      and panel.result.checkpoint.plane_z_mm == 102.)
    assert [z for z, _ in calls] == [100., 102.]
    assert 100. in panel._cache


def test_requested_cached_plane_takes_precedence_over_late_intermediate(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    cached_result = panel.result
    started, released = Event(), Event()
    calls = []

    def solve(_state, _source, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_mixed_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    try:
        panel.set_selected_z(100.)
        assert panel.result is cached_result
        assert not calls[0][1]()
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel.pool.waitForDone(2000)
    assert panel.result is cached_result
    assert panel._cache[105.][0].checkpoint.plane_z_mm == 105.
    assert [z for z, _ in calls] == [105.]
    assert "100" in panel.plot.getPlotItem().titleLabel.text


def test_return_to_inflight_plane_does_not_recompute_completed_plane(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    started, released = Event(), Event()
    calls = []

    def solve(_state, _source, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_mixed_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    try:
        panel.set_selected_z(106.)
        panel.set_selected_z(105.)
        assert not calls[0][1]()
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel._worker is None)
    qtbot.wait(100)
    assert [z for z, _ in calls] == [105.]
    assert panel.result.checkpoint.plane_z_mm == 105.
    assert not panel.timer.isActive()


def test_mirror_shares_one_session_and_retains_completed_image_while_pending(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    mirror = module.CoherentBeamMirror(panel)
    qtbot.addWidget(mirror)
    mirror.show()
    panel.hide()
    qtbot.waitUntil(lambda: mirror.screen.image.image is not None)
    initial = mirror.screen.image.image
    initial_main = panel.image.image
    calls = []
    released = Event()

    def solve(_state, _source, request, **kwargs):
        calls.append(request.observation_z_mm)
        assert released.wait(5.)
        assert not kwargs["cancelled"]()
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_mixed_wave", solve)
    panel.set_selected_z(101.)
    qtbot.waitUntil(lambda: calls == [101.])
    try:
        panel.set_selected_z(102.)
        panel.set_selected_z(103.)
        assert mirror.screen.image.image is initial
        assert "103" in mirror.status.text() and "100" in mirror.status.text()
        assert mirror.plane_positions.text() == "Requested Z 103 mm | Calculating Z 101 mm | Displayed Z 100 mm"
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 103.)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert calls == [101., 103.]
    assert "103" in mirror.screen.plot.getPlotItem().titleLabel.text
    assert mirror.screen._pending[1] is panel.preview
    assert panel.image.image is initial_main  # Hidden page has no redundant upload.
    panel.show()
    qtbot.waitUntil(lambda: "103" in panel.plot.getPlotItem().titleLabel.text)
    assert mirror.readout.text() == panel.readout.text()


def test_both_coherent_screens_fit_only_on_explicit_request(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    mirror = module.CoherentBeamMirror(panel)
    qtbot.addWidget(mirror)
    mirror.show()
    qtbot.wait(20)
    original_result, original_session = panel.result, panel._observation_session
    calls = []
    monkeypatch.setattr(module, "simulate_mixed_wave", lambda *_args, **_kwargs: calls.append(1))
    for host in (panel, mirror):
        view = host.screen
        view.plot.setRange(xRange=(-.1, .1), yRange=(-.1, .1), padding=0.)
        qtbot.mouseClick(view.fit_button, Qt.MouseButton.LeftButton)
        fitted = np.asarray(view.plot.getViewBox().viewRange())
        assert np.all(fitted[:, 0] <= preview().bounds_um[:, 0])
        assert np.all(fitted[:, 1] >= preview().bounds_um[:, 1])
        assert not any(view.plot.getViewBox().autoRangeEnabled())
    assert calls == [] and panel._worker is None
    assert panel.result is original_result and panel._observation_session is original_session


def test_worker_progress_is_throttled_and_cancelled_progress_suppressed(panel, monkeypatch):
    monkeypatch.setattr(module, "monotonic", lambda: 1.)
    request = panel._make_request()
    worker = module._WaveWorker(1, 1, instrument(), request, Event())
    messages = []
    worker.signals.progress.connect(lambda generation, text: messages.append((generation, text)))
    for index in range(100):
        worker._report_progress(index, 100, "Field propagation")
    worker._report_progress(100, 100, "Complete")
    assert len(messages) == 2
    assert messages[-1][1] == "Complete (100/100)"
    worker.event.set()
    worker._report_progress(100, 100, "Cancelled")
    assert len(messages) == 2


def test_idle_wave_cache_is_inventoried_for_other_shared_job_owners(panel, qtbot):
    calculate(panel, qtbot)
    coordinator = panel.pool.coordinator
    assert any(owner() is panel and method == "_retained_wave_roots"
               for owner, method in coordinator._providers)
    before = coordinator._retained_bytes()
    array = np.zeros(100_000)
    panel.result.checkpoint.test_array = array
    after = coordinator._retained_bytes()
    assert after-before >= array.nbytes
    # The same checkpoint is referenced by result and its exact-Z cache.
    # Adding another reference is ownership aliasing, not another wave array.
    panel._cache[101.] = (panel.result, panel.preview)
    aliased = coordinator._retained_bytes()
    assert aliased-after < array.nbytes
