"""Camera reception UI keeps capture provenance and never runs a scan loop."""
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui import receiver_imaging as module
from temsim.detector import receiver_scan
from temsim.detector.receiver_image import ReceiverImage
from test_receiver_image import capture


def _capture():
    planes = [SimpleNamespace(key=key, name=name, z_mm=z, inserted=True)
              for key, name, z in (("flu_screen", "Fluorescent Screen", 2.), ("camera", "Camera", 3.))]
    state = SimpleNamespace(recording_planes=planes, projector_mode="image")
    return SimpleNamespace(state_snapshot=state, scan_geometry=None)


def _image(key="camera", *, zero=False, status="AVAILABLE", exposure_s=1.):
    values = np.zeros((16, 16))
    if not zero:
        values[8, 9] = .25
    return ReceiverImage(key, key, status, "Captured physical hits", values, values,
        np.linspace(-1., 1., 16), np.linspace(-1., 1., 16), exposure_s,
        None if status != "AVAILABLE" else values * (exposure_s or 1.) * 100.,
        (0. if zero else 1.) if status == "AVAILABLE" else None,
        {"capture_request_identity": "original-capture"})


@pytest.fixture
def page(qtbot, monkeypatch):
    view = module.ReceiverImagingView()
    qtbot.addWidget(view)
    calls = []
    def sampled(result, key, **kw):
        calls.append((result, key, kw))
        return _image(key, exposure_s=kw.get("exposure_s"))
    monkeypatch.setattr(module, "receiver_image", sampled)
    monkeypatch.setattr(receiver_scan, "receiver_image", sampled)
    view._test_calls = calls
    return view


def test_opening_and_selecting_receiver_never_requests_transport(page, qtbot):
    capture = _capture()
    requested = []
    page.calculation_requested.connect(lambda: requested.append(True))
    page.set_state(capture.state_snapshot)
    page.resize(1000, 600)
    page.show()
    qtbot.wait(100)
    assert page._test_calls == []
    assert page._displayed_image is None
    assert "No reception calculated" in page.notice.text()
    page.receiver.setCurrentIndex(1)
    qtbot.wait(100)
    assert not requested and not page._test_calls
    page.calculate_button.click()
    assert requested == [True]


def test_hidden_result_is_lazy_switch_preserves_snapshot_and_stale_status(page, qtbot):
    capture = _capture()
    page.display_result(capture)
    assert not page._test_calls
    page.show()
    qtbot.waitUntil(lambda: page._displayed_image is not None)
    page.receiver.setCurrentIndex(1)
    qtbot.waitUntil(lambda: page._displayed_image.key == "camera")
    assert all(row[0] is capture for row in page._test_calls)
    live = _capture().state_snapshot
    live.recording_planes.clear()
    page.set_state(live)
    page.mark_result_stale()
    qtbot.waitUntil(lambda: "inputs changed" in page.summary.text())
    assert page.receiver.count() == 2
    assert all(plane.inserted for plane in capture.state_snapshot.recording_planes)
    assert "Inputs changed" in page.calculation_bar.status.text()
    page.display_result(None)
    qtbot.waitUntil(lambda: page._displayed_image is None)
    assert not page.export_button.isEnabled()


def test_real_zero_and_unavailable_are_distinct(page, monkeypatch):
    capture = _capture()
    monkeypatch.setattr(module, "receiver_image", lambda *a, **kw: _image(zero=True))
    page.display_result(capture)
    page._render()
    assert page._displayed_image is not None
    assert "No electrons received" in page.summary.text()
    assert page.export_button.isEnabled()
    assert len(page.sensor_outline.getData()[0]) > 0
    monkeypatch.setattr(module, "receiver_image", lambda *a, **kw: _image(status="NOT_REACHED"))
    page.display_result(capture)
    page._render()
    assert page._displayed_image is None
    assert "NOT REACHED" in page.summary.text()
    assert not page.export_button.isEnabled()
    assert len(page.sensor_outline.getData()[0]) == 0


def test_manual_scan_is_explicit_preview_and_export_has_no_dose(page, monkeypatch, tmp_path, qtbot):
    capture = _capture()
    times = np.array([[0., .25], [.5, .75]])
    capture.scan_geometry = SimpleNamespace(times_s=times)
    calls = []
    def scan(index=None, *, accumulate=False):
        key = page.receiver.currentData()
        calls.append(index)
        image = _image(key, status="GEOMETRIC_PREVIEW")
        return SimpleNamespace(key=key, image=image, status="GEOMETRIC_PREVIEW", detail="Preview only",
            times_s=times, displacement_x_mm=times, displacement_y_mm=times*0,
            trajectory_x_mm=times, trajectory_y_mm=times*0)
    prepared = SimpleNamespace(preview=scan, times_s=times,
        base_image=_image(), metrics={"frame_period_s": 1.})
    monkeypatch.setattr(module, "prepare_receiver_scan", lambda *a, **kw: prepared)
    page.display_result(capture)
    page.resize(1000, 600)
    page.show()
    qtbot.waitUntil(lambda: page._displayed_image is not None)
    assert page.mode.currentData() == "frame"
    assert calls == [None]
    qtbot.wait(150)
    assert calls == [None] and not page._timer.isActive()
    assert not page.exposure.isEnabled() and not page.quantity.isEnabled()
    assert "Geometry preview" in page.notice.text()
    page.mode.setCurrentIndex(2)
    page.position.setValue(3)
    qtbot.waitUntil(lambda: calls[-1] == 3)
    assert "4/4" in page.position_text.text()
    assert page.marker.getData()[0].size == 1
    path = tmp_path / "preview.npz"
    page.export_result(path)
    with np.load(path, allow_pickle=False) as data:
        assert "expected_electrons" not in data
        assert np.array_equal(data["times_s"], times)
        assert json.loads(str(data["metadata_json"]))["status"] == "GEOMETRIC_PREVIEW"


def test_scan_preview_never_keeps_an_electron_count_display_label(page, monkeypatch):
    page.quantity.setCurrentIndex(1)
    page.mode.setCurrentIndex(1)
    page._render()
    assert page.quantity.currentData() == "probability"
    assert not page.quantity.isEnabled()


def test_static_export_keeps_raw_values_after_log_display(page, tmp_path):
    page.display_result(_capture())
    page.log_scale.setChecked(True)
    page._render()
    path = tmp_path / "physical.npz"
    page.export_result(path)
    with np.load(path, allow_pickle=False) as data:
        assert data["response_probability"].sum() == .25
        assert data["expected_electrons"].sum() == 25.
        assert json.loads(str(data["metadata_json"]))["capture_request_identity"] == "original-capture"
    assert page.plot.getViewBox().state["yInverted"] is False


def test_pixel_selection_rebins_retained_hits_and_caches_each_grid(qtbot, capture, monkeypatch, tmp_path):
    view = module.ReceiverImagingView()
    qtbot.addWidget(view)
    original = module.receiver_image
    calls, requested = [], []
    def sampled(result, key, **kwargs):
        calls.append((result, kwargs))
        return original(result, key, **kwargs)
    monkeypatch.setattr(module, "receiver_image", sampled)
    view.calculation_requested.connect(lambda: requested.append(True))
    before = capture.simulation.branches["post"].x.copy()
    view.display_result(capture)
    view.receiver.setCurrentIndex(view.receiver.findData("camera"))
    view._fit_next = False
    extent = (-.5, .5, -.5, .5)
    view._view_extent = extent
    view._render()
    default = view._displayed_image
    assert view.pixels.currentData() == 512
    assert [view.pixels.itemData(i) for i in range(view.pixels.count())] == [32, 64, 128, 256, 512, 1024]
    view.pixels.setCurrentIndex(view.pixels.findData(64))
    view._render()
    coarse = view._displayed_image
    assert coarse.ideal_probability.shape == (64, 64)
    assert coarse.metrics["display_extent_mm"] == default.metrics["display_extent_mm"] == extent
    assert coarse.current_pa == default.current_pa
    assert coarse.ideal_probability.sum() == default.ideal_probability.sum()
    view.pixels.setCurrentIndex(view.pixels.findData(512))
    view._render()
    assert view._displayed_image is default
    assert len(calls) == 2 and all(result is capture for result, _ in calls)
    assert not requested
    np.testing.assert_array_equal(capture.simulation.branches["post"].x, before)
    assert "Hits in view 1 / 1" in view.summary.text()
    assert "Median weighted hits/bin 1" in view.summary.text()
    assert "not a convergence proof" in view.summary.toolTip()
    path = tmp_path/"sampling.npz"
    view.export_result(path)
    with np.load(path) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["sampling"]["occupied_bin_count"] == 1
        assert metadata["display_pixels"] == 512


def test_static_pixel_choice_does_not_change_scan_preview_grid(page, monkeypatch):
    capture = _capture()
    capture.scan_geometry = SimpleNamespace(times_s=np.array([[0.]]))
    calls = []
    preview = SimpleNamespace(image=_image(status="GEOMETRIC_PREVIEW"), status="GEOMETRIC_PREVIEW", detail="Preview")
    prepared = SimpleNamespace(preview=lambda *a, **kw: preview, times_s=np.array([[0.]]),
                               metrics={}, base_image=_image())
    def prepare(*args, **kwargs):
        calls.append(kwargs)
        return prepared
    monkeypatch.setattr(module, "prepare_receiver_scan", prepare)
    monkeypatch.setattr(page, "_draw_trajectory", lambda value: None)
    page.pixels.setCurrentIndex(page.pixels.findData(1024))
    page.display_result(capture)
    page._render()
    assert page.mode.currentData() == "frame" and not page.pixels.isEnabled()
    assert calls[0]["pixels"] == 192
    page.pixels.setCurrentIndex(page.pixels.findData(32))
    page._render()
    assert len(calls) == 1 and page._displayed_image is preview.image


def test_actual_hits_rebin_on_zoom_and_keep_fixed_view_for_new_capture(qtbot, capture, tmp_path):
    view = module.ReceiverImagingView()
    qtbot.addWidget(view)
    view.resize(1000, 700)
    view.display_result(capture)
    view.receiver.setCurrentIndex(view.receiver.findData("camera"))
    view.show()
    qtbot.waitUntil(lambda: view._displayed_image is not None)
    full = view._displayed_image
    assert full.ideal_probability.shape == (512, 512)
    view.plot.setRange(xRange=(-.02, .02), yRange=(-.02, .02), padding=0.)
    qtbot.waitUntil(lambda: view._displayed_image is not None and
                   view._displayed_image.metrics["pixel_size_x_mm"] < .001)
    zoomed = view._displayed_image
    extent = zoomed.metrics["display_extent_mm"]
    assert zoomed is not full and zoomed.current_pa == full.current_pa
    shifted = deepcopy(capture)
    shifted.simulation.branches["post"].x[:, 0] += 5e-6
    view.display_result(shifted)
    qtbot.waitUntil(lambda: view._displayed_image is not None)
    changed = view._displayed_image
    assert changed.metrics["display_extent_mm"] == pytest.approx(extent)
    assert not np.array_equal(changed.ideal_probability, zoomed.ideal_probability)
    # Sensor outline remains at the physical position while the viewport moves.
    outline_x, outline_y = view.sensor_outline.getData()
    assert (min(outline_x), max(outline_x), min(outline_y), max(outline_y)) == pytest.approx((-1, 1, -1, 1))
    path = tmp_path/"crop.npz"
    view.export_result(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["display_extent_mm"] == pytest.approx(extent)
        assert metadata["physical_area_extent_mm"] == [-1, 1, -1, 1]
        assert metadata["current_pa"] == full.current_pa
        assert metadata["pixel_size_x_mm"] < .001
    view.fit_receiver()
    qtbot.waitUntil(lambda: view._displayed_image is not None and
                   view._displayed_image.metrics["display_extent_mm"] == (-1., 1., -1., 1.))


def test_fit_beam_uses_actual_hit_bounds_and_blur_toggle_has_separate_cache(qtbot, capture, tmp_path):
    view = module.ReceiverImagingView()
    qtbot.addWidget(view)
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm, camera.point_spread_sigma_y_mm = .02, .03
    camera.point_spread_rotation_deg = 37.
    branch = capture.simulation.branches["post"]
    branch.blocked_key[1] = "camera"
    branch.blocked_z[1] = 3.
    branch.x[:, 0], branch.x[:, 1] = 10e-6, 12e-6
    view.display_result(capture)
    view.receiver.setCurrentIndex(view.receiver.findData("camera"))
    view.resize(1000, 700)
    view.show()
    qtbot.waitUntil(lambda: view._displayed_image is not None)
    view.detector_blur.setChecked(False)
    view.fit_beam()
    qtbot.waitUntil(lambda: view._displayed_image is not None and
                   view._displayed_image.metrics["pixel_size_x_mm"] < 1e-4)
    arrivals = view._displayed_image
    assert np.count_nonzero(arrivals.ideal_probability) == 2
    assert arrivals.current_pa == pytest.approx(1.)
    np.testing.assert_array_equal(view.image.getImageItem().image, arrivals.ideal_probability)
    assert arrivals.response_probability is None and arrivals.expected_electrons is None
    path = tmp_path/"arrivals.npz"
    view.export_result(path)
    with np.load(path, allow_pickle=False) as data:
        assert "response_probability" not in data and "expected_electrons" not in data
        assert data["expected_arrival_electrons"].sum() == pytest.approx(1e-12/1.602176634e-19)
        assert json.loads(str(data["metadata_json"]))["response_computed"] is False
    view.detector_blur.setChecked(True)
    qtbot.waitUntil(lambda: view._displayed_image is not None and
                   view._displayed_image.response_probability is not None)
    response = view._displayed_image
    assert response is not arrivals and response.current_pa == arrivals.current_pa
    np.testing.assert_array_equal(view.image.getImageItem().image, response.response_probability)
    view.detector_blur.setChecked(False)
    qtbot.waitUntil(lambda: view._displayed_image is arrivals)
    view.detector_blur.setChecked(True)
    qtbot.waitUntil(lambda: view._displayed_image is response)
    view.fit_beam()
    qtbot.waitUntil(lambda: view._displayed_image is not None and
                   view._displayed_image.metrics["display_extent_mm"][1]-
                   view._displayed_image.metrics["display_extent_mm"][0] >= .2)
    assert "full receiver" in view.notice.text()


def test_later_optical_preview_keeps_completed_physical_receiver(page):
    capture = _capture()
    capture.workflow = "receiver"
    page.display_result(capture)
    page._render()
    expected = page._displayed_image
    optical = _capture()
    optical.workflow = "rays"
    optical.simulation = SimpleNamespace(metrics={"optical_tuning": True})
    page.display_result(optical)
    page._render()
    assert page._result is capture
    assert page._displayed_image is expected
    assert page._stale and "Previous calculation" in page.summary.text()


def test_small_screen_leaves_room_for_receiver_image(page, qtbot, qapp):
    from temsim.app import APPLICATION_STYLE
    old = qapp.styleSheet()
    qapp.setStyleSheet(APPLICATION_STYLE)
    try:
        page.display_result(_capture())
        page.resize(1000, 550)
        page.show()
        qtbot.waitUntil(lambda: page._displayed_image is not None)
        assert page.width() == 1000 and page.height() == 550
        assert page.image.height() >= 250
    finally:
        qapp.setStyleSheet(old)


def test_workspace_routes_receiver_calculation_without_wave_request(qtbot):
    from temsim.gui.visualization import VisualizationWorkspace
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    requests = []
    workspace.calculation_requested.connect(requests.append)
    assert workspace.illuminating_page.widget(0) is workspace.receiver_imaging
    assert workspace.illuminating_page.widget(2) is workspace.wave_imaging
    workspace.receiver_imaging.calculate_button.click()
    assert requests == ["receiver"]
    assert workspace.selected_plane_readout.shutdown()
    assert workspace.conjugate_planes.shutdown()
