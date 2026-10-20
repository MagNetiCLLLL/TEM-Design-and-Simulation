"""Bounded viewport sampling of captured fields, without optical solves."""

from contextlib import contextmanager
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.gui.diagnostic_tabs import MagneticFieldView
from temsim.magnetic_field_scene import MagneticDiagnosticProfile


def synthetic_profile(z):
    z = np.asarray(z, dtype=float)
    on_axis = np.column_stack((z * .001, z * -.002, np.sin(z)))
    return MagneticDiagnosticProfile(
        z, on_axis, np.full(len(z), .02), np.full(len(z), .1),
        np.full(len(z), 1e-5), np.ones(len(z), dtype=bool),
    )


@pytest.fixture
def view(qtbot, monkeypatch):
    page = MagneticFieldView()
    qtbot.addWidget(page)
    page.resize(1000, 600)
    calls, budgets = [], []

    @contextmanager
    def job(requested, *, cancelled=None):
        budgets.append(requested)
        yield

    def sample(scene, z):
        calls.append((scene, np.array(z), get_ident()))
        return synthetic_profile(z)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Display detail must not prepare or republish a field scene")

    monkeypatch.setattr("temsim.cpu_resources.numerical_job", job)
    monkeypatch.setattr("temsim.magnetic_field_scene.sample_magnetic_diagnostic", sample)
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", forbidden)
    monkeypatch.setattr(page.field_lines, "update_snapshot", forbidden)
    monkeypatch.setattr(page.field_vectors, "update_snapshot", forbidden)
    monkeypatch.setattr(page.field_lines, "set_active", lambda *_a: None)
    monkeypatch.setattr(page.field_vectors, "set_active", lambda *_a: None)
    page._scene = SimpleNamespace(source_keys=("synthetic",), source_regions=(), notes=())
    page._profile = synthetic_profile(np.linspace(0., 1000., 3000))
    page._z_limits_mm = (0., 1000.)
    page._render_cached_profile()
    page.plot.setRange(xRange=(0., 1000.), yRange=(-2., 2.), padding=0)
    page.show()
    qtbot.wait(20)
    page._cancel_profile_detail()
    page.test_calls = calls
    page.test_budgets = budgets
    yield page
    page.hide()
    qtbot.waitUntil(lambda: page._detail_worker is None, timeout=5000)


def completed(page, qtbot):
    qtbot.waitUntil(
        lambda: page._detail_profile is not None and page._detail_worker is None,
        timeout=5000,
    )


def test_zoom_samples_captured_scene_locally_on_one_cpu_without_republishing(view, qtbot):
    overview, scene = view._profile, view._scene
    reference = view.field_lines.reference.value()
    view.plot.setXRange(25., 25.1, padding=0)
    ranges = np.array(view.plot.viewRange())
    completed(view, qtbot)
    used_scene, z, thread = view.test_calls[0]
    assert used_scene is scene and view._scene is scene
    assert view._profile is overview
    assert 128 <= len(z) <= 4096 and len(z) > 1000
    np.testing.assert_allclose(z[[0, -1]], (25., 25.1))
    assert thread != get_ident()
    assert view.test_budgets == [1]
    np.testing.assert_allclose(view.plot.viewRange(), ranges)
    assert view.field_lines.reference.value() == reference
    np.testing.assert_allclose(view._curves["bz"].xData, z)
    assert "local detail" in view.status.text()


def test_rapid_range_changes_debounce_to_only_latest_view(view, qtbot):
    for offset in range(8):
        view.plot.setXRange(100. + offset, 110. + offset, padding=0)
        qtbot.wait(10)
    assert not view.test_calls
    completed(view, qtbot)
    assert len(view.test_calls) == 1
    np.testing.assert_allclose(view.test_calls[0][1][[0, -1]], (107., 117.))


def test_large_viewport_still_has_a_fixed_sampling_budget(view, qtbot, monkeypatch):
    monkeypatch.setattr(view.plot.getViewBox(), "width", lambda: 12000.)
    view.plot.setXRange(20., 21., padding=0)
    completed(view, qtbot)
    assert len(view.test_calls[0][1]) == 4096
    assert len(view._detail_profile.z_mm) == 4096


def test_lru_bounded_and_auto_detail_off_restores_overview(view, qtbot):
    for index in range(5):
        view.plot.setXRange(10. + index, 11. + index, padding=0)
        completed(view, qtbot)
    assert len(view._detail_cache) == 4
    assert len(view.test_calls) == 5
    view.plot.setXRange(13., 14., padding=0)
    completed(view, qtbot)
    assert len(view.test_calls) == 5
    assert sum(len(profile.z_mm) for profile in view._detail_cache.values()) <= 4 * 4096
    view.auto_detail.setChecked(False)
    assert view._detail_profile is None
    np.testing.assert_array_equal(view._curves["bz"].xData, view._profile.z_mm)
    view.auto_detail.setChecked(True)
    completed(view, qtbot)
    assert len(view.test_calls) == 5


def test_shared_ray_axis_zoom_requests_detail_without_changing_ray_range(view, qtbot):
    ray_plot = pg.PlotWidget()
    qtbot.addWidget(ray_plot)
    ray_plot.setRange(xRange=(0., 1000.), yRange=(-.1, .1), padding=0)
    view.link_axial_axis(ray_plot, align_pixels=False)
    ray_plot.setXRange(300., 301., padding=0)
    expected = np.array(ray_plot.viewRange())
    completed(view, qtbot)
    np.testing.assert_allclose(view._detail_profile.z_mm[[0, -1]], (300., 301.))
    np.testing.assert_allclose(ray_plot.viewRange(), expected)


def test_hidden_page_and_other_modes_do_not_sample(view, qtbot):
    view.hide()
    view.plot.setXRange(40., 41., padding=0)
    qtbot.wait(view.DETAIL_DELAY_MS + 30)
    assert not view.test_calls
    view.display_mode.setCurrentIndex(view.display_mode.findData("3d"))
    view.show()
    qtbot.wait(view.DETAIL_DELAY_MS + 30)
    assert not view.test_calls
    view.display_mode.setCurrentIndex(view.display_mode.findData("2d"))
    completed(view, qtbot)
    assert len(view.test_calls) == 1
    view.hide()
    assert not view._detail_timer.isActive()
    np.testing.assert_array_equal(view._curves["bz"].xData, view._profile.z_mm)


def test_obsolete_worker_cannot_publish_and_only_latest_request_is_queued(view, qtbot, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def sample(scene, z):
        calls.append(np.array(z))
        if len(calls) == 1:
            entered.set()
            assert release.wait(5.)
        return synthetic_profile(z)

    monkeypatch.setattr("temsim.magnetic_field_scene.sample_magnetic_diagnostic", sample)
    try:
        view.plot.setXRange(40., 41., padding=0)
        qtbot.waitUntil(entered.is_set)
        worker = view._detail_worker
        for index in range(3):
            view.plot.setXRange(50. + index, 51. + index, padding=0)
        qtbot.wait(view.DETAIL_DELAY_MS + 30)
        assert view._detail_worker is worker and worker.cancelled.is_set()
        assert len(calls) == 1
        assert view._detail_request[1][1:3] == (52., 53.)
    finally:
        release.set()
    completed(view, qtbot)
    assert len(calls) == 2
    assert len(view._detail_cache) == 1
    np.testing.assert_allclose(view._detail_profile.z_mm[[0, -1]], (52., 53.))


def test_input_invalidation_cancels_detail_and_rejects_late_result(view, qtbot, monkeypatch):
    entered, release = Event(), Event()

    def sample(scene, z):
        entered.set()
        assert release.wait(5.)
        return synthetic_profile(z)

    monkeypatch.setattr("temsim.magnetic_field_scene.sample_magnetic_diagnostic", sample)
    try:
        view.plot.setXRange(40., 41., padding=0)
        qtbot.waitUntil(entered.is_set)
        view.mark_presentation_pending()
        assert not view._detail_timer.isActive()
        assert view._detail_worker.cancelled.is_set()
    finally:
        release.set()
    qtbot.waitUntil(lambda: view._detail_worker is None)
    assert view._profile is None and view._detail_profile is None
    assert not view._detail_cache
    assert view._curves["bz"].xData is None or not len(view._curves["bz"].xData)


def test_hiding_during_sampling_cancels_worker_and_leaves_overview(view, qtbot, monkeypatch):
    entered, release = Event(), Event()

    def sample(scene, z):
        entered.set()
        assert release.wait(5.)
        return synthetic_profile(z)

    monkeypatch.setattr("temsim.magnetic_field_scene.sample_magnetic_diagnostic", sample)
    try:
        view.plot.setXRange(40., 41., padding=0)
        qtbot.waitUntil(entered.is_set)
        worker = view._detail_worker
        view.hide()
        assert worker.cancelled.is_set()
        assert view._detail_request is None
    finally:
        release.set()
    qtbot.waitUntil(lambda: view._detail_worker is None)
    assert view._detail_profile is None and not view._detail_cache
    np.testing.assert_array_equal(view._curves["bz"].xData, view._profile.z_mm)


def test_viewport_clipping_overview_and_rotation_reuse(view, qtbot):
    view.plot.setXRange(-10., 1., padding=0)
    completed(view, qtbot)
    np.testing.assert_allclose(view._detail_profile.z_mm[[0, -1]], (0., 1.))
    calls = len(view.test_calls)
    detail = view._detail_profile
    view.set_projection_angle(90.)
    assert view._detail_profile is detail and len(view.test_calls) == calls
    np.testing.assert_allclose(view._curves["bu"].yData, detail.on_axis_t[:, 1])
    view.plot.setXRange(0., 1000., padding=0)
    assert view._detail_profile is None
    qtbot.wait(view.DETAIL_DELAY_MS + 30)
    assert len(view.test_calls) == calls
    np.testing.assert_array_equal(view._curves["bz"].xData, view._profile.z_mm)


def test_stale_inputs_restore_complete_prior_overview_without_new_sampling(view, qtbot):
    view.plot.setXRange(40., 41., padding=0)
    completed(view, qtbot)
    calls = len(view.test_calls)
    view.mark_inputs_stale()
    view.plot.setXRange(0., 1000., padding=0)
    qtbot.wait(view.DETAIL_DELAY_MS + 30)
    assert view._detail_profile is None and not view._detail_cache
    assert len(view.test_calls) == calls
    np.testing.assert_array_equal(view._curves["bz"].xData, view._profile.z_mm)
    assert "Stale magnetic field" in view.status.text()

