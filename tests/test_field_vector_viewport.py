"""Bounded CPU viewport sampling, cancellation and captured-scene reuse."""
from threading import Event

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QWheelEvent

from test_field_vector_page import activate, setup_page, wait_sample


def ready(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.show()
    activate(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)
    # Real widget layout/projection, with no expensive optical or FEM solve.
    qtbot.wait(50)
    return page, scene, captures


def zoom(page, amount=120):
    position = page.canvas._plot_rect().center()
    event = QWheelEvent(position, position, QPointF().toPoint(), QPointF(0, amount).toPoint(),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.ScrollUpdate, False)
    page.canvas.wheelEvent(event)


def wait_detail(qtbot, page):
    qtbot.waitUntil(lambda: page._detail_key is not None and page._detail_worker is None
                   and not page._detail_dirty and not page._detail_timer.isActive(), timeout=10000)


def test_wheel_refines_visible_section_with_same_capture_mask_and_references(qtbot, monkeypatch):
    page, scene, captures = ready(qtbot, monkeypatch)
    base = page._base_sample_data
    refs = page.canvas.references.copy()
    n_captures, n_calls = len(captures), len(scene.calls)
    # A deliberately nonrectangular physical envelope must still clip detail.
    page._display_regions.mask = lambda p, key=None: p[:, 0] >= 0.
    zoom(page, 480)
    qtbot.wait(120)
    assert len(scene.calls) == n_calls  # Debounce retains cached vectors.
    wait_detail(qtbot, page)
    assert len(captures) == n_captures
    assert page._base_sample_data is base
    assert page.canvas.references == refs
    np.testing.assert_array_equal(page.canvas._bounds_m, base[1])
    points = page._sample.points_m
    assert 0 < len(points) <= 1600
    assert np.all(points[:, 0] >= 0.)
    assert np.all(points[:, 1] == 0.)
    assert np.ptp(points[:, 2]) < np.ptp(base[0].points_m[:, 2])
    assert all(len(call[0]) <= 512 for call in scene.calls[n_calls:])
    assert all(call[1] == ("magnet", "electrode") for call in scene.calls[n_calls:])


def test_repeated_wheels_coalesce_and_returning_view_uses_cache(qtbot, monkeypatch):
    page, scene, _ = ready(qtbot, monkeypatch)
    calls = len(scene.calls)
    for _ in range(4):
        zoom(page)
        qtbot.wait(20)
    assert len(scene.calls) == calls
    wait_detail(qtbot, page)
    first, first_key = page._sample, page._detail_key
    camera = page.canvas._zoom, QPointF(page.canvas._pan)
    zoom(page, 240)
    wait_detail(qtbot, page)
    assert page._detail_key != first_key
    calls = len(scene.calls)
    page.canvas._zoom, page.canvas._pan = camera
    page.canvas._view_updated()
    wait_detail(qtbot, page)
    assert page._sample is first
    assert page._detail_key == first_key
    assert len(scene.calls) == calls
    assert "cache" in page.detail_status.text()


def test_slow_detail_keeps_one_job_and_cannot_publish_obsolete_view(qtbot, monkeypatch):
    page, scene, _ = ready(qtbot, monkeypatch)
    base = page._sample
    entered, release = Event(), Event()
    scene.block = (entered, release)
    try:
        zoom(page, 240)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        old_worker = page._detail_worker
        for _ in range(3):
            zoom(page)
        qtbot.wait(320)
        assert old_worker.cancelled.is_set()
        assert len(page._detail_jobs) == 1
        assert page._sample is base
        scene.block = None
        release.set()
        wait_detail(qtbot, page)
        assert page._detail_key != old_worker.request_key
        assert not page._detail_jobs
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page._detail_jobs, timeout=10000)


@pytest.mark.parametrize("action", ["hide", "stale", "off"])
def test_pending_detail_cannot_publish_after_hide_stale_or_disable(qtbot, monkeypatch, action):
    page, scene, _ = ready(qtbot, monkeypatch)
    base = page._sample
    entered, release = Event(), Event()
    scene.block = (entered, release)
    try:
        zoom(page, 240)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        if action == "hide":
            page.set_active(False)
        elif action == "stale":
            page.mark_inputs_stale()
        else:
            page.auto_detail.setChecked(False)
        release.set()
        qtbot.waitUntil(lambda: not page._detail_jobs, timeout=10000)
        assert page._detail_key is None
        assert page._sample is (None if action == "stale" else base)
        calls = len(scene.calls)
        zoom(page)
        qtbot.wait(350)
        assert len(scene.calls) == calls
    finally:
        release.set()


def test_auto_off_restores_base_and_empty_physical_view_never_samples(qtbot, monkeypatch):
    page, scene, _ = ready(qtbot, monkeypatch)
    base = page._base_sample_data
    zoom(page, 240)
    wait_detail(qtbot, page)
    page.auto_detail.setChecked(False)
    assert page._sample is base[0]
    assert "Fixed section samples" in page.detail_status.text()
    calls = len(scene.calls)
    zoom(page)
    qtbot.wait(350)
    assert len(scene.calls) == calls
    page._display_regions.mask = lambda points, key=None: np.zeros(len(points), bool)
    page.auto_detail.setChecked(True)
    wait_detail(qtbot, page)
    assert len(scene.calls) == calls
    assert len(page._sample.points_m) == 0
    assert "No locations" in page.detail_status.text()
    assert page.canvas.grab().width() > 0


def test_returning_to_page_restarts_cancelled_view_and_source_change_refines_current_zoom(qtbot, monkeypatch):
    page, scene, captures = ready(qtbot, monkeypatch)
    entered, release = Event(), Event()
    scene.block = (entered, release)
    try:
        zoom(page, 480)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        page.set_active(False)
        scene.block = None
        release.set()
        qtbot.waitUntil(lambda: not page._detail_jobs, timeout=10000)
        assert page._detail_dirty
        page.set_active(True)
        wait_detail(qtbot, page)
        page.sources.item(1).setCheckState(Qt.CheckState.Unchecked)
        wait_sample(qtbot, page)
        wait_detail(qtbot, page)
        assert scene.calls[-1][1] == ("magnet",)
        assert np.ptp(page._sample.points_m[:, 2]) < np.ptp(page._base_sample_data[0].points_m[:, 2])
        assert len(captures) == 1
    finally:
        release.set()


def test_three_dimensional_view_never_resamples_on_navigation(qtbot, monkeypatch):
    page, scene, _ = ready(qtbot, monkeypatch)
    page.section.setCurrentIndex(page.section.findData("3D"))
    wait_sample(qtbot, page)
    sample, count = page._sample, len(scene.calls)
    zoom(page, 480)
    qtbot.wait(350)
    assert page._sample is sample
    assert len(scene.calls) == count
    assert "3D uses cached" in page.detail_status.text()


def test_view_cache_is_bounded_and_sampling_failure_retains_current_data(qtbot, monkeypatch):
    page, scene, _ = ready(qtbot, monkeypatch)
    for _ in range(6):
        zoom(page)
        wait_detail(qtbot, page)
    assert len(page._detail_cache) == 4
    assert all(len(item[0].points_m) <= 1600 for item in page._detail_cache.values())
    previous = page._sample

    def failed_sample(*args, **kwargs):
        raise ValueError("synthetic provider failure")

    monkeypatch.setattr(scene, "sample", failed_sample)
    zoom(page)
    qtbot.waitUntil(lambda: "synthetic provider failure" in page.detail_status.text(), timeout=10000)
    assert page._sample is previous
    assert not page._detail_jobs


def test_expanded_controls_preserve_plot_area_in_small_window(qtbot, monkeypatch):
    page, _scene, _ = ready(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    page.resize(980, 575)
    page.controls_toggle.setChecked(True)
    qtbot.wait(30)
    assert page.size().width() == 980 and page.size().height() == 575
    assert page.minimumSizeHint().height() == 0
    assert page.canvas.height() >= 220
    assert page.canvas._plot_rect().height() >= 109
    assert page.control_scroll.verticalScrollBar().maximum() > 0
    assert page.details_toggle.geometry().bottom() < page.height()
