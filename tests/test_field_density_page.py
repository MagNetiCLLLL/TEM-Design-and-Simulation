"""Density presentation reuses the field snapshot and retains masked grid topology."""
from threading import Event
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt

from test_field_vector_page import activate, setup_page, wait_sample


def test_rotation_keeps_inflight_provider_capture_and_samples_latest_angle(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    import temsim.field_vector_scene as field_scene
    prepare = field_scene.prepare_field_vector_scene
    entered, release = Event(), Event()

    def blocked_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        entered.set()
        assert release.wait(10.)
        return result

    monkeypatch.setattr(field_scene, "prepare_field_vector_scene", blocked_prepare)
    try:
        page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.), prepared_scene="captured-B")
        page.set_active(True)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        worker, generation = page._worker, page._generation
        page.set_projection_angle(37.)
        page.set_projection_angle(90.)
        qtbot.wait(350)
        assert page._worker is worker and page._generation == generation
        assert not worker.cancelled.is_set()
        assert len(captures) == 1 and not scene.calls
        release.set()
        wait_sample(qtbot, page)
        assert len(captures) == 1 and len(scene.calls) == 1
        assert page._sample.section_angle_deg == 90.
        np.testing.assert_allclose(page._sample.points_m[:, 0], 0., atol=1e-16)
        assert page.canvas.sample is page._sample
    finally:
        release.set()


def test_arrows_remain_default_and_hatching_switches_without_sampling(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)
    assert page.render_style.currentData() == "arrows"
    assert page.quantity.currentData() == "overlay"
    assert not page.quantity.isHidden()
    assert not page.render_style.isHidden()
    sample = page._sample
    count, capture_count = len(scene.calls), len(captures)
    reference = page.canvas.references.copy()

    page.render_style.setCurrentIndex(page.render_style.findData("density"))
    assert page.quantity.currentData() == "B"
    assert not page.quantity.model().item(page.quantity.findData("overlay")).isEnabled()
    assert "different units" in page.quantity.toolTip()
    for quantity in ("E", "force", "B"):
        page.quantity.setCurrentIndex(page.quantity.findData(quantity))
        assert page.canvas._mode == quantity
    page.render_style.setCurrentIndex(page.render_style.findData("arrows"))
    assert page.quantity.model().item(page.quantity.findData("overlay")).isEnabled()
    page.quantity.setCurrentIndex(page.quantity.findData("overlay"))
    qtbot.wait(350)

    assert page._sample is sample
    assert len(scene.calls) == count
    assert len(captures) == capture_count
    assert page.canvas.references == reference
    assert page.canvas._mode == "overlay"


def test_shown_small_page_style_switch_does_not_trigger_auto_detail(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    assert page.auto_detail.isChecked()
    page.resize(700, 550)
    page.show()
    activate(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)

    def settled():
        return (page._worker is None and not page._jobs
                and not page._request_timer.isActive()
                and page._detail_worker is None and not page._detail_jobs
                and not page._detail_timer.isActive() and not page._detail_dirty)

    # Let deferred layout/resize events and their 250 ms refinement debounce
    # finish before recording the baseline. Auto detail remains enabled.
    qtbot.wait(500)
    qtbot.waitUntil(settled, timeout=10000)
    calls_before, captures_before = len(scene.calls), len(captures)
    page.render_style.setCurrentIndex(page.render_style.findData("density"))
    qtbot.wait(500)
    qtbot.waitUntil(settled, timeout=10000)
    assert page.isVisible() and page.auto_detail.isChecked()
    assert len(scene.calls) == calls_before
    assert len(captures) == captures_before

    page.render_style.setCurrentIndex(page.render_style.findData("arrows"))
    page.quantity.setCurrentIndex(page.quantity.findData("overlay"))
    qtbot.wait(500)
    qtbot.waitUntil(settled, timeout=10000)
    assert len(scene.calls) == calls_before
    assert len(captures) == captures_before


def test_hatching_is_limited_to_2d_and_overlay_cannot_bypass_units(qtbot, monkeypatch):
    page, _scene, _captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    page.render_style.setCurrentIndex(page.render_style.findData("density"))
    # A programmatic change must enforce the same restriction as a user click.
    page.quantity.setCurrentIndex(page.quantity.findData("overlay"))
    assert page.quantity.currentData() == "B"

    page.section.setCurrentIndex(page.section.findData("3D"))
    wait_sample(qtbot, page)
    assert page.render_style.currentData() == "arrows"
    assert not page.render_style.model().item(page.render_style.findData("density")).isEnabled()
    assert "3D" in page.render_style.toolTip()
    assert page._sample.grid_axes_m is None
    page.render_style.setCurrentIndex(page.render_style.findData("density"))
    assert page.render_style.currentData() == "arrows"

    page.section.setCurrentIndex(page.section.findData("UZ"))
    wait_sample(qtbot, page)
    assert page.render_style.model().item(page.render_style.findData("density")).isEnabled()
    page.render_style.setCurrentIndex(page.render_style.findData("density"))
    assert page.render_style.currentData() == "density"


def test_grid_metadata_keeps_rows_removed_by_the_physical_mask(qtbot, monkeypatch):
    page, _scene, _captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    full_points, _bounds, _section = page._sample_grid()
    full_x, full_z = np.unique(full_points[:, 0]), np.unique(full_points[:, 2])
    removed_x = full_x[len(full_x) // 2]
    page._display_regions.mask = lambda points, key=None: points[:, 0] != removed_x
    page._controls_changed()
    wait_sample(qtbot, page)

    assert not np.any(page._sample.points_m[:, 0] == removed_x)
    np.testing.assert_array_equal(page._sample.grid_axes_m[0], full_x)
    np.testing.assert_array_equal(page._sample.grid_axes_m[1], full_z)
    assert page._base_sample_data[0].grid_axes_m is page._sample.grid_axes_m


def test_visible_detail_and_cache_retain_the_unmasked_grid(qtbot, monkeypatch):
    page, scene, _captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    xs, zs = np.linspace(-.001, .001, 5), np.linspace(0., .01, 4)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    full_points = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
    key = ("density-test-visible-grid",)
    page._display_regions.mask = lambda points, component=None: points[:, 0] != 0.
    monkeypatch.setattr(page.canvas, "viewport_sample_grid",
                        lambda *_args, **_kwargs: (full_points, key))
    page.auto_detail.setChecked(True)
    qtbot.waitUntil(lambda: page._detail_key == key and page._detail_worker is None,
                    timeout=10000)

    assert len(page._sample.points_m) == 16
    np.testing.assert_array_equal(page._sample.grid_axes_m[0], xs)
    np.testing.assert_array_equal(page._sample.grid_axes_m[1], zs)
    cached = page._detail_cache[key][0]
    assert cached is page._sample
    call_count = len(scene.calls)
    # Force the same viewport to be looked up in the existing cache.
    page._detail_key = None
    page._request_view_detail()
    assert page._sample is cached
    assert len(scene.calls) == call_count
