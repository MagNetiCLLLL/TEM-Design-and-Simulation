"""Offscreen lifecycle/selection/units checks for physical vector presentation."""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QWheelEvent

from temsim.gui.field_vector_page import FieldVectorPage


def make_source(key, category="magnetic", supported=True):
    return SimpleNamespace(key=key, label=key.title(), category=category,
                           component_keys=(key,), bounds_m=np.array([[-.001, -.001, 0.], [.001, .001, .01]]),
                           model="Synthetic constant provider", notes=("Test-only model.",), supported=supported)


class Scene:
    def __init__(self, block=None, value=1.):
        self.sources = (make_source("magnet"), make_source("electrode", "electric"),
                        make_source("missing", "unmodelled", False))
        self.bounds_m = self.sources[0].bounds_m
        self.notes = ("Shared physical channels included once.",)
        self.calls = []
        self.block, self.value = block, value

    def sample(self, points, source_keys, *, kinetic_energy_ev, direction):
        self.calls.append((points.copy(), source_keys, kinetic_energy_ev, direction))
        if self.block:
            self.block[0].set()
            assert self.block[1].wait(10.)
        count = len(points)
        b = np.zeros((count, 3))
        e = np.zeros((count, 3))
        if "magnet" in source_keys:
            b[:, 2] = self.value
        if "electrode" in source_keys:
            e[:, 0] = 2.
        valid = np.full(count, "missing" not in source_keys)
        force = np.ones((count, 3)) * kinetic_energy_ev * 1e-20
        b[~valid] = np.nan
        force[~valid] = np.nan
        return SimpleNamespace(points_m=points, b_t=b, e_v_per_m=e, force_n=force,
                               magnetic_valid=valid, electric_valid=np.ones(count, bool),
                               force_valid=valid, notes=("Synthetic sampling.",))


def setup_page(qtbot, monkeypatch, scene=None):
    scene = scene or Scene()
    calls = []

    def prepare(state, **kwargs):
        calls.append((state, kwargs))
        return scene

    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene", prepare)
    # Lifecycle fixtures supply an explicit observation region. Production
    # snapshots without captured mechanical geometry are tested separately.
    region = SimpleNamespace(
        notes=("Synthetic physical observation region.",),
        mask=lambda points, key=None: np.ones(len(points), dtype=bool),
        bounds_for_component=lambda key=None: scene.bounds_m,
        source_keys_for_component=lambda key: (key,) if key else (),
        regions_for_component=lambda key=None: (),
    )
    monkeypatch.setattr("temsim.field_display_regions.build_field_display_regions", lambda *_args: region)
    page = FieldVectorPage()
    qtbot.addWidget(page)
    page.resize(900, 650)
    return page, scene, calls


def wait_sample(qtbot, page):
    qtbot.waitUntil(lambda: page._sample is not None and page._worker is None, timeout=10000)


def activate(qtbot, page):
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.), prepared_scene="captured-B", result_reference="rays")
    page.set_active(True)
    wait_sample(qtbot, page)


def test_lazy_capture_and_fixed_detail_navigation_never_resamples(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.), prepared_scene="captured-B", result_reference="rays")
    qtbot.wait(150)
    assert not captures and not scene.calls
    page.set_active(True)
    wait_sample(qtbot, page)
    assert captures[0][1] == {"magnetic_scene": "captured-B", "z_limits_mm": None}
    assert page._result_reference == "rays" and page.energy.value() == 200000.
    call_count = len(scene.calls)
    page.quantity.setCurrentIndex(page.quantity.findData("force"))
    page.fit_view.click()
    event = QWheelEvent(QPointF(100, 100), QPointF(100, 100), QPointF(0, 0).toPoint(), QPointF(0, 120).toPoint(),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
    page.canvas.wheelEvent(event)
    page.resize(1000, 700)
    page.canvas.grab()
    page.set_active(False)
    page.set_active(True)
    qtbot.wait(300)
    assert len(scene.calls) == call_count and len(captures) == 1


def test_source_checklist_passes_exact_keys_and_unmodelled_is_not_zero(qtbot, monkeypatch):
    page, scene, _captures = setup_page(qtbot, monkeypatch)
    activate(qtbot, page)
    assert page.selected_source_keys == ("magnet", "electrode", "missing")
    assert np.isnan(page._sample.b_t).all()
    assert "unmodelled" in page.sources.item(2).text()
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)
    assert scene.calls[-1][1] == ("magnet", "electrode")
    assert np.all(page._sample.b_t[:, 2] == 1.)
    assert np.all(page._sample.e_v_per_m[:, 0] == 2.)
    assert "B [T]" in page.legend.text() and "E [V/m]" in page.legend.text()
    assert "enclosing" in page.details.toPlainText().lower()
    page.select_none.click()
    wait_sample(qtbot, page)
    assert scene.calls[-1][1] == ()
    assert np.all(page._sample.b_t == 0.) and np.all(page._sample.e_v_per_m == 0.)


@pytest.mark.parametrize("section, angle", [("UZ", 0.), ("UZ", 90.), ("3D", 0.)])
def test_slice_coordinates_and_3d_budget(qtbot, monkeypatch, section, angle):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    activate(qtbot, page)
    page.centre_x.setValue(.1)
    page.centre_y.setValue(.2)
    page.centre_z.setValue(4.)
    page.section.setCurrentIndex(page.section.findData(section))
    page.set_projection_angle(angle)
    page.density.setCurrentIndex(2)
    wait_sample(qtbot, page)
    points = page._sample.points_m
    if section == "UZ":
        theta = np.deg2rad(angle)
        u = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
        v = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
        assert np.allclose(v, 0., rtol=0., atol=1e-16)
        assert np.isclose(np.mean(u), .0001*np.cos(theta) + .0002*np.sin(theta))
        assert np.isclose(np.mean(points[:, 2]), .004)
        assert len(points) == 441
        assert page._sample.section_angle_deg == angle
        # Increasing axial position always moves right, even at 90°.
        pair = np.array([[0., 0., .003], [0., 0., .005]])
        projected = page.canvas._project_positions(pair, page.canvas._plot_rect())
        assert projected[1, 0] > projected[0, 0]
        assert np.isclose(projected[0, 1], projected[1, 1])
    else:
        assert len(points) == 729
        assert all(len(np.unique(points[:, axis])) == 9 for axis in range(3))
    assert page.canvas._section == section
    assert page.canvas.grab().width() > 0


def test_longitudinal_view_replaces_transverse_and_fixed_axis_selectors(qtbot, monkeypatch):
    page, _scene, _captures = setup_page(qtbot, monkeypatch)
    assert [page.section.itemData(index) for index in range(page.section.count())] == ["UZ", "3D"]
    assert page.section.currentData() == "UZ"
    assert all(page.section.findData(old) == -1 for old in ("XY", "XZ", "YZ"))
    assert page.angle.value() == 0.


def test_rotating_section_reuses_capture_with_auto_detail_disabled(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    before, capture_count = len(scene.calls), len(captures)
    changed = []
    page.settings_changed.connect(lambda: changed.append(True))
    page.set_projection_angle(90.)
    assert page._sample is None and page.canvas.sample is None
    wait_sample(qtbot, page)
    assert len(scene.calls) == before + 1
    assert len(captures) == capture_count
    assert page._sample.section_angle_deg == 90.
    assert np.allclose(page._sample.points_m[:, 0], 0., atol=1e-16)
    assert page.angle.value() == 90.
    assert not changed
    page.set_projection_angle(90.)
    page.set_projection_angle(450.)
    qtbot.wait(350)
    assert len(scene.calls) == before + 1
    assert len(captures) == capture_count
    assert not changed


def test_fast_rotations_sample_only_latest_plane(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    before = len(scene.calls)
    for angle in (10., 25., 50., 80.):
        page.set_projection_angle(angle)
    wait_sample(qtbot, page)
    assert len(scene.calls) == before + 1
    assert len(captures) == 1
    assert page._sample.section_angle_deg == 80.
    theta = np.deg2rad(80.)
    points = page._sample.points_m
    np.testing.assert_allclose(-points[:, 0]*np.sin(theta) + points[:, 1]*np.cos(theta), 0., atol=1e-16)


def test_hidden_rotation_defers_sampling_until_page_activation(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    before = len(scene.calls)
    page.set_active(False)
    page.hide()
    page.set_projection_angle(60.)
    qtbot.wait(350)
    assert len(scene.calls) == before
    assert page._sample is None
    page.set_active(True)
    wait_sample(qtbot, page)
    assert len(scene.calls) == before + 1
    assert len(captures) == 1
    assert page._sample.section_angle_deg == 60.


def test_angle_control_and_canvas_requests_update_page_before_forwarding(qtbot, monkeypatch):
    page, _scene, _captures = setup_page(qtbot, monkeypatch)
    received = []
    page.projection_angle_requested.connect(
        lambda angle: received.append((angle, page.angle.value(), page._projection_angle_deg)))
    page.angle.setValue(45.)
    page.canvas.projection_angle_requested.emit(90.)
    assert received == [(45., 45., 45.), (90., 90., 90.)]
    assert page.canvas._projection_angle_deg == 90.


def test_inflight_old_angle_cannot_overwrite_latest_plane(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    activate(qtbot, page)
    entered, release = Event(), Event()
    scene.block = entered, release
    published = []
    publish = page._publish_sample

    def record(result):
        published.append(result[0].section_angle_deg)
        publish(result)

    monkeypatch.setattr(page, "_publish_sample", record)
    try:
        page.set_projection_angle(30.)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        page.set_projection_angle(60.)
        release.set()
        wait_sample(qtbot, page)
        qtbot.waitUntil(lambda: not page._jobs, timeout=10000)
        assert published == [60.]
        assert page._sample.section_angle_deg == 60.
        assert len(captures) == 1
    finally:
        release.set()


def test_force_energy_direction_and_zero_direction_validation(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    activate(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    page.energy.setValue(100000.)
    page.direction_x.setValue(1.)
    page.direction_z.setValue(-1.)
    page.quantity.setCurrentIndex(page.quantity.findData("force"))
    wait_sample(qtbot, page)
    assert scene.calls[-1][2:] == (100000., (1., 0., -1.))
    assert "F [N]" in page.legend.text()
    assert "uniform test" in page.status.text()
    page.direction_x.setValue(0.)
    page.direction_z.setValue(0.)
    qtbot.waitUntil(lambda: "nonzero" in page.status.text(), timeout=2000)
    assert page._sample is None and page.canvas.sample is None


def test_stale_inputs_clear_data_and_block_all_sampling_until_new_snapshot(qtbot, monkeypatch):
    page, scene, captures = setup_page(qtbot, monkeypatch)
    activate(qtbot, page)
    page.mark_inputs_stale()
    count = len(scene.calls)
    assert page._sample is None and page.canvas.sample is None
    assert "Stale" in page.status.text()
    page.centre_z.setValue(2.)
    page.quantity.setCurrentIndex(1)
    page._request_sample()
    qtbot.wait(150)
    assert len(scene.calls) == count
    assert "no current" in page.legend.text() or "B [T]" in page.legend.text()
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=300.), prepared_scene="new-B")
    wait_sample(qtbot, page)
    assert len(captures) == 2 and captures[-1][1]["magnetic_scene"] == "new-B"
    assert "Current snapshot" in page.status.text()
    page.invalidate("No accepted result")
    assert page.canvas.sample is None and page._state is None
    assert page.sources.count() == 0 and not page.details.toPlainText()


def test_inflight_sample_cannot_publish_after_snapshot_replacement(qtbot, monkeypatch):
    entered, release = Event(), Event()
    first, latest = Scene((entered, release), 1.), Scene(value=7.)
    page, _scene, _captures = setup_page(qtbot, monkeypatch, first)
    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene",
                        lambda state, **kwargs: first if state == "old" else latest)
    page.update_snapshot("old")
    page.set_active(True)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    old_generation = page._generation
    page.mark_inputs_stale()
    page.update_snapshot("new")
    release.set()
    wait_sample(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)
    assert page._generation > old_generation
    assert np.all(page._sample.b_t[:, 2] == 7.)
    assert page._scene is latest
    assert not page._jobs


def test_late_capture_cannot_restore_invalidated_display(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    entered, release = Event(), Event()

    def prepare(*_args, **_kwargs):
        entered.set()
        assert release.wait(10.)
        return scene

    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene", prepare)
    page.update_snapshot(object())
    page.set_active(True)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    page.invalidate()
    release.set()
    qtbot.waitUntil(lambda: not page._jobs, timeout=10000)
    assert page._scene is None and page.canvas.sample is None and page.sources.count() == 0
    assert not scene.calls


def test_control_burst_is_debounced_and_hidden_page_has_no_minimum_layout_pressure(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    activate(qtbot, page)
    count = len(scene.calls)
    for value in np.linspace(1., 2., 12):
        page.centre_z.setValue(value)
    wait_sample(qtbot, page)
    assert len(scene.calls) == count + 1
    assert page.minimumSizeHint().height() == 0
    assert page.status.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert page.legend.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse


def test_fixed_arrow_references_preserve_strength_changes_across_snapshots(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    activate(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    wait_sample(qtbot, page)
    assert page.canvas.references["B"] == 1.
    scene.value = 2.
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.))
    wait_sample(qtbot, page)
    assert page.canvas.references["B"] == 1.
    assert "max 2" in page.legend.text() and "arrow ref 1" in page.legend.text()
    # A vector in the displayed plane doubles in length at a fixed reference.
    values = np.array([[1., 0., 0.], [2., 0., 0.]])
    arrows = page.canvas._display_vectors(values, np.ones(2, bool), np.eye(3), np.ones(3), np.ones(2), 10., 1.)
    assert np.allclose(np.linalg.norm(arrows, axis=1), [10., 20.])
    count = len(scene.calls)
    page.match_references.click()
    assert page.canvas.references["B"] == 2.
    page.reference_inputs["B"].setText("5e-3")
    page.reference_inputs["B"].editingFinished.emit()
    assert page.canvas.references["B"] == .005
    page.reference_inputs["B"].setText("nan")
    page.reference_inputs["B"].editingFinished.emit()
    assert page.canvas.references["B"] == .005
    assert "positive" in page.status.text()
    qtbot.wait(150)
    assert len(scene.calls) == count


def test_parent_destruction_cancels_inflight_job_without_close_event(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    entered, release = Event(), Event()

    def prepare(*_args, **_kwargs):
        entered.set()
        assert release.wait(10.)
        return scene

    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene", prepare)
    page.update_snapshot(object())
    page.set_active(True)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    worker = page._worker
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert worker.cancelled.is_set()
    release.set()
    qtbot.wait(100)
    assert not scene.calls


@pytest.mark.parametrize("intent, expected", [("all", ("magnet", "electrode", "missing")),
                                               ("none", ()), ("subset", ("magnet",))])
def test_selection_intent_survives_invalidation_and_new_source_catalogue(qtbot, monkeypatch, intent, expected):
    scene = Scene()
    scene.sources = scene.sources[:2]
    page, _, _ = setup_page(qtbot, monkeypatch, scene)
    activate(qtbot, page)
    if intent == "none":
        page.select_none.click()
    elif intent == "subset":
        page.sources.item(1).setCheckState(Qt.CheckState.Unchecked)
    else:
        page.select_all.click()
    wait_sample(qtbot, page)
    page.invalidate()
    scene.sources += (make_source("missing", "unmodelled", False),)
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.))
    wait_sample(qtbot, page)
    assert page.selected_source_keys == expected
    assert scene.calls[-1][1] == expected
    if intent == "all":
        assert np.isnan(page._sample.b_t).all()


def test_section_normal_magnetic_field_has_visible_marks_distinct_from_zero(qtbot, monkeypatch):
    page, scene, _ = setup_page(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    original_sample = scene.sample

    def normal_sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        # The default is now the longitudinal UZ plane at zero degrees:
        # its normal is global Y, not the old transverse XY view's Z.
        result.b_t = result.b_t[:, [0, 2, 1]]
        return result

    monkeypatch.setattr(scene, "sample", normal_sample)
    activate(qtbot, page)
    page.sources.item(2).setCheckState(Qt.CheckState.Unchecked)
    page.quantity.setCurrentIndex(page.quantity.findData("B"))
    wait_sample(qtbot, page)
    page.show()
    qtbot.wait(30)
    # Compare glyph pixels at the same viewport, independently of wrapping
    # status labels that can change height as a snapshot is replaced.
    page.canvas.setFixedSize(page.canvas.size())
    np.testing.assert_array_equal(page._sample.b_t @ page.canvas._basis()[2], 1.)
    normal_image = page.canvas.grab().toImage()
    normal_pixels = np.frombuffer(normal_image.constBits(), np.uint8).copy()
    scene.value = 0.
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.))
    wait_sample(qtbot, page)
    zero_image = page.canvas.grab().toImage()
    zero_pixels = np.frombuffer(zero_image.constBits(), np.uint8).copy()
    assert normal_pixels.shape == zero_pixels.shape
    assert np.count_nonzero(normal_pixels != zero_pixels) > 200
