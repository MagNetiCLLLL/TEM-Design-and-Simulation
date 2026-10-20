"""Physical observation region is independent of which fields are summed."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from temsim.gui.field_vector_page import FieldVectorPage
from temsim.field_display_regions import build_field_display_regions
from test_field_vector_geometry import _part, _state
from test_field_vector_page import Scene, make_source, wait_sample


def captured_state():
    rows = [_part("magnet"), _part("electrode", local_start_z_mm=20.,
                 local_center_z_mm=22., local_end_z_mm=24.)]
    rows.append(_part("passive", local_start_z_mm=30., local_center_z_mm=32.,
                      local_end_z_mm=34., mechanical_profile="cylinder",
                      mechanical_only=True, mechanical_part_role="custom_mechanical_copy"))
    state = _state(rows, shift=0.)
    state.beam_voltage_kv = 200.
    return state


def page_fixture(qtbot, monkeypatch):
    state, scene = captured_state(), Scene()
    scene.sources = (make_source("magnet"), make_source("electrode", "electric"))
    scene.bounds_m = np.array([[-.1, -.1, 0.], [.1, .1, .05]])
    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene", lambda *_a, **_k: scene)
    page = FieldVectorPage()
    qtbot.addWidget(page)
    page.update_snapshot(state)
    page.focus_component("magnet")
    page.set_active(True)
    wait_sample(qtbot, page)
    return page, state, scene


def test_all_and_component_modes_keep_same_observation_points(qtbot, monkeypatch):
    page, state, scene = page_fixture(qtbot, monkeypatch)
    total = page._sample
    index = build_field_display_regions(state, scene.sources)
    assert index.mask(total.points_m, "magnet").all()
    assert np.all(total.b_t[:, 2] == 1.) and np.all(total.e_v_per_m[:, 0] == 2.)
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    wait_sample(qtbot, page)
    assert scene.calls[-1][1] == ("magnet",)
    np.testing.assert_array_equal(page._sample.points_m, total.points_m)
    np.testing.assert_array_equal(page._sample.b_t, total.b_t)
    assert np.all(page._sample.e_v_per_m == 0.)
    # The original total and captured state are not mutated by a display filter.
    assert np.all(total.e_v_per_m[:, 0] == 2.)
    assert state._resolved_assembly.parts[0].center_z_mm == 12.


def test_physical_envelope_includes_bore_and_excludes_external_3d_corners(qtbot, monkeypatch):
    page, state, scene = page_fixture(qtbot, monkeypatch)
    page.section.setCurrentIndex(page.section.findData("3D"))
    wait_sample(qtbot, page)
    points = page._sample.points_m
    assert 0 < len(points) < 729
    assert build_field_display_regions(state, scene.sources).mask(points, "magnet").all()
    assert np.any(np.linalg.norm(points[:, :2], axis=1) == 0.)
    # Camera-only rotation and material opacity preserve the exact sample object.
    sample, calls = page._sample, len(scene.calls)
    page.set_projection_angle(57.)
    page.model_opacity.setValue(100)
    qtbot.wait(150)
    assert page._sample is sample and len(scene.calls) == calls
    assert page.canvas.projection_angle_deg == 57.
    assert page.angle.value() == 57.


def test_longitudinal_rotation_samples_new_physical_plane_without_repreparing_scene(qtbot, monkeypatch):
    page, state, scene = page_fixture(qtbot, monkeypatch)
    page.auto_detail.setChecked(False)
    assert page.section.currentData() == "UZ"
    original = page._sample
    np.testing.assert_allclose(original.points_m[:, 1], 0., atol=1e-15)
    page.canvas.set_reference("B", 3.)
    references = page.canvas.references.copy()
    regions = build_field_display_regions(state, scene.sources)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Rotating the section must reuse its captured field providers")

    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene", forbidden)
    previous = original
    for angle in (37.5, 90.):
        calls = len(scene.calls)
        page.set_projection_angle(angle)
        # Never label the old plane's samples as belonging to the new section.
        assert page._sample is None and page.canvas.sample is None
        assert len(scene.calls) == calls
        wait_sample(qtbot, page)
        sample = page._sample
        assert sample is not previous and page._scene is scene
        assert len(scene.calls) > calls
        assert sample.section_angle_deg == angle
        radians = np.deg2rad(angle)
        normal = -sample.points_m[:, 0] * np.sin(radians) + sample.points_m[:, 1] * np.cos(radians)
        np.testing.assert_allclose(normal, 0., atol=1e-15)
        assert np.ptp(sample.points_m[:, 1]) > 0.
        assert np.ptp(sample.points_m[:, 2]) == pytest.approx(np.ptp(original.points_m[:, 2]))
        assert regions.mask(sample.points_m, "magnet").all()
        assert page.canvas.references == references
        assert page.angle.value() == angle
        previous = sample
    # Rotation affects observation only; neither the saved part nor old sample moves.
    np.testing.assert_allclose(original.points_m[:, 1], 0., atol=1e-15)
    assert state._resolved_assembly.parts[0].center_z_mm == 12.


def test_default_fit_resolves_near_axis_field_without_expanding_its_support(qtbot, monkeypatch):
    page, state, scene = page_fixture(qtbot, monkeypatch)
    scene.sources[0].bounds_m = np.array([[-.0001, -.0001, .008], [.0001, .0001, .016]])
    page.fit_selected.click()
    wait_sample(qtbot, page)
    points = page._sample.points_m.copy()
    assert np.max(np.abs(points[:, 0])) == .0001
    np.testing.assert_allclose([points[:, 2].min(), points[:, 2].max()], [.010, .014])
    assert build_field_display_regions(state, scene.sources).mask(points, "magnet").all()
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    wait_sample(qtbot, page)
    np.testing.assert_array_equal(page._sample.points_m, points)


def test_passive_component_has_no_own_field_but_can_observe_total(qtbot, monkeypatch):
    page, _state, scene = page_fixture(qtbot, monkeypatch)
    page.focus_component("passive")
    wait_sample(qtbot, page)
    assert np.all(page._sample.b_t[:, 2] == 1.)
    assert page.show_models.isChecked()
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    wait_sample(qtbot, page)
    assert scene.calls[-1][1] == ()
    assert np.all(page._sample.b_t == 0.) and np.all(page._sample.e_v_per_m == 0.)
    assert "no independent field source" in page.status.text()
    qtbot.waitUntil(lambda: page._geometry is not None, timeout=10000)
    assert {mesh.key for mesh in page.canvas._physical_meshes} == {"passive"}


def test_source_row_selects_real_component_and_context_uses_captured_z(qtbot, monkeypatch):
    page, _state, _scene = page_fixture(qtbot, monkeypatch)
    selected, navigations = [], []
    page.component_selected.connect(selected.append)
    page.navigation_requested.connect(lambda *args: navigations.append(args))
    page.sources.setCurrentRow(1)
    wait_sample(qtbot, page)
    assert page._selected_component == "electrode" and selected == ["electrode"]
    menu = page.component_navigation_menu("electrode")
    assert menu is not None
    menu.actions()[0].trigger()
    assert navigations == [("electrode", "ray", 22.)]
    assert page.component_navigation_menu("not_installed") is None
    page.mark_inputs_stale()
    assert page.component_navigation_menu("electrode") is None


def test_missing_captured_geometry_does_not_borrow_field_support(qtbot, monkeypatch):
    scene = Scene()
    monkeypatch.setattr("temsim.field_vector_scene.prepare_field_vector_scene", lambda *_a, **_k: scene)
    page = FieldVectorPage()
    qtbot.addWidget(page)
    page.update_snapshot(SimpleNamespace(beam_voltage_kv=200.))
    page.set_active(True)
    qtbot.waitUntil(lambda: "No physical observation" in page.status.text(), timeout=10000)
    assert page._sample is None and not scene.calls
    assert page._display_regions.bounds_for_component() is None


def test_single_source_intent_survives_snapshot_and_selected_part_removal(qtbot, monkeypatch):
    page, state, _scene = page_fixture(qtbot, monkeypatch)
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    wait_sample(qtbot, page)
    page.update_snapshot(state)
    wait_sample(qtbot, page)
    assert page.selected_source_keys == ("magnet",)
    page.focus_component("removed_component")
    qtbot.waitUntil(lambda: "No physical observation" in page.status.text(), timeout=10000)
    assert page._sample is None
    assert page.selected_source_keys == ()


def test_quick_picker_works_with_advanced_controls_collapsed(qtbot, monkeypatch):
    page, state, _scene = page_fixture(qtbot, monkeypatch)
    page.resize(1000, 650)
    page.show()
    qtbot.waitUntil(lambda: page._geometry is not None, timeout=10000)
    assert not page.controls_toggle.isChecked() and page.control_scroll.isHidden()
    assert page.component_picker.currentData() == "magnet"
    selected = []
    page.component_selected.connect(selected.append)
    page.component_picker.setCurrentIndex(page.component_picker.findData("passive"))
    wait_sample(qtbot, page)
    assert selected == ["passive"] and page._selected_component == "passive"
    assert page.canvas.height() >= 220
    # Capturing a new snapshot restores intent without issuing a new user selection.
    page.update_snapshot(state)
    wait_sample(qtbot, page)
    assert page.component_picker.currentData() == "passive" and selected == ["passive"]
    page.mark_inputs_stale()
    assert not page.component_picker.isEnabled()


def test_picker_excludes_virtual_channels_and_does_not_invent_zero_selection(qtbot, monkeypatch):
    page, state, scene = page_fixture(qtbot, monkeypatch)
    rows = [dict(part.data) for part in state._resolved_assembly.parts]
    rows.append(_part("virtual_channel", layout_role="control_channel"))
    page.update_snapshot(_state(rows, shift=0.))
    wait_sample(qtbot, page)
    assert page.component_picker.findData("virtual_channel") == -1
    page.field_scope.setCurrentIndex(page.field_scope.findData("component"))
    wait_sample(qtbot, page)
    calls = len(scene.calls)
    page.clear_selection.click()
    qtbot.waitUntil(lambda: "Select a physical component" in page.status.text(), timeout=10000)
    assert page.component_picker.currentData() is None
    assert page._sample is None and len(scene.calls) == calls
