"""Optical-only assembly display reuses geometry and leaves saved inputs alone."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from temsim.gui import assembly_model_page as gui
from temsim.gui import part_model_view as renderer
from test_assembly_model_3d import _assembly, _part


def scene(shift=100.):
    rows = [_part("coil", mechanical_part_role="excitation_coil"),
            _part("housing", mechanical_profile="magnetic_lens_housing",
                  mechanical_part_role="housing", mechanical_inner_diameter_mm=9.,
                  mechanical_outer_diameter_mm=11.)]
    liner = SimpleNamespace(key="@liner:test", name="Vacuum liner", start_z_mm=100.,
        end_z_mm=120., inner_diameter_mm=2., outer_diameter_mm=3.)
    return _assembly(rows, shift=shift, liners=(liner,))


def show(qtbot, assembly):
    page = gui.AssemblyModelPage()
    qtbot.addWidget(page)
    page.resize(1000, 650)
    page.set_assembly(assembly)
    page.show()
    qtbot.waitUntil(lambda: page.mesh_builds == 1)
    return page


def test_optical_filter_is_display_only_and_reuses_validated_meshes(qtbot, monkeypatch):
    assembly = scene()
    from temsim.assembly_model_3d import assembly_model_fingerprint
    fingerprint = assembly_model_fingerprint(assembly)
    page = show(qtbot, assembly)
    original = {mesh.key: mesh for mesh in page.view._meshes}
    assert set(original) == {"coil", "housing", "@liner:test"}
    page.view._zoom = 2.3
    page.view._pan[:] = (11., -9.)
    camera = page.view.camera_rotation.copy(), page.view._center.copy()
    monkeypatch.setattr(renderer, "_mesh", lambda *a: pytest.fail("Display filter rebuilt mesh snapshots"))
    monkeypatch.setattr(gui, "assembly_model_from_assembly", lambda *a, **k: pytest.fail("Filter rebuilt geometry"))
    page.optical_only.setChecked(True)
    assert [mesh.key for mesh in page.view._meshes] == ["coil"]
    assert page.view._meshes[0] is original["coil"]
    assert page._tree_items["housing"].isHidden()
    assert page._tree_items["@liner:test"].isHidden()
    assert not page._tree_items["coil"].isHidden()
    assert "Optical parts only" in page.status.text()
    page.optical_only.setChecked(False)
    assert all(mesh is original[mesh.key] for mesh in page.view._meshes)
    assert not page._tree_items["housing"].isHidden()
    assert page.mesh_builds == 1
    assert page.view.zoom_factor == 2.3
    np.testing.assert_array_equal(page.view._pan, (11., -9.))
    np.testing.assert_array_equal(page.view.camera_rotation, camera[0])
    np.testing.assert_array_equal(page.view._center, camera[1])
    assert assembly_model_fingerprint(assembly) == fingerprint


def test_filter_preserves_individual_visibility_and_section_selection(qtbot):
    page = show(qtbot, scene())
    page._tree_items["housing"].setCheckState(0, Qt.CheckState.Unchecked)
    page.focus_component("coil")
    page.section.setChecked(True)
    page.optical_only.setChecked(True)
    assert page.fit_current_selection()
    assert page.view.selection[0] == "coil"
    assert all((triangles[..., 1] <= 1e-12).all() for triangles in page.view._triangles)
    page._tree_items["coil"].setCheckState(0, Qt.CheckState.Unchecked)
    assert not page.view._meshes
    assert "0 visible parts" in page.status.text()
    page.optical_only.setChecked(False)
    assert {mesh.key for mesh in page.view._meshes} == {"@liner:test"}
    assert page._hidden_keys == {"coil", "housing"}


def test_geometry_edit_replaces_pool_while_optical_filter_stays_active(qtbot):
    page = show(qtbot, scene())
    page.optical_only.setChecked(True)
    old = page.view._meshes[0]
    page.set_assembly(scene(shift=150.))
    qtbot.waitUntil(lambda: page.mesh_builds == 2)
    assert page.optical_only.isChecked()
    assert [mesh.key for mesh in page.view._meshes] == ["coil"]
    assert page.view._meshes[0] is not old
    np.testing.assert_array_equal(page.view._meshes[0].vertices, old.vertices + (0., 0., 50.))


def test_combined_cell_and_optical_filters_keep_existing_material_context(qtbot):
    from temsim.cell_geometry import CellPhysicalContext
    from temsim.optics.column import default_state
    # Applied membranes are a separate retained presentation context, not a
    # reason to change the source assembly or rebuild column hardware.
    state = default_state()
    state.vacuum_map.cell.inserted = True
    state.vacuum_map.cell.diameter_mm = .002
    state.vacuum_map.cell.length_mm = 100e-6
    state.sample.thickness_nm = 10
    state.vacuum_map.cell.upstream_window.thickness_nm = 8
    state.vacuum_map.cell.downstream_window.thickness_nm = .3354
    page = show(qtbot, scene())
    page.set_cell_context(CellPhysicalContext.capture(state))
    qtbot.waitUntil(lambda: page.mesh_builds == 2)
    page.cell_only.setChecked(True)
    page.optical_only.setChecked(True)
    assert {mesh.key for mesh in page.view._meshes} >= {"cell_window_upstream", "cell_window_downstream"}
    assert not any(mesh.key in ("coil", "housing", "@liner:test") for mesh in page.view._meshes)
