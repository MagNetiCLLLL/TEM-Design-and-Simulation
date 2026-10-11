"""Real editor controls preserve native/STEP/physical-placement contracts."""
from copy import deepcopy

import numpy as np
import pytest
import tomli_w
from PySide6.QtCore import Qt

from temsim.gui.cad_export import draft_catalog
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.shared_tip import copy_catalog_tree


@pytest.fixture
def catalog_root(tmp_path):
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return root


def _page(qtbot, path, key):
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    assert page.open_path(path, selected_key=key), page.status.text()
    return page


def _edit(page, key, field, value):
    path = ("parts", key, field)
    for row in range(page.dimensions.rowCount()):
        item = page.dimensions.item(row, 1)
        if tuple(item.data(Qt.ItemDataRole.UserRole)) == path:
            assert item.flags() & Qt.ItemFlag.ItemIsEditable
            item.setText(str(value))
            return
    raise AssertionError(f"Missing editable dimension {path}")


def _vertices(page, key):
    return np.concatenate([row["vertices"] for row in page._mesh_records if row["key"] == key])


def _working_radius(page, key):
    vertices = []
    for row in page._mesh_records:
        if row["key"] == key:
            faces = row["faces"][row["face_groups"] == "working_opening"]
            vertices.extend(row["vertices"][np.unique(faces)].tolist())
    assert vertices, "The selected working opening needs an actual inner surface"
    return np.linalg.norm(np.asarray(vertices)[:, :2], axis=1)


def _reject_destination(*_args, **_kwargs):
    pytest.fail("An invalid or unsupported draft must fail before choosing an export destination")


def test_selected_aperture_is_identical_in_part_module_and_step_draft(qtbot, tmp_path):
    key = "condenser_aperture_2"
    source = PartModelDocument(INSTRUMENT_CONFIG_ROOT / "column/C2.toml")
    path = tmp_path / "aperture.toml"
    source.save_copy(path)
    document = deepcopy(source.document)
    part = next(p for p in document["parts"] if p["key"] == key)
    part["aperture_hole_diameters_mm"] = [.05, .3]
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    page = _page(qtbot, path, key)
    original = path.read_bytes()
    page.aperture.setCurrentIndex(1)
    np.testing.assert_allclose(_working_radius(page, key), .15, atol=1e-7)
    page.scope.setCurrentIndex(page.scope.findData("module"))
    np.testing.assert_allclose(_working_radius(page, key), .15, atol=1e-7)
    catalog = draft_catalog(page.session.document, source=path, aperture_indices={key: page._aperture_index})
    solid = next(s for s in catalog["modules"][0]["solids"] if s["key"] == key)
    assert solid["parameters"]["working_diameter_mm"] == .3
    page.aperture.setCurrentIndex(0)
    np.testing.assert_allclose(_working_radius(page, key), .025, atol=1e-7)
    assert path.read_bytes() == original
    assert not page.session.dirty


def test_invalid_shared_mechanical_draft_is_rejected_before_step_export(qtbot, catalog_root, monkeypatch):
    from temsim.gui import part_model_editor
    key = "sample_holder"
    path = catalog_root / "column/C2.toml"
    page = _page(qtbot, path, key)
    files = {p: p.read_bytes() for p in catalog_root.parent.rglob("*.toml")}
    original = deepcopy(page.session.document)
    # Equal shaft/guide diameters have no insertion clearance. The edit remains
    # an undoable draft, but cannot be saved, assembled or exported.
    _edit(page, key, "holder_shaft_diameter_mm", 9.)
    assert page.session.dirty
    monkeypatch.setattr(part_model_editor.QFileDialog, "getExistingDirectory", _reject_destination)
    qtbot.mouseClick(page.export_step_button, Qt.MouseButton.LeftButton)
    assert page._cad_export_dialog is None
    assert "shaft" in page.status.text().lower() or "clearance" in page.status.text().lower()
    assert all(p.read_bytes() == value for p, value in files.items())
    qtbot.mouseClick(page.undo_button, Qt.MouseButton.LeftButton)
    assert page.session.document == original
    page.session.validate()


def test_native_shape_edit_saves_undoes_and_prepares_same_step_geometry(qtbot, catalog_root, monkeypatch):
    from temsim.gui import part_model_editor
    key = "condenser_lens_1_excitation_coil"
    path = catalog_root / "column/C2.toml"
    page = _page(qtbot, path, key)
    original = _vertices(page, key).copy()
    page.base_shape.setCurrentIndex(page.base_shape.findData("box"))
    qtbot.mouseClick(page.apply_shape_button, Qt.MouseButton.LeftButton)
    assert page.session.part(key)["model_3d"]["base"]["kind"] == "box"
    changed = _vertices(page, key).copy()
    assert original.shape != changed.shape or not np.array_equal(original, changed)
    destinations = []
    monkeypatch.setattr(part_model_editor.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: destinations.append(True) or "")
    qtbot.mouseClick(page.export_step_button, Qt.MouseButton.LeftButton)
    assert destinations == [True]
    assert page._cad_export_dialog is None
    catalog = draft_catalog(page.session.document, source=path)
    shape = next(s["shape"] for s in catalog["modules"][0]["solids"] if s["key"] == key)
    assert shape["kind"] == "affine" and shape["shape"]["kind"] == "box"
    qtbot.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert not page.session.dirty, page.status.text()
    assert PartModelDocument(path).part(key)["model_3d"]["base"]["kind"] == "box"
    qtbot.mouseClick(page.undo_button, Qt.MouseButton.LeftButton)
    np.testing.assert_allclose(_vertices(page, key), original)
    qtbot.mouseClick(page.redo_button, Qt.MouseButton.LeftButton)
    np.testing.assert_allclose(_vertices(page, key), changed)


def test_pose_controls_save_shared_offsets_and_optical_tilt_without_rotating_cad(qtbot, catalog_root):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm, lens_pose_registration
    from temsim.optics.column import default_state
    from temsim.cad_catalog import posed_solids
    key, child = "condenser_lens_3", "condenser_lens_3_excitation_coil"
    path = catalog_root / "column/C3_ProbeCorrector.toml"
    catalog = AssemblyCatalog(catalog_root)
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    sample_z = next(p.center_z_mm for p in state._resolved_assembly.parts if p.key == "sample")
    page = _page(qtbot, path, child)
    original = _vertices(page, child).copy()
    _edit(page, key, "offset_x_mm", .025)
    _edit(page, key, "rotation_y_mrad", 2.5)
    rows = {p["key"]: p for p in page.session.document["parts"]}
    mechanical_rotation, mechanical_offset = mechanical_part_transform_mm(rows[child], rows)
    np.testing.assert_array_equal(mechanical_rotation, np.eye(3))
    np.testing.assert_allclose(mechanical_offset, (.025, 0., 0.), atol=1e-14)
    np.testing.assert_allclose(_vertices(page, child), original + mechanical_offset, atol=1e-10)
    # STEP and GUI bodies remain unrotated; the saved error angle is consumed
    # separately by the solver rather than redefining the material envelope.
    solids = posed_solids(rows[child], rows)
    assert solids
    for solid in solids:
        np.testing.assert_allclose(solid.shape["rotation"], mechanical_rotation)
        np.testing.assert_allclose(solid.shape["offset_mm"], mechanical_offset)
    qtbot.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert not page.session.dirty, page.status.text()
    saved = PartModelDocument(path).part(key)
    assert saved["offset_x_mm"] == .025 and saved["rotation_y_mrad"] == 2.5
    catalog.apply(state, catalog.default_selection(), preserve_operating_parameters=True)
    part = next(p for p in state._resolved_assembly.parts if p.key == key)
    registration = lens_pose_registration(state, key)
    rotation, offset = effective_part_transform_mm(rows[key], rows)
    assert not np.allclose(rotation, mechanical_rotation)
    shift = np.array((0., 0., part.center_z_mm - saved["local_center_z_mm"]))
    np.testing.assert_allclose(registration.rotation_array, rotation)
    np.testing.assert_allclose(registration.origin_global_m,
                               (shift - rotation @ shift + offset) * 1e-3, atol=1e-14)
    assert next(p.center_z_mm for p in state._resolved_assembly.parts if p.key == "sample") == sample_z


@pytest.mark.parametrize("module,key,field,value,region", [
    ("column/C2.toml", "eds_detector_system", "mechanical_housing_wall_mm", .2, "segment_1_sensor"),
    ("project_and_recording_system/EnergyFilter.toml", "energy_filter_zebra",
     "mechanical_housing_depth_mm", 7., "housing"),
])
def test_detector_material_controls_save_without_changing_acceptance_or_readout(
        qtbot, catalog_root, module, key, field, value, region):
    from dataclasses import replace
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cad_detectors import EDS_FIELDS, EELS_FIELDS, detector_solids
    from temsim.detector.eds_geometry import EDSDetectorArrayGeometry, resolve_eds_detector_part_data
    from temsim.energy_filter_model_3d import energy_filter_frame_mm, energy_filter_meshes
    from temsim.optics.column import default_state
    path = catalog_root / module
    page = _page(qtbot, path, key)
    before = deepcopy(page.session.part(key))
    fields = EDS_FIELDS if key == "eds_detector_system" else EELS_FIELDS
    rows = {p["key"]: p for p in page.session.document["parts"]}
    expected = detector_solids(before, rows)
    assert expected
    for name in fields:
        entries = [page.dimensions.item(i, 1) for i in range(page.dimensions.rowCount())]
        item = next(i for i in entries if tuple(i.data(Qt.ItemDataRole.UserRole)) == ("parts", key, name))
        assert item.flags() & Qt.ItemFlag.ItemIsEditable
        assert "does not redefine" in item.toolTip()
    if key == "eds_detector_system":
        def angular(part):
            data = part if "eds_system_key" in part else resolve_eds_detector_part_data(part)
            return EDSDetectorArrayGeometry.from_part_data(data)
        acceptance = angular(before)
        sensor = next(r for r in page._mesh_records if r["region"] == "segment_1_sensor")
        assert sensor["surfaces"]
        for surface in sensor["surfaces"].values():
            assert surface["kind"] == "cad_material"
            assert ("parts", key, "mechanical_sensor_thickness_mm") in surface["parameter_paths"]
        triangles = sensor["vertices"][sensor["faces"]]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for i, group in enumerate(sensor["face_groups"]):
            np.testing.assert_allclose(sensor["surfaces"][group]["normal"], normals[i], atol=1e-9)
    else:
        guides, _ = energy_filter_meshes(before, rows, 16)
        alignment = next(g for g in guides if g.region == "alignment_active_area")
        visible = next(r for r in page._mesh_records if r["region"] == "alignment_active_area")
        assert visible["wireframe"]
        np.testing.assert_allclose(visible["vertices"], alignment.vertices)
        catalog = AssemblyCatalog(catalog_root)
        selection = replace(catalog.default_selection(), recording="Energy Filter")
        state = default_state()
        catalog.apply(state, selection)
        readout = state.energy_filter.zebra_detector_d_mm
        coordinates = np.linspace(-.02, .02, 41)
        xx, yy = np.meshgrid(coordinates, coordinates)
        acceptance = state.energy_filter.zebra_detector.recording_mask(xx, yy)
        assert acceptance.any() and not acceptance.all()
    page.select_part(key, region=region)
    assert not page.assign_button.isEnabled()
    assert "Read-only reconstruction" in page.material_current.text()
    _edit(page, key, field, value)
    rows = {p["key"]: p for p in page.session.document["parts"]}
    actual = detector_solids(rows[key], rows)
    assert [s.region for s in actual] == [s.region for s in expected]
    assert all(s.parameters[field] == value for s in actual)
    if key == "eds_detector_system":
        assert angular(rows[key]) == acceptance
    else:
        after_guides, _ = energy_filter_meshes(rows[key], rows, 16)
        for old, new in zip(guides, after_guides, strict=True):
            np.testing.assert_array_equal(old.vertices, new.vertices)
    qtbot.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert not page.session.dirty, page.status.text()
    saved = PartModelDocument(path).part(key)
    assert saved[field] == value
    assert {k: v for k, v in saved.items() if k != field} == {k: v for k, v in before.items() if k != field}
    if key == "energy_filter_zebra":
        catalog.apply(state, selection, preserve_operating_parameters=True)
        assert state.energy_filter.zebra_detector_d_mm == readout
        np.testing.assert_array_equal(state.energy_filter.zebra_detector.recording_mask(xx, yy), acceptance)
        # A real axial path edit moves the material/guide and the recording
        # plane together, unlike the preceding material-depth-only edit.
        old_vertices = _vertices(page, key).copy()
        _, tangent_frame = energy_filter_frame_mm(saved, rows)
        _edit(page, key, "path_center_mm", saved["path_center_mm"] + 1.)
        np.testing.assert_allclose(_vertices(page, key) - old_vertices,
                                   np.broadcast_to(tangent_frame[:, 2], old_vertices.shape), atol=1e-10)
        qtbot.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
        assert not page.session.dirty, page.status.text()
        catalog.apply(state, selection, preserve_operating_parameters=True)
        assert state.energy_filter.zebra_detector_d_mm == pytest.approx(readout + 1.)
        np.testing.assert_array_equal(state.energy_filter.zebra_detector.recording_mask(xx, yy), acceptance)
