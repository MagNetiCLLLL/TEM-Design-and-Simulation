"""Gun display/CAD share configured electrodes, without changing field inputs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_auxiliary import (
    GUN_ELECTRODE_KEYS, gun_electrode_solids, validate_gun_electrode_geometry,
)
from temsim.cad_catalog import native_reconstruction, part_solids, posed_solids
from temsim.part_model_3d import part_model_from_document


ROOT = Path(__file__).resolve().parents[1]
ELECTRODES = [
    ("FEG.toml", "feg_extractor"),
    ("FEG.toml", "feg_electrostatic_lens"),
    ("FEG_Mono.toml", "feg_extractor"),
    ("FEG_Mono.toml", "feg_electrostatic_lens"),
    ("Thermionic.toml", "thermionic_wehnelt"),
    ("Thermionic.toml", "thermionic_gun_lens"),
]


def _document(filename):
    return tomllib.loads((ROOT / "configs/instruments/gun" / filename).read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename,key", ELECTRODES)
def test_native_and_cad_electrodes_preserve_configured_closed_annulus(filename, key):
    document = _document(filename)
    original = deepcopy(document)
    parts = {p["key"]: p for p in document["parts"]}
    part = parts[key]
    spec, = part_solids(part, parts)
    assert native_reconstruction(part, parts) == (spec,)
    assert spec.status == "configured_geometry"
    assert spec.material == "electrode"
    assert "OEM nose and support details are not established" in spec.note
    lo, hi = part["local_start_z_mm"], part["local_end_z_mm"]
    inner, outer = part["bore_diameter_mm"]/2, part["outer_diameter_mm"]/2
    np.testing.assert_allclose(spec.shape["section_zr_mm"],
                               [(lo, inner), (lo, outer), (hi, outer), (hi, inner)])
    mesh, = part_model_from_document(document, key, angular_segments=32).meshes
    assert mesh.is_exact and mesh.material_class == "electrode" and not mesh.wireframe
    radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    assert radius.min() == pytest.approx(inner)
    assert radius.max() == pytest.approx(outer)
    assert mesh.vertices[:, 2].min() == pytest.approx(lo)
    assert mesh.vertices[:, 2].max() == pytest.approx(hi)
    # Every mesh edge is shared by two faces: a closed material shell, not
    # an open tube or a disk that obstructs the configured beam bore.
    edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    triangles = mesh.vertices[mesh.faces]
    volume = np.einsum("ij,ij->i", triangles[:, 0],
                       np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
    polygon_area_factor = 32*np.sin(2*np.pi/32)/2
    assert volume == pytest.approx(polygon_area_factor*(outer**2-inner**2)*(hi-lo), rel=1e-12)
    assert document == original


@pytest.mark.parametrize("filename,key", ELECTRODES)
def test_gun_surfaces_expose_actual_electrode_faces_bore_and_dimension_ownership(filename, key):
    document = _document(filename)
    part = next(p for p in document["parts"] if p["key"] == key)
    mesh, = part_model_from_document(document, key, angular_segments=32).meshes
    assert set(mesh.surfaces) == {
        "electrode_upstream", "electrode_downstream", "electrode_bore", "electrode_outer"}
    for group, z, normal in (
        ("electrode_upstream", part["local_start_z_mm"], (0., 0., -1.)),
        ("electrode_downstream", part["local_end_z_mm"], (0., 0., 1.)),
    ):
        points = mesh.vertices[mesh.faces[mesh.face_groups == group]]
        np.testing.assert_allclose(points[:, :, 2], z, atol=1e-12)
        np.testing.assert_allclose(mesh.surfaces[group]["normal"], normal, atol=1e-12)
    assert ("parts", key, "local_start_z_mm") in mesh.surfaces["electrode_upstream"]["parameter_paths"]
    assert ("parts", key, "local_end_z_mm") in mesh.surfaces["electrode_downstream"]["parameter_paths"]
    for group, field in (("electrode_bore", "bore_diameter_mm"),
                         ("electrode_outer", "outer_diameter_mm")):
        surface = mesh.surfaces[group]
        assert surface["parameter_paths"] == (("parts", key, field),)
        assert "normal" not in surface  # a cylinder is not a single planar face
        points = mesh.vertices[mesh.faces[mesh.face_groups == group]]
        np.testing.assert_allclose(np.linalg.norm(points[:, :, :2], axis=2), part[field]/2)
    assert len(mesh.edges) == 4  # bore/outer circle at each axial face


def test_gun_surface_metadata_uses_the_same_radius_priority_as_geometry():
    document = _document("FEG.toml")
    part = next(p for p in document["parts"] if p["key"] == "feg_extractor")
    part.update(mechanical_outer_diameter_mm=26., mechanical_bore_diameter_mm=6.)
    mesh, = part_model_from_document(document, part["key"], angular_segments=32).meshes
    assert np.linalg.norm(mesh.vertices[:, :2], axis=1).min() == pytest.approx(3.)
    assert np.linalg.norm(mesh.vertices[:, :2], axis=1).max() == pytest.approx(13.)
    assert mesh.surfaces["electrode_bore"]["parameter_paths"] == (
        ("parts", part["key"], "mechanical_bore_diameter_mm"),)
    assert mesh.surfaces["electrode_outer"]["parameter_paths"] == (
        ("parts", part["key"], "mechanical_outer_diameter_mm"),)


@pytest.mark.parametrize("gun_name,filename", [("FEG", "FEG.toml"), ("FEG + Mono", "FEG_Mono.toml")])
def test_displayed_electrodes_match_consumed_field_boundaries_without_mutating_runtime(gun_name, filename):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    from temsim.physics.grounded_tip_field import field_request
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun=gun_name),
                  preserve_operating_parameters=True)
    gun = state.electron_gun
    gun.emitter.surface_model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    before = deepcopy(field_request(gun))
    document = _document(filename)
    for key, start_m, end_m, inner_m, outer_m, _ in before["rings"]:
        if key not in GUN_ELECTRODE_KEYS:
            continue
        mesh, = part_model_from_document(document, key, angular_segments=32).meshes
        np.testing.assert_allclose(
            (mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max(),
             np.linalg.norm(mesh.vertices[:, :2], axis=1).min(),
             np.linalg.norm(mesh.vertices[:, :2], axis=1).max()),
            np.array((start_m, end_m, inner_m, outer_m))*1000, rtol=0., atol=1e-11)
    assert field_request(gun) == before


@pytest.mark.parametrize("field,value", [
    ("bore_diameter_mm", 0.), ("bore_diameter_mm", 21.),
    ("outer_diameter_mm", float("nan")), ("outer_diameter_mm", True),
    ("local_end_z_mm", 5.), ("local_end_z_mm", 6.),
])
def test_invalid_gun_electrodes_are_rejected_before_mesh_or_cad(field, value):
    part = next(p for p in _document("FEG.toml")["parts"] if p["key"] == "feg_extractor")
    part[field] = value
    with pytest.raises(ValueError):
        gun_electrode_solids(part)


def test_unknown_gun_internals_do_not_gain_electrodes_or_guessed_poles():
    document = _document("FEG.toml")
    parts = {p["key"]: p for p in document["parts"]}
    part = parts["feg_stigmator"]
    # A legacy gun without the new explicit construction declaration stays
    # unknown; an opt-in engineering adaptation does not rewrite old files.
    for field in tuple(part):
        if field.startswith("gun_stigmator_"):
            del part[field]
    assert gun_electrode_solids(part) is None
    spec, = part_solids(part, parts)
    assert spec.status == "configured_envelope"
    mesh, = part_model_from_document(document, part["key"], angular_segments=32).meshes
    assert not mesh.is_exact


@pytest.mark.parametrize("value", [0., -1., 20., 21., float("nan"), float("inf"), True, 10**1000])
def test_gun_electrode_bad_bore_edit_rejects_before_draft_or_history_changes(value):
    from temsim.part_model_document import PartModelDocument

    draft = PartModelDocument(ROOT/"configs/instruments/gun/FEG.toml")
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    raw, index = draft.path.read_bytes(), draft._history_index
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", "feg_extractor", "bore_diameter_mm"), value)
    assert draft.document == before and draft._history == history
    assert draft._history_index == index and not draft.dirty
    assert draft.path.read_bytes() == raw
    part = deepcopy(draft.part("feg_extractor"))
    part["bore_diameter_mm"] = value
    with pytest.raises(ValueError):
        validate_gun_electrode_geometry(part)


@pytest.mark.parametrize("value", [0., 20., 21.])
def test_gun_electrode_bad_document_cannot_validate_or_be_saved(tmp_path, value):
    from temsim.module_manifest import validate_document
    from temsim.part_model_document import PartModelDocument

    draft = PartModelDocument(ROOT/"configs/instruments/gun/FEG.toml")
    draft.part("feg_extractor")["bore_diameter_mm"] = value
    with pytest.raises(ValueError, match="bore"):
        validate_document(draft.document)
    destination = tmp_path/"invalid-gun.toml"
    with pytest.raises(ValueError, match="bore"):
        draft.save_copy(destination)
    assert not destination.exists()


@pytest.mark.parametrize("filename,key", ELECTRODES)
def test_valid_electrode_edit_changes_shared_geometry_and_survives_save_reopen(tmp_path, filename, key):
    from temsim.part_model_document import PartModelDocument

    draft = PartModelDocument(ROOT/"configs/instruments/gun"/filename)
    original = deepcopy(draft.document)
    raw = draft.path.read_bytes()
    before = gun_electrode_solids(draft.part(key))
    bore = (draft.part(key)["bore_diameter_mm"]+draft.part(key)["outer_diameter_mm"])/2
    draft.set_dimension(("parts", key, "bore_diameter_mm"), bore)
    after = gun_electrode_solids(draft.part(key))
    assert after != before and after[0].parameters["bore_radius_mm"] == bore/2
    expected = deepcopy(original)
    next(row for row in expected["parts"] if row["key"] == key)["bore_diameter_mm"] = bore
    assert draft.document == expected  # No excitation/calibration/other geometry edit.
    mesh, = part_model_from_document(draft.document, key, include_children=False).meshes
    assert np.linalg.norm(mesh.vertices[:, :2], axis=1).min() == pytest.approx(bore/2)
    draft.undo()
    assert draft.document == original
    draft.redo()
    destination = tmp_path/"edited-gun.toml"
    draft.save_copy(destination)
    reopened = PartModelDocument(destination)
    reopened.validate()
    assert reopened.part(key)["bore_diameter_mm"] == bore
    assert gun_electrode_solids(reopened.part(key)) == after
    assert (ROOT/"configs/instruments/gun"/filename).read_bytes() == raw


def test_independent_electrode_replacement_defers_native_validation_until_atomic_restore(tmp_path):
    from temsim.cad_mesh import _manifold
    from temsim.part_model_document import PartModelDocument

    draft = PartModelDocument(ROOT/"configs/instruments/gun/FEG.toml")
    original = deepcopy(draft.part("feg_extractor"))
    key = draft.copy_component_from(draft.document, "feg_extractor", "copied_electrode", 2500.,
                                    include_children=False)
    assert gun_electrode_solids(draft.part(key))[0].material == "electrode"
    draft.update_model_3d(key, {"base": {"kind": "box", "width_mm": 10.,
                                       "height_mm": 10., "length_mm": 1.}})
    draft.set_dimension(("parts", key, "bore_diameter_mm"), 21.)
    assert validate_gun_electrode_geometry(draft.part(key)) is None
    assert gun_electrode_solids(draft.part(key)) is None
    draft.validate()
    mesh, = part_model_from_document(draft.document, key, include_children=False).meshes
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), [10., 10., 1.])
    solid, = posed_solids(draft.part(key), {p["key"]: p for p in draft.document["parts"]})
    assert _manifold(solid.shape, 32).volume() == pytest.approx(100.)
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    index = draft._history_index
    native = deepcopy(draft.part(key)["model_3d"])
    native["base"] = {"kind": "existing"}
    with pytest.raises(ValueError, match="bore"):
        draft.set_model_3d(key, native)
    assert draft.document == before and draft._history == history and draft._history_index == index
    assert draft.part("feg_extractor") == original
    target = tmp_path/"replacement-gun.toml"
    draft.save_copy(target)
    reopened = PartModelDocument(target)
    assert len(part_model_from_document(reopened.document, key, include_children=False).meshes) == 1
    reopened.set_dimension(("parts", key, "bore_diameter_mm"), 6.)
    reopened.set_model_3d(key, native)
    assert gun_electrode_solids(reopened.part(key))[0].parameters["bore_radius_mm"] == 3.


@pytest.mark.parametrize("bore", [0., 20., 21.])
def test_installed_electrode_replacement_still_validates_live_optical_boundaries(bore):
    from temsim.module_manifest import validate_document
    from temsim.part_model_document import PartModelDocument

    draft = PartModelDocument(ROOT/"configs/instruments/gun/FEG.toml")
    key = "feg_extractor"
    draft.update_model_3d(key, {"base": {"kind": "box", "width_mm": 10.,
                                       "height_mm": 10., "length_mm": 1.}})
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    assert gun_electrode_solids(draft.part(key)) is None
    assert validate_gun_electrode_geometry(draft.part(key)) is not None
    with pytest.raises(ValueError, match="bore"):
        draft.set_dimension(("parts", key, "bore_diameter_mm"), bore)
    assert draft.document == before and draft._history == history
    invalid = deepcopy(before)
    next(part for part in invalid["parts"] if part["key"] == key)["bore_diameter_mm"] = bore
    with pytest.raises(ValueError, match="bore"):
        validate_document(invalid)
