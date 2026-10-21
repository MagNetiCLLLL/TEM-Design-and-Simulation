"""Upper Objective details through actual editors, face controls and map identity."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from temsim.cad_catalog import part_solids
from temsim.cad_objective_upper_hardware import PREFIX
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.parameter_impact import describe_parameter_impact
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector.toml"
COIL = "objective_lens_excitation_coil"
PARENT = "objective_lens"
DETAILS = {PREFIX + "insulation_thickness_mm": .3,
           PREFIX + "inner_insulation_thickness_mm": .3,
           PREFIX + "cooling_plate_thickness_mm": 1.2}
EDITS = {**{("parts", COIL, field): value for field, value in DETAILS.items()},
         ("parts", PARENT, "mechanical_coil_axial_inset_mm"): 6.,
         ("parts", COIL, "mechanical_inner_diameter_mm"): 61.,
         # The existing objective installation contract requires both shank
         # ODs to match the winding ID; do not silently bypass that constraint.
         ("parts", "objective_upper_pole", "pole_stem_outer_diameter_mm"): 61.,
         ("parts", "objective_lower_pole", "pole_stem_outer_diameter_mm"): 61.}


def rows(document):
    return {row["key"]: row for row in document["parts"]}


def sections(document):
    index = rows(document)
    return {spec.region: spec.shape["section_zr_mm"] for spec in part_solids(index[COIL], index)}


@pytest.fixture
def draft(tmp_path):
    from temsim.shared_tip import copy_catalog_tree
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return PartModelDocument(root / MODULE)


@pytest.mark.parametrize("route", ("part_model", "manifest"))
def test_real_editors_roundtrip_detail_parent_and_radial_bindings(draft, route):
    before = deepcopy(draft.document)
    previous = sections(before)
    editor = ManifestEditor(draft.path.parents[1])
    for key in dict.fromkeys(path[1] for path in EDITS):
        native = {spec.path: spec for spec in part_dimension_specs(before, key)}
        manifest = {spec.path: spec for spec in editor.fields(ManifestTarget(MODULE, key))}
        for path in EDITS:
            if path[1] == key:
                assert native[path].editable and manifest[path].editable
                assert native[path].unit == "mm"
    if route == "part_model":
        for path, value in EDITS.items():
            old = sections(draft.document)
            draft.set_dimension(path, value)
            new = sections(draft.document)
            assert (new != old) == (path[1] in (COIL, PARENT))
            if path[2] in DETAILS:
                assert new["lower"] == old["lower"]
        changed = deepcopy(draft.document)
        draft.undo()
        assert draft.document != changed
        draft.redo()
        assert draft.document == changed
        draft.save()
        assert not draft.dirty
    else:
        editor.save(ManifestTarget(MODULE, COIL), EDITS, None)
    loaded = PartModelDocument(draft.path)
    expected = deepcopy(before)
    for (_, key, field), value in EDITS.items():
        rows(expected)[key][field] = value
    # Direct coil ID is linked to its parent's saved radial winding thickness.
    rows(expected)[PARENT]["mechanical_coil_radial_thickness_mm"] = 35.5
    assert loaded.document == expected
    actual = sections(loaded.document)
    assert len(actual) == 7
    np.testing.assert_allclose(actual["upper"],
        [[947.5, 30.8], [947.5, 66.], [1090.5, 66.], [1090.5, 30.8]])
    np.testing.assert_allclose(actual["upper_winding_inner_insulation"],
        [[947.5, 30.5], [947.5, 30.8], [1090.5, 30.8], [1090.5, 30.5]])
    np.testing.assert_allclose(actual["upper_cooling_plate"],
        [[946., 30.8], [946., 66.], [947.2, 66.], [947.2, 30.8]])
    assert min(z for z, _ in actual["lower_cooling_plate"]) == pytest.approx(1090.8)
    assert max(z for z, _ in actual["lower_cooling_plate"]) == pytest.approx(1092.)
    np.testing.assert_allclose(actual["lower"],
        [[1207.9, 30.5], [1207.9, 66.], [1350.9, 66.], [1350.9, 30.5]])
    assert all(actual[region] != previous[region] for region in actual)


def test_unmatched_objective_id_remains_a_draft_and_cannot_be_saved(draft):
    before = deepcopy(draft.document)
    data = draft.path.read_bytes()
    draft.set_dimension(("parts", COIL, "mechanical_inner_diameter_mm"), 61.)
    assert rows(draft.document)[PARENT]["mechanical_coil_radial_thickness_mm"] == 35.5
    assert sections(draft.document) != sections(before)
    with pytest.raises(ValueError, match="mounting-shank OD.*excitation-coil ID"):
        draft.save()
    assert draft.dirty and draft.path.read_bytes() == data
    with pytest.raises(ValueError, match="mounting-shank OD.*excitation-coil ID"):
        ManifestEditor(draft.path.parents[1]).save(ManifestTarget(MODULE, COIL),
            {("parts", COIL, "mechanical_inner_diameter_mm"): 61.}, None)
    assert draft.path.read_bytes() == data
    draft.undo()
    assert draft.document == before and not draft.dirty


def test_native_faces_keep_parent_interval_and_upper_inner_insulation_controls():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)
    meshes = {mesh.region: mesh for mesh in part_model_from_document(
        draft.document, COIL, include_children=False).meshes}
    radial_path = ("parts", COIL, PREFIX + "inner_insulation_thickness_mm")
    for region in ("upper", "lower"):
        for side, field in (("axial_negative", f"{region}_yoke_start_local_z_mm"),
                            ("axial_positive", f"{region}_yoke_end_local_z_mm")):
            paths = set(meshes[region].surfaces[side]["parameter_paths"])
            assert ("parts", PARENT, field) in paths
            assert ("parts", PARENT, "mechanical_coil_axial_inset_mm") in paths
            assert ("parts", COIL, "length_mm") not in paths
        assert (radial_path in meshes[region].surfaces["inner"]["parameter_paths"]) == (region == "upper")
    for region, mesh in meshes.items():
        if region in ("upper", "lower"):
            continue
        for surface in mesh.surfaces.values():
            paths = set(surface["parameter_paths"])
            assert {("parts", COIL, field) for field in DETAILS} <= paths
            assert {("parts", PARENT, field) for field in (
                "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
                "mechanical_coil_axial_inset_mm")} <= paths


def test_detail_dimensions_stale_map_identity_but_keep_homogenized_field_domain():
    from temsim.optics.column import default_state
    from temsim.physics.axisymmetric_magnetostatics import _part_mask
    from temsim.physics.lens_field_provider import lens_geometry_binding

    state = default_state()
    assembly = state._resolved_assembly
    baseline = lens_geometry_binding(state, PARENT)
    provider = next(lens for lens in state.lenses if lens.key == PARENT)
    z_mm = np.linspace(1350., 1850., 251)
    field = provider.magnetic_field_t(z_mm).copy()

    def bound_coil(binding):
        parts = json.loads(binding.canonical_geometry_json)["lens_assembly"]["parts"]
        windings = [part for part in parts if part["data"].get("mechanical_profile") == "magnetic_excitation_coil"]
        assert len(windings) == 1 and windings[0]["key"] == COIL
        return windings[0]

    coil = bound_coil(baseline)
    radial, axial = np.meshgrid(np.array([29.9, 30.05, 30.25, 31., 65.9, 66.1])*1.e-3,
                               np.linspace(1390., 1810., 137)*1.e-3)
    mask = _part_mask(coil, radial, axial)
    for field_name, value in DETAILS.items():
        state._resolved_assembly = replace(assembly, parts=tuple(
            replace(part, data={**part.data, field_name: value}) if part.key == COIL else part
            for part in assembly.parts))
        updated = lens_geometry_binding(state, PARENT)
        assert updated.geometry_fingerprint != baseline.geometry_fingerprint
        updated_coil = bound_coil(updated)
        assert updated_coil["data"]["material_intervals_mm"] == coil["data"]["material_intervals_mm"]
        np.testing.assert_array_equal(_part_mask(updated_coil, radial, axial), mask)
        np.testing.assert_array_equal(provider.magnetic_field_t(z_mm), field)
    source_field = PREFIX + "geometry_source"
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(part, data={**part.data, source_field: part.data[source_field] + " Clarified engineering note."})
        if part.key == COIL else part for part in assembly.parts))
    assert lens_geometry_binding(state, PARENT).geometry_fingerprint == baseline.geometry_fingerprint


@pytest.mark.parametrize("role", ("custom_mechanical_copy", "custom_mechanical"))
def test_detached_winding_faces_use_owned_interval_controls(role):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)
    key = draft.copy_component_from(draft.document, COIL, "detached_winding", 2400., include_children=False)
    draft.part(key)["mechanical_part_role"] = role
    for mesh in part_model_from_document(draft.document, key, include_children=False).meshes:
        paths = {tuple(path) for surface in mesh.surfaces.values() for path in surface["parameter_paths"]}
        index = 1 if mesh.region == "lower" else 0
        assert {("parts", key, "material_intervals_mm", index, side) for side in (0, 1)} <= paths
        assert all(path[1] == key for path in paths if path[0] == "parts")


@pytest.mark.parametrize("field", tuple(DETAILS))
def test_detail_impact_requires_matching_imported_map_without_claiming_extra_fem_region(field):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / MODULE)
    index = rows(draft.document)
    path = ("parts", COIL, field)
    for mode in ("ideal", "linear_geometry", "nonlinear_material", "custom"):
        impact = describe_parameter_impact(index[COIL], path, by_key=index, simulation_mode=mode,
                                           descriptors={PARENT: {"solver": "axisymmetric_linear_fem"}})
        assert impact.status == "inactive" and not impact.active
        assert "field_map_identity" in impact.effects and "magnetic_field" not in impact.effects
        assert "not additional FEM material regions" in impact.detail
    imported = describe_parameter_impact(index[COIL], path, by_key=index, simulation_mode="custom",
        descriptors={PARENT: {"source_path": "not_read_existing_objective.npz",
                              "map_type": "axisymmetric_rz", "geometry_fingerprint": "earlier"}})
    assert imported.status == "configuration_required"
    assert "field_map_identity" in imported.effects and "magnetic_field" not in imported.effects
    assert "Field-map validity" in imported.affected_results
