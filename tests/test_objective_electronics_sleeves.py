"""Passive polymer sleeves cannot become optics, bores or magnetic material."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_catalog import native_reconstruction, part_solids
from temsim.cad_mesh import meshes_from_solids
from temsim.cad_objective_sleeves import (
    COMPONENTS, MATERIAL, PROFILE, validate_nominal_sleeve_clearance, validate_objective_sleeves,
)
from temsim.lens_pose import mechanical_part_transform_mm
from temsim.module_manifest import read_document, validate_document


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = sorted((ROOT/"configs/instruments/column").glob("*.toml"))


def sleeve_rows(document):
    return [row for row in document["parts"] if row.get("mechanical_profile") == PROFILE]


@pytest.mark.parametrize("path", COLUMNS, ids=lambda p: p.stem)
def test_all_column_sleeves_cover_package_clear_bores_and_stop_outside_poles(path):
    doc = read_document(path)
    before = deepcopy(doc)
    validate_document(doc)
    rows = {row["key"]: row for row in doc["parts"]}
    assert len(sleeve_rows(doc)) == 2
    for row in sleeve_rows(doc):
        validate_nominal_sleeve_clearance(row, rows)
        side = row["sleeve_side"]
        solid, = part_solids(row, rows)
        assert (solid,) == native_reconstruction(row, rows)
        assert solid.material == MATERIAL and solid.status == "engineering_reconstruction"
        assert solid.parameters["wall_thickness_mm"] == pytest.approx(.3)
        assert row["mechanical_only"] and row["axial_vacuum_context_only"]
        assert row["parent_key"] == "objective_lens"
        assert "optical_reference_local_z_mm" not in row
        for key in COMPONENTS[side]:
            assert row["local_start_z_mm"] <= rows[key]["local_start_z_mm"]
            assert row["local_end_z_mm"] >= rows[key]["local_end_z_mm"]
            assert row["mechanical_inner_diameter_mm"] > rows[key]["mechanical_outer_diameter_mm"]
        assert row["mechanical_outer_diameter_mm"] < rows["objective_lens_excitation_coil"]["mechanical_inner_diameter_mm"]
        if side == "upper":
            assert row["local_end_z_mm"] < rows["objective_upper_pole"]["local_start_z_mm"]
        else:
            assert row["local_start_z_mm"] > rows["objective_lower_pole"]["local_end_z_mm"]
    assert doc == before
    legacy = deepcopy(doc)
    legacy["parts"] = [r for r in legacy["parts"] if r.get("mechanical_profile") != PROFILE]
    validate_document(legacy)  # Optional in earlier instrument files.


def test_native_mesh_is_opaque_open_annulus_and_excluded_from_optical_only_view():
    from temsim.assembly_optical_visibility import optical_mesh_visible
    from temsim.cad_mesh import _manifold
    from temsim.cad_geometry import annulus
    from temsim.cad_catalog import is_reconstruction_region
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    rows = {r["key"]: r for r in doc["parts"]}
    for row in sleeve_rows(doc):
        assert is_reconstruction_region(row, "body")
        specs = part_solids(row, rows)
        mesh, = meshes_from_solids(specs, 32)
        assert mesh.color[3] == 1.
        assert not optical_mesh_visible(mesh, row, parts_by_key=rows)
        radial = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        assert radial.min() == pytest.approx(29.6)
        assert radial.max() == pytest.approx(29.9)
        cavity = _manifold(annulus(row["local_start_z_mm"]-1, row["local_end_z_mm"]+1, 0., 29.5), 32)
        material = _manifold(specs[0].shape, 32)
        assert material.volume() > 0.
        assert (material ^ cavity).volume() < 1e-10  # No hidden end disks.


@pytest.mark.parametrize("change,reason", [
    ({"mechanical_inner_diameter_mm": 0.}, "open bore"),
    ({"mechanical_outer_diameter_mm": 59.1}, "open bore"),
    ({"mechanical_only": False}, "passive polymer"),
    ({"axial_vacuum_context_only": False}, "passive polymer"),
    ({"field_source_key": "mini_condenser"}, "optical field"),
    ({"material_class": "soft_magnetic"}, "passive polymer"),
    ({"material_regions": {"body": {"material_key": "femm_pure_iron"}}}, "magnetic material assignment"),
])
def test_invalid_carriers_cannot_change_bores_fields_or_material_semantics(change, reason):
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    row = next(r for r in sleeve_rows(doc) if r["sleeve_side"] == "upper")
    row.update(change)
    with pytest.raises(ValueError, match=reason):
        validate_objective_sleeves(doc["parts"])


def test_carrier_navigation_parent_does_not_inherit_objective_optical_placement():
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    rows = {r["key"]: r for r in doc["parts"]}
    rows["objective_lens"].update(offset_x_mm=1., offset_y_mm=2., offset_z_mm=3., rotation_x_mrad=50.)
    for row in sleeve_rows(doc):
        rotation, offset = mechanical_part_transform_mm(row, rows)
        np.testing.assert_array_equal(rotation, np.eye(3))
        np.testing.assert_array_equal(offset, np.zeros(3))


@pytest.mark.parametrize("key", ["ac_deflector", "mini_condenser"])
def test_passive_sleeve_does_not_block_moving_existing_electronics(key):
    from temsim.component_operations import placed_component_document
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    original = next(r for r in doc["parts"] if r["key"] == key)
    # Respect the pre-existing upstream-corrector and objective-pole limits.
    delta = -.5 if key == "ac_deflector" else -200.
    doc, _ = placed_component_document(doc, key, original["local_center_z_mm"]+delta)
    rows = {r["key"]: r for r in doc["parts"]}
    validate_document(doc)
    sleeve = rows["objective_upper_electronics_sleeve"]
    assert part_solids(sleeve, rows)  # The view remains available after the edit.
    with pytest.raises(ValueError, match="must cover"):
        validate_nominal_sleeve_clearance(sleeve, rows)


def test_passive_sleeve_dimensions_are_editable_even_when_nominal_clearance_is_lost():
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    rows = {r["key"]: r for r in doc["parts"]}
    sleeve = rows["objective_upper_electronics_sleeve"]
    sleeve["mechanical_inner_diameter_mm"] = 58.5
    validate_document(doc)
    assert part_solids(sleeve, rows)
    with pytest.raises(ValueError, match="bore must clear"):
        validate_nominal_sleeve_clearance(sleeve, rows)


@pytest.mark.parametrize("source_key", ["objective_upper_electronics_sleeve", "objective_lens"])
def test_copy_operations_keep_sleeves_as_detached_passive_annuli(source_key):
    from temsim.component_operations import copied_component_document
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    before = deepcopy(doc)
    copied, _ = copied_component_document(doc, doc, source_key, "mechanical_test_copy", 4000.)
    validate_document(copied)
    assert doc == before
    rows = {r["key"]: r for r in copied["parts"]}
    copies = [r for r in sleeve_rows(copied) if r.get("mechanical_part_role") == "custom_mechanical_copy"]
    assert len(copies) == (1 if source_key.endswith("sleeve") else 2)
    for row in copies:
        solid, = part_solids(row, rows)
        assert solid.material == MATERIAL
        assert solid.parameters["coverage_status"] == "detached_mechanical_copy"
        assert not solid.parameters["covered_component_keys"]
        assert solid.parameters["wall_thickness_mm"] == pytest.approx(.3)
        assert row["axial_vacuum_context_only"] and row["mechanical_only"]


def test_native_material_guard_preserves_detached_copy_material_editor_contract():
    from temsim.cad_catalog import is_reconstruction_region, module_catalog
    from temsim.component_operations import copied_component_document
    from temsim.part_materials import is_magnetostatic_body, part_material_updates
    doc = read_document(ROOT/"configs/instruments/column/C3_ProbeCorrector.toml")
    native = next(r for r in sleeve_rows(doc) if r["sleeve_side"] == "upper")
    assert is_reconstruction_region(native, "body")
    native["material_regions"] = next(iter(part_material_updates(native, "femm_pure_iron").values()))
    with pytest.raises(ValueError, match="magnetic material assignment"):
        validate_document(doc)
    del native["material_regions"]
    copied, key = copied_component_document(doc, doc, native["key"], "sleeve_material_copy", 4000.)
    row = next(r for r in copied["parts"] if r["key"] == key)
    assert not is_reconstruction_region(row, "body")
    row["material_regions"] = next(iter(part_material_updates(row, "femm_pure_iron").values()))
    validate_document(copied)
    assert not is_magnetostatic_body(row)
    # Export reports the copy's actual independent CAD assignment.
    catalog = module_catalog({**copied, "parts": [row]})
    solid, = catalog["solids"]
    assert solid["material"] == "femm_pure_iron"
    assert solid["parameters"]["coverage_status"] == "detached_mechanical_copy"


def test_sleeves_preserve_vacuum_segments_module_ends_and_objective_navigation(monkeypatch):
    from temsim.assembly_structure import build_assembly_structure
    from temsim.column.layout import LayoutConfiguration
    from temsim.column import module_assembly
    configuration = LayoutConfiguration(energy_filter_selected=False)
    with_sleeves = module_assembly.resolve_module_assembly(configuration)
    structure = build_assembly_structure(with_sleeves)
    for side in COMPONENTS:
        record = structure.by_key[f"objective_{side}_electronics_sleeve"]
        assert record.group_key == "objective"
        assert record.parent_id == structure.by_key["objective_lens"].instance_id
    read = module_assembly._read
    def old_document(path):
        document = deepcopy(read(path))
        if "parts" in document:
            document["parts"] = [r for r in document["parts"] if r.get("mechanical_profile") != PROFILE]
        return document
    monkeypatch.setattr(module_assembly, "_read", old_document)
    without_sleeves = module_assembly.resolve_module_assembly(configuration)
    assert with_sleeves.exit_z_mm == without_sleeves.exit_z_mm
    assert with_sleeves.vacuum_bore_segments == without_sleeves.vacuum_bore_segments
    assert with_sleeves.vacuum_liner_segments == without_sleeves.vacuum_liner_segments
    retained = tuple(part for part in with_sleeves.parts if part.data.get("mechanical_profile") != PROFILE)
    assert retained == without_sleeves.parts


def test_sleeves_create_no_runtime_fields_or_change_current_geometry_and_strengths():
    from temsim.optics.column import default_state
    from temsim.column.module_assembly import _state_targets, _apply_manifest_runtime_geometry
    state = default_state()
    assembly = state._resolved_assembly
    stripped = replace(assembly, parts=tuple(part for part in assembly.parts
                                            if part.data.get("mechanical_profile") != PROFILE))
    def providers():
        result = []
        for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
            result.append(tuple((id(item), item.key, tuple((name, getattr(item, name)) for name in
                ("z_mm", "upper_z_mm", "lower_z_mm", "length_mm", "percent", "enabled", "max_strength_m2",
                 "strength_x_percent", "strength_y_percent", "channel_x_angle_deg", "channel_y_angle_deg")
                if hasattr(item, name))) for item in collection))
        return result
    before = providers()
    assert not any(key.endswith("electronics_sleeve") for key in _state_targets(state))
    _apply_manifest_runtime_geometry(state, {part.key: part for part in stripped.parts}, stripped)
    assert providers() == before
    _apply_manifest_runtime_geometry(state, {part.key: part for part in assembly.parts}, assembly)
    assert providers() == before
