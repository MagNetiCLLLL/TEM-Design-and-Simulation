"""Opt-in DPH2 laminate edge seals preserve copper and optical ownership."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import part_solids, posed_solids
from temsim.cad_geometry import annulus, transformed
from temsim.cad_mesh import _manifold
from temsim.corrector_model_3d import deflector_coil_meshes
from temsim.deflector_support_geometry import (
    END_COVER_FIELD, END_METADATA_FIELDS, END_REGIONS, END_STATUS,
    deflector_support_dimensions, full_material_axial_extent,
    validate_support_installations,
)
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector.toml"
KEY = "probe_dph2_coils"
CHANNEL = "probe_dph2_deflector"
HOST = "probe_hp2_hexapole"
END_FIELDS = (END_COVER_FIELD, *END_METADATA_FIELDS)


def rows(document):
    return {row["key"]: row for row in document["parts"]}


@pytest.fixture
def document():
    return read_document(INSTRUMENT_CONFIG_ROOT/MODULE)


def without_ends(part):
    return {key: value for key, value in part.items() if key not in END_FIELDS}


def test_end_covers_are_explicit_family_inference_and_leave_all_old_eleven_regions_unchanged(document):
    by_key = rows(document)
    row = by_key[KEY]
    before = deepcopy(row)
    old = {spec.region: spec for spec in part_solids(without_ends(row), by_key)}
    new = {spec.region: spec for spec in part_solids(row, by_key)}
    assert len(old) == 11 and len(new) == 13
    assert set(new)-set(old) == set(END_REGIONS)
    for region, spec in old.items():
        assert new[region].shape == spec.shape and new[region].material == spec.material
    assert row == before
    assert row["local_start_z_mm"] == 687.25 and row["local_end_z_mm"] == 701.75
    assert full_material_axial_extent(row) == pytest.approx((687.21, 701.79))
    assert full_material_axial_extent(without_ends(row)) == pytest.approx((687.25, 701.75))
    for region in END_REGIONS:
        spec = new[region]
        assert spec.material == "insulator" and spec.status == END_STATUS
        assert "not a documented DPH2" in spec.note and spec.evidence
        assert spec.parameters["optical_field_unchanged"]
    assert by_key[CHANNEL]["effective_thickness_mm"] == 4.
    assert by_key[CHANNEL]["physical_host_key"] == KEY
    # No automatic rollout to the other four inserts.
    for key in ("probe_dph1_coils", "probe_dp22_coils", "probe_dp21_coils", "probe_dp11_coils"):
        assert all(field not in by_key[key] for field in END_FIELDS)
        assert len(part_solids(by_key[key], by_key)) == 11


def test_caps_touch_copper_ends_without_overlap_and_preserve_open_bore(document):
    by_key = rows(document)
    part = by_key[KEY]
    specs = {spec.region: spec for spec in part_solids(part, by_key)}
    d = deflector_support_dimensions(part)
    bore = _manifold(annulus(d["material_start_mm"]-.01, d["material_end_mm"]+.01,
        0., d["material_inner_radius_mm"]-.001), 144)
    for region in END_REGIONS:
        spec = specs[region]
        cap = _manifold(spec.shape, 144)
        expected_volume = np.pi*(d["material_outer_radius_mm"]**2-d["material_inner_radius_mm"]**2)*.04
        assert cap.volume() == pytest.approx(expected_volume, rel=.0004)
        assert len(cap.decompose()) == 1 and cap.volume() > 0.
        assert (cap ^ bore).volume() < 1.e-9
        for other_region, other in specs.items():
            if other_region == region:
                continue
            material = _manifold(other.shape, 144)
            assert (cap ^ material).volume() < 1.e-9
            if "winding" in other_region:
                assert cap.min_gap(material, 1.) == pytest.approx(0., abs=1.e-9)
    old_meshes, _ = deflector_coil_meshes(without_ends(part))
    new_meshes, notes = deflector_coil_meshes(part)
    by_region = {mesh.region: mesh for mesh in new_meshes}
    assert len(new_meshes) == 13
    for old in old_meshes:
        np.testing.assert_array_equal(old.vertices, by_region[old.region].vertices)
        np.testing.assert_array_equal(old.faces, by_region[old.region].faces)
    assert any("687.21" in note and "701.79" in note for note in notes)


@pytest.mark.parametrize("change", [
    {END_COVER_FIELD: False}, {END_COVER_FIELD: None}, {END_COVER_FIELD: 0.},
    {END_COVER_FIELD: -0.04}, {END_COVER_FIELD: float("nan")},
    {END_COVER_FIELD: float("inf")}, {END_COVER_FIELD: 10**1000},
    {END_METADATA_FIELDS[0]: "OEM"}, {END_METADATA_FIELDS[1]: "other"},
    {END_METADATA_FIELDS[2]: ""}, {"winding_support_end_cvoer_thickness_mm": .04},
    {"layout_role": "control_channel"},
])
def test_bad_end_metadata_and_nonfinite_dimensions_fail_before_geometry(document, change):
    by_key = rows(document)
    by_key[KEY].update(change)
    for validate in (deflector_support_dimensions, full_material_axial_extent):
        with pytest.raises(ValueError):
            validate(by_key[KEY])
    with pytest.raises(ValueError):
        validate_document(document)


def test_axial_caps_must_fit_real_hp_host_even_when_copper_allocation_still_fits(document):
    by_key = rows(document)
    by_key[KEY][END_COVER_FIELD] = 1.86
    assert by_key[HOST]["local_start_z_mm"] < by_key[KEY]["local_start_z_mm"]
    assert by_key[KEY]["local_end_z_mm"] < by_key[HOST]["local_end_z_mm"]
    with pytest.raises(ValueError, match="full axial winding support"):
        validate_support_installations(document["parts"])
    with pytest.raises(ValueError, match="full axial winding support"):
        validate_document(document)


def test_full_material_extent_rejects_a_neighbour_outside_the_copper_allocation(document):
    from temsim.module_manifest import _validate_column_mechanical_overlaps

    row = deepcopy(rows(document)[KEY])
    row.pop("parent_key", None)
    obstacle = {"key": "edge_obstacle", "local_start_z_mm": 701.76,
                "local_end_z_mm": 701.78, "length_mm": .02,
                "mechanical_inner_diameter_mm": 22.2,
                "mechanical_outer_diameter_mm": 23.8}
    _validate_column_mechanical_overlaps([without_ends(row), obstacle])
    with pytest.raises(ValueError, match="Undeclared mechanical overlap"):
        _validate_column_mechanical_overlaps([row, obstacle])


def test_independent_copy_replacement_owns_geometry_and_restore_revalidates_atomically(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    root = tmp_path/"instruments"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (MODULE,))
    draft = PartModelDocument(root/MODULE)
    key = draft.copy_component_from(draft.document, KEY, "copied_dph2", 2500., include_children=False)
    assert len(part_model_from_document(draft.document, key, include_children=False).meshes) == 13
    draft.update_model_3d(key, {"base": {"kind": "box", "width_mm": 10.,
                                      "height_mm": 10., "length_mm": 1.}})
    validate_document(draft.document)
    model = part_model_from_document(draft.document, key, include_children=False)
    assert len(model.meshes) == 1
    np.testing.assert_allclose(np.ptp(model.meshes[0].vertices, axis=0), [10., 10., 1.])
    before = deepcopy(draft.document)
    history_index = draft._history_index
    native_configuration = deepcopy(draft.part(key)["model_3d"])
    native_configuration["base"] = {"kind": "existing"}
    with pytest.raises(ValueError, match="winding band width"):
        draft.set_model_3d(key, native_configuration)
    assert draft.document == before and draft._history_index == history_index
    draft.undo()
    assert len(part_model_from_document(draft.document, key, include_children=False).meshes) == 13
    draft.redo()
    assert draft.document == before
    target = tmp_path/"replacement-dph2.toml"
    draft.save_copy(target)
    restored = PartModelDocument(target)
    assert len(part_model_from_document(restored.document, key, include_children=False).meshes) == 1


def test_end_cover_edit_is_atomic_undoable_persisted_and_has_native_face_provenance(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    root = tmp_path/"instruments"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (MODULE,))
    draft = PartModelDocument(root/MODULE)
    before = deepcopy(draft.document)
    specs = {spec.path[-1]: spec for spec in part_dimension_specs(draft.document, KEY)}
    assert specs[END_COVER_FIELD].editable and specs[END_COVER_FIELD].unit == "mm"
    assert specs[END_COVER_FIELD].meaning.category == "physical"
    assert specs[END_COVER_FIELD].meaning.source_kind == "estimated"
    for invalid in (1.86, False, 10**1000):
        with pytest.raises(ValueError):
            draft.set_dimension(("parts", KEY, END_COVER_FIELD), invalid)
        assert draft.document == before and not draft.dirty and not draft.can_undo
    draft.set_dimension(("parts", KEY, END_COVER_FIELD), .05)
    model = part_model_from_document(draft.document, KEY, include_children=False)
    assert len(model.meshes) == 13
    for mesh in model.meshes:
        if mesh.region in END_REGIONS:
            paths = {tuple(path) for metadata in mesh.surfaces.values() for path in metadata["parameter_paths"]}
            assert ("parts", KEY, END_COVER_FIELD) in paths
            assert not mesh.is_exact
    assert draft.part(CHANNEL) == rows(before)[CHANNEL]
    assert draft.part(HOST) == rows(before)[HOST]
    draft.undo()
    assert draft.document == before
    draft.redo()
    destination = tmp_path/"independent-dph2.toml"
    draft.save_copy(destination)
    loaded = PartModelDocument(destination)
    assert full_material_axial_extent(loaded.part(KEY)) == pytest.approx((687.20, 701.80))
    assert loaded.part(CHANNEL) == rows(before)[CHANNEL]


def test_resolved_material_clears_true_tube_hp2_and_tl22_and_is_not_a_duplicate_channel():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cad_stage_holder import physical_liner_rows
    from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
    from temsim.optics.column import default_state

    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    by_key = {part.key: part.data for part in assembly.parts}

    def material(part):
        dz = part.center_z_mm-part.data["local_center_z_mm"]
        return [_manifold(transformed(spec.shape, offset=(0., 0., dz)), 144)
                for spec in posed_solids(part.data, by_key)]

    part = assembly.part(KEY)
    before = deepcopy(dict(part.data))
    insert = material(part)
    tube_row = next(p for p in physical_liner_rows(assembly) if p["key"] == CONTINUOUS_TUBE_KEY)
    tube, = tube_solids(tube_row)
    tube_body = _manifold(tube.shape, 144)
    hp = material(assembly.part(HOST))
    tl = [solid for p in assembly.parts if p.data.get("parent_key") == "probe_tl22_lens"
          for solid in material(p)]
    for neighbours, expected in (([tube_body], 1.46), (hp, .06), (tl, 1.81)):
        assert min(a.min_gap(b, 10.) for a in insert for b in neighbours) == pytest.approx(expected, abs=.004)
        assert max((a ^ b).volume() for a in insert for b in neighbours) < 1.e-8
    assert len(insert) == 13
    assert not part_solids(by_key[CHANNEL], by_key)
    assert dict(part.data) == before
