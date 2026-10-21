"""Actual objective intervals, insulated copper clearance and detached copies."""

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_catalog import native_reconstruction, part_solids
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_objective_upper_hardware import (
    FIELDS, PREFIX, objective_upper_hardware_solids, validate_objective_upper_hardware,
)
from temsim.cad_round_lenses import round_lens_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument


PATH = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector.toml"
COIL = "objective_lens_excitation_coil"
PARENT = "objective_lens"


@pytest.fixture
def document():
    return read_document(PATH)


@pytest.fixture
def rows(document):
    return {row["key"]: row for row in document["parts"]}


def test_only_selected_profile_opts_in_and_keeps_single_shared_field(rows):
    for path in sorted(PATH.parent.glob("*.toml")):
        doc = read_document(path)
        validate_document(doc)
        selected = {row["key"]: row for row in doc["parts"]}
        assert any(name.startswith(PREFIX) for row in selected.values() for name in row) == (path == PATH)
    legacy = deepcopy(rows)
    for name in FIELDS:
        legacy[COIL].pop(name)
    assert circuit_inventory(list(rows.values())) == circuit_inventory(list(legacy.values()))
    assert rows[COIL]["field_source_key"] == PARENT
    for key in ("objective_upper_pole", "objective_lower_pole", "objective_lens_yoke", "objective_lens_housing"):
        assert part_solids(rows[key], rows) == part_solids(legacy[key], legacy)


@pytest.mark.parametrize("count", (16, 32, 64))
def test_inner_insulation_preserves_pack_envelope_and_separates_copper_from_pole(rows, count):
    original = deepcopy(rows)
    specs = objective_upper_hardware_solids(rows[COIL], rows)
    by_region = {spec.region: _manifold(spec.shape, count) for spec in specs}
    assert len(specs) == 7
    assert all((a ^ b).volume() < 1e-7 for a, b in combinations(by_region.values(), 2))
    upper_pack = _manifold(annulus(946.5, 1091.5, 30., 66.), count)
    new_pack = by_region["upper"] + by_region["upper_winding_inner_insulation"]
    assert (upper_pack - new_pack).volume() < 1e-7
    assert (new_pack - upper_pack).volume() < 1e-7
    assert specs[1].shape == annulus(1206.9, 1351.9, 30., 66.)
    pole = _manifold(part_solids(rows["objective_upper_pole"], rows)[0].shape, count)
    assert all((pole ^ solid).volume() < 1e-7 for solid in by_region.values())
    assert pole.min_gap(by_region["upper"], .3) == pytest.approx(.2 * np.cos(np.pi/count), abs=2e-8)
    assert pole.min_gap(by_region["upper_winding_inner_insulation"], .1) == pytest.approx(0., abs=1e-10)
    for mesh in meshes_from_solids(specs, count):
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        _, multiplicities = np.unique(edges, axis=0, return_counts=True)
        assert np.all(multiplicities == 2)
    assert rows == original


def test_native_cad_and_material_interval_sources_are_identical(document, rows):
    specs = objective_upper_hardware_solids(rows[COIL], rows)
    assert native_reconstruction(rows[COIL], rows) == part_solids(rows[COIL], rows) == specs
    native = part_model_from_document(document, COIL).meshes
    cad = meshes_from_solids(specs)
    assert len(native) == len(cad) == 7
    for a, b in zip(native, cad):
        assert (a.region, a.material_class) == (b.region, b.material_class)
        np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-10)
        np.testing.assert_array_equal(a.faces, b.faces)
    expected = {"upper_cooling_plate": (945.3, 946.3), "upper_end_insulation": (946.3, 946.5),
                "lower_end_insulation": (1091.5, 1091.7), "lower_cooling_plate": (1091.7, 1092.7)}
    for spec in specs:
        if spec.region in expected:
            zs = [point[0] for point in spec.shape["section_zr_mm"]]
            assert (min(zs), max(zs)) == pytest.approx(expected[spec.region])
        assert "homogenized" in spec.note and "not another current source" in spec.note


def test_parent_inset_and_coil_radius_drive_every_upper_layer_without_saved_contacts(rows):
    row = rows[COIL]
    baseline = objective_upper_hardware_solids(row, rows)
    assert not any("contact" in name or "local_z" in name for name in FIELDS)
    # The old coil allocation envelope is not the actual winding geometry.
    row["local_start_z_mm"] += 10.
    row["local_end_z_mm"] -= 10.
    assert [s.shape for s in objective_upper_hardware_solids(row, rows)] == [s.shape for s in baseline]
    rows[PARENT]["mechanical_coil_axial_inset_mm"] += 1.
    row["mechanical_inner_diameter_mm"] += 1.
    row["mechanical_outer_diameter_mm"] -= 1.
    detailed = {s.region: s.shape["section_zr_mm"] for s in objective_upper_hardware_solids(row, rows)}
    assert detailed["upper"] == [[947.5, 30.7], [947.5, 65.5], [1090.5, 65.5], [1090.5, 30.7]]
    assert detailed["upper_winding_inner_insulation"] == [[947.5, 30.5], [947.5, 30.7], [1090.5, 30.7], [1090.5, 30.5]]
    assert min(p[0] for p in detailed["upper_cooling_plate"]) == pytest.approx(946.3)
    assert max(p[0] for p in detailed["lower_cooling_plate"]) == pytest.approx(1091.7)


@pytest.mark.parametrize("suffix,value,match", [
    ("insulation_thickness_mm", 0., "positive"),
    ("insulation_thickness_mm", 1e-30, "representable"),
    ("inner_insulation_thickness_mm", float("nan"), "finite"),
    ("inner_insulation_thickness_mm", True, "finite"),
    ("inner_insulation_thickness_mm", 36., "positive"),
    ("cooling_plate_thickness_mm", 5., "end spaces"),
    ("cooling_plate_thickness_mm", float("inf"), "finite"),
    ("geometry_status", "oem_drawing", "not OEM"),
    ("geometry_source_urls", [], "source_urls"),
    ("insullation_thickness_mm", .2, "unknown"),
])
def test_invalid_detail_does_not_silently_render(rows, suffix, value, match):
    rows[COIL][PREFIX + suffix] = value
    with pytest.raises(ValueError, match=match):
        objective_upper_hardware_solids(rows[COIL], rows)


def test_missing_parent_cannot_fall_back_to_allocation_envelope(rows):
    with pytest.raises(ValueError, match="parent geometry"):
        objective_upper_hardware_solids(rows[COIL], {})
    with pytest.raises(ValueError, match="parent geometry"):
        objective_upper_hardware_solids(rows[COIL])
    rows[PARENT]["upper_yoke_end_local_z_mm"] = True
    with pytest.raises(ValueError, match="finite"):
        objective_upper_hardware_solids(rows[COIL], rows)


def test_detached_copy_keeps_intervals_without_parent_and_no_field_owner():
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, COIL, "copied_upper_winding", 2400., include_children=False)
    row = draft.part(key)
    assert "field_source_key" not in row
    assert row["mechanical_part_role"] == "custom_mechanical_copy"
    before = objective_upper_hardware_solids(row, {})
    assert len(before) == 7
    draft.place_component(key, 2500.)
    after = objective_upper_hardware_solids(draft.part(key), {})
    for old, new in zip(before, after):
        np.testing.assert_allclose(np.array(new.shape["section_zr_mm"]),
                                   np.array(old.shape["section_zr_mm"]) + [100., 0.])
    broken = deepcopy(draft.part(key))
    broken.pop("material_intervals_mm")
    with pytest.raises(ValueError, match="two saved material intervals"):
        objective_upper_hardware_solids(broken, {})


def test_legacy_and_explicit_replacement_keep_precedence(rows):
    row = rows[COIL]
    row["model_3d"] = {"base": {"kind": "box", "width_mm": 10., "height_mm": 10., "length_mm": 1.}}
    row[PREFIX + "inner_insulation_thickness_mm"] = -1.
    assert objective_upper_hardware_solids(row, rows) is None
    validate_objective_upper_hardware(row, rows)
    del row["model_3d"]
    for field in FIELDS:
        row.pop(field)
    assert objective_upper_hardware_solids(row, rows) is None
    assert len(round_lens_solids(row, rows)) == 2


@pytest.mark.parametrize("key,profile", [
    ("objective_upper_pole", [[0., 2.88, 40.], [62., 2.88, 40.]]),
    ("objective_lens_yoke", [[0., 60., 85.], [420.4, 60., 85.]]),
])
def test_neighbor_radial_profiles_cannot_hide_actual_winding_intersection(document, rows, key, profile):
    rows[key]["magnetic_radial_profile_mm"] = profile
    # Independently establish the positive CAD intersection without invoking
    # the refinement validator when constructing the reference winding pack.
    other = _manifold(part_solids(rows[key], rows)[0].shape, 32)
    pack = _manifold(annulus(946.5, 1091.5, 30.2, 66.), 32)
    assert (other ^ pack).volume() > 100.
    with pytest.raises(ValueError, match="lack clearance from " + key):
        validate_document(document)


@pytest.mark.parametrize("key,field,value,profile", [
    ("objective_upper_pole", "pole_stem_outer_diameter_mm", 62., [[0., 2.88, 20.], [62., 2.88, 20.]]),
    ("objective_lens_yoke", "mechanical_inner_diameter_mm", 120., [[0., 70., 85.], [420.4, 70., 85.]]),
])
def test_native_radial_profile_takes_precedence_over_inactive_pole_or_yoke_contour(document, rows, key, field, value, profile):
    rows[key][field] = value
    rows[key]["magnetic_radial_profile_mm"] = profile
    # The new detail validator respects actual CAD precedence. Older global
    # objective sizing contracts independently constrain these legacy scalars.
    validate_objective_upper_hardware(rows[COIL], rows)
    other = _manifold(part_solids(rows[key], rows)[0].shape, 32)
    assert all((other ^ _manifold(spec.shape, 32)).volume() < 1e-7
               for spec in objective_upper_hardware_solids(rows[COIL], rows))


@pytest.mark.parametrize("key,field,value", [
    ("objective_upper_pole", "pole_stem_outer_diameter_mm", 62.),
    ("objective_lens_yoke", "mechanical_inner_diameter_mm", 120.),
])
def test_neighbor_replacement_does_not_validate_inactive_native_contour(document, rows, key, field, value):
    from temsim.cad_catalog import posed_solids

    rows[key]["model_3d"] = {"schema_version": 1,
        "base": {"kind": "box", "width_mm": 2., "height_mm": 2., "length_mm": 1.}}
    rows[key][field] = value
    validate_objective_upper_hardware(rows[COIL], rows)
    replacement, = posed_solids(rows[key], rows)
    box = _manifold(replacement.shape, 32)
    assert all((box ^ _manifold(spec.shape, 32)).volume() < 1e-7
               for spec in objective_upper_hardware_solids(rows[COIL], rows))


def test_neighbor_cut_is_qualified_by_final_cad_not_uncut_section(document, rows):
    import manifold3d as manifold

    pole = rows["objective_upper_pole"]
    pole["magnetic_radial_profile_mm"] = [[0., 2.88, 40.], [62., 2.88, 40.]]
    with pytest.raises(ValueError, match="lack clearance"):
        validate_objective_upper_hardware(rows[COIL], rows)
    pole["model_3d"] = {"schema_version": 1, "base": {"kind": "existing"},
        "features": [{"id": "remove_upper_interference", "kind": "hole", "axis": "z",
                      "center_mm": [0., 0., -24.], "depth_mm": 18., "diameter_mm": 84.}]}
    validate_objective_upper_hardware(rows[COIL], rows)
    mesh, = part_model_from_document(document, "objective_upper_pole").meshes
    solid = manifold.Manifold(manifold.Mesh64(mesh.vertices.copy(), mesh.faces.astype(np.uint64)))
    assert solid.volume() > 1.
    assert all((solid ^ _manifold(spec.shape, 32)).volume() < 1e-7
               for spec in objective_upper_hardware_solids(rows[COIL], rows))


def test_neighbor_transform_defers_to_actual_cad_overlap_instead_of_certifying_clearance(rows):
    from temsim.cad_catalog import posed_solids

    pole = rows["objective_upper_pole"]
    pole["model_3d"] = {"schema_version": 1, "base": {"kind": "existing"},
                        "transform": {"offset_mm": [40., 0., 0.]}}
    validate_objective_upper_hardware(rows[COIL], rows)
    shifted, = posed_solids(pole, rows)
    actual = _manifold(shifted.shape, 32)
    # The fast native check is deliberately not a final mechanical clearance
    # certificate. The existing posed-CAD overlap path sees this moved solid.
    assert any((actual ^ _manifold(spec.shape, 32)).volume() > 1.
               for spec in objective_upper_hardware_solids(rows[COIL], rows))
