"""Adapter Lens material detail retains its native optical and coil geometry."""

from copy import deepcopy
from functools import reduce
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_adapter_lens_hardware import (
    PREFIX, FIELDS, adapter_lens_hardware_solids, validate_adapter_lens_hardware,
)
from temsim.cad_c3_lens_hardware import c3_lens_hardware_solids, validate_c3_lens_hardware
from temsim.cad_catalog import native_reconstruction, part_solids, posed_solids
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_round_lenses import round_lens_solids
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/instruments/column/C3_ProbeCorrector.toml"
HOUSING = "adapter_lens_housing"
COIL = "adapter_lens_excitation_coil"
CHILDREN = (HOUSING, COIL, "adapter_lens_upper_pole", "adapter_lens_lower_pole", "adapter_lens_yoke")
COLUMNS = sorted(PATH.parent.glob("*.toml"))


@pytest.fixture
def document():
    return read_document(PATH)


@pytest.fixture
def rows(document):
    return {row["key"]: row for row in document["parts"]}


@pytest.mark.parametrize("path", COLUMNS, ids=lambda path: path.stem)
def test_only_three_probe_variants_have_adapter_hardware_with_identical_saved_geometry(rows, path):
    doc = read_document(path)
    validate_document(doc)
    variant = {row["key"]: row for row in doc["parts"]}
    if "ProbeCorrector" not in path.stem:
        assert HOUSING not in variant
        assert not any(name.startswith(PREFIX) for row in variant.values() for name in row)
        return
    assert adapter_lens_hardware_solids(variant[HOUSING], variant) == adapter_lens_hardware_solids(rows[HOUSING], rows)
    assert variant["adapter_lens"]["pole_gap_mm"] == 14.
    assert (variant[COIL]["local_start_z_mm"], variant[COIL]["local_end_z_mm"]) == (621.82, 654.58)


@pytest.mark.parametrize("count", (16, 32, 64))
def test_nine_closed_material_regions_preserve_original_five_and_do_not_intersect(rows, count):
    before = deepcopy(rows)
    legacy = deepcopy(rows)
    for name in FIELDS:
        legacy[HOUSING].pop(name)
    specs = adapter_lens_hardware_solids(rows[HOUSING], rows)
    assert [s.region for s in specs] == ["body", "upper_cooling_plate", "upper_coil_insulation",
                                       "lower_coil_insulation", "lower_cooling_plate"]
    assert [s.material for s in specs] == ["nonmagnetic_housing", "copper_cooling_plate", "insulator",
                                         "insulator", "copper_cooling_plate"]
    assert specs[0].shape == annulus(620., 656.4, 87., 90.)
    assert specs[0].shape == round_lens_solids(legacy[HOUSING], legacy)[0].shape
    for key in CHILDREN[1:]:
        assert round_lens_solids(rows[key], rows) == round_lens_solids(legacy[key], legacy)
    all_specs = [s for key in CHILDREN for s in part_solids(rows[key], rows)]
    assert len(all_specs) == 9
    solids = [_manifold(s.shape, count) for s in all_specs]
    assert all((a ^ b).volume() < 1e-9 for a, b in combinations(solids, 2))
    for mesh in meshes_from_solids(all_specs, count):
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        _, multiplicities = np.unique(edges, axis=0, return_counts=True)
        assert np.all(multiplicities == 2)
        assert mesh.vertices[:, 2].min() >= 620. - 1e-10
        assert mesh.vertices[:, 2].max() <= 656.4 + 1e-10
    polygon_area = count * .5 * np.sin(2 * np.pi / count) * ((91.05/2)**2 - (78.33/2)**2)
    assert sum(s.volume() for s in solids[1:5]) == pytest.approx(1.2 * polygon_area, rel=1e-12)
    assert circuit_inventory(list(legacy.values())) == circuit_inventory(list(rows.values()))
    assert rows == before


def test_native_and_cad_use_identical_regions_and_preserve_passive_material_ownership(document, rows):
    specs = adapter_lens_hardware_solids(rows[HOUSING], rows)
    assert native_reconstruction(rows[HOUSING], rows) == specs
    assert part_solids(rows[HOUSING], rows) == specs
    cad = meshes_from_solids(specs, 32)
    native = part_model_from_document(document, HOUSING, angular_segments=32).meshes
    assert len(native) == len(cad) == 5
    for mesh, reference in zip(native, cad):
        assert (mesh.key, mesh.region, mesh.material_class) == (reference.key, reference.region, reference.material_class)
        np.testing.assert_allclose(mesh.vertices, reference.vertices, atol=1e-10)
        np.testing.assert_array_equal(mesh.faces, reference.faces)
    spans = [(min(p[0] for p in s.shape["section_zr_mm"]), max(p[0] for p in s.shape["section_zr_mm"]))
             for s in specs[1:]]
    np.testing.assert_allclose(spans, [(621.22, 621.72), (621.72, 621.82),
                                     (654.58, 654.68), (654.68, 655.18)], atol=1e-10)
    assert all(s.material != "soft_magnetic" and "not another current source" in s.note for s in specs[1:])
    assert all(s.status == "engineering_reconstruction_not_oem" for s in specs[1:])


def test_constant_tube_clearance_converges_to_radial_limit_and_material_is_disjoint(document, rows):
    geometry = document["geometry"]
    tube_spec, = tube_solids({**geometry, "key": CONTINUOUS_TUBE_KEY,
        "mechanical_inner_diameter_mm": geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"]})
    radial = (20. - 19.2) / 2
    assert radial == pytest.approx(.4)
    for key in ("adapter_lens_upper_pole", "adapter_lens_lower_pole"):
        pole_spec, = round_lens_solids(rows[key], rows)
        gaps = []
        for count in (32, 64, 128):
            tube, pole = _manifold(tube_spec.shape, count), _manifold(pole_spec.shape, count)
            assert (tube ^ pole).volume() < 1e-9
            gap = tube.min_gap(pole, .5)
            assert gap == pytest.approx(radial * np.cos(np.pi / count), abs=2e-8)
            gaps.append(gap)
        assert 0 < gaps[0] < gaps[1] < gaps[2] < radial
        assert radial - gaps[-1] < .000121


@pytest.mark.parametrize("neighbor,expected_gap", [
    # DPH2's new 0.04 mm upstream end seal reduces the earlier 30.85 mm gap.
    ("beam_deflector", 10.), ("probe_dph2_coils", 30.81), ("probe_hp2_hexapole", 29.),
])
def test_complete_material_union_keeps_existing_neighbor_clearance(rows, neighbor, expected_gap):
    adapter = reduce(lambda a, b: a + b, (_manifold(s.shape, 32)
        for key in CHILDREN for s in posed_solids(rows[key], rows)))
    other = reduce(lambda a, b: a + b, (_manifold(s.shape, 32)
        for s in posed_solids(rows[neighbor], rows)))
    assert (adapter ^ other).volume() < 1e-9
    assert adapter.min_gap(other, expected_gap + 1.) == pytest.approx(expected_gap, abs=1e-8)


def test_missing_partial_context_and_detached_copy_do_not_consult_default_coil(rows):
    expected = adapter_lens_hardware_solids(rows[HOUSING], rows)
    for context in (None, {}, {HOUSING: rows[HOUSING]}):
        assert adapter_lens_hardware_solids(rows[HOUSING], context) == expected
    legacy = {name: value for name, value in rows[HOUSING].items() if name not in FIELDS}
    assert adapter_lens_hardware_solids(legacy, rows) is None
    detached = {**rows[HOUSING], "key": "copied_adapter", "mechanical_part_role": "custom_mechanical_copy"}
    rows[COIL]["local_start_z_mm"] += 100.
    assert [s.shape for s in adapter_lens_hardware_solids(detached, rows)] == [s.shape for s in expected]


def test_independent_copy_move_and_saved_readback_keep_both_contact_planes(tmp_path):
    source = PartModelDocument(PATH)
    target = PartModelDocument(ROOT / "configs/instruments/project_and_recording_system/NoEnergyFilter.toml")
    key = target.copy_component_from(source.document, HOUSING, "copied_adapter", 2400., include_children=False)
    fields = (PREFIX + "upper_contact_local_z_mm", PREFIX + "lower_contact_local_z_mm")
    np.testing.assert_allclose([target.part(key)[f] for f in fields], [2383.62, 2416.38])
    original = adapter_lens_hardware_solids(target.part(key), {})
    target.place_component(key, 2700.)
    np.testing.assert_allclose([target.part(key)[f] for f in fields], [2683.62, 2716.38])
    target.save_copy(tmp_path / "copied-adapter.toml")
    reread = PartModelDocument(target.path)
    placed = adapter_lens_hardware_solids(reread.part(key), {})
    for old, new in zip(original, placed):
        old_zr, new_zr = np.array(old.shape["section_zr_mm"]), np.array(new.shape["section_zr_mm"])
        np.testing.assert_allclose(new_zr[:, 0] - old_zr[:, 0], 300.)
        np.testing.assert_array_equal(new_zr[:, 1], old_zr[:, 1])


@pytest.mark.parametrize("field,value,match", [
    ("upper_contact_local_z_mm", float("nan"), "finite"),
    ("cooling_plate_thickness_mm", 1 << 4096, "finite"),
    ("lower_contact_local_z_mm", True, "finite"),
    ("upper_contact_local_z_mm", 655., "ordered"),
    ("cooling_plate_thickness_mm", 1.8, "ordered"),
    ("insulation_thickness_mm", 0., "positive thicknesses"),
    ("outer_diameter_mm", 176., "plate OD"),
    ("inner_diameter_mm", 19., "plate ID"),
    ("geometry_status", "oem", "engineering"),
    ("geometry_source", "", "geometry_source"),
    ("geometry_source_urls", [], "geometry_source_urls"),
    ("structure", True, "declared structure"),
    ("uppper_contact_local_z_mm", 621.82, "unknown Adapter Lens"),
])
def test_invalid_or_misspelled_recipe_is_rejected(rows, field, value, match):
    rows[HOUSING][PREFIX + field] = value
    with pytest.raises(ValueError, match=match):
        validate_adapter_lens_hardware(rows[HOUSING], rows)


@pytest.mark.parametrize("key,prefix,validate,builder", [
    (HOUSING, PREFIX, validate_adapter_lens_hardware, adapter_lens_hardware_solids),
    ("condenser_lens_3_housing", "mechanical_c3_end_", validate_c3_lens_hardware, c3_lens_hardware_solids),
])
def test_generic_native_context_rejects_missing_coil_and_conflicting_families(rows, key, prefix, validate, builder):
    row = rows[key]
    coil_key = row[prefix + "coil_key"]
    row[prefix + "coil_key"] = "nonexistent_coil"
    with pytest.raises(ValueError, match="references missing coil"):
        validate(row, rows)
    row[prefix + "coil_key"] = coil_key
    only_sibling = {coil_key: rows[coil_key]}
    row[prefix + "coil_key"] = "misspelled_coil"
    with pytest.raises(ValueError, match="references missing coil"):
        validate(row, only_sibling)
    row[prefix + "coil_key"] = coil_key
    other_prefix = "mechanical_c3_end_" if prefix == PREFIX else PREFIX
    row[other_prefix + "structure"] = "insulated_coil_end_cooling_plates"
    with pytest.raises(ValueError, match="conflicting round-lens"):
        validate(row, rows)
    with pytest.raises(ValueError, match="conflicting round-lens"):
        builder(row, rows)
    with pytest.raises(ValueError, match="conflicting round-lens"):
        part_solids(row, rows)


@pytest.mark.parametrize("source", (HOUSING, "condenser_lens_3_housing"))
def test_explicit_replacement_suppresses_native_recipe_and_invalid_restore_is_atomic(source):
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, source, "copied_housing", 2500., include_children=False)
    draft.update_model_3d(key, {"base": {"kind": "box", "width_mm": 10., "height_mm": 10., "length_mm": 1.}})
    rows = {row["key"]: row for row in draft.document["parts"]}
    specs = posed_solids(rows[key], rows)
    assert len(specs) == 1
    assert _manifold(specs[0].shape, 16).volume() == pytest.approx(100.)
    assert len(part_model_from_document(draft.document, key).meshes) == 1
    snapshot, history = deepcopy(draft.document), deepcopy(draft._history)
    with pytest.raises(ValueError, match="ordered contacts"):
        draft.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"}})
    assert draft.document == snapshot and draft._history == history
    draft.set_dimension(("parts", key, "model_3d", "base", "length_mm"), 170. if source.startswith("condenser") else 36.4)
    draft.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"}})
    assert len(part_model_from_document(draft.document, key).meshes) == 5
