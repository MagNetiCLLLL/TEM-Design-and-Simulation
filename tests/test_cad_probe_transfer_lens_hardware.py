"""Lower probe relay material detail, interfaces and edit preservation."""

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_catalog import native_reconstruction, part_solids, posed_solids
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_probe_transfer_lens_hardware import (
    FIELDS, PREFIX, probe_transfer_lens_hardware_solids, validate_probe_transfer_lens_hardware,
)
from temsim.cad_round_lenses import round_lens_solids
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document
from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/instruments/column/C3_ProbeCorrector.toml"
LENSES = ("probe_tl21_lens", "probe_tl12_lens")
EXPECTED = {
    "probe_tl21_lens": (787.2, 812.7, 788.475, 811.425, 93.3164759, .675),
    "probe_tl12_lens": (903.6, 929., 904.87, 927.73, 93.63, .67),
}


@pytest.fixture
def document():
    return read_document(PATH)


@pytest.fixture
def rows(document):
    return {part["key"]: part for part in document["parts"]}


@pytest.mark.parametrize("path", sorted(PATH.parent.glob("*.toml")), ids=lambda p: p.stem)
def test_saved_variants_opt_in_with_their_own_winding_dimensions(rows, path):
    doc = read_document(path)
    validate_document(doc)
    other = {part["key"]: part for part in doc["parts"]}
    if "ProbeCorrector" not in path.stem:
        assert not any(name.startswith(PREFIX) for part in other.values() for name in part)
        return
    for lens in LENSES:
        housing, coil = lens + "_housing", lens + "_excitation_coil"
        actual = probe_transfer_lens_hardware_solids(other[housing], other)
        assert actual == probe_transfer_lens_hardware_solids(rows[housing], rows)
        assert other[housing][PREFIX + "outer_diameter_mm"] == other[coil]["mechanical_outer_diameter_mm"]
        assert other[housing][PREFIX + "outer_diameter_mm"] == EXPECTED[lens][4]
        assert other[housing][PREFIX + "outer_diameter_mm"] != rows["probe_tl22_lens_excitation_coil"]["mechanical_outer_diameter_mm"]


@pytest.mark.parametrize("lens", LENSES)
@pytest.mark.parametrize("segments", (16, 32, 64))
def test_original_five_materials_unchanged_and_nine_total_regions_closed_disjoint(rows, lens, segments):
    original = deepcopy(rows)
    legacy = deepcopy(rows)
    housing = lens + "_housing"
    for field in FIELDS:
        legacy[housing].pop(field)
    actual = probe_transfer_lens_hardware_solids(rows[housing], rows)
    assert actual[0].shape == round_lens_solids(legacy[housing], legacy)[0].shape
    suffixes = ("housing", "excitation_coil", "upper_pole", "lower_pole", "yoke")
    for suffix in suffixes[1:]:
        key = lens + "_" + suffix
        assert part_solids(rows[key], rows) == part_solids(legacy[key], legacy)
    specs = [s for suffix in suffixes for s in part_solids(rows[lens + "_" + suffix], rows)]
    assert len(specs) == 9
    bodies = [_manifold(s.shape, segments) for s in specs]
    assert all(s.volume() > 0. for s in bodies)
    assert all((a ^ b).volume() < 1e-9 for a, b in combinations(bodies, 2))
    for mesh in meshes_from_solids(specs, segments):
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    outer = EXPECTED[lens][4]
    polygon_area = segments * .5 * np.sin(2 * np.pi / segments) * ((outer / 2)**2 - (78.33 / 2)**2)
    assert sum(s.volume() for s in bodies[1:5]) == pytest.approx(1.2 * polygon_area, rel=2e-12)
    assert circuit_inventory(list(rows.values())) == circuit_inventory(list(legacy.values()))
    assert rows == original


@pytest.mark.parametrize("lens", LENSES)
def test_same_native_cad_materials_preserve_contacts_and_end_clearances(document, rows, lens):
    housing = lens + "_housing"
    specs = probe_transfer_lens_hardware_solids(rows[housing], rows)
    assert native_reconstruction(rows[housing], rows) == part_solids(rows[housing], rows) == specs
    native = part_model_from_document(document, housing, angular_segments=32).meshes
    cad = meshes_from_solids(specs, 32)
    assert len(native) == len(cad) == 5
    for a, b in zip(native, cad):
        assert (a.region, a.material_class) == (b.region, b.material_class)
        np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-10)
        np.testing.assert_array_equal(a.faces, b.faces)
    lo, hi, a, b, _, clearance = EXPECTED[lens]
    spans = [(min(p[0] for p in s.shape["section_zr_mm"]), max(p[0] for p in s.shape["section_zr_mm"]))
             for s in specs[1:]]
    np.testing.assert_allclose(spans, [(a-.6, a-.1), (a-.1, a), (b, b+.1), (b+.1, b+.6)], atol=1e-10)
    assert spans[0][0] - lo == pytest.approx(clearance)
    assert hi - spans[-1][1] == pytest.approx(clearance)
    assert all(s.status == "engineering_reconstruction_not_oem" and "not another current source" in s.note
               for s in specs[1:])


@pytest.mark.parametrize("lens,deflector", ((LENSES[0], "probe_dp21_coils"), (LENSES[1], "probe_dp11_coils")))
def test_upstream_deflector_copper_and_seal_keep_positive_gap_at_all_mesh_resolutions(rows, lens, deflector):
    pole_spec, = posed_solids(rows[lens + "_upper_pole"], rows)
    specs = posed_solids(rows[deflector], rows)
    for segments in (32, 64, 128):
        pole = _manifold(pole_spec.shape, segments)
        copper = [s for s in specs if s.material == "copper_winding"]
        assert len(copper) == 4
        for spec in copper:
            other = _manifold(spec.shape, segments)
            assert (pole ^ other).volume() < 1e-9
            assert pole.min_gap(other, .3) == pytest.approx(.14, abs=1e-9)
        cap = _manifold(next(s.shape for s in specs if s.region == "downstream_end_cover"), segments)
        assert (pole ^ cap).volume() < 1e-9
        assert pole.min_gap(cap, .3) == pytest.approx(.10, abs=1e-9)


@pytest.mark.parametrize("lens", LENSES)
def test_real_tube_clearance_converges_to_point_four_mm(document, rows, lens):
    g = document["geometry"]
    tube_spec, = tube_solids({**g, "key": CONTINUOUS_TUBE_KEY,
        "mechanical_inner_diameter_mm": g["c1_c2_objective_vacuum_tube_inner_diameter_mm"]})
    for suffix in ("upper_pole", "lower_pole"):
        pole_spec, = posed_solids(rows[lens + "_" + suffix], rows)
        for segments in (32, 64, 128):
            tube, pole = _manifold(tube_spec.shape, segments), _manifold(pole_spec.shape, segments)
            assert (tube ^ pole).volume() < 1e-9
            assert tube.min_gap(pole, .5) == pytest.approx(.4 * np.cos(np.pi / segments), abs=2e-8)


@pytest.mark.parametrize("lens", LENSES)
@pytest.mark.parametrize("field,value,match", (
    ("cooling_plate_thickness_mm", 1.2, "ordered"),
    ("insulation_thickness_mm", 0., "positive"),
    ("outer_diameter_mm", float("nan"), "finite"),
    ("coil_key", "unknown_coil", "missing coil"),
    ("uppper_contact_local_z_mm", 800., "unknown Probe transfer lens"),
))
def test_invalid_saved_recipe_is_rejected(rows, lens, field, value, match):
    row = rows[lens + "_housing"]
    row[PREFIX + field] = value
    with pytest.raises(ValueError, match=match):
        validate_probe_transfer_lens_hardware(row, rows)


@pytest.mark.parametrize("housing,prefix", (
    ("condenser_lens_3_housing", "mechanical_c3_end_"),
    ("adapter_lens_housing", "mechanical_adapter_end_"),
    ("probe_tl22_lens_housing", "mechanical_tl22_end_"),
    ("probe_tl21_lens_housing", PREFIX),
    ("probe_tl12_lens_housing", PREFIX),
))
@pytest.mark.parametrize("suffix", ("insulation_thickness_mm", "cooling_plate_thickness_mm"))
def test_sub_ulp_layer_thickness_is_rejected_before_build_and_edit_atomically(document, housing, prefix, suffix):
    invalid = deepcopy(document)
    rows = {part["key"]: part for part in invalid["parts"]}
    field = prefix + suffix
    rows[housing][field] = 1e-20
    with pytest.raises(ValueError, match="positive thicknesses and ordered contacts"):
        validate_document(invalid)
    with pytest.raises(ValueError, match="positive thicknesses and ordered contacts"):
        part_solids(rows[housing], rows)
    draft = PartModelDocument(PATH)
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    with pytest.raises(ValueError, match="positive thicknesses and ordered contacts"):
        draft.set_dimension(("parts", housing, field), 1e-20)
    assert draft.document == before and draft._history == history
    assert not draft.dirty and not draft.can_undo and not draft.can_redo


@pytest.mark.parametrize("lens", LENSES)
def test_conflicting_family_and_native_coil_drift_cannot_silently_render(rows, lens):
    row = rows[lens + "_housing"]
    row["mechanical_tl22_end_structure"] = "insulated_coil_end_cooling_plates"
    with pytest.raises(ValueError, match="conflicting round-lens"):
        part_solids(row, rows)
    row.pop("mechanical_tl22_end_structure")
    rows[lens + "_excitation_coil"]["local_start_z_mm"] += .1
    with pytest.raises(ValueError, match="synchronize"):
        validate_probe_transfer_lens_hardware(row, rows)


@pytest.mark.parametrize("lens", LENSES)
def test_direct_coil_edits_synchronize_hardware_and_gap_keeps_optical_center(lens, tmp_path):
    draft = PartModelDocument(PATH)
    housing, coil = lens + "_housing", lens + "_excitation_coil"
    center = draft.part(lens)["local_center_z_mm"]
    expected_coil_outer = draft.part(coil)["mechanical_outer_diameter_mm"] + .2
    draft.set_dimension(("parts", coil, "mechanical_outer_diameter_mm"), expected_coil_outer)
    assert draft.part(housing)[PREFIX + "outer_diameter_mm"] == expected_coil_outer
    before_gap = deepcopy(draft.document)
    gap = draft.part(lens)["pole_gap_mm"] + .2
    draft.set_dimension(("parts", lens, "pole_gap_mm"), gap)
    assert draft.part(lens)["local_center_z_mm"] == center
    assert draft.part(lens + "_upper_pole")["local_end_z_mm"] == pytest.approx(center-gap/2)
    assert draft.part(lens + "_lower_pole")["local_start_z_mm"] == pytest.approx(center+gap/2)
    draft.undo()
    assert draft.document == before_gap
    draft.redo()
    draft.validate()
    draft.save_copy(tmp_path / (lens + ".toml"))
    reopened = PartModelDocument(draft.path)
    assert reopened.part(housing)[PREFIX + "outer_diameter_mm"] == expected_coil_outer
    assert reopened.part(lens)["pole_gap_mm"] == gap


@pytest.mark.parametrize("lens", LENSES)
def test_parent_copy_remaps_and_moves_its_own_coil_interfaces(lens, tmp_path):
    draft = PartModelDocument(PATH)
    old_parts = deepcopy(draft.document["parts"])
    copied = draft.copy_component_from(draft.document, lens, "relay_copy", 2500., include_children=True)
    draft.place_component(copied, 2600.)
    housing, coil = copied + "_housing", copied + "_excitation_coil"
    assert draft.part(housing)[PREFIX + "coil_key"] == coil
    for suffix, name in (("upper", "local_start_z_mm"), ("lower", "local_end_z_mm")):
        assert draft.part(housing)[PREFIX + suffix + "_contact_local_z_mm"] == draft.part(coil)[name]
    diameter = draft.part(coil)["mechanical_outer_diameter_mm"] + .1
    draft.set_dimension(("parts", coil, "mechanical_outer_diameter_mm"), diameter)
    assert draft.part(housing)[PREFIX + "outer_diameter_mm"] == diameter
    for old in old_parts:
        assert draft.part(old["key"]) == old
    draft.save_copy(tmp_path / (lens + "-copy.toml"))
    reopened = PartModelDocument(draft.path)
    assert reopened.part(housing)[PREFIX + "coil_key"] == coil
    assert len(part_model_from_document(reopened.document, housing).meshes) == 5


@pytest.mark.parametrize("lens", LENSES)
def test_passive_dimensions_have_visible_meaning_and_replacement_restoration_is_atomic(rows, lens):
    housing = lens + "_housing"
    field = PREFIX + "cooling_plate_thickness_mm"
    meaning = describe_parameter(rows[housing], (field,), by_key=rows)
    assert (meaning.category, meaning.unit, meaning.source_kind) == ("physical", "mm", "estimated")
    impact = describe_parameter_impact(rows[housing], ("parts", housing, field), by_key=rows)
    assert "field_map_identity" in impact.effects
    draft = PartModelDocument(PATH)
    visible = {item.name: item for item in part_model_from_document(draft.document, housing).fields}
    for suffix in ("upper_contact_local_z_mm", "lower_contact_local_z_mm", "inner_diameter_mm",
                   "outer_diameter_mm", "insulation_thickness_mm", "cooling_plate_thickness_mm"):
        assert PREFIX + suffix in visible
        assert visible[PREFIX + suffix].unit == "mm"
    copied = draft.copy_component_from(draft.document, housing, "passive_copy", 2500., include_children=False)
    draft.update_model_3d(copied, {"base": {"kind": "box", "width_mm": 10., "height_mm": 10., "length_mm": 1.}})
    draft.set_dimension(("parts", copied, field), 100.)
    assert len(part_model_from_document(draft.document, copied).meshes) == 1
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    with pytest.raises(ValueError, match="ordered contacts"):
        draft.set_model_3d(copied, {"schema_version": 1, "base": {"kind": "existing"}})
    assert draft.document == before and draft._history == history
