"""TL22-specific saved geometry, material contacts and native/CAD integration."""

from copy import deepcopy
from functools import reduce
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_catalog import native_reconstruction, part_solids, posed_solids
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_round_lenses import round_lens_solids
from temsim.cad_tl22_lens_hardware import PREFIX, FIELDS, tl22_lens_hardware_solids, validate_tl22_lens_hardware
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/instruments/column/C3_ProbeCorrector.toml"
HOUSING = "probe_tl22_lens_housing"
COIL = "probe_tl22_lens_excitation_coil"
CHILDREN = (HOUSING, COIL, "probe_tl22_lens_upper_pole", "probe_tl22_lens_lower_pole", "probe_tl22_lens_yoke")


@pytest.fixture
def document():
    return read_document(PATH)


@pytest.fixture
def rows(document):
    return {part["key"]: part for part in document["parts"]}


@pytest.mark.parametrize("path", sorted(PATH.parent.glob("*.toml")), ids=lambda p: p.stem)
def test_three_probe_variants_share_their_own_explicit_tl22_hardware_and_others_do_not(rows, path):
    other_doc = read_document(path)
    validate_document(other_doc)
    other = {part["key"]: part for part in other_doc["parts"]}
    if "ProbeCorrector" not in path.stem:
        assert HOUSING not in other
        assert not any(name.startswith(PREFIX) for part in other.values() for name in part)
        return
    assert tl22_lens_hardware_solids(other[HOUSING], other) == tl22_lens_hardware_solids(rows[HOUSING], rows)
    # The transfer winding differs from ADL; recipes must not copy its OD/Z.
    assert other[HOUSING][PREFIX + "outer_diameter_mm"] == other[COIL]["mechanical_outer_diameter_mm"] == 93.5109425
    assert other["probe_tl22_lens"]["pole_gap_mm"] == 12.


@pytest.mark.parametrize("count", (16, 32, 64))
def test_added_layers_are_closed_disjoint_and_preserve_original_five_solids(rows, count):
    snapshot = deepcopy(rows)
    legacy = deepcopy(rows)
    for field in FIELDS:
        legacy[HOUSING].pop(field)
    specs = tl22_lens_hardware_solids(rows[HOUSING], rows)
    assert len(specs) == 5
    assert specs[0].shape == annulus(703.6, 729.1, 87., 90.)
    assert round_lens_solids(legacy[HOUSING], legacy)[0].shape == specs[0].shape
    for key in CHILDREN[1:]:
        assert round_lens_solids(rows[key], rows) == round_lens_solids(legacy[key], legacy)
    assert circuit_inventory(list(rows.values())) == circuit_inventory(list(legacy.values()))
    all_specs = [s for key in CHILDREN for s in part_solids(rows[key], rows)]
    assert len(all_specs) == 9
    solids = [_manifold(s.shape, count) for s in all_specs]
    assert all((a ^ b).volume() < 1e-9 for a, b in combinations(solids, 2))
    for mesh in meshes_from_solids(all_specs, count):
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        _, multiplicities = np.unique(edges, axis=0, return_counts=True)
        assert np.all(multiplicities == 2)
        assert mesh.vertices[:, 2].min() >= 703.6 - 1e-10
        assert mesh.vertices[:, 2].max() <= 729.1 + 1e-10
    expected_area = count * .5 * np.sin(2 * np.pi / count) * ((93.5109425/2)**2 - (78.33/2)**2)
    assert sum(s.volume() for s in solids[1:5]) == pytest.approx(1.2 * expected_area, rel=1e-12)
    assert rows == snapshot


def test_native_and_cad_share_the_four_passive_layers_at_saved_tl22_contacts(document, rows):
    specs = tl22_lens_hardware_solids(rows[HOUSING], rows)
    assert part_solids(rows[HOUSING], rows) == native_reconstruction(rows[HOUSING], rows) == specs
    native = part_model_from_document(document, HOUSING).meshes
    cad = meshes_from_solids(specs)
    assert len(native) == len(cad) == 5
    for a, b in zip(native, cad):
        assert (a.region, a.material_class) == (b.region, b.material_class)
        np.testing.assert_allclose(a.vertices, b.vertices, atol=1e-10)
        np.testing.assert_array_equal(a.faces, b.faces)
    spans = [(min(p[0] for p in s.shape["section_zr_mm"]), max(p[0] for p in s.shape["section_zr_mm"])) for s in specs[1:]]
    np.testing.assert_allclose(spans, [(704.275, 704.775), (704.775, 704.875),
                                     (727.825, 727.925), (727.925, 728.425)], atol=1e-10)
    assert spans[0][0] - rows[HOUSING]["local_start_z_mm"] == pytest.approx(.675)
    assert rows[HOUSING]["local_end_z_mm"] - spans[-1][1] == pytest.approx(.675)
    assert all(s.status == "engineering_reconstruction_not_oem" and "not another current source" in s.note
               for s in specs[1:])


@pytest.mark.parametrize("neighbor,pole,z,side", [
    ("probe_hp2_hexapole", "probe_tl22_lens_upper_pole", 703.6, -1),
    ("probe_dp22_coils", "probe_tl22_lens_lower_pole", 729.1, 1),
])
def test_existing_neighbor_materials_have_face_contacts_not_assembly_clearance(rows, neighbor, pole, z, side):
    pole_spec, = posed_solids(rows[pole], rows)
    other_specs = posed_solids(rows[neighbor], rows)
    contact_areas = []
    for count in (32, 64, 128):
        lens = _manifold(pole_spec.shape, count)
        opposite = [_manifold(s.shape, count) for s in other_specs]
        combined = reduce(lambda a, b: a + b, opposite)
        assert (lens ^ combined).volume() < 1e-9
        assert lens.min_gap(combined, .1) == pytest.approx(0., abs=1e-10)
        # Sample inside opposite sides of the common plane, then intersect the
        # XY sections. This detects positive-area contact of actual material.
        section = lens.slice(z - side * 1e-7)
        contact = [(spec.region, (section ^ solid.slice(z + side * 1e-7)).area())
                   for spec, solid in zip(other_specs, opposite)]
        positive = {name: area for name, area in contact if area > 1e-7}
        contact_areas.append(sum(positive.values()))
        if neighbor == "probe_hp2_hexapole":
            assert set(positive) == {"return_yoke"}
            expected = count * .5 * np.sin(2 * np.pi/count) * (33.165**2 - 31.89**2)
            assert positive["return_yoke"] == pytest.approx(expected, abs=2e-6)
        else:
            assert {"x_winding_1", "x_winding_2", "y_winding_1", "y_winding_2"} <= positive.keys()
            assert all(positive[region] > 7. for region in positive if "winding" in region)
    assert all(area > 60. for area in contact_areas)


def test_constant_tube_retains_04_mm_nominal_clearance_through_both_poles(document, rows):
    g = document["geometry"]
    tube, = tube_solids({**g, "key": CONTINUOUS_TUBE_KEY,
        "mechanical_inner_diameter_mm": g["c1_c2_objective_vacuum_tube_inner_diameter_mm"]})
    for key in ("probe_tl22_lens_upper_pole", "probe_tl22_lens_lower_pole"):
        pole, = part_solids(rows[key], rows)
        gaps = []
        for count in (32, 64, 128):
            a, b = _manifold(tube.shape, count), _manifold(pole.shape, count)
            assert (a ^ b).volume() < 1e-9
            gap = a.min_gap(b, .5)
            assert gap == pytest.approx(.4 * np.cos(np.pi/count), abs=2e-8)
            gaps.append(gap)
        assert 0 < gaps[0] < gaps[1] < gaps[2] < .4


@pytest.mark.parametrize("field,value,match", [
    ("cooling_plate_thickness_mm", 1.2, "ordered"),
    ("upper_contact_local_z_mm", float("nan"), "finite"),
    ("outer_diameter_mm", True, "finite"),
    ("coil_key", "missing_coil", "missing coil"),
    ("uppper_contact_local_z_mm", 704.875, "unknown TL22"),
])
def test_invalid_tl22_recipe_is_rejected(rows, field, value, match):
    rows[HOUSING][PREFIX + field] = value
    with pytest.raises(ValueError, match=match):
        validate_tl22_lens_hardware(rows[HOUSING], rows)


@pytest.mark.parametrize("other_prefix", ("mechanical_c3_end_", "mechanical_adapter_end_"))
def test_tl22_recipe_cannot_be_silently_hidden_by_another_family(rows, other_prefix):
    rows[HOUSING][other_prefix + "structure"] = "insulated_coil_end_cooling_plates"
    with pytest.raises(ValueError, match="conflicting round-lens"):
        validate_tl22_lens_hardware(rows[HOUSING], rows)
    with pytest.raises(ValueError, match="conflicting round-lens"):
        part_solids(rows[HOUSING], rows)


def test_detached_copy_moves_its_contacts_and_explicit_replacement_keeps_priority(tmp_path):
    draft = PartModelDocument(PATH)
    key = draft.copy_component_from(draft.document, HOUSING, "copied_tl22_housing", 2400., include_children=False)
    draft.place_component(key, 2500.)
    fields = (PREFIX + "upper_contact_local_z_mm", PREFIX + "lower_contact_local_z_mm")
    np.testing.assert_allclose([draft.part(key)[f] for f in fields], [2488.525, 2511.475])
    expected = tl22_lens_hardware_solids(draft.part(key), {})
    draft.save_copy(tmp_path / "tl22-copy.toml")
    reread = PartModelDocument(draft.path)
    assert tl22_lens_hardware_solids(reread.part(key), {}) == expected
    reread.update_model_3d(key, {"base": {"kind": "box", "width_mm": 10., "height_mm": 10., "length_mm": 1.}})
    assert len(part_model_from_document(reread.document, key).meshes) == 1
    before, history = deepcopy(reread.document), deepcopy(reread._history)
    with pytest.raises(ValueError, match="ordered contacts"):
        reread.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"}})
    assert reread.document == before and reread._history == history
