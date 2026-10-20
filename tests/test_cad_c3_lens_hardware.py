"""C3 end hardware preserves winding/optics and the actual tube clearance."""

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_c3_lens_hardware import (
    PREFIX, FIELDS, c3_lens_hardware_solids, validate_c3_lens_hardware,
)
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_round_lenses import round_lens_solids
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_document import PartModelDocument


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/instruments/column/C3_ProbeCorrector.toml"
HOUSING = "condenser_lens_3_housing"
COIL = "condenser_lens_3_excitation_coil"
C3_COLUMNS = sorted(PATH.parent.glob("C3*.toml"))


@pytest.fixture
def document():
    return read_document(PATH)


@pytest.fixture
def rows(document):
    return {row["key"]: row for row in document["parts"]}


@pytest.mark.parametrize("path", C3_COLUMNS, ids=lambda path: path.stem)
def test_all_six_c3_variants_use_identical_explicit_end_hardware(rows, path):
    doc = read_document(path)
    validate_document(doc)
    other = {row["key"]: row for row in doc["parts"]}
    assert len(C3_COLUMNS) == 6
    assert c3_lens_hardware_solids(other[HOUSING], other) == c3_lens_hardware_solids(rows[HOUSING], rows)
    assert other["condenser_lens_3"]["pole_gap_mm"] == 18.
    assert (other[COIL]["local_start_z_mm"], other[COIL]["local_end_z_mm"]) == (388.5, 541.5)


@pytest.mark.parametrize("count", (16, 32, 64))
def test_original_wall_and_copper_survive_without_material_intersections(rows, count):
    before = deepcopy(rows)
    specs = c3_lens_hardware_solids(rows[HOUSING], rows)
    assert [s.region for s in specs] == ["body", "upper_cooling_plate", "upper_coil_insulation",
                                      "lower_coil_insulation", "lower_cooling_plate"]
    assert [s.material for s in specs] == ["nonmagnetic_housing", "copper_cooling_plate", "insulator",
                                         "insulator", "copper_cooling_plate"]
    assert specs[0].shape == annulus(380., 550., 87., 90.)
    legacy = {key: {k: v for k, v in row.items() if k not in FIELDS} for key, row in rows.items()}
    assert round_lens_solids(legacy[HOUSING], legacy)[0].shape == specs[0].shape
    keys = (COIL, "condenser_lens_3_upper_pole", "condenser_lens_3_lower_pole", "condenser_lens_3_yoke")
    for key in keys:
        assert round_lens_solids(rows[key], rows) == round_lens_solids(legacy[key], legacy)
    all_specs = [*specs, *(s for key in keys for s in round_lens_solids(rows[key], rows))]
    solids = [_manifold(s.shape, count) for s in all_specs]
    for first, second in combinations(solids, 2):
        assert (first ^ second).volume() < 1e-9
    for mesh in meshes_from_solids(all_specs, count):
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        _, multiplicities = np.unique(edges, axis=0, return_counts=True)
        assert np.all(multiplicities == 2)
    polygon_annulus_area = count * .5 * np.sin(2 * np.pi/count) * (78.**2 - 48.**2)
    assert sum(s.volume() for s in solids[1:5]) == pytest.approx(5. * polygon_annulus_area, rel=1e-12)
    assert rows == before


def test_end_layers_touch_coil_without_claiming_new_magnetic_sources(rows):
    old = deepcopy(rows)
    for name in FIELDS:
        old[HOUSING].pop(name)
    assert circuit_inventory(list(old.values())) == circuit_inventory(list(rows.values()))
    specs = c3_lens_hardware_solids(rows[HOUSING], rows)
    z_spans = [(min(point[0] for point in s.shape["section_zr_mm"]),
                max(point[0] for point in s.shape["section_zr_mm"])) for s in specs[1:]]
    assert z_spans == [(386., 388.), (388., 388.5), (541.5, 542.), (542., 544.)]
    assert all(s.material != "soft_magnetic" and "not another current source" in s.note for s in specs[1:])
    assert specs[1].parameters[PREFIX + "cooling_plate_thickness_mm"] == 2.


def test_legacy_missing_context_and_detached_copy_are_self_contained(rows):
    legacy = {key: value for key, value in rows[HOUSING].items() if key not in FIELDS}
    assert c3_lens_hardware_solids(legacy, {}) is None
    expected = c3_lens_hardware_solids(rows[HOUSING], rows)
    assert c3_lens_hardware_solids(rows[HOUSING], None) == expected
    assert c3_lens_hardware_solids(rows[HOUSING], {}) == expected
    detached = {**rows[HOUSING], "key": "copied_c3_housing", "mechanical_part_role": "custom_mechanical_copy"}
    rows[COIL]["local_start_z_mm"] += 100.
    assert [s.shape for s in c3_lens_hardware_solids(detached, rows)] == [s.shape for s in expected]
    assert all(s.key == detached["key"] for s in c3_lens_hardware_solids(detached, {}))


def test_copy_and_move_translate_both_saved_contacts_without_original_coil():
    source = PartModelDocument(PATH)
    target = PartModelDocument(ROOT/"configs/instruments/project_and_recording_system/NoEnergyFilter.toml")
    key = target.copy_component_from(source.document, HOUSING, "copied_c3_housing", 2400., include_children=False)
    fields = (PREFIX + "upper_contact_local_z_mm", PREFIX + "lower_contact_local_z_mm")
    assert tuple(target.part(key)[field] for field in fields) == (2323.5, 2476.5)
    original = c3_lens_hardware_solids(target.part(key), {})
    target.place_component(key, 2700.)
    assert tuple(target.part(key)[field] for field in fields) == (2623.5, 2776.5)
    placed = c3_lens_hardware_solids(target.part(key), {})
    for before, after in zip(original, placed):
        b, a = np.array(before.shape["section_zr_mm"]), np.array(after.shape["section_zr_mm"])
        np.testing.assert_allclose(a[:, 0] - b[:, 0], 300.)
        np.testing.assert_array_equal(a[:, 1], b[:, 1])


@pytest.mark.parametrize("field,value,match", [
    ("upper_contact_local_z_mm", float("nan"), "finite"),
    ("cooling_plate_thickness_mm", 1 << 4096, "finite"),
    ("lower_contact_local_z_mm", True, "finite"),
    ("upper_contact_local_z_mm", 542., "ordered"),
    ("cooling_plate_thickness_mm", 10., "ordered"),
    ("insulation_thickness_mm", 0., "positive thicknesses"),
    ("outer_diameter_mm", 176., "plate OD"),
    ("inner_diameter_mm", 17., "plate ID"),
    ("geometry_status", "oem", "engineering"),
    ("geometry_source", "", "geometry_source"),
    ("geometry_source_urls", [], "geometry_source_urls"),
    ("structure", True, "declared structure"),
    ("uppper_contact_local_z_mm", 388.5, "unknown C3"),
])
def test_invalid_saved_hardware_is_rejected_before_render(rows, field, value, match):
    rows[HOUSING][PREFIX + field] = value
    with pytest.raises(ValueError, match=match):
        validate_c3_lens_hardware(rows[HOUSING], rows)


def test_incompatible_coil_or_surrounding_material_requires_explicit_recipe_sync(rows):
    rows[COIL]["local_start_z_mm"] += 1.
    with pytest.raises(ValueError, match="synchronize the housing hardware recipe"):
        validate_c3_lens_hardware(rows[HOUSING], rows)
    rows[COIL]["local_start_z_mm"] -= 1.
    rows["condenser_lens_3_yoke"]["mechanical_inner_diameter_mm"] = 156.
    with pytest.raises(ValueError, match="clearance from yoke"):
        validate_c3_lens_hardware(rows[HOUSING], rows)


def test_tube_minimum_three_dimensional_clearance_converges_to_conical_transition_limit(document, rows):
    geometry = document["geometry"]
    tube_row = {**geometry, "key": CONTINUOUS_TUBE_KEY,
                "mechanical_inner_diameter_mm": geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"]}
    tube_spec, = tube_solids(tube_row)
    # Slots are far from the C3 poles and do not set this local minimum gap.
    radial = (18. - 17.2) / 2
    slope = ((19.2 - 17.2) / 2) / 10.
    analytic_gap = radial / np.sqrt(1 + slope**2)
    assert analytic_gap == pytest.approx(.398014876084, abs=1e-12)
    for key in ("condenser_lens_3_upper_pole", "condenser_lens_3_lower_pole"):
        pole_spec, = round_lens_solids(rows[key], rows)
        gaps = []
        for n in (32, 64, 128):
            tube, pole = _manifold(tube_spec.shape, n), _manifold(pole_spec.shape, n)
            assert (tube ^ pole).volume() < 1e-9
            gap = tube.min_gap(pole, .5)
            c = np.cos(np.pi / n)
            expected_faceted_gap = radial * c / np.sqrt(1 + (slope * c)**2)
            assert gap == pytest.approx(expected_faceted_gap, abs=2e-8)
            gaps.append(gap)
        assert 0 < gaps[0] < gaps[1] < gaps[2] < analytic_gap < radial
        assert analytic_gap - gaps[-1] < .00012
