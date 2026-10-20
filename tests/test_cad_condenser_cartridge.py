"""Independent checks for the shared carrier's explicit mechanical partition."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_condenser_cartridge import FIELDS, cartridge_solids, validate_cartridge_geometry
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_round_lenses import round_lens_solids
from temsim.magnetic_circuits import circuit_inventory
from temsim.module_manifest import read_document, validate_document


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/instruments/column/C3_ProbeCorrector.toml"
COLUMNS = sorted(PATH.parent.glob("*.toml"))
KEY = "c1_c2_pole_piece_cartridge"


@pytest.fixture
def rows():
    return {row["key"]: row for row in read_document(PATH)["parts"]}


@pytest.mark.parametrize("segments", (16, 32, 64))
def test_partition_preserves_volume_envelope_bore_and_closed_nonoverlap(rows, segments):
    before = deepcopy(rows)
    specs = cartridge_solids(rows[KEY], rows)
    assert [s.region for s in specs] == ["c1_carrier_unresolved", "nonmagnetic_separator_assumed", "c2_carrier_unresolved"]
    assert [s.material for s in specs] == [rows[KEY]["material_class"], "nonmagnetic_housing", rows[KEY]["material_class"]]
    materials = [_manifold(s.shape, segments) for s in specs]
    old = _manifold(annulus(0., 300., 30., 45.375), segments)
    assert sum(m.volume() for m in materials) == pytest.approx(old.volume(), rel=1e-12)
    combined = materials[0] + materials[1] + materials[2]
    assert (old - combined).volume() < 1e-9
    assert (combined - old).volume() < 1e-9
    for i, first in enumerate(materials):
        for second in materials[i+1:]:
            assert (first ^ second).volume() < 1e-9
    for mesh in meshes_from_solids(specs, segments):
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)
        assert np.linalg.norm(mesh.vertices[:, :2], axis=1).min() == pytest.approx(30.)
    assert rows == before


def test_cartridge_partition_adds_no_poles_coils_or_solver_materials(rows):
    before = deepcopy(rows)
    for field in FIELDS:
        before[KEY].pop(field)
    assert circuit_inventory(list(before.values())) == circuit_inventory(list(rows.values()))
    assert all(KEY not in circuit.body_keys for circuit in circuit_inventory(list(rows.values())))
    for key, row in rows.items():
        if key != KEY:
            assert row == before[key]
            assert round_lens_solids(row, rows) == round_lens_solids(before[key], before)
    specs = cartridge_solids(rows[KEY], rows)
    assert all(spec.material != "soft_magnetic" and "not a known OEM" in spec.note for spec in specs)


def test_explicit_separator_position_changes_geometry_without_changing_poles(rows):
    poles = deepcopy([rows[key] for key in rows[KEY]["mechanical_cartridge_pole_keys"]])
    row = rows[KEY]
    row["mechanical_cartridge_spacer_start_local_z_mm"] = 92.
    row["mechanical_cartridge_spacer_end_local_z_mm"] = 108.
    specs = cartridge_solids(row, rows)
    assert np.asarray(specs[1].shape["section_zr_mm"])[:, 0].tolist() == [92., 92., 108., 108.]
    assert [rows[key] for key in row["mechanical_cartridge_pole_keys"]] == poles
    specs[0].parameters["mechanical_cartridge_geometry_source_urls"].append("changed")
    assert len(row["mechanical_cartridge_geometry_source_urls"]) == 1


@pytest.mark.parametrize("name,value,match", [
    ("mechanical_cartridge_spacer_start_local_z_mm", True, "finite number"),
    ("mechanical_cartridge_spacer_end_local_z_mm", float("nan"), "finite number"),
    ("mechanical_cartridge_spacer_start_local_z_mm", 110., "spacer start"),
    ("mechanical_cartridge_spacer_end_local_z_mm", 300., "spacer start"),
    ("mechanical_cartridge_spacer_material", "soft_magnetic", "spacer material"),
    ("mechanical_cartridge_structure", True, "unsupported cartridge"),
    ("mechanical_cartridge_geometry_status", "oem", "engineering reconstruction"),
    ("mechanical_cartridge_geometry_source", "", "geometry_source"),
    ("mechanical_cartridge_geometry_source_urls", [], "geometry_source_urls"),
    ("mechanical_cartridge_pole_keys", ["missing", "other"], "missing"),
    ("mechanical_cartridge_spacer_strat_z_mm", 90., "unknown cartridge"),
    ("material_class", "soft_magnetic", "unresolved mixed"),
    ("magnetic_radial_profile_mm", [[0, 30, 45.375], [300, 30, 45.375]], "compete"),
])
def test_invalid_or_misspelled_details_fail_before_rendering(rows, name, value, match):
    rows[KEY][name] = value
    with pytest.raises(ValueError, match=match):
        validate_cartridge_geometry(rows[KEY], rows)


def test_legacy_and_independent_copy_do_not_acquire_optical_children(rows):
    legacy = {name: value for name, value in rows[KEY].items() if name not in FIELDS}
    assert cartridge_solids(legacy, rows) is None
    copy = {**rows[KEY], "key": "copied_carrier", "geometry_template_key": KEY,
            "mechanical_part_role": "custom_mechanical_copy"}
    assert len(cartridge_solids(copy, {copy["key"]: copy})) == 3
    assert all(spec.key == copy["key"] for spec in cartridge_solids(copy, {}))
    assert cartridge_solids({**rows[KEY], "layout_role": "control_channel"}, rows) == ()


def test_installed_manifest_keeps_and_validates_explicit_partition():
    document = read_document(PATH)
    validate_document(document)
    row = next(row for row in document["parts"] if row["key"] == KEY)
    row["mechanical_cartridge_spacer_start_local_z_mm"] = -1.
    with pytest.raises(ValueError, match="spacer start"):
        validate_document(document)


@pytest.mark.parametrize("path", COLUMNS, ids=lambda path: path.stem)
def test_all_column_variants_keep_same_shared_cartridge_detail(rows, path):
    document = read_document(path)
    validate_document(document)
    variant = {row["key"]: row for row in document["parts"]}
    assert len(COLUMNS) == 7
    assert cartridge_solids(variant[KEY], variant) == cartridge_solids(rows[KEY], rows)
    for key in rows[KEY]["mechanical_cartridge_pole_keys"]:
        for name in ("local_start_z_mm", "local_end_z_mm", "mechanical_outer_diameter_mm",
                     "mechanical_bore_diameter_mm", "mechanical_container_key", "parent_key"):
            assert variant[key][name] == rows[key][name]
