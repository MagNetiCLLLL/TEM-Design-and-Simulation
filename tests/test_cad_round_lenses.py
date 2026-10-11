"""Round-lens CAD keeps native surfaces, material ownership and local frames."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from temsim.cad_round_lenses import SUPPORTED_PROFILES, round_lens_solids
from temsim.module_manifest import read_document
from temsim.part_model_3d import _legacy_part_meshes, revolve_section


ROOT = Path(__file__).resolve().parents[1]


def _meshes(specs):
    return [revolve_section(s.shape["section_zr_mm"], key=s.key, region=s.region,
                            angular_segments=16) for s in specs]


def _bbox(meshes):
    points = np.concatenate([x.vertices for x in meshes])
    return np.array([points.min(axis=0), points.max(axis=0)])


@pytest.mark.parametrize("relative", [
    "column/C2.toml", "column/C3.toml", "column/C3_ProbeCorrector.toml",
    "column/C3_ImageCorrector.toml", "column/C3_ProbeCorrector_ImageCorrector.toml",
    "project_and_recording_system/NoEnergyFilter.toml",
])
def test_all_native_round_material_base_shapes_match_existing_view(relative):
    doc = read_document(ROOT / "configs/instruments" / relative)
    before = deepcopy(doc)
    by_key = {row["key"]: row for row in doc["parts"]}
    checked = []
    for part in doc["parts"]:
        if part.get("mechanical_profile") not in SUPPORTED_PROFILES:
            continue
        specs = round_lens_solids(part, by_key)
        assert specs, part["key"]
        # Compare the uncut native round base. The separately tested holder
        # access port intentionally changes the final objective shell surface.
        with patch("temsim.cad_catalog.native_reconstruction", return_value=None):
            old, _ = _legacy_part_meshes(part, by_key, 16)
        new = _meshes(specs)
        assert np.allclose(_bbox(new), _bbox(old), atol=1e-10, rtol=0), part["key"]
        # Comparing the actual vertex set checks noses, bores and gaps, not only
        # the outer envelope. All current regions use the same 16-angle grid.
        vertices = lambda ms: set(map(tuple, np.round(np.concatenate([m.vertices for m in ms]), 9)))
        assert vertices(new) == vertices(old), part["key"]
        assert all(s.key == part["key"] for s in specs)
        assert all(s.evidence and "OEM" in s.note or "unresolved" in s.note for s in specs)
        checked.append(part["key"])
    assert checked
    assert len(set(checked)) == len(checked)
    assert doc == before


def test_all_18_round_optical_parents_add_material_only_through_their_children():
    parents = set()
    for p in (ROOT / "configs/instruments/column").glob("*.toml"):
        doc = read_document(p)
        for row in doc["parts"]:
            if row.get("mechanical_profile") == "magnetic_lens_assembly":
                parents.add(row["key"])
                assert round_lens_solids(row, {}) == ()
    doc = read_document(ROOT / "configs/instruments/project_and_recording_system/NoEnergyFilter.toml")
    for row in doc["parts"]:
        if row.get("mechanical_profile") == "magnetic_lens_assembly":
            parents.add(row["key"])
            assert round_lens_solids(row, {}) == ()
    assert len(parents) == 18
    assert "image_tl12_lens" in parents
    assert "image_ol_post_lens" not in parents


def test_objective_split_intervals_and_mixed_cartridge_are_preserved():
    doc = read_document(ROOT / "configs/instruments/column/C3_ProbeCorrector.toml")
    rows = {r["key"]: r for r in doc["parts"]}
    for key in ("objective_lens_yoke", "objective_lens_excitation_coil"):
        specs = round_lens_solids(rows[key], rows)
        assert [s.region for s in specs] == ["upper", "lower"]
        a, b = [np.asarray(s.shape["section_zr_mm"])[:, 0] for s in specs]
        assert a.max() < b.min()  # Never fill the sample interval with material.
    key = "c1_c2_pole_piece_cartridge"
    spec, = round_lens_solids(rows[key], rows)
    assert spec.status == "mixed_material_cartridge_unresolved"
    assert spec.material == "mixed_material_cartridge_user_photo_unresolved"
    assert spec.material != "soft_magnetic"


def test_explicit_radial_profile_keeps_inner_outer_knots_and_local_origin():
    row = {"key": "test_yoke", "mechanical_profile": "magnetic_lens_yoke",
           "material_class": "soft_magnetic", "local_start_z_mm": 100.,
           "local_end_z_mm": 110., "mechanical_outer_diameter_mm": 12.,
           "vacuum_inner_diameter_mm": 2.,
           "magnetic_radial_profile_mm": [[0., 1., 5.], [4., 2., 6.], [10., 1.5, 5.]]}
    spec, = round_lens_solids(row, {})
    assert spec.shape["section_zr_mm"] == [[100., 5.], [104., 6.], [110., 5.],
                                         [110., 1.5], [104., 2.], [100., 1.]]
    spec.parameters["magnetic_radial_profile_mm"][0][1] = 999
    assert row["magnetic_radial_profile_mm"][0][1] == 1


def test_explicit_material_intervals_remain_separate_and_reject_overlap():
    row = {"key": "copied_coil", "mechanical_profile": "magnetic_excitation_coil",
           "mechanical_part_role": "custom_mechanical_copy",
           "local_start_z_mm": 0., "local_end_z_mm": 20.,
           "mechanical_inner_diameter_mm": 4., "mechanical_outer_diameter_mm": 8.,
           "material_class": "insulated_copper_winding",
           "material_intervals_mm": [[0., 3.], [6., 10.], [15., 20.]]}
    specs = round_lens_solids(row, {})
    assert len(specs) == 3
    assert all(s.material == "copper_winding" for s in specs)
    row["material_intervals_mm"][1][0] = 2.
    with pytest.raises(ValueError, match="overlap"):
        round_lens_solids(row, {})


def test_unsupported_parts_and_nonmaterial_channels_are_not_invented():
    assert round_lens_solids({"key": "scan_coil", "mechanical_profile": "unknown"}, {}) is None
    assert round_lens_solids({"key": "virtual_round", "layout_role": "control_channel",
                             "mechanical_profile": "magnetic_pole_piece"}, {}) == ()
