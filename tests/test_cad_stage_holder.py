"""Real material openings, holder motion and shared CAD/native geometry."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.cad_catalog import module_catalog, part_solids, posed_solids
from temsim.cad_geometry import box
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.cad_stage_holder import physical_liner_rows, liner_material_decisions
from temsim.module_manifest import read_document
from temsim.part_model_3d import part_model_from_document, part_dimension_specs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def document():
    return read_document(ROOT / "configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml")


def material(specs):
    result = _manifold(specs[0].shape, 64)
    for spec in specs[1:]:
        result = result + _manifold(spec.shape, 64)
    return result


def probe_volume(solid, center, size=(.01, .01, .01)):
    return (solid ^ _manifold(box(center, size), 64)).volume()


def test_holder_has_open_beam_path_and_actual_seat_support(document):
    rows = {p["key"]: p for p in document["parts"]}
    z = rows["sample"]["local_center_z_mm"]
    holder = material(part_solids(rows["sample_holder"], rows))
    assert probe_volume(holder, (0., 0., z), (2.35, .05, 2.)) < 1e-12
    assert probe_volume(holder, (1.4, 0., z)) < 1e-12  # Seat above floor.
    assert probe_volume(holder, (1.4, 0., z-.1)) > 0.  # Supporting annulus.
    assert probe_volume(holder, (5., 0., z)) > 0.  # Flat arm.
    assert probe_volume(holder, (5., 0., z+1.)) < 1e-12
    assert probe_volume(holder, (20., 0., z+3.)) > 0.  # Full rod outside tip.


def test_shell_port_is_through_material_and_keeps_opposite_wall(document):
    rows = {p["key"]: p for p in document["parts"]}
    z = rows["sample"]["local_center_z_mm"]
    shell = material(part_solids(rows["objective_lens_housing"], rows))
    assert probe_volume(shell, (88.5, 0., z), (2., 2., 2.)) < 1e-10
    assert probe_volume(shell, (-88.5, 0., z)) > 0.
    assert probe_volume(shell, (88.5, 0., z+7.)) > 0.


def test_native_and_step_specs_share_geometry_and_stage_is_a_separate_unit(document):
    rows = {p["key"]: p for p in document["parts"]}
    units = {row["key"]: row for row in module_catalog(document)["units"]}
    for key in ("sample_stage", "sample_holder", "objective_lens_housing"):
        expected = meshes_from_solids(posed_solids(rows[key], rows), 32)
        shown = part_model_from_document(document, key, angular_segments=32, include_children=False)
        assert len(shown.meshes) == len(expected)
        for left, right in zip(shown.meshes, expected):
            np.testing.assert_allclose(left.vertices, right.vertices, atol=1e-10, rtol=0)
    assert len(units["sample_holder"]["solid_ids"]) == 3
    assert len(units["sample_stage"]["solid_ids"]) == 3
    paths = {field.path[-1] for field in part_dimension_specs(document, "sample_holder")}
    assert {"holder_retraction_mm", "holder_tip_seat_depth_mm", "holder_shaft_end_x_mm"} <= paths


def test_retraction_moves_only_holder_and_does_not_inherit_objective_tilt(document):
    rows = {p["key"]: p for p in document["parts"]}
    before = deepcopy(rows)
    original = posed_solids(rows["sample_holder"], rows)
    rows["objective_lens"]["rotation_x_mrad"] = 2.
    rows["objective_lens"]["rotation_y_mrad"] = -3.
    assert posed_solids(rows["sample_holder"], rows) == original
    rows["sample_holder"]["holder_retraction_mm"] = 50.
    shifted = meshes_from_solids(posed_solids(rows["sample_holder"], rows), 32)
    unshifted = meshes_from_solids(original, 32)
    for a, b in zip(shifted, unshifted):
        np.testing.assert_allclose(a.vertices, b.vertices + (50., 0., 0.), atol=1e-10, rtol=0)
    assert rows["sample"] == before["sample"]
    assert rows["sample_stage"] == before["sample_stage"]


def test_material_liner_has_access_without_mutating_equivalent_solver_boundary(document):
    rows = {p["key"]: p for p in document["parts"]}
    stage = SimpleNamespace(key="sample_stage", data=rows["sample_stage"], center_z_mm=2000.)
    liner = SimpleNamespace(key="liner", name="liner", start_z_mm=1997.3,
                            end_z_mm=2002.7, inner_diameter_mm=5., outer_diameter_mm=5.5)
    assembly = SimpleNamespace(parts=[stage], vacuum_liner_segments=[liner])
    before = deepcopy(vars(liner))
    row, = physical_liner_rows(assembly)
    solid = material(part_solids(row, {row["key"]: row}))
    assert probe_volume(solid, (2.6, 0., 2000.)) < 1e-12
    assert probe_volume(solid, (-2.6, 0., 2000.)) > 0.
    assert vars(liner) == before


def test_equivalent_component_boundaries_do_not_duplicate_material():
    def tube(key):
        return SimpleNamespace(key=key, name=key, start_z_mm=0., end_z_mm=10.,
                               inner_diameter_mm=4., outer_diameter_mm=4.5)
    segments = [tube("@vacuum_liner:feg_extractor"), tube("@vacuum_liner:sample_stage"),
                tube("@vacuum_liner:c1_c2_to_upper_objective"),
                tube("@vacuum_liner:@vacuum_drift:gun:feg")]
    assembly = SimpleNamespace(parts=[], vacuum_liner_segments=segments)
    decisions = liner_material_decisions(assembly)
    assert [r["independent_material"] for r in decisions] == [False, False, True, True]
    assert [r["key"] for r in physical_liner_rows(assembly)] == [s.key for s in segments[2:]]
    assert len(assembly.vacuum_liner_segments) == 4
