"""Mechanical winding partition, display and native catalog integration."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from temsim.cad_catalog import native_reconstruction, part_solids
from temsim.cad_geometry import SolidSpec, annulus, torus, winding_insulation
from temsim.cad_mesh import _check_winding_mesh_budget, _manifold, meshes_from_solids
from temsim.cad_round_lenses import round_lens_solids
from temsim.cad_round_winding import resolve_round_winding_geometry, round_winding_solids, round_winding_envelope


def coil(**changes):
    return dict(key="test_winding", mechanical_profile="magnetic_excitation_coil",
                material_class="insulated_copper_winding", local_start_z_mm=10., local_end_z_mm=18.,
                mechanical_inner_diameter_mm=4., mechanical_outer_diameter_mm=8., **changes)


def test_unspecified_legacy_winding_keeps_its_envelope_and_has_no_invented_turns():
    row = coil()
    assert resolve_round_winding_geometry(row, {}) is None
    assert round_winding_solids(row, {}) is None
    assert round_winding_envelope(row, {}) == dict(inner_diameter_mm=4., outer_diameter_mm=8.,
                                                  intervals_mm=[[10., 18.]])
    spec, = round_lens_solids(row, {})
    assert spec.shape == annulus(10., 18., 2., 4.)
    assert "winding_turn_count" not in spec.parameters


def test_turn_count_only_retains_native_envelope_without_guessing_wire_diameter():
    row = coil(winding_turn_count=586, winding_interval_turn_counts=[586])
    before = deepcopy(row)
    assert resolve_round_winding_geometry(row, {}) is None
    assert round_winding_solids(row, {}) is None
    spec, = round_lens_solids(row, {})
    assert spec.shape == annulus(10., 18., 2., 4.)
    assert row == before


def test_wire_diameter_without_turn_count_is_rejected():
    with pytest.raises(ValueError, match="requires winding_turn_count"):
        round_winding_solids(coil(winding_wire_diameter_mm=.4), {})


def test_metal_area_input_builds_round_turns_and_fills_all_remaining_insulation():
    row = coil(winding_turn_count=12,winding_wire_cross_section_mm2=.2)
    before = deepcopy(row)
    geometry = resolve_round_winding_geometry(row,{})
    witness = geometry.layout.area
    assert geometry.to_metadata()['wire_cross_section_mm2'] == .2
    assert geometry.layout.wire_diameter_mm == pytest.approx(math.sqrt(.8/math.pi))
    assert witness.cross_section_area_mm2 == 16.
    assert witness.conductor_cross_section_area_mm2 == pytest.approx(12*.2)
    assert witness.insulation_cross_section_area_mm2 == pytest.approx(16.-12*.2)
    assert witness.conductor_volume_mm3+witness.insulation_volume_mm3 == pytest.approx(witness.allocation_volume_mm3)
    specs = round_winding_solids(row,{})
    wires = [spec for spec in specs if spec.material=='copper_winding']
    remaining, = [spec for spec in specs if spec.material=='insulator']
    assert len(wires)==12 and len(remaining.shape['turns_rz_mm'])==12
    assert all(spec.parameters['winding_wire_cross_section_mm2']==.2 for spec in wires)
    # The same polygonal profile is used for each conductor and its insulation
    # cavity, so the displayed CAD partition closes without changing the shell.
    pack = _manifold(annulus(10.,18.,2.,4.),32)
    solids = [_manifold(spec.shape,32) for spec in specs]
    assert sum(s.volume() for s in solids)==pytest.approx(pack.volume(),rel=2e-9)
    assert row==before


def test_metal_area_and_legacy_diameter_have_equivalent_cad_geometry():
    diameter = .4
    area = math.pi*(diameter/2)**2
    modern = resolve_round_winding_geometry(coil(winding_turn_count=12,winding_wire_cross_section_mm2=area),{})
    old = resolve_round_winding_geometry(coil(winding_turn_count=12,winding_wire_diameter_mm=diameter),{})
    assert modern.field_envelope==old.field_envelope
    assert modern.conductor_allocations==old.conductor_allocations
    assert modern.layout.wire_diameter_mm==pytest.approx(old.layout.wire_diameter_mm,rel=2e-15)
    np.testing.assert_allclose([(t.radius_mm,t.z_mm) for t in modern.layout.turns],
                               [(t.radius_mm,t.z_mm) for t in old.layout.turns],rtol=2e-15,atol=0)


def test_turn_or_metal_area_edit_refills_insulation_inside_fixed_envelope():
    base = coil(winding_turn_count=12,winding_wire_cross_section_mm2=.2)
    larger_wire = dict(base,winding_wire_cross_section_mm2=.4)
    more_turns = dict(base,winding_turn_count=24)
    geometries = [resolve_round_winding_geometry(row,{}) for row in (base,larger_wire,more_turns)]
    first,bigger,more = geometries
    assert first.field_envelope==bigger.field_envelope==more.field_envelope
    assert first.conductor_allocations==bigger.conductor_allocations==more.conductor_allocations
    assert bigger.layout.area.insulation_cross_section_area_mm2 < first.layout.area.insulation_cross_section_area_mm2
    assert more.layout.area.insulation_cross_section_area_mm2 < first.layout.area.insulation_cross_section_area_mm2
    assert bigger.layout.interwire_gap_mm != first.layout.interwire_gap_mm
    assert more.layout.interwire_gap_mm != first.layout.interwire_gap_mm
    for geometry in geometries:
        radius=geometry.layout.wire_diameter_mm/2
        assert geometry.layout.area.conductor_volume_mm3+geometry.layout.area.insulation_volume_mm3==pytest.approx(
            geometry.layout.area.allocation_volume_mm3)
        guard=8*np.finfo(float).eps*18.
        for turn in geometry.layout.turns:
            assert 2.-guard <= turn.radius_mm-radius <= turn.radius_mm+radius <= 4.+guard
            assert 10.-guard <= turn.z_mm-radius <= turn.z_mm+radius <= 18.+guard


@pytest.mark.parametrize('changes',[
    {'winding_turn_count':1000,'winding_wire_cross_section_mm2':1.},
    {'winding_turn_count':12,'winding_wire_cross_section_mm2':100.},
])
def test_impossible_metal_area_or_turn_count_rejects_without_resizing(changes):
    row=coil(**changes);before=deepcopy(row)
    with pytest.raises(ValueError,match='capacity'):
        resolve_round_winding_geometry(row,{})
    assert row==before


def test_area_only_input_keeps_fixed_objective_turns_and_passive_layers():
    from temsim.module_manifest import read_document
    path=Path(__file__).resolve().parents[1]/'configs/instruments/column/C3_ProbeCorrector.toml'
    rows={part['key']:part for part in read_document(path)['parts']}
    row=deepcopy(rows['objective_lens_excitation_coil'])
    row.pop('winding_wire_diameter_mm',None)
    row.update(winding_turn_count=1172,winding_interval_turn_counts=[586,586],
               winding_wire_cross_section_mm2=.5)
    small=resolve_round_winding_geometry(row,rows)
    row['winding_wire_cross_section_mm2']=1.
    larger=resolve_round_winding_geometry(row,rows)
    assert small.passive_solids==larger.passive_solids
    assert small.field_envelope==larger.field_envelope
    assert small.layout.interval_turn_counts==larger.layout.interval_turn_counts==(586,586)
    assert len(larger.layout.turns)==1172
    assert larger.layout.area.insulation_volume_mm3 < small.layout.area.insulation_volume_mm3


def test_n_only_interval_count_uses_actual_envelope_not_guessed_split():
    row = coil(winding_turn_count=1172, winding_interval_turn_counts=[586, 586])
    with pytest.raises(ValueError, match="one count per actual winding interval"):
        resolve_round_winding_geometry(row, {})


def test_n_only_still_rejects_invalid_count_and_partial_interval_sum():
    for changes in (dict(winding_turn_count=True),
                    dict(winding_turn_count=1172, winding_interval_turn_counts=[586, 585])):
        with pytest.raises(ValueError):
            resolve_round_winding_geometry(coil(**changes), {})


def test_all_turns_and_exact_remaining_material_are_shared_by_catalog_and_view():
    row = coil(winding_turn_count=12, winding_wire_diameter_mm=.4)
    before = deepcopy(row)
    specs = round_winding_solids(row, {})
    assert specs == part_solids(row, {}) == native_reconstruction(row, {}) == round_lens_solids(row, {})
    wires = [spec for spec in specs if spec.material == "copper_winding"]
    insulation, = [spec for spec in specs if spec.material == "insulator"]
    assert 0 < insulation.color[-1] < 1
    assert len(wires) == 12
    assert len({spec.region for spec in specs}) == 13
    assert all(spec.shape["kind"] == "torus" for spec in wires)
    assert len(insulation.shape["turns_rz_mm"]) == 12
    pack = _manifold(annulus(10., 18., 2., 4.), 32)
    wire_solids = [_manifold(spec.shape, 32) for spec in wires]
    remaining = _manifold(insulation.shape, 32)
    assert sum(s.volume() for s in wire_solids) + remaining.volume() == pytest.approx(pack.volume(), rel=2e-9)
    # Clipper2 quantizes planar display Booleans; exact STEP independently
    # checks the analytic zero-overlap partition below its BRep tolerance.
    assert all((s ^ remaining).volume() < 1e-7 for s in wire_solids)
    assert all((a ^ b).volume() < 1e-8 for i, a in enumerate(wire_solids) for b in wire_solids[i+1:])
    meshes = meshes_from_solids(specs, 24)
    assert len(meshes) == 13 and all(np.isfinite(mesh.vertices).all() for mesh in meshes)
    assert all("wire profile 12 facets, azimuth 24 facets" in mesh.description for mesh in meshes)
    from temsim.part_model_3d import part_model_from_document
    model = part_model_from_document({"parts": [row]}, row["key"], include_children=False)
    assert {mesh.region for mesh in model.meshes} == {spec.region for spec in specs}
    assert all("wire profile 12 facets, azimuth 32 facets" in mesh.description for mesh in model.meshes)
    assert row == before


def test_round_wire_display_retains_declared_profile_facets_and_torus_bounds():
    shape = torus(3., .2, 13.)
    mesh = _manifold(shape, 64)
    expected = 2*math.pi**2*3.*.2**2
    # Polygonal display volume includes both known angular discretizations;
    # the independent STEP tests check the continuous analytic torus instead.
    polygonal = expected*math.sin(2*math.pi/12)/(2*math.pi/12)*math.sin(2*math.pi/64)/(2*math.pi/64)
    assert mesh.volume() == pytest.approx(polygonal, rel=2e-8)
    points = np.asarray(mesh.to_mesh64().vert_properties)[:, :3]
    profile_angles = np.arctan2(points[:, 2]-13., np.hypot(points[:, 0], points[:, 1])-3.)
    major_angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2*math.pi)
    assert len(np.unique(np.round(np.mod(profile_angles, 2*math.pi), 7))) == 12
    assert len(np.unique(np.round(major_angles, 7))) == 64
    bounds = mesh.bounding_box()
    assert bounds == pytest.approx((-3.2, -3.2, 12.8, 3.2, 3.2, 13.2), abs=2e-8)


def test_fully_occupied_orthogonal_grid_retains_every_turn_and_closed_insulation():
    row = coil(winding_turn_count=4, winding_wire_diameter_mm=1.)
    row["local_end_z_mm"] = 12.
    specs = round_winding_solids(row, {})
    assert len(specs) == 5
    meshes = meshes_from_solids(specs, 24)
    assert len(meshes) == 5 and all(len(mesh.faces) for mesh in meshes)
    solids = [_manifold(spec.shape, 24) for spec in specs]
    pack = _manifold(annulus(10., 12., 2., 4.), 24)
    assert sum(s.volume() for s in solids) == pytest.approx(pack.volume(), rel=2e-9)


def test_split_objective_preserves_passive_layers_and_original_field_envelope():
    from temsim.cad_objective_upper_hardware import objective_upper_hardware_solids
    from temsim.module_manifest import read_document
    path = Path(__file__).resolve().parents[1]/"configs/instruments/column/C3_ProbeCorrector.toml"
    document = read_document(path)
    rows = {row["key"]: row for row in document["parts"]}
    row = deepcopy(rows["objective_lens_excitation_coil"])
    original_passive = tuple(spec for spec in objective_upper_hardware_solids(row, rows)
                            if spec.material != "copper_winding")
    row.update(winding_turn_count=24, winding_wire_diameter_mm=1.,
               winding_interval_turn_counts=[7, 17])
    rows[row["key"]] = row
    geometry = resolve_round_winding_geometry(row, rows)
    receipt = geometry.to_metadata()
    assert receipt["field_envelope"]["inner_diameter_mm"] == row["mechanical_inner_diameter_mm"]
    assert receipt["conductor_allocations"][0]["inner_mm"] > row["mechanical_inner_diameter_mm"]/2
    assert receipt["conductor_allocations"][1]["inner_mm"] == row["mechanical_inner_diameter_mm"]/2
    assert geometry.passive_solids == original_passive
    assert geometry.layout.interval_turn_counts == (7, 17)
    specs = round_winding_solids(row, rows)
    assert tuple(spec for spec in specs if not spec.region.startswith("winding_")) == original_passive
    assert len([spec for spec in specs if spec.shape["kind"] == "torus"]) == 24
    assert len([spec for spec in specs if spec.shape["kind"] == "round_winding_insulation"]) == 2
    for turn in geometry.layout.turns:
        matches = [item for item in geometry.conductor_allocations
                   if item["start_mm"] < turn.z_mm < item["end_mm"]]
        item, = matches
        assert turn.radius_mm-.5 >= item["inner_mm"]
        assert turn.radius_mm+.5 <= item["outer_mm"]


def test_objective_fixed_photo_counts_survive_unknown_wire_then_later_cad_input():
    from temsim.module_manifest import read_document
    path = Path(__file__).resolve().parents[1]/"configs/instruments/column/C3_ProbeCorrector.toml"
    rows = {row["key"]: row for row in read_document(path)["parts"]}
    row = deepcopy(rows["objective_lens_excitation_coil"])
    row.pop("winding_wire_diameter_mm", None)
    row.update(winding_turn_count=1172, winding_interval_turn_counts=[586, 586])
    rows[row["key"]] = row
    assert resolve_round_winding_geometry(row, rows) is None
    row["winding_wire_diameter_mm"] = .5
    geometry = resolve_round_winding_geometry(row, rows)
    assert geometry.layout.interval_turn_counts == (586, 586)
    assert len(geometry.layout.turns) == 1172
    assert sum(turn.interval_index == 0 for turn in geometry.layout.turns) == 586
    assert sum(turn.interval_index == 1 for turn in geometry.layout.turns) == 586


def test_fixed_source_counts_reject_ambiguous_conductor_subdivision(monkeypatch):
    from temsim.cad_geometry import revolve
    row = coil(winding_turn_count=12, winding_wire_diameter_mm=.2,
               winding_interval_turn_counts=[12])
    def divided(*args):
        return tuple(SolidSpec(row["key"], name, "copper_winding",
            revolve(((a, 2.), (a, 4.), (b, 4.), (b, 2.))))
            for name, a, b in (("first",10.,13.),("second",14.,18.)))
    monkeypatch.setattr("temsim.cad_objective_upper_hardware.objective_upper_hardware_solids", divided)
    with pytest.raises(ValueError, match="one unambiguous conductor allocation"):
        resolve_round_winding_geometry(row, {})


def test_display_budget_rejects_explicitly_instead_of_omitting_turns(monkeypatch):
    spec = SolidSpec("coil", "winding_insulation_body", "insulator",
                     winding_insulation(0., 1., 2., 4., .01, ()),
                     parameters={"winding_turn_count": 10_000})
    monkeypatch.setattr("temsim.cad_mesh._manifold", lambda *args: pytest.fail("Budget must be checked first"))
    with pytest.raises(ValueError, match="No turns were omitted"):
        meshes_from_solids((spec,), 32)


def test_display_budget_accepts_two_thousand_turn_estimate_without_generating_meshes():
    spec = SolidSpec("coil", "winding_insulation_body", "insulator", {},
                     parameters={"winding_turn_count": 2048})
    _check_winding_mesh_budget((spec,), 32)


def test_detailed_winding_rejects_arbitrary_replacement_geometry():
    row = coil(winding_turn_count=12, winding_wire_diameter_mm=.4,
               model_3d={"base": {"kind": "box"}})
    with pytest.raises(ValueError, match="native annular base"):
        resolve_round_winding_geometry(row, {})
