"""Saved beam-deflector registration slots, independent of optical steering."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from temsim.cad_catalog import part_solids, posed_solids
from temsim.cad_geometry import annulus, transformed
from temsim.cad_mesh import _manifold
from temsim.module_manifest import read_document, validate_document
from temsim.ordinary_deflector_geometry import SLOT_SWEEP_FIELD, support_dimensions
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector.toml"
KEY = "beam_deflector"
CHANGED = {f"coil_{station}_{suffix}" for station in (1, 2)
           for suffix in ("y_carrier", "interaxis_spacer")}


def document_rows():
    document = read_document(INSTRUMENT_CONFIG_ROOT/MODULE)
    return document, {row["key"]: row for row in document["parts"]}


def test_only_outer_carriers_and_spacers_gain_slots_with_unchanged_copper_and_pins():
    _, rows = document_rows()
    part = rows[KEY]
    old = {spec.region: spec for spec in part_solids({**part, SLOT_SWEEP_FIELD: 0.}, rows)}
    new = {spec.region: spec for spec in part_solids(part, rows)}
    assert len(old) == len(new) == 20 and set(new) == set(old)
    assert support_dimensions(part)["registration_slot_sweep_deg"] == 6.
    assert {region for region in new if new[region].shape != old[region].shape} == CHANGED
    for region, spec in new.items():
        if region in CHANGED:
            assert "outer carrier Y" in spec.note
            assert "not an OEM" in spec.note
            assert _manifold(spec.shape, 72).volume() < _manifold(old[region].shape, 72).volume()
        else:
            assert spec.shape == old[region].shape
    no_slot = {name: value for name, value in part.items() if name != SLOT_SWEEP_FIELD}
    assert [s.shape for s in part_solids(no_slot, rows)] == [s.shape for s in old.values()]


def test_all_twenty_solids_are_closed_disjoint_and_leave_the_material_bore_open():
    _, rows = document_rows()
    part = rows[KEY]
    d = support_dimensions(part)
    specs = part_solids(part, rows)
    solids = [_manifold(spec.shape, 72) for spec in specs]
    bore = _manifold(annulus(d["start_mm"]-1., d["end_mm"]+1., 0.,
                             d["material_inner_radius_mm"]), 72)
    for index, solid in enumerate(solids):
        assert len(solid.decompose()) == 1 and solid.volume() > 0.
        mesh = solid.to_mesh64()
        edges = np.sort(np.concatenate([mesh.tri_verts[:, pair] for pair in
                        ([0, 1], [1, 2], [2, 0])]), axis=1)
        assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
        assert (solid ^ bore).volume() < 1.e-8
        for other in solids[index+1:]:
            assert (solid ^ other).volume() < 1.e-8
    # Off-centre radial probes traverse only the outer adjustment slot; the
    # inner carrier still has material where its round hole does not reach.
    for stage, centre in enumerate(d["finite_coil_centres_mm"], 1):
        angle = np.deg2rad(47.5)
        for suffix, radii, should_hit in (
                ("y_carrier", d["y_carrier_radii_mm"], False),
                ("interaxis_spacer", d["interaxis_spacer_radii_mm"], False),
                ("x_carrier", d["x_carrier_radii_mm"], True)):
            radius = sum(radii)/2.
            probe = _manifold(annulus(centre-.02, centre+.02, 0., .02), 32).translate(
                (radius*np.cos(angle), radius*np.sin(angle), 0.))
            solid = solids[next(i for i, s in enumerate(specs) if s.region == f"coil_{stage}_{suffix}")]
            volume = (solid ^ probe).volume()
            assert (volume > 1.e-6) == should_hit


@pytest.mark.parametrize("value", [False, None, -1., float("nan"), float("inf"), 24., 360.])
def test_invalid_sweep_fails_loading_geometry_and_transactionally(value, tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    document, rows = document_rows()
    rows[KEY][SLOT_SWEEP_FIELD] = value
    with pytest.raises(ValueError):
        support_dimensions(rows[KEY])
    with pytest.raises(ValueError):
        validate_document(document)
    root = tmp_path/"instruments"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (MODULE,))
    draft = PartModelDocument(root/MODULE)
    before = deepcopy(draft.document)
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", KEY, SLOT_SWEEP_FIELD), value)
    assert draft.document == before and not draft.dirty and not draft.can_undo


def test_angle_edit_is_native_cad_consistent_undoable_and_preserves_effective_field(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    root = tmp_path/"instruments"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (MODULE,))
    draft = PartModelDocument(root/MODULE)
    before = deepcopy(draft.document)
    field = next(spec for spec in part_dimension_specs(draft.document, KEY)
                 if spec.path[-1] == SLOT_SWEEP_FIELD)
    assert field.editable and field.unit == "deg"
    draft.set_dimension(("parts", KEY, SLOT_SWEEP_FIELD), 8.)
    after = deepcopy(draft.part(KEY))
    expected = deepcopy(next(p for p in before["parts"] if p["key"] == KEY))
    expected[SLOT_SWEEP_FIELD] = 8.
    assert after == expected
    model = part_model_from_document(draft.document, KEY, include_children=False)
    assert len(model.meshes) == 20
    for mesh in model.meshes:
        if mesh.region in CHANGED:
            paths = {tuple(path) for surface in mesh.surfaces.values() for path in surface["parameter_paths"]}
            assert ("parts", KEY, SLOT_SWEEP_FIELD) in paths
    draft.undo()
    assert draft.document == before
    draft.redo()
    destination = tmp_path/"independent-beam-deflector.toml"
    draft.save_copy(destination)
    assert PartModelDocument(destination).part(KEY)[SLOT_SWEEP_FIELD] == 8.
    assert after["mechanical_coil_length_mm"] == after["effective_thickness_mm"] == 15.
    rows = {row["key"]: row for row in draft.document["parts"]}
    changed_effective = {**after, "effective_thickness_mm": 7.5}
    assert [spec.shape for spec in part_solids(changed_effective, rows)] == [
        spec.shape for spec in part_solids(after, rows)]


def test_installed_beam_support_uses_true_tube_profile_and_clear_neighbour_material():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cad_stage_holder import physical_liner_rows
    from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
    from temsim.optics.column import default_state

    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    rows = {part.key: part.data for part in assembly.parts}
    part = assembly.part(KEY)
    before = deepcopy(dict(part.data))
    assert (part.start_z_mm, part.end_z_mm) == (1020., 1060.)
    offset = part.center_z_mm-part.data["local_center_z_mm"]
    solids = [_manifold(transformed(spec.shape, offset=(0., 0., offset)), 72)
              for spec in posed_solids(part.data, rows)]
    tube_row = next(row for row in physical_liner_rows(assembly) if row["key"] == CONTINUOUS_TUBE_KEY)
    tube, = tube_solids(tube_row)
    tube_solid = _manifold(tube.shape, 72)
    assert min(solid.min_gap(tube_solid, 100.) for solid in solids) == pytest.approx(2.9, abs=.02)
    assert all((solid ^ tube_solid).volume() < 1.e-8 for solid in solids)
    # Test material-bearing children rather than a zero-material optical lens
    # parent, and include the aperture's currently selected hardware recipe.
    neighbours = [p for p in assembly.parts if p.key == "condenser_aperture_3"
                  or p.data.get("parent_key") == "adapter_lens"]
    checked = set()
    for neighbour in neighbours:
        shift = neighbour.center_z_mm-neighbour.data["local_center_z_mm"]
        for spec in posed_solids(neighbour.data, rows):
            material = _manifold(transformed(spec.shape, offset=(0., 0., shift)), 72)
            assert all((solid ^ material).volume() < 1.e-8 for solid in solids)
            assert min(solid.min_gap(material, 100.) for solid in solids) > 0.
            checked.add(neighbour.key)
    assert "condenser_aperture_3" in checked
    assert any(key.startswith("adapter_lens_") for key in checked)
    assert dict(part.data) == before
    assert state.descan_deflector._physical_host is state.image_diffraction_deflector


def test_actual_beam_slots_survive_strict_step_volume_and_metadata_roundtrip(tmp_path):
    """A native closed mesh alone missed a real pre-fused-slot STEP failure."""
    root = Path(__file__).resolve().parents[1]
    python = root/"environment/cad/Scripts/python.exe"
    if not python.is_file():
        pytest.skip("Optional isolated CadQuery environment is not installed")
    _, rows = document_rows()
    solids = [{**asdict(s), "color": s.color} for s in part_solids(rows[KEY], rows)]
    catalog = {"units": "mm", "modules": [{"key": "beam_slot_roundtrip", "solids": solids}]}
    # Import only the STEP adapter in its isolated CAD process; keep the
    # simulator/Qt runtime and its dependency graph out of this interpreter.
    code = """import importlib.util,json,sys
from pathlib import Path
source=Path(sys.argv[1])/'src/temsim/cad_step.py'
definition=importlib.util.spec_from_file_location('isolated_beam_cad',source)
cad=importlib.util.module_from_spec(definition)
definition.loader.exec_module(cad)
report=cad.export_catalog(json.load(sys.stdin),Path(sys.argv[2]))
print(json.dumps(report))
"""
    result = subprocess.run([str(python), "-c", code, str(root), str(tmp_path/"step")],
                            input=json.dumps(catalog), text=True, capture_output=True,
                            timeout=120, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "passed"
    for exported in report["exports"]:
        assert exported["leaf_count"] == 20
        assert all(row["solid_count"] == 1 for row in exported["records"])
        assert all(row["volume_difference_mm3"] <= max(1.e-12, 1.e-7*row["volume_mm3"])
                   for row in exported["records"])
