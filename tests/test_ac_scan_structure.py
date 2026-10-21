"""Installed AC support details retain the calibrated two-station scan geometry."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import part_solids, posed_solids
from temsim.cad_geometry import annulus, transformed
from temsim.cad_mesh import _manifold
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.module_manifest import read_document
from temsim.ordinary_deflector_geometry import SLOT_SWEEP_FIELD, support_dimensions
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector.toml"
KEY = "ac_deflector"
PREFIX = "deflector_support_"
SLOTTED = {f"coil_{stage}_{region}" for stage in (1, 2)
           for region in ("y_carrier", "interaxis_spacer")}
COPPER = {f"coil_{stage}_{axis}_winding_{side}" for stage in (1, 2)
          for axis in "xy" for side in (1, 2)}
SUPPORT_EDITS = {PREFIX + "radial_skin_thickness_mm": .22,
                 PREFIX + "casing_thickness_mm": .45,
                 PREFIX + "pin_diameter_mm": .42,
                 SLOT_SWEEP_FIELD: 8.}


def rows(document):
    return {p["key"]: p for p in document["parts"]}


def shapes(document, key=KEY):
    index = rows(document)
    return {s.region: s.shape for s in part_solids(index[key], index)}


@pytest.fixture
def document():
    return read_document(INSTRUMENT_CONFIG_ROOT / MODULE)


@pytest.fixture
def draft(tmp_path):
    from temsim.shared_tip import copy_catalog_tree
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return PartModelDocument(root / MODULE)


def test_default_ac_slots_change_only_outer_carriers_and_spacers(document):
    part = rows(document)[KEY]
    assert support_dimensions(part)["registration_slot_sweep_deg"] == 6.
    before = deepcopy(document)
    rows(before)[KEY][SLOT_SWEEP_FIELD] = 0.
    old, new = shapes(before), shapes(document)
    assert len(new) == len(old) == 20
    assert {region for region in new if new[region] != old[region]} == SLOTTED
    assert all(new[region] == old[region] for region in COPPER)
    # Missing optional sweep preserves the previous round-hole recipe.
    rows(before)[KEY].pop(SLOT_SWEEP_FIELD)
    assert shapes(before) == old
    # Other Probe configurations have different installations, without this
    # rigid-carrier opt-in; do not silently install a researched default there.
    for suffix in ("_BCOR", "_ImageCorrector"):
        other = read_document(INSTRUMENT_CONFIG_ROOT / f"column/C3_ProbeCorrector{suffix}.toml")
        assert support_dimensions(rows(other)[KEY]) is None


def test_slots_leave_inner_round_holes_and_copper_clear(document):
    part = rows(document)[KEY]
    d = support_dimensions(part)
    solids = {region: _manifold(shape, 72) for region, shape in shapes(document).items()}
    bore = _manifold(annulus(d["start_mm"]-1, d["end_mm"]+1,
                             0, d["material_inner_radius_mm"]), 72)
    for index, solid in enumerate(solids.values()):
        assert len(solid.decompose()) == 1 and solid.volume() > 0
        mesh = solid.to_mesh64()
        edges = np.sort(np.concatenate([mesh.tri_verts[:, pair]
                        for pair in ([0, 1], [1, 2], [2, 0])]), axis=1)
        assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
        assert (solid ^ bore).volume() < 1.e-8
        assert all((solid ^ other).volume() < 1.e-8
                   for other in list(solids.values())[index+1:])
    for stage, center in enumerate(d["finite_coil_centres_mm"], 1):
        for region, radii, filled in (("x_carrier", d["x_carrier_radii_mm"], True),
                                    ("y_carrier", d["y_carrier_radii_mm"], False),
                                    ("interaxis_spacer", d["interaxis_spacer_radii_mm"], False)):
            radius = sum(radii)/2
            angle = np.deg2rad(47.5)
            probe = _manifold(annulus(center-.02, center+.02, 0, .02), 32).translate(
                (radius*np.cos(angle), radius*np.sin(angle), 0))
            assert ((solids[f"coil_{stage}_{region}"] ^ probe).volume() > 1.e-6) == filled


def test_support_dimensions_follow_both_editors_and_preserve_runtime_planes(draft):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.parameter_impact import describe_parameter_impact

    original = deepcopy(draft.document)
    before = shapes(original)
    fields = {s.path[-1]: s for s in part_dimension_specs(original, KEY)}
    editor = ManifestEditor(draft.path.parents[1])
    target = ManifestTarget(MODULE, KEY)
    manifest_fields = {s.path[-1]: s for s in editor.fields(target)}
    for field, value in SUPPORT_EDITS.items():
        assert fields[field].editable and manifest_fields[field].editable
        assert fields[field].unit == ("deg" if field == SLOT_SWEEP_FIELD else "mm")
        impact = describe_parameter_impact(rows(original)[KEY], ("parts", KEY, field), by_key=rows(original))
        assert impact.status != "unknown" and "Ray trajectories" not in impact.affected_results
        old_shapes = shapes(draft.document)
        draft.set_dimension(("parts", KEY, field), value)
        assert shapes(draft.document) != old_shapes
    expected = deepcopy(original)
    rows(expected)[KEY].update(SUPPORT_EDITS)
    assert draft.document == expected
    assert all(shapes(expected)[region] == before[region] for region in COPPER)
    model = part_model_from_document(expected, KEY, include_children=False)
    for mesh in model.meshes:
        paths = {tuple(path) for surface in mesh.surfaces.values()
                 for path in surface["parameter_paths"]}
        if mesh.region in SLOTTED:
            assert ("parts", KEY, SLOT_SWEEP_FIELD) in paths
        if mesh.region in COPPER:
            assert ("parts", KEY, "mechanical_coil_length_mm") in paths
    # Save through the actual component editor, then restore via the manifest
    # editor. Both routes must retain every non-support field and field plane.
    draft.save()
    assert PartModelDocument(draft.path).document == expected
    editor.save(target, {("parts", KEY, field): rows(original)[KEY][field]
                         for field in SUPPORT_EDITS}, None)
    assert PartModelDocument(draft.path).document == original
    catalog = AssemblyCatalog(editor.root)
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())
    part = assembly.part(KEY)
    shift = part.center_z_mm-part.data["local_center_z_mm"]
    assert (state.ac_deflector.upper_z_mm, state.ac_deflector.lower_z_mm) == tuple(
        shift + z for z in rows(original)[KEY]["interaction_centers_local_z_mm"])
    assert state.ac_deflector.mechanical_coil_length_mm == 15.
    assert state.ac_deflector.mechanical_inter_coil_gap_mm == 10.
    assert state.descan_deflector._physical_host is state.image_diffraction_deflector


@pytest.mark.parametrize("value", [-1., False, float("nan"), 24.])
def test_invalid_ac_slot_edits_are_rejected_without_mutating_draft_or_file(draft, value):
    before = deepcopy(draft.document)
    data = draft.path.read_bytes()
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", KEY, SLOT_SWEEP_FIELD), value)
    assert draft.document == before and not draft.dirty and not draft.can_undo
    with pytest.raises(ValueError):
        ManifestEditor(draft.path.parents[1]).save(
            ManifestTarget(MODULE, KEY), {("parts", KEY, SLOT_SWEEP_FIELD): value}, None)
    assert draft.path.read_bytes() == data


def test_ac_coil_length_is_not_an_independent_passive_edit(draft):
    from temsim.parameter_impact import describe_parameter_impact

    data = draft.path.read_bytes()
    for field in ("mechanical_coil_length_mm", "mechanical_inter_coil_gap_mm"):
        impact = describe_parameter_impact(draft.part(KEY), ("parts", KEY, field),
                                           by_key=rows(draft.document))
        assert impact.status == "active"
        assert {"Ray trajectories", "Magnetic field", "Optical transfer"} <= set(impact.affected_results)
        assert "station separation" in impact.detail
    # A shorter physical coil can be previewed, but the calibrated finite
    # thickness, envelope, gap and interaction centers require a coherent edit.
    draft.set_dimension(("parts", KEY, "mechanical_coil_length_mm"), 14.)
    with pytest.raises(ValueError, match="field thickness|coil.*gap|interaction centres"):
        draft.save()
    assert draft.dirty and draft.path.read_bytes() == data
    with pytest.raises(ValueError, match="field thickness|coil.*gap|interaction centres"):
        ManifestEditor(draft.path.parents[1]).save(ManifestTarget(MODULE, KEY),
            {("parts", KEY, "mechanical_coil_length_mm"): 14.}, None)
    assert draft.path.read_bytes() == data
    draft.undo()
    assert not draft.dirty


def test_ac_copy_retains_slots_and_replacement_inactivates_native_support(draft):
    before = deepcopy(draft.document)
    original = part_model_from_document(before, KEY, include_children=False)
    key = draft.copy_component_from(draft, KEY, "ac_support_copy", 4000., include_children=False)
    copy = draft.part(key)
    assert copy["mechanical_only"] and copy["axial_vacuum_context_only"]
    assert copy["geometry_template_key"] == KEY
    assert support_dimensions(copy)["registration_slot_sweep_deg"] == 6.
    copied = part_model_from_document(draft.document, key, include_children=False)
    assert len(copied.meshes) == len(original.meshes) == 20
    delta = copy["local_center_z_mm"]-rows(before)[KEY]["local_center_z_mm"]
    source_shapes, copied_shapes = shapes(before), shapes(draft.document, key)
    for source, target in zip(original.meshes, copied.meshes):
        assert target.region == source.region
        # Boolean triangulation may reorder vertices after an axial move;
        # compare material volumes and bounds rather than triangle numbering.
        a = _manifold(source_shapes[source.region], 72).translate((0, 0, delta))
        b = _manifold(copied_shapes[target.region], 72)
        assert (a-b).volume() + (b-a).volume() < 1.e-7
        np.testing.assert_allclose(target.vertices.min(axis=0), source.vertices.min(axis=0)+(0, 0, delta), atol=1.e-8)
        np.testing.assert_allclose(target.vertices.max(axis=0), source.vertices.max(axis=0)+(0, 0, delta), atol=1.e-8)
        assert all(path[1] == key for surface in target.surfaces.values()
                   for path in surface["parameter_paths"] if path[0] == "parts")
    draft.undo()
    assert draft.document == before
    replacement = {"schema_version": 1,
                   "base": {"kind": "elliptic_cylinder", "length_mm": 40.,
                            "width_mm": 30., "height_mm": 30.},
                   "features": [{"id": "open_bore", "kind": "hole", "axis": "z",
                                 "center_mm": [0., 0., 0.], "diameter_mm": 21.,
                                 "depth_mm": 42., "enabled": True}]}
    draft.set_model_3d(KEY, replacement)
    assert support_dimensions(draft.part(KEY)) is None
    assert not set(shapes(draft.document)) & SLOTTED
    assert draft.part(KEY)[SLOT_SWEEP_FIELD] == 6.
    draft.set_model_3d(KEY, None)
    assert draft.document == before and shapes(draft.document) == shapes(before)


def test_installed_ac_uses_true_liner_and_neighbour_material(document):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cad_stage_holder import physical_liner_rows
    from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
    from temsim.optics.column import default_state

    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(state, catalog.default_selection())
    index = {part.key: part.data for part in assembly.parts}
    part = assembly.part(KEY)
    assert (part.start_z_mm, part.end_z_mm) == (1380., 1420.)

    def material(part):
        shift = part.center_z_mm-part.data["local_center_z_mm"]
        return [_manifold(transformed(spec.shape, offset=(0, 0, shift)), 72)
                for spec in posed_solids(part.data, index)]

    ac = material(part)
    tube_row = next(row for row in physical_liner_rows(assembly)
                    if row["key"] == CONTINUOUS_TUBE_KEY)
    tube, = tube_solids(tube_row)
    liner = _manifold(tube.shape, 72)
    assert min(solid.min_gap(liner, 100.) for solid in ac) == pytest.approx(1.56, abs=.02)
    assert all((solid ^ liner).volume() < 1.e-8 for solid in ac)
    neighbours = [p for p in assembly.parts if p.key in (
        "objective_upper_electronics_sleeve", "condenser_stigmator", "mini_condenser")
        or p.data.get("parent_key") in ("probe_tl12_lens", "mini_condenser")
        or p.key == "objective_lens_excitation_coil"]
    checked = set()
    for neighbour in neighbours:
        for solid in material(neighbour):
            assert all((station ^ solid).volume() < 1.e-8 for station in ac)
            assert min(station.min_gap(solid, 500.) for station in ac) > 0
            checked.add(neighbour.key)
    assert {"objective_upper_electronics_sleeve", "condenser_stigmator"} <= checked
    assert any(key.startswith("mini_condenser_") for key in checked)
    assert any(key.startswith("probe_tl12_") for key in checked)
    assert state.descan_deflector._physical_host is state.image_diffraction_deflector
