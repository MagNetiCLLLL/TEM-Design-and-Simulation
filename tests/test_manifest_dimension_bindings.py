"""Generic manifest edits obey the same material law as the 3-D editor."""
from copy import deepcopy
import tomllib

import pytest
import tomli_w
import numpy as np

from temsim.manifest_editor import ManifestEditor, ManifestTarget, _complete_part_length_updates
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from test_step_import_gui import actual_step


COLUMN = "column/C3_ProbeCorrector.toml"
RECORDING = "project_and_recording_system/NoEnergyFilter.toml"
INNER = "mechanical_inner_diameter_mm"
OUTER = "mechanical_outer_diameter_mm"
THICKNESS = "mechanical_coil_radial_thickness_mm"


def complete(document, updates):
    before = deepcopy((document, updates))
    completed = _complete_part_length_updates(document, updates)
    result = deepcopy(document)
    rows = {part["key"]: part for part in result["parts"]}
    for (section, key, field), value in completed.items():
        assert section == "parts"
        rows[key][field] = deepcopy(value)
    assert (document, updates) == before
    return result, completed


@pytest.mark.parametrize("module,key,field,factor", [
    (COLUMN, "condenser_lens_3", "pole_gap_mm", 1.01),
    (COLUMN, "probe_tl22_lens", "pole_gap_mm", 1.01),
    (RECORDING, "intermediate_lens", "pole_gap_mm", 1.01),
    (COLUMN, "condenser_lens_1", THICKNESS, 1.01),
    (COLUMN, "objective_lens", THICKNESS, 1.01),
    (COLUMN, "condenser_lens_3", THICKNESS, 1.01),
    (COLUMN, "condenser_lens_3_excitation_coil", INNER, 1.01),
    (COLUMN, "condenser_lens_3_excitation_coil", OUTER, 1.01),
    (COLUMN, "condenser_lens_1_excitation_coil", INNER, 1.01),
    (COLUMN, "probe_tl22_lens_excitation_coil", OUTER, .999),
])
def test_single_field_matches_public_part_document_transaction(module, key, field, factor):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    before = deepcopy(draft.document)
    value = draft.part(key)[field]*factor
    draft.set_dimension(("parts", key, field), value)
    candidate, _ = complete(before, {("parts", key, field): value})
    # Preserve the requested number, allowing the existing single-field
    # editor's subtraction roundoff only on the derived explicit field.
    expected = deepcopy(draft.document)
    row = next(part for part in expected["parts"] if part["key"] == key)
    assert row[field] == pytest.approx(value, rel=1e-14, abs=1e-14)
    row[field] = value
    assert candidate == expected
    validate_document(candidate)


@pytest.mark.parametrize("constraints", [(INNER, OUTER), (INNER, THICKNESS),
                                         (OUTER, THICKNESS), (INNER, OUTER, THICKNESS)])
@pytest.mark.parametrize("reverse", [False, True])
def test_consistent_simultaneous_radials_are_order_independent(constraints, reverse):
    document = read_document(INSTRUMENT_CONFIG_ROOT/COLUMN)
    rows = {part["key"]: part for part in document["parts"]}
    lens, coil = "condenser_lens_3", "condenser_lens_3_excitation_coil"
    inner = rows[coil][INNER]*1.001
    thickness = rows[lens][THICKNESS]*.999
    values = {INNER: inner, OUTER: inner+2*thickness, THICKNESS: thickness}
    updates = {("parts", lens if field == THICKNESS else coil, field): values[field]
               for field in (reversed(constraints) if reverse else constraints)}
    candidate, result = complete(document, updates)
    rows = {part["key"]: part for part in candidate["parts"]}
    assert rows[coil][INNER] == pytest.approx(inner)
    assert rows[coil][OUTER] == pytest.approx(inner+2*thickness)
    assert rows[lens][THICKNESS] == pytest.approx(thickness)
    housing = rows[lens+"_housing"]
    assert housing["mechanical_c3_end_inner_diameter_mm"] == pytest.approx(inner)
    assert housing["mechanical_c3_end_outer_diameter_mm"] == pytest.approx(inner+2*thickness)
    assert all(result[path] == value for path, value in updates.items())
    validate_document(candidate)


def test_complete_gap_and_pole_batch_accepts_consistency_and_rejects_explicit_conflict():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/COLUMN)
    before = deepcopy(draft.document)
    key = "condenser_lens_3"
    draft.set_dimension(("parts", key, "pole_gap_mm"), draft.part(key)["pole_gap_mm"]*1.01)
    previous = {part["key"]: part for part in before["parts"]}
    updates = {("parts", part["key"], field): value for part in draft.document["parts"]
               for field, value in part.items() if previous[part["key"]].get(field) != value}
    for batch in (updates, dict(reversed(list(updates.items())))):
        candidate, _ = complete(before, batch)
        assert candidate == draft.document
    bad = dict(updates)
    bad[("parts", key+"_upper_pole", "local_end_z_mm")] += .01
    with pytest.raises(ValueError, match="explicit value conflicts"):
        complete(before, bad)


@pytest.mark.parametrize("case", ["three_radials", "shared_radials", "shared_thickness", "passive_interface"])
def test_contradictory_explicit_constraints_are_rejected_without_overwriting(case):
    document = read_document(INSTRUMENT_CONFIG_ROOT/COLUMN)
    rows = {part["key"]: part for part in document["parts"]}
    if case == "three_radials":
        key = "condenser_lens_3"
        updates = {("parts", key, THICKNESS): rows[key][THICKNESS]*1.01,
                   ("parts", key+"_excitation_coil", INNER): rows[key+"_excitation_coil"][INNER],
                   ("parts", key+"_excitation_coil", OUTER): rows[key+"_excitation_coil"][OUTER]}
    elif case == "shared_radials":
        updates = {("parts", key+"_excitation_coil", OUTER): rows[key+"_excitation_coil"][OUTER]*factor
                   for key, factor in (("condenser_lens_1", 1.01), ("condenser_lens_2", 1.02))}
    elif case == "shared_thickness":
        updates = {("parts", key, THICKNESS): rows[key][THICKNESS]*factor
                   for key, factor in (("condenser_lens_1", 1.01), ("condenser_lens_2", 1.02))}
    else:
        updates = {("parts", "condenser_lens_3", THICKNESS): rows["condenser_lens_3"][THICKNESS]*1.01,
                   ("parts", "condenser_lens_3_housing", "mechanical_c3_end_outer_diameter_mm"):
                       rows["condenser_lens_3_housing"]["mechanical_c3_end_outer_diameter_mm"]}
    before = deepcopy((document, updates))
    with pytest.raises(ValueError, match="[Cc]onflict"):
        complete(document, updates)
    assert (document, updates) == before


@pytest.mark.parametrize("key,field", [
    ("condenser_lens_1", "pole_gap_mm"),
    ("objective_upper_pole", "pole_cone_angle_to_axis_deg"),
    ("condenser_aperture_2", "mechanical_bore_diameter_mm"),
    ("condenser_deflector", "length_mm"),
])
def test_reference_fields_are_visible_read_only_and_changed_values_rejected(key, field):
    editor = ManifestEditor(INSTRUMENT_CONFIG_ROOT)
    target = ManifestTarget(COLUMN, key)
    entry = next(item for item in editor.fields(target) if item.label == field)
    assert not entry.editable
    document = read_document(INSTRUMENT_CONFIG_ROOT/COLUMN)
    with pytest.raises(ValueError, match="[Rr]eference|allocation"):
        complete(document, {entry.path: entry.value*1.01})
    candidate, _ = complete(document, {entry.path: entry.value})
    assert candidate == document


def test_replacement_metadata_and_unrelated_optical_fields_do_not_trigger_native_binding():
    document = read_document(INSTRUMENT_CONFIG_ROOT/COLUMN)
    key = "condenser_lens_3"
    row = next(part for part in document["parts"] if part["key"] == key)
    replacement = dict(schema_version=1, base=dict(kind="box", width_mm=10., height_mm=10., length_mm=10.))
    updates = {("parts", key, "model_3d"): replacement,
               ("parts", key, THICKNESS): row[THICKNESS]*1.01,
               ("parts", key, "focal_length_mm"): 55.}
    candidate, completed = complete(document, updates)
    assert completed == updates
    assert next(part for part in candidate["parts"] if part["key"] == key)["model_3d"] == replacement


@pytest.fixture
def copied_catalog(tmp_path):
    from temsim.shared_tip import copy_catalog_tree
    root = tmp_path/"instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return root


def test_manifest_save_uses_existing_shared_storage_transaction(copied_catalog):
    root = copied_catalog
    source_before = read_document(INSTRUMENT_CONFIG_ROOT/COLUMN)
    # Exercise storage ownership explicitly; the current default embeds TL22
    # and only composes sample_stage, so it is not itself a TL22 sharing test.
    parent = root/COLUMN
    raw = tomllib.loads(parent.read_text(encoding="utf-8-sig"))
    key = "probe_tl22_lens"
    selected = [row for row in raw["parts"] if row["key"] == key or row.get("parent_key") == key]
    assert len(selected) == 6
    raw["parts"] = [row for row in raw["parts"] if row not in selected]
    dependency = root.parent/"subassemblies"/"manifest_tl22_fixture.toml"
    dependency.write_text(tomli_w.dumps(dict(coordinate_system="module_local_z_mm",
        module=dict(type="subassembly"), parts=selected)), encoding="utf-8")
    raw.setdefault("subassemblies", []).append(dict(key="manifest_tl22_fixture",
        file="../../subassemblies/manifest_tl22_fixture.toml",
        placement=dict(mode="fixed", origin_z_mm=0.)))
    parent.write_text(tomli_w.dumps(raw), encoding="utf-8")
    editor = ManifestEditor(root)
    target = ManifestTarget(COLUMN, "probe_tl22_lens")
    original = read_document(root/COLUMN)
    rows = {part["key"]: part for part in original["parts"]}
    raw_parent = (root/COLUMN).read_bytes()
    updates = {("parts", target.part_key, "pole_gap_mm"): rows[target.part_key]["pole_gap_mm"]*1.001,
               ("parts", target.part_key+"_excitation_coil", OUTER): rows[target.part_key+"_excitation_coil"][OUTER]*.999}
    expected, _ = complete(original, updates)
    originals = editor.save(target, updates, None)
    assert originals and all("subassemblies/" in path for path in originals)
    assert (root/COLUMN).read_bytes() == raw_parent
    assert read_document(root/COLUMN) == expected
    assert read_document(INSTRUMENT_CONFIG_ROOT/COLUMN) == source_before


@pytest.mark.parametrize("reverse", [False, True])
def test_batch_policy_uses_final_replacement_base_and_rechecks_restored_native(reverse):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ImageCorrector.toml")
    source = next(part["key"] for part in draft.document["parts"] if "winding_axial_length_mm" in part)
    key = draft.copy_component_from(draft.document, source, "manifest_copy", 2500., include_children=False)
    original = deepcopy(draft.document)
    replacement = dict(schema_version=1, base=dict(kind="box", width_mm=10., height_mm=10., length_mm=1.))
    updates = {("parts", key, "model_3d"): replacement, ("parts", key, "length_mm"): 1.}
    if reverse:
        updates = dict(reversed(list(updates.items())))
    candidate, _ = complete(original, updates)
    row = next(part for part in candidate["parts"] if part["key"] == key)
    assert row["length_mm"] == 1. and row["model_3d"] == replacement
    assert row["local_end_z_mm"]-row["local_start_z_mm"] == 1.
    with pytest.raises(ValueError, match="allocation"):
        complete(candidate, {("parts", key, "model_3d"): dict(schema_version=1, base=dict(kind="existing")),
                             ("parts", key, "length_mm"): 2.})


def test_save_rejects_conflicting_batch_before_any_shared_file_changes(copied_catalog):
    root = copied_catalog
    editor = ManifestEditor(root)
    target = ManifestTarget(COLUMN, "condenser_lens_3")
    before = {path: path.read_bytes() for path in root.parent.rglob("*.toml")}
    rows = {part["key"]: part for part in read_document(root/COLUMN)["parts"]}
    updates = {("parts", target.part_key, THICKNESS): rows[target.part_key][THICKNESS]*1.01,
               ("parts", target.part_key+"_excitation_coil", OUTER): rows[target.part_key+"_excitation_coil"][OUTER]}
    updates[("parts", target.part_key+"_excitation_coil", INNER)] = rows[target.part_key+"_excitation_coil"][INNER]
    with pytest.raises(ValueError, match="conflict"):
        editor.save(target, updates, None)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("key,field", [("condenser_lens_1", "pole_gap_mm"),
                                      ("condenser_deflector", "length_mm")])
def test_public_manifest_save_rejects_reference_change_without_writing(copied_catalog, key, field):
    editor = ManifestEditor(copied_catalog)
    target = ManifestTarget(COLUMN, key)
    entry = next(item for item in editor.fields(target) if item.label == field)
    before = {path: path.read_bytes() for path in copied_catalog.parent.rglob("*.toml")}
    with pytest.raises(ValueError, match="[Rr]eference|allocation"):
        editor.save(target, {entry.path: entry.value*1.01}, None)
    assert all(path.read_bytes() == data for path, data in before.items())


def add_explicit_fixture(root, shape="box", *, copied=False, base=None):
    from temsim.component_operations import make_component
    path = root/COLUMN
    if copied:
        draft = PartModelDocument(path)
        key = draft.copy_component_from(draft.document, "probe_dph2_coils", "manifest_body", 2500.,
                                        include_children=False)
        draft.update_model_3d(key, dict(base=dict(kind=shape, width_mm=10., height_mm=8., length_mm=4.)))
        row = deepcopy(draft.part(key))
    else:
        row = make_component(key="manifest_body", shape=shape, width_mm=10., height_mm=8.,
                             center_z_mm=2500., length_mm=4.)
    if base is not None:
        row["model_3d"]["base"] = deepcopy(base)
    raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    row["order"] = max(part["order"] for part in read_document(path)["parts"])+1
    raw["parts"].append(row)
    path.write_text(tomli_w.dumps(raw), encoding="utf-8")
    return row["key"]


@pytest.mark.parametrize("shape,copied", [("box", False), ("box", True), ("elliptic_cylinder", True)])
def test_public_custom_length_save_changes_real_mesh_and_preserves_native_metadata(copied_catalog, shape, copied):
    from temsim.part_model_3d import part_model_from_document
    root = copied_catalog
    key = add_explicit_fixture(root, shape, copied=copied)
    editor = ManifestEditor(root)
    target = ManifestTarget(COLUMN, key)
    original = read_document(root/COLUMN)
    old = next(row for row in original["parts"] if row["key"] == key)
    original_vertices = np.concatenate([mesh.vertices for mesh in
        part_model_from_document(original, key, include_children=False).meshes])
    fields = {item.label: item for item in editor.fields(target)}
    assert fields["length_mm"].editable and fields["name"].editable
    if copied:
        assert not fields[OUTER].editable
    editor.save(target, {("parts", key, "length_mm"): 6.35}, None)
    saved = read_document(root/COLUMN)
    row = next(row for row in saved["parts"] if row["key"] == key)
    assert row["model_3d"]["base"]["length_mm"] == row["length_mm"] == 6.35
    assert row["local_center_z_mm"] == old["local_center_z_mm"]
    vertices = np.concatenate([mesh.vertices for mesh in
        part_model_from_document(saved, key, include_children=False).meshes])
    assert np.ptp(original_vertices[:, 2]) == pytest.approx(4.)
    assert np.ptp(vertices[:, 2]) == pytest.approx(6.35)
    np.testing.assert_array_equal(vertices[:, :2], original_vertices[:, :2])
    if copied:
        assert row[OUTER] == old[OUTER]
        # UI-inactive recipe values remain an allowed metadata API edit.
        editor.save(target, {("parts", key, OUTER): old[OUTER]*1.01}, None)
        metadata = read_document(root/COLUMN)
        meshes = part_model_from_document(metadata, key, include_children=False).meshes
        np.testing.assert_array_equal(np.concatenate([mesh.vertices for mesh in meshes]), vertices)


def test_custom_length_rejects_conflicting_explicit_base_before_save(copied_catalog):
    key = add_explicit_fixture(copied_catalog)
    editor = ManifestEditor(copied_catalog)
    path = copied_catalog/COLUMN
    original = path.read_bytes()
    row = next(row for row in read_document(path)["parts"] if row["key"] == key)
    for reverse in (False, True):
        updates = {("parts", key, "length_mm"): 6., ("parts", key, "model_3d"): row["model_3d"]}
        if reverse:
            updates = dict(reversed(list(updates.items())))
        with pytest.raises(ValueError, match="base length conflicts"):
            editor.save(ManifestTarget(COLUMN, key), updates, None)
        assert path.read_bytes() == original


@pytest.mark.parametrize("model", [[], dict(schema_version=1, base=[])])
def test_invalid_explicit_model_is_a_value_error_and_preserves_all_storage(copied_catalog, model):
    key = add_explicit_fixture(copied_catalog)
    editor = ManifestEditor(copied_catalog)
    before = {path: path.read_bytes() for path in copied_catalog.parent.rglob("*.toml")}
    with pytest.raises(ValueError, match="model_3d.*table"):
        editor.save(ManifestTarget(COLUMN, key), {("parts", key, "model_3d"): model,
                    ("parts", key, "length_mm"): 6.}, None)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_public_imported_step_length_save_scales_mesh_but_preserves_source_asset(copied_catalog, actual_step):
    from temsim.part_model_3d import part_model_from_document
    from temsim.step_assets import base_from_asset
    _, asset = actual_step
    base = base_from_asset(asset)
    key = add_explicit_fixture(copied_catalog, base=base)
    editor = ManifestEditor(copied_catalog)
    editor.save(ManifestTarget(COLUMN, key), {("parts", key, "length_mm"): 6.35}, None)
    document = read_document(copied_catalog/COLUMN)
    row = next(part for part in document["parts"] if part["key"] == key)
    assert row["model_3d"]["base"]["asset_json"] == base["asset_json"]
    assert row["model_3d"]["base"]["length_mm"] == 6.35
    vertices = np.concatenate([mesh.vertices for mesh in
        part_model_from_document(document, key, include_children=False).meshes])
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [base["width_mm"], base["height_mm"], 6.35])


def test_removing_replacement_in_same_batch_restores_native_dimension_binding():
    document = read_document(INSTRUMENT_CONFIG_ROOT/COLUMN)
    key = "condenser_lens_3"
    row = next(part for part in document["parts"] if part["key"] == key)
    row["model_3d"] = dict(schema_version=1, base=dict(kind="box", width_mm=10., height_mm=10., length_mm=10.))
    value = row[THICKNESS]*1.01
    candidate, updates = complete(document, {("parts", key, "model_3d"): None,
                                             ("parts", key, THICKNESS): value})
    # The text writer consumes None as deletion, while this simple helper
    # applies fields literally. Inspect the completed child interfaces here.
    assert updates[("parts", key, "model_3d")] is None
    rows = {part["key"]: part for part in candidate["parts"]}
    assert rows[key+"_excitation_coil"][OUTER] == pytest.approx(rows[key+"_excitation_coil"][INNER]+2*value)


@pytest.mark.parametrize("key,operating_fields", [
    ("objective_lens", ("nominal_focal_length_mm", "spherical_aberration_mm")),
    ("probe_hp2_hexapole", ("effective_length_mm",)),
])
def test_replacement_hides_only_inactive_material_dimensions_not_optical_controls(copied_catalog, key, operating_fields):
    path = copied_catalog/COLUMN
    raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    row = next(part for part in raw["parts"] if part["key"] == key)
    row["model_3d"] = dict(schema_version=1, base=dict(kind="box", width_mm=10., height_mm=10., length_mm=10.))
    path.write_text(tomli_w.dumps(raw), encoding="utf-8")
    editor = ManifestEditor(copied_catalog)
    fields = {item.label: item for item in editor.fields(ManifestTarget(COLUMN, key))}
    assert not fields[OUTER].editable
    assert fields["name"].editable
    assert all(fields[field].editable for field in operating_fields)
    document = read_document(path)
    updates = {fields[field].path: fields[field].value*1.01 for field in operating_fields}
    _, completed = complete(document, updates)
    assert completed == updates
