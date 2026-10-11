"""Corrector drafts edit one physical owner and preserve that relation on save."""
import tomllib
from copy import deepcopy

import pytest

from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.part_model_document import PartModelDocument
from temsim.part_model_3d import part_dimension_specs
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.fixture
def draft(tmp_path):
    source = INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml"
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    return PartModelDocument(target)


def test_bound_geometry_is_readonly_without_disabling_all_fields(draft):
    editor = ManifestEditor(draft.path.parent)
    target = ManifestTarget(draft.path.name, "probe_dph2_deflector")
    fields = {f.path[-1]: f for f in editor.fields(target)}
    assert not fields["effective_thickness_mm"].editable
    assert fields["effective_thickness_mm"].owner_key == "probe_dph2_coils"
    assert fields["name"].editable
    with pytest.raises(ValueError, match="probe_dph2_coils"):
        editor.save(target, {("parts", target.part_key, "effective_thickness_mm"): 6.}, None)
    assert not draft.dirty


def test_coil_edit_updates_field_undo_redo_and_save_copy(draft, tmp_path):
    draft.set_dimension(("parts", "probe_dph2_coils", "effective_thickness_mm"), 3.)
    assert draft.part("probe_dph2_deflector")["effective_thickness_mm"] == 3.
    assert ("parts", "probe_dph2_deflector", "effective_thickness_mm") not in draft.updates()
    draft.undo()
    assert draft.part("probe_dph2_deflector")["effective_thickness_mm"] == 4.
    draft.redo()
    target = tmp_path / "copy.toml"
    draft.save_copy(target)
    assert not draft.dirty
    reloaded = PartModelDocument(target)
    assert reloaded.part("probe_dph2_deflector")["effective_thickness_mm"] == 3.
    assert tomllib.loads(target.read_text())["parts"]


def test_hp_move_carries_insert_and_field_without_independent_reposition(draft):
    hp = draft.part("probe_hp2_hexapole")
    z = hp["local_center_z_mm"]
    draft.place_component(hp["key"], z - 1.)
    for key in ("probe_dph2_coils", "probe_dph2_deflector"):
        assert draft.part(key)["local_center_z_mm"] == z - 1.
        with pytest.raises(ValueError, match="follows"):
            draft.place_component(key, z + 2.)
    derived = {p for p in draft.updates() if p[1] in {"probe_dph2_coils", "probe_dph2_deflector"}}
    assert not derived


def test_channel_cad_edit_rejected_and_dimension_view_routes_to_coil(draft):
    with pytest.raises(ValueError, match="control channel"):
        draft.set_model_3d("probe_dph2_deflector", None)
    fields = part_dimension_specs(draft.document, "probe_dph2_deflector")
    thickness = next(f for f in fields if f.path[-1] == "effective_thickness_mm")
    assert not thickness.editable
    assert "probe_dph2_coils" in thickness.reason
    pose = [f for f in fields if f.path[-1] == "rotation_x_mrad"]
    assert len(pose) == 1
    assert pose[0].path[1] == "probe_dph2_coils"
    assert pose[0].editable


def test_anchored_coil_length_survives_save_copy(draft, tmp_path):
    hp = draft.part("probe_hp2_hexapole")
    draft.place_component(hp["key"], hp["local_center_z_mm"] - 1.)
    draft.save_copy(tmp_path / "moved.toml")
    draft.set_dimension(("parts", "probe_dph2_coils", "length_mm"), 15.)
    draft.save_copy(tmp_path / "resized.toml")
    for key in ("probe_dph2_coils", "probe_dph2_deflector"):
        part = draft.part(key)
        assert part["length_mm"] == 15.
        assert part["local_end_z_mm"] - part["local_start_z_mm"] == 15.
        assert part["local_center_z_mm"] == draft.part("probe_hp2_hexapole")["local_center_z_mm"]
    assert not draft.dirty


def test_mechanical_copy_detaches_original_hp_and_channel(draft):
    draft.copy_component_from(draft.document, "probe_dph2_coils", "coil_copy", 670., include_children=False)
    row = draft.part("coil_copy")
    assert row["local_center_z_mm"] == 670.
    assert "optical_reference_host_key" not in row
    assert "physical_host_key" not in row
    assert row["mechanical_only"]
    assert row["mechanical_part_role"] == "custom_mechanical_copy"
    assert draft.part("probe_dph2_deflector")["physical_host_key"] == "probe_dph2_coils"


def test_hp_mechanical_subtree_copy_has_no_new_field_bindings(draft):
    from temsim.component_operations import copied_component_document
    from temsim.optics.field_hardware_binding import resolve_field_hardware_parts
    from temsim.part_model_3d import part_model_from_document
    copied, root = copied_component_document(draft.document, draft.document,
        "probe_hp2_hexapole", "hp_copy", 670.)
    resolved = resolve_field_hardware_parts(copied)
    copied_rows = [p for p in resolved["parts"] if p["key"].startswith(root)]
    assert len(copied_rows) == 3
    for row in copied_rows:
        assert row["mechanical_only"]
        assert "physical_host_key" not in row
        assert "optical_reference_host_key" not in row
        if row.get("layout_role") == "control_channel":
            assert not part_model_from_document(resolved, row["key"]).meshes


def test_native_winding_dimensions_use_common_editor_units_and_save_chain(draft, tmp_path):
    key = "probe_dph2_coils"
    original_channel = deepcopy(draft.part("probe_dph2_deflector"))
    original_hp = deepcopy(draft.part("probe_hp2_hexapole"))
    values = {"winding_layer_thickness_mm": (.28, "mm"),
              "winding_angular_span_deg": (110., "deg"), "winding_band_width_mm": (.9, "mm")}
    fields = {field.path[-1]: field for field in part_dimension_specs(draft.document, key)}
    manifest = ManifestEditor(draft.path.parent)
    editable = {f.path[-1]: f for f in manifest.fields(ManifestTarget(draft.path.name, key))}
    for name, (value, unit) in values.items():
        assert fields[name].editable and fields[name].unit == unit
        assert fields[name].meaning.category == "physical"
        assert fields[name].meaning.source_kind == "estimated"
        assert editable[name].editable
        draft.set_dimension(("parts", key, name), value)
    draft.undo()
    assert draft.part(key)["winding_band_width_mm"] == 1.16
    draft.redo()
    target = tmp_path / "native_coils.toml"
    draft.save_copy(target)
    loaded = PartModelDocument(target)
    for name, (value, _) in values.items():
        assert loaded.part(key)[name] == value
    assert loaded.part("probe_dph2_deflector") == original_channel
    assert loaded.part("probe_hp2_hexapole") == original_hp
    assert loaded.part(key)["physical_host_status"] == "modelled"
    assert loaded.part(key)["optical_reference_host_key"] == original_hp["key"]


def test_invalid_winding_edit_is_atomic_and_generic_manifest_save_rejects_it(draft):
    from temsim import module_manifest
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match="winding band width"):
        draft.set_dimension(("parts", "probe_dph2_coils", "winding_band_width_mm"), 20.)
    assert draft.document == before and not draft.dirty
    candidate = deepcopy(before)
    row = next(row for row in candidate["parts"] if row["key"] == "probe_dph2_coils")
    row["winding_angular_span_deg"] = 180.
    with pytest.raises(ValueError, match="winding angular span"):
        module_manifest.validate_document(candidate)
    editor = ManifestEditor(draft.path.parent)
    target = ManifestTarget(draft.path.name, "probe_dph2_coils")
    original_bytes = draft.path.read_bytes()
    with pytest.raises(ValueError, match="winding layer thickness"):
        editor.save(target, {("parts", target.part_key, "winding_layer_thickness_mm"): .8}, None)
    assert draft.path.read_bytes() == original_bytes


def test_legacy_profile_file_roundtrip_remains_readable(draft, tmp_path):
    from temsim.part_model_3d import part_model_from_document
    raw = draft.path.read_text(encoding="utf-8")
    raw = raw.replace('mechanical_profile = "deflector_coil_insert"',
                      'mechanical_profile = "corrector_deflector_coils"')
    raw = "\n".join(line for line in raw.splitlines() if not line.startswith("winding_")) + "\n"
    path = tmp_path / "legacy.toml"
    path.write_text(raw, encoding="utf-8")
    loaded = PartModelDocument(path)
    loaded.set_dimension(("parts", "probe_dph1_coils", "effective_thickness_mm"), 3.)
    saved = tmp_path / "legacy_saved.toml"
    loaded.save_copy(saved)
    reloaded = PartModelDocument(saved)
    assert reloaded.part("probe_dph1_coils")["mechanical_profile"] == "corrector_deflector_coils"
    assert reloaded.part("probe_dph1_deflector")["effective_thickness_mm"] == 3.
    assert len(part_model_from_document(reloaded.document, "probe_dph1_coils").meshes) == 4
