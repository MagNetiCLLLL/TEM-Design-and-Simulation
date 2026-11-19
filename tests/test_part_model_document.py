"""Real module-file drafts: dimensions, arrays, materials and safe persistence."""

from copy import deepcopy
from pathlib import Path
import shutil
import tomllib

import pytest

from temsim import module_manifest
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = "intermediate_lens_excitation_coil"
MODULE = "project_and_recording_system/EnergyFilter.toml"


@pytest.fixture
def document(tmp_path, request):
    source = INSTRUMENT_CONFIG_ROOT / MODULE
    original = source.read_bytes()
    path = tmp_path / "model.toml"
    # A standalone copy must materialize the composed module and detach links.
    PartModelDocument(source).save_copy(path)
    if "model_3d" in request.fixturenames:
        # These legacy CAD tests deliberately replace the annular coil by a
        # box. They do not describe a configured native round winding source.
        _legacy_winding_inputs(path)
    yield PartModelDocument(path)
    assert source.read_bytes() == original


def _legacy_winding_inputs(path):
    import tomli_w
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    parts = {part["key"]: part for part in data["parts"]}
    coil = parts[KEY]
    for field in ("winding_turn_count", "winding_interval_turn_counts", "winding_wire_diameter_mm", "winding_wire_cross_section_mm2"):
        coil.pop(field, None)
        coil.get("parameter_metadata", {}).pop(field, None)
    owner = parts[coil["field_source_key"]]
    owner.pop("ideal_coil_current_a_at_100_percent", None)
    owner.get("parameter_metadata", {}).pop("ideal_coil_current_a_at_100_percent", None)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


def test_declared_winding_rejects_nonannular_replacement_without_mutating_draft(document):
    if "winding_turn_count" not in document.part(KEY):
        pytest.fail("The installed default winding must retain its declared physical turns")
    before = deepcopy(document.document)
    history = deepcopy(document._history)
    with pytest.raises(ValueError, match="native annular"):
        document.set_model_3d(KEY, {"schema_version": 1,
            "base": {"kind": "box", "width_mm": 40., "height_mm": 30., "length_mm": 50.}})
    assert document.document == before and document._history == history


def test_turns_current_and_unknown_wire_size_form_one_atomic_reversible_draft(document):
    original = deepcopy(document.document)
    saved_bytes = document.path.read_bytes()
    document.configure_winding(KEY, winding_turn_count=1172,
        winding_interval_turn_counts=[1172], ideal_coil_current_a_at_100_percent=1575.233)
    coil = document.part(KEY)
    assert coil["winding_turn_count"] == 1172
    assert coil["winding_interval_turn_counts"] == [1172]
    assert "winding_wire_diameter_mm" not in coil
    assert document.part("intermediate_lens")["ideal_coil_current_a_at_100_percent"] == 1575.233
    assert document.path.read_bytes() == saved_bytes
    document.undo()
    assert document.document == original
    document.redo()
    document.save()
    restored = PartModelDocument(document.path)
    assert restored.part(KEY)["winding_interval_turn_counts"] == [1172]
    assert "winding_wire_diameter_mm" not in restored.part(KEY)


def test_single_interval_turn_count_follows_total_without_a_duplicate_input(document):
    document.configure_winding(KEY, winding_turn_count=12, winding_wire_diameter_mm=.2,
        winding_interval_turn_counts=[12], ideal_coil_current_a_at_100_percent=2.5)
    before = deepcopy(document.document)
    history = deepcopy(document._history)
    document.configure_winding(KEY, winding_turn_count=13,
        ideal_coil_current_a_at_100_percent=1000.)
    assert document.part(KEY)["winding_interval_turn_counts"] == [13]
    assert document.part(KEY)["winding_wire_diameter_mm"] == .2
    document.undo()
    assert document.document == before


def test_split_physical_counts_are_not_redistributed_by_a_total_turn_edit():
    document=PartModelDocument(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')
    before=deepcopy(document.document)
    with pytest.raises(ValueError,match='sum'):
        document.configure_winding('objective_lens_excitation_coil',winding_turn_count=1173,
            ideal_coil_current_a_at_100_percent=1575.2330285262728)
    assert document.document==before


def test_area_only_input_rebuilds_insulation_and_roundtrips_without_a_diameter_alias(document):
    from temsim.coil_winding import WIRE_AREA_FIELD
    from temsim.cad_round_winding import resolve_round_winding_geometry
    original=deepcopy(document.document)
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.5,
        ideal_coil_current_a_at_100_percent=2.5)
    coil=document.part(KEY)
    assert coil[WIRE_AREA_FIELD]==.5 and 'winding_wire_diameter_mm' not in coil
    by_key={p['key']:p for p in document.document['parts']}
    first=resolve_round_winding_geometry(coil,by_key)
    document.configure_winding(KEY,winding_turn_count=20,winding_wire_cross_section_mm2=1.,
        ideal_coil_current_a_at_100_percent=2.5)
    by_key={p['key']:p for p in document.document['parts']}
    second=resolve_round_winding_geometry(document.part(KEY),by_key)
    assert second.field_envelope==first.field_envelope
    assert second.layout.area.insulation_volume_mm3 < first.layout.area.insulation_volume_mm3
    document.undo();document.undo()
    assert document.document==original
    document.redo();document.redo()
    document.save()
    restored=PartModelDocument(document.path)
    assert restored.part(KEY)[WIRE_AREA_FIELD]==1.
    assert 'winding_wire_diameter_mm' not in restored.part(KEY)


def test_area_only_noop_preserves_exact_area_metadata_and_undo_history(document):
    area=.12345678901234567
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=area,
        ideal_coil_current_a_at_100_percent=2.5)
    document.save()
    before=deepcopy(document.document);history=deepcopy(document._history)
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=area,
        ideal_coil_current_a_at_100_percent=2.5)
    assert document.document==before and document._history==history
    assert not document.dirty


def test_turn_only_api_edit_retains_metal_area_and_current_without_double_entry(document):
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.25,
        ideal_coil_current_a_at_100_percent=2.5)
    document.configure_winding(KEY,winding_turn_count=20,
        ideal_coil_current_a_at_100_percent=2.5)
    coil=document.part(KEY)
    assert coil['winding_wire_cross_section_mm2']==.25
    assert coil['winding_interval_turn_counts']==[20]
    assert document.part('intermediate_lens')['ideal_coil_current_a_at_100_percent']==2.5


def test_same_area_dimension_value_preserves_source_alias_and_history(document):
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.25,
        ideal_coil_current_a_at_100_percent=2.5)
    document.save()
    before=deepcopy(document.document);history=deepcopy(document._history)
    document.set_dimension(('parts',KEY,'winding_wire_cross_section_mm2'),.25)
    assert document.document==before and document._history==history


def test_turn_dimension_edit_retains_envelope_area_and_current_and_updates_insulation(document):
    from temsim.cad_round_winding import resolve_round_winding_geometry
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.25,
        ideal_coil_current_a_at_100_percent=2.5)
    before=deepcopy(document.document)
    first=resolve_round_winding_geometry(document.part(KEY),{p['key']:p for p in document.document['parts']})
    document.set_dimension(('parts',KEY,'winding_turn_count'),20.)
    coil=document.part(KEY)
    assert coil['winding_turn_count']==20 and type(coil['winding_turn_count']) is int
    assert coil['winding_interval_turn_counts']==[20]
    assert coil['winding_wire_cross_section_mm2']==.25
    second=resolve_round_winding_geometry(coil,{p['key']:p for p in document.document['parts']})
    assert second.field_envelope==first.field_envelope
    assert second.layout.area.insulation_volume_mm3 < first.layout.area.insulation_volume_mm3
    owner=document.part('intermediate_lens')
    assert owner['ideal_coil_current_a_at_100_percent']==2.5
    assert owner['parameter_metadata']['ideal_coil_current_a_at_100_percent'][
        'calibration_status']=='invalidated_by_manual_turn_edit'
    document.undo();assert document.document==before


@pytest.mark.parametrize('turns',[0.,1.5,-1.])
def test_invalid_turn_dimension_edit_is_atomic(document,turns):
    before=deepcopy(document.document);history=deepcopy(document._history)
    with pytest.raises(ValueError):
        document.set_dimension(('parts',KEY,'winding_turn_count'),turns)
    assert document.document==before and document._history==history


def test_split_turn_dimension_edit_keeps_regional_counts_and_rejects_inconsistent_total():
    document=PartModelDocument(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')
    before=deepcopy(document.document);history=deepcopy(document._history)
    with pytest.raises(ValueError,match='sum'):
        document.set_dimension(('parts','objective_lens_excitation_coil','winding_turn_count'),1173.)
    assert document.document==before and document._history==history


def test_actual_area_edit_replaces_legacy_alias_and_old_size_source(document):
    _saved_estimated_winding(document)
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.25,
        ideal_coil_current_a_at_100_percent=2.5)
    coil=document.part(KEY)
    assert coil['winding_wire_cross_section_mm2']==.25
    assert 'winding_wire_diameter_mm' not in coil
    assert 'winding_wire_diameter_mm' not in coil['parameter_metadata']
    assert coil['parameter_metadata']['winding_wire_cross_section_mm2']['source_kind']=='user_defined'
    document.save()
    assert PartModelDocument(document.path).part(KEY)==coil


def test_area_can_be_cleared_from_a_batch_and_absence_survives_save(document):
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.25,
        ideal_coil_current_a_at_100_percent=2.5)
    document.configure_windings('intermediate_lens',{KEY:dict(winding_turn_count=12,
        winding_wire_cross_section_mm2=None)},ideal_coil_current_a_at_100_percent=2.5)
    document.save()
    coil=PartModelDocument(document.path).part(KEY)
    assert 'winding_wire_cross_section_mm2' not in coil
    assert 'winding_wire_diameter_mm' not in coil


def test_area_dimension_edit_rebuilds_cad_clears_alias_and_keeps_optical_controls(document):
    import math
    document.configure_winding(KEY,winding_turn_count=12,winding_wire_cross_section_mm2=.25,
        ideal_coil_current_a_at_100_percent=2.5)
    # A compatible old dual declaration must not block the canonical area edit.
    document.part(KEY)['winding_wire_diameter_mm']=math.sqrt(1/math.pi)
    document.part(KEY)['parameter_metadata']['winding_wire_diameter_mm']=dict(
        source_kind='measured',source_note='Legacy diameter fixture.')
    document.save()
    before=deepcopy(document.document)
    document.set_dimension(('parts',KEY,'winding_wire_cross_section_mm2'),.5)
    coil=document.part(KEY)
    assert coil['winding_wire_cross_section_mm2']==.5
    assert 'winding_wire_diameter_mm' not in coil
    assert 'winding_wire_diameter_mm' not in coil['parameter_metadata']
    assert coil['parameter_metadata']['winding_wire_cross_section_mm2']['source_kind']=='user_defined'
    assert document.part('intermediate_lens')==next(p for p in before['parts'] if p['key']=='intermediate_lens')
    changed=deepcopy(document.document)
    document.undo();assert document.document==before
    document.redo();assert document.document==changed
    document.save();assert PartModelDocument(document.path).document==changed


@pytest.mark.parametrize('size',[
    dict(winding_wire_cross_section_mm2=1000000.),
    dict(winding_wire_cross_section_mm2=.1,winding_wire_diameter_mm=2.),
])
def test_unfit_area_and_conflicting_size_aliases_reject_atomically(document,size):
    before=deepcopy(document.document);history=deepcopy(document._history)
    with pytest.raises(ValueError):
        document.configure_winding(KEY,winding_turn_count=12,
            ideal_coil_current_a_at_100_percent=2.5,**size)
    assert document.document==before and document._history==history


def test_fixed_interval_counts_can_be_cleared_only_by_explicit_input(document):
    document.configure_winding(KEY, winding_turn_count=12, winding_interval_turn_counts=[12],
        ideal_coil_current_a_at_100_percent=2.5)
    document.configure_winding(KEY, winding_turn_count=13, winding_interval_turn_counts=None,
        ideal_coil_current_a_at_100_percent=2.5)
    assert document.part(KEY)["winding_turn_count"] == 13
    assert "winding_interval_turn_counts" not in document.part(KEY)
    document.undo()
    assert document.part(KEY)["winding_interval_turn_counts"] == [12]


def _saved_estimated_winding(document):
    document.configure_winding(KEY, winding_turn_count=12, winding_wire_diameter_mm=.2,
        winding_interval_turn_counts=[12], ideal_coil_current_a_at_100_percent=2.5)
    coil = document.part(KEY)
    coil["parameter_metadata"] = {
        "winding_turn_count": dict(source_kind="estimated", source_note="Area-ratio estimate."),
        "winding_interval_turn_counts": dict(source_kind="estimated", source_note="Saved fixed allocation."),
        "winding_wire_diameter_mm": dict(source_kind="measured", source_note="Fixture conductor measurement."),
    }
    document.part("intermediate_lens")["parameter_metadata"] = {
        "ideal_coil_current_a_at_100_percent": dict(source_kind="estimated", source_note="Fixture field calibration."),
        "air_core_reference_ampere_turns": dict(source_kind="estimated", source_note="Unrelated retained reference."),
    }
    document.save()


def test_winding_noop_preserves_exact_inputs_provenance_and_history(document):
    _saved_estimated_winding(document)
    before = deepcopy(document.document)
    history = deepcopy(document._history)
    document.configure_winding(KEY, winding_turn_count=12, winding_wire_diameter_mm=.2,
        winding_interval_turn_counts=[12], ideal_coil_current_a_at_100_percent=2.5)
    assert document.document == before
    assert document._history == history
    assert not document.dirty


def test_manual_current_updates_only_its_source_and_remains_atomic_after_save(document):
    _saved_estimated_winding(document)
    before = deepcopy(document.document)
    document.configure_windings("intermediate_lens", {KEY: dict(winding_turn_count=12)},
        ideal_coil_current_a_at_100_percent=1575.2330285262728)
    source = document.part("intermediate_lens")["parameter_metadata"]
    assert source["ideal_coil_current_a_at_100_percent"]["source_kind"] == "user_defined"
    assert "calibrations are not asserted" in source["ideal_coil_current_a_at_100_percent"]["source_note"]
    previous_owner = next(p for p in before["parts"] if p["key"] == "intermediate_lens")
    assert source["air_core_reference_ampere_turns"] == previous_owner["parameter_metadata"]["air_core_reference_ampere_turns"]
    assert document.part(KEY) == next(p for p in before["parts"] if p["key"] == KEY)
    changed = deepcopy(document.document)
    document.undo()
    assert document.document == before
    document.redo()
    assert document.document == changed
    document.save()
    restored = PartModelDocument(document.path)
    assert restored.document == changed
    assert restored.part("intermediate_lens")["ideal_coil_current_a_at_100_percent"] == 1575.2330285262728


def test_manual_turn_change_marks_affected_sources_without_recalibrating_current(document):
    _saved_estimated_winding(document)
    previous_owner = deepcopy(document.part("intermediate_lens"))
    previous_wire = deepcopy(document.part(KEY)["parameter_metadata"]["winding_wire_diameter_mm"])
    document.configure_winding(KEY, winding_turn_count=13, winding_wire_diameter_mm=.2,
        winding_interval_turn_counts=[13], ideal_coil_current_a_at_100_percent=2.5)
    source = document.part(KEY)["parameter_metadata"]
    assert all(source[field]["source_kind"] == "user_defined" for field in
               ("winding_turn_count", "winding_interval_turn_counts"))
    assert "calibrations are not asserted" in source["winding_turn_count"]["source_note"]
    assert source["winding_wire_diameter_mm"] == previous_wire
    owner = document.part("intermediate_lens")
    current_field = "ideal_coil_current_a_at_100_percent"
    assert owner[current_field] == previous_owner[current_field]
    original_source = previous_owner["parameter_metadata"][current_field]
    current_source = owner["parameter_metadata"][current_field]
    assert current_source["source_kind"] == original_source["source_kind"]
    assert original_source["source_note"] in current_source["source_note"]
    assert "no field match is asserted" in current_source["source_note"]
    assert current_source["calibration_status"] == "invalidated_by_manual_turn_edit"
    assert owner["parameter_metadata"]["air_core_reference_ampere_turns"] == previous_owner["parameter_metadata"]["air_core_reference_ampere_turns"]
    document.save()
    assert PartModelDocument(document.path).part(KEY)["parameter_metadata"] == source


def test_cleared_optional_winding_inputs_and_their_sources_survive_save_reload(document):
    _saved_estimated_winding(document)
    document.configure_winding(KEY, winding_turn_count=13, winding_wire_diameter_mm=None,
        winding_interval_turn_counts=None, ideal_coil_current_a_at_100_percent=2.5)
    document.save()
    restored = PartModelDocument(document.path).part(KEY)
    for field in ("winding_wire_diameter_mm", "winding_interval_turn_counts"):
        assert field not in restored
        assert field not in restored["parameter_metadata"]
    assert restored["winding_turn_count"] == 13


def test_invalid_batch_preserves_every_value_source_and_undo_state(document):
    _saved_estimated_winding(document)
    before = deepcopy(document.document)
    history = deepcopy(document._history)
    with pytest.raises(ValueError, match="must belong"):
        document.configure_windings("intermediate_lens", {
            KEY: dict(winding_turn_count=13, winding_interval_turn_counts=[13]),
            "diffraction_lens_excitation_coil": dict(winding_turn_count=14),
        }, ideal_coil_current_a_at_100_percent=1000.)
    assert document.document == before
    assert document._history == history


def test_manual_input_source_can_replace_expanded_metadata_tables(document):
    import tomli_w
    _saved_estimated_winding(document)
    document.path.write_text(tomli_w.dumps(document.document), encoding="utf-8")
    assert "[parts.parameter_metadata." in document.path.read_text(encoding="utf-8")
    document.reload()
    document.configure_winding(KEY, winding_turn_count=12, winding_wire_diameter_mm=.2,
        winding_interval_turn_counts=[12], ideal_coil_current_a_at_100_percent=3.)
    document.save()
    restored = PartModelDocument(document.path)
    assert restored.part("intermediate_lens")["parameter_metadata"][
        "ideal_coil_current_a_at_100_percent"]["source_kind"] == "user_defined"
    assert restored.part(KEY)["parameter_metadata"]["winding_turn_count"]["source_kind"] == "estimated"


def test_dimensions_are_staged_across_parts_then_saved_with_comments_retained(document):
    original = document.path.read_bytes()
    before = deepcopy(document.document)
    center = document.part(KEY)["local_center_z_mm"]
    document.set_dimension(("parts", KEY, "length_mm"), 178.0)
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    document.set_dimension(("parts", "intermediate_lens_housing", "mechanical_inner_diameter_mm"), 176.0)
    assert document.path.read_bytes() == original
    assert document.part(KEY)["local_center_z_mm"] == center
    assert document.part(KEY)["local_start_z_mm"] == center - 89.0
    assert document.dirty
    document.save()
    assert not document.dirty
    loaded = module_manifest.read_document(document.path)
    assert loaded == document.document
    for key in ("module", "geometry", "ports"):
        assert loaded[key] == before[key]
    original_comments = [line for line in original.decode().splitlines() if line.lstrip().startswith("#")]
    assert all(line in document.path.read_text(encoding="utf-8") for line in original_comments)


def test_conflicting_dimensions_stay_as_draft_and_do_not_write(document):
    original = document.path.read_bytes()
    conflicting_outer = document.part("intermediate_lens_yoke")["mechanical_inner_diameter_mm"] + 10.0
    document.set_dimension(("parts", KEY, "mechanical_outer_diameter_mm"), conflicting_outer)
    with pytest.raises(ValueError, match="overlap"):
        document.save()
    assert document.dirty
    assert document.path.read_bytes() == original
    document.set_dimension(("parts", KEY, "mechanical_outer_diameter_mm"), 95.0)
    document.save()
    assert not document.dirty


def test_undo_redo_revert_and_two_saves_preserve_original_precision(document):
    original = deepcopy(document.document)
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    document.undo()
    assert document.document == original
    document.redo()
    document.save()
    document.set_dimension(("parts", KEY, "length_mm"), 178.0)
    document.revert()
    assert document.part(KEY)["length_mm"] == next(p for p in original["parts"] if p["key"] == KEY)["length_mm"]
    assert document.part(KEY)["mechanical_inner_diameter_mm"] == 70.0
    document.set_dimension(("parts", KEY, "length_mm"), 178.0)
    document.save()
    original_coil = next(p for p in original["parts"] if p["key"] == KEY)
    assert document.part(KEY)["mechanical_outer_diameter_mm"] == original_coil["mechanical_outer_diameter_mm"]


@pytest.mark.parametrize("value", [True, "70", float("nan"), float("inf")])
def test_invalid_input_does_not_mutate_draft(document, value):
    before = deepcopy(document.document)
    with pytest.raises(ValueError):
        document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), value)
    assert document.document == before


def test_physical_lens_pose_virtual_defaults_roundtrip_and_legacy_migration(document):
    import math
    from temsim.lens_pose import PHYSICAL_POSE_FIELDS
    # Exercise an old file with no canonical pose assignments, even though
    # newly bundled templates now expose explicit zero defaults.
    lines = document.path.read_text(encoding="utf-8").splitlines()
    document.path.write_text("\n".join(line for line in lines
        if line.partition("=")[0].strip() not in PHYSICAL_POSE_FIELDS) + "\n", encoding="utf-8")
    document = PartModelDocument(document.path)
    key = "intermediate_lens"
    before = deepcopy(document.part(key))
    document.set_model_3d(key, {"schema_version": 1, "base": {"kind": "existing"},
        "transform": {"scale_xy": [1.0, 1.0], "offset_mm": [0.03, -0.02, 0.04],
                      "rotation_deg": [0.1, -0.2, 0.3]}, "features": []})
    document.set_dimension(("parts", key, "rotation_y_mrad"), -1.25)
    part = document.part(key)
    assert part["rotation_x_mrad"] == pytest.approx(math.radians(0.1) * 1000)
    assert part["rotation_y_mrad"] == -1.25
    assert part["offset_x_mm"] == 0.03
    assert part["offset_z_mm"] == 0.04
    assert part["model_3d"]["transform"] == {"scale_xy": [1.0, 1.0]}
    assert all(part[name] == before[name] for name in before if name not in PHYSICAL_POSE_FIELDS)
    document.save()
    reopened = PartModelDocument(document.path)
    assert reopened.part(key) == document.part(key)
    assert reopened.part(key)["rotation_y_mrad"] == -1.25


@pytest.mark.parametrize("value", [True, "2.5", float("nan"), float("inf")])
def test_invalid_physical_lens_pose_is_atomic(document, value):
    before = deepcopy(document.document)
    with pytest.raises(ValueError, match="finite"):
        document.set_physical_pose("intermediate_lens", {"rotation_x_mrad": value})
    assert document.document == before
    invalid = deepcopy(before)
    next(part for part in invalid["parts"] if part["key"] == "intermediate_lens")["rotation_x_mrad"] = value
    with pytest.raises(ValueError, match="finite"):
        module_manifest.validate_document(invalid)


def test_lens_pose_and_axial_placement_are_one_undoable_operation(document):
    before = deepcopy(document.document)
    key = "intermediate_lens"
    document.place_component(key, document.part(key)["local_center_z_mm"],
                             physical_pose={"offset_x_mm": 0.025, "rotation_y_mrad": 2.5})
    assert document.part(key)["offset_x_mm"] == 0.025
    assert document.part(key)["rotation_y_mrad"] == 2.5
    document.undo()
    assert document.document == before
    document.redo()
    assert document.part(key)["rotation_y_mrad"] == 2.5


def test_manifest_editor_exposes_zero_pose_and_preserves_legacy_when_one_angle_changes():
    import math
    from temsim.lens_pose import PHYSICAL_POSE_FIELDS
    from temsim.manifest_editor import ManifestEditor, ManifestTarget, _complete_part_length_updates
    editor = ManifestEditor(INSTRUMENT_CONFIG_ROOT)
    fields = {field.path[-1]: field for field in editor.fields(ManifestTarget(MODULE, "intermediate_lens"))}
    assert fields["rotation_x_mrad"].editable
    assert fields["rotation_x_mrad"].value == 0.0
    assert fields["rotation_x_mrad"].meaning.category == "placement"
    source = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / MODULE)
    part = next(part for part in source["parts"] if part["key"] == "intermediate_lens")
    for name in PHYSICAL_POSE_FIELDS:
        part.pop(name, None)
    part["model_3d"] = {"schema_version": 1, "transform": {
        "offset_mm": [0.03, 0.02, -0.01], "rotation_deg": [0.1, 0.2, 0.3], "scale_xy": [1.0, 1.0]}}
    prefix = ("parts", "intermediate_lens")
    changes = _complete_part_length_updates(source, {(*prefix, "rotation_y_mrad"): 2.5})
    assert changes[(*prefix, "rotation_x_mrad")] == pytest.approx(math.radians(0.1) * 1000)
    assert changes[(*prefix, "rotation_y_mrad")] == 2.5
    assert changes[(*prefix, "offset_x_mm")] == 0.03
    assert changes[(*prefix, "model_3d")]["transform"] == {"scale_xy": [1.0, 1.0]}


def test_existing_array_dimensions_are_updated_by_index(document):
    key = "projector_lens_1_upper_pole"
    field = "magnetic_radial_profile_mm"
    expected = deepcopy(document.part(key)[field])
    expected[0][1] = 20.5
    document.set_dimension(("parts", key, field, 0, 1), 20.5)
    assert document.part(key)[field] == expected
    assert document.updates() == {("parts", key, field): expected}


def test_source_change_is_detected_but_save_copy_can_preserve_draft(document, tmp_path):
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    document.path.write_bytes(document.path.read_bytes() + b"\n# external edit\n")
    external = document.path.read_bytes()
    with pytest.raises(ValueError, match="changed outside"):
        document.save()
    path = document.path
    document.save_copy(tmp_path / "copy.toml")
    assert path.read_bytes() == external
    assert document.part(KEY)["mechanical_inner_diameter_mm"] == 70.0
    assert not document.dirty


def test_per_part_material_roundtrip_uses_new_optional_field(document):
    before = deepcopy(document.part(KEY))
    document.assign_material(KEY, "copper")
    assert document.path.read_bytes().find(b"material_regions") == -1
    document.save()
    assert document.part(KEY)["material_class"] == before["material_class"]
    assert document.part(KEY)["material_regions"]["body"]["material_key"] == "copper"
    assert document.part(KEY)["length_mm"] == before["length_mm"]
    document.undo()
    document.save()
    assert document.part(KEY)["material_regions"] == {}
    document.redo()
    document.save()
    assert document.part(KEY)["material_regions"]["body"]["material_key"] == "copper"


def test_existing_pole_bore_can_be_enlarged_without_changing_vacuum(document):
    key = "intermediate_lens_upper_pole"
    original_vacuum = document.part(key)["vacuum_inner_diameter_mm"]
    document.set_dimension(("parts", key, "mechanical_bore_diameter_mm"), 22.0)
    document.save()
    assert document.part(key)["mechanical_bore_diameter_mm"] == 22.0
    assert document.part(key)["vacuum_inner_diameter_mm"] == original_vacuum


def test_project_save_failure_keeps_draft(document):
    before = document.path.read_bytes()
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    def save(path, updates):
        assert path == document.path
        assert updates[("parts", KEY, "mechanical_inner_diameter_mm")] == 70.0
        raise ValueError("Assembly load failed")
    with pytest.raises(ValueError, match="Assembly load failed"):
        document.save(project_save=save)
    assert document.dirty
    assert document.path.read_bytes() == before


def test_non_module_file_and_fixed_centre_edits_are_rejected(tmp_path, document):
    path = tmp_path / "other.toml"
    path.write_text("[profile]\nname='other'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="module TOML"):
        PartModelDocument(path)
    with pytest.raises(ValueError, match="centre is fixed"):
        document.set_dimension(("parts", KEY, "local_center_z_mm"), 0.0)


@pytest.fixture
def model_3d():
    return {
        "schema_version": 1,
        "base": {"kind": "box", "width_mm": 40.0, "height_mm": 30.0, "length_mm": 50.0},
        "transform": {"scale_xy": [1.2, 0.85], "offset_mm": [4.0, -2.0, 3.0],
                      "rotation_deg": [12.0, 25.0, 40.0]},
        "features": [
            {"id": "hole_1", "kind": "hole", "axis": "z", "center_mm": [-8.0, 0.0, 0.0],
             "depth_mm": 60.0, "diameter_mm": 8.0, "enabled": True},
            {"id": "slot_1", "kind": "slot", "axis": "z", "center_mm": [9.0, 0.0, 0.0],
             "depth_mm": 60.0, "width_mm": 4.0, "length_mm": 12.0,
             "rotation_deg": 0.0, "enabled": True},
        ],
    }


def test_model_configuration_and_nested_dimensions_are_independent_of_solver_fields(document, model_3d):
    original = deepcopy(document.part(KEY))
    expected = deepcopy(model_3d)
    document.set_model_3d(KEY, model_3d)
    model_3d["transform"]["offset_mm"][0] = 999.0
    assert document.part(KEY)["model_3d"] == expected
    document.set_dimension(("parts", KEY, "model_3d", "base", "width_mm"), 42.5)
    document.set_dimension(("parts", KEY, "model_3d", "transform", "rotation_deg", 1), 37.5)
    document.set_dimension(("parts", KEY, "model_3d", "features", 0, "diameter_mm"), 9.0)
    expected["base"]["width_mm"] = 42.5
    expected["transform"]["rotation_deg"][1] = 37.5
    expected["features"][0]["diameter_mm"] = 9.0
    assert document.part(KEY) == {**original, "model_3d": expected}
    assert document.updates() == {("parts", KEY, "model_3d"): expected}
    document.undo()
    assert document.part(KEY)["model_3d"]["features"][0]["diameter_mm"] == 8.0
    document.redo()
    document.save()
    assert PartModelDocument(document.path).part(KEY) == {**original, "model_3d": expected}
    assert not document.dirty


@pytest.mark.parametrize("suffix,value", [
    (("base", "width_mm"), -1.0),
    (("transform", "scale_xy", 0), 0.0),
    (("features", 1, "length_mm"), 1.0),
    (("features", 0, "diameter_mm"), float("nan")),
    (("features", 0, "depth_mm"), 10**1000),
    (("features", -1, "width_mm"), 6.0),
    (("features", 30, "diameter_mm"), 6.0),
    (("features", 0, "enabled"), 1.0),
    (("transform", "missing", 0), 6.0),
], ids=["negative-size", "zero-scale", "short-slot", "nan", "overflow",
        "negative-index", "missing-index", "boolean-field", "missing-path"])
def test_model_numeric_errors_leave_document_and_history_unchanged(document, model_3d, suffix, value):
    document.set_model_3d(KEY, model_3d)
    before, history = deepcopy(document.document), deepcopy(document._history)
    with pytest.raises(ValueError):
        document.set_dimension(("parts", KEY, "model_3d", *suffix), value)
    assert document.document == before
    assert document._history == history


def test_model_feature_transactions_keep_stable_order_and_undo_as_one_change(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    document.update_model_3d(KEY, {"transform": {"offset_mm": [1.0, 2.0, 3.0]}})
    current = document.part(KEY)["model_3d"]
    assert current["base"] == model_3d["base"]
    assert current["features"] == model_3d["features"]
    assert current["transform"] == {**model_3d["transform"], "offset_mm": [1.0, 2.0, 3.0]}
    before = deepcopy(document.document)
    replacement = {**model_3d["features"][0], "diameter_mm": 9.0}
    document.upsert_model_feature(KEY, replacement)
    replacement["diameter_mm"] = 100.0
    assert [feature["id"] for feature in document.part(KEY)["model_3d"]["features"]] == ["hole_1", "slot_1"]
    assert document.part(KEY)["model_3d"]["features"][0]["diameter_mm"] == 9.0
    document.undo()
    assert document.document == before
    document.redo()
    before = deepcopy(document.document)
    document.remove_model_feature(KEY, "hole_1")
    assert [feature["id"] for feature in document.part(KEY)["model_3d"]["features"]] == ["slot_1"]
    document.undo()
    assert document.document == before
    with pytest.raises(ValueError, match="Unknown model feature"):
        document.remove_model_feature(KEY, "absent")
    assert document.document == before
    document.redo()
    document.save()
    document.undo()
    document.save()
    assert document.part(KEY)["model_3d"]["features"][0]["id"] == "hole_1"


@pytest.mark.parametrize("operation", ["update", "feature"])
def test_first_model_edit_includes_editable_transform_defaults(document, model_3d, operation):
    from temsim.part_model_features import default_model_3d

    defaults = default_model_3d(document.part(KEY))
    if operation == "update":
        document.update_model_3d(KEY, {"transform": {"offset_mm": [0.0, 1.0, 0.0]}})
        defaults["transform"]["offset_mm"] = [0.0, 1.0, 0.0]
    else:
        document.upsert_model_feature(KEY, model_3d["features"][0])
        defaults["features"] = [model_3d["features"][0]]
    assert document.part(KEY)["model_3d"] == defaults
    document.undo()
    assert "model_3d" not in document.part(KEY)


def test_model_removal_undo_redo_and_revert_round_trip_without_empty_table(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    document.save()
    document.set_model_3d(KEY, None)
    assert document.updates() == {("parts", KEY, "model_3d"): None}
    document.revert()
    assert document.part(KEY)["model_3d"] == model_3d
    document.set_model_3d(KEY, None)
    document.save()
    assert "model_3d" not in PartModelDocument(document.path).part(KEY)
    document.undo()
    document.save()
    assert document.part(KEY)["model_3d"] == model_3d
    document.redo()
    document.save()
    assert "model_3d" not in document.part(KEY)


@pytest.mark.parametrize("outside", [True, False])
def test_boolean_failure_preserves_schema_valid_draft_and_file(document, model_3d, outside):
    feature = model_3d["features"][0]
    feature["center_mm"] = [1000.0, 0.0, 0.0] if outside else [0.0, 0.0, 0.0]
    feature["diameter_mm"] = 8.0 if outside else 100.0
    model_3d["features"] = [feature]
    before = document.path.read_bytes()
    document.set_model_3d(KEY, model_3d)
    with pytest.raises(ValueError):
        document.save()
    assert document.path.read_bytes() == before
    assert document.dirty
    assert document.part(KEY)["model_3d"] == model_3d
    feature["enabled"] = False
    document.upsert_model_feature(KEY, feature)
    document.save()
    assert document.part(KEY)["model_3d"]["features"] == [feature]


def test_invalid_schema_replacement_and_duplicate_feature_are_atomic(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    before, history = deepcopy(document.document), deepcopy(document._history)
    with pytest.raises(ValueError):
        document.update_model_3d(KEY, {"base": {"kind": "existing"}})
    with pytest.raises(ValueError):
        document.upsert_model_feature(KEY, {**model_3d["features"][0], "width_mm": 2.0})
    invalid = deepcopy(model_3d)
    invalid["features"].append(deepcopy(invalid["features"][0]))
    with pytest.raises(ValueError):
        document.set_model_3d(KEY, invalid)
    assert document.document == before
    assert document._history == history


@pytest.mark.parametrize("returned,error", [(False, "did not save"), (None, "does not match")])
def test_project_callback_must_persist_the_validated_draft(document, model_3d, returned, error):
    document.set_model_3d(KEY, model_3d)
    original, baseline = document.path.read_bytes(), deepcopy(document._baseline)
    with pytest.raises(ValueError, match=error):
        document.save(project_save=lambda path, updates: returned)
    assert document.dirty
    assert document._baseline == baseline
    assert document.path.read_bytes() == original


def test_source_change_during_validation_prevents_project_callback(document, model_3d, monkeypatch):
    document.set_model_3d(KEY, model_3d)
    validate = document.validate
    callbacks = []

    def validate_and_external_edit():
        validate()
        document.path.write_bytes(document.path.read_bytes() + b"\n# external edit during validation\n")

    monkeypatch.setattr(document, "validate", validate_and_external_edit)
    with pytest.raises(ValueError, match="changed outside"):
        document.save(project_save=lambda *args: callbacks.append(args))
    assert not callbacks
    assert document.dirty
    assert document.path.read_bytes().endswith(b"# external edit during validation\n")


def test_external_reload_is_atomic_and_cannot_discard_model_draft(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    document.path.write_bytes(document.path.read_bytes() + b'\n# edited externally\n')
    external = document.path.read_bytes()
    with pytest.raises(ValueError, match="current draft"):
        document.reload()
    with pytest.raises(ValueError, match="changed outside"):
        document.save()
    assert document.part(KEY)["model_3d"] == model_3d
    document.revert()
    document.reload()
    assert document._source_bytes == external
    assert not document.can_undo and not document.dirty
    baseline = deepcopy(document.document)
    document.path.write_text("[[parts]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        document.reload()
    assert document.document == baseline
    assert document._source_bytes == external


def test_model_save_copy_preserves_full_features_after_external_source_edit(document, model_3d, tmp_path):
    document.set_model_3d(KEY, model_3d)
    source = document.path
    source.write_bytes(source.read_bytes() + b"\n# user's independent edit\n")
    external = source.read_bytes()
    destination = tmp_path / "retained_model.toml"
    document.save_copy(destination)
    assert source.read_bytes() == external
    assert document.path == destination.resolve()
    assert PartModelDocument(destination).part(KEY)["model_3d"] == model_3d
    assert not document.dirty


def test_nested_model_tables_and_feature_arrays_are_replaced_without_stale_values(document, model_3d):
    import tomli_w

    text = document.path.read_text(encoding="utf-8")
    # Exercise externally authored TOML, including quoted keys, array tables,
    # retained comments and an unrelated nested table after the model.
    text = text.replace(f'key = "{KEY}"', f"key = '{KEY}' # stable component key")
    lines = text.splitlines(keepends=True)
    _, end = module_manifest._part_span(lines, KEY)
    expanded = tomli_w.dumps({"parts": [{"model_3d": model_3d}]}).split("\n", 1)[1]
    expanded = expanded.replace("parts.model_3d", 'parts."model_3d"')
    expanded = "\n# keep CAD design comment\n" + expanded
    expanded += '\n[parts.custom_geometry_notes]\nlabel = "keep this table"\n'
    lines[end:end] = [expanded + "\n"]
    document.path.write_text("".join(lines), encoding="utf-8")
    loaded = PartModelDocument(document.path)
    before = deepcopy(loaded.part(KEY))
    loaded.remove_model_feature(KEY, "hole_1")
    loaded.set_dimension(("parts", KEY, "model_3d", "base", "height_mm"), 32.0)
    loaded.save()
    written = loaded.path.read_text(encoding="utf-8")
    assert "# keep CAD design comment" in written
    assert "# stable component key" in written
    assert loaded.part(KEY)["custom_geometry_notes"] == before["custom_geometry_notes"]
    assert loaded.part(KEY)["model_3d"]["features"] == [model_3d["features"][1]]
    assert loaded.part(KEY)["model_3d"]["base"]["height_mm"] == 32.0
    assert tomllib.loads(written) == loaded.document
    loaded.set_model_3d(KEY, None)
    loaded.save()
    assert "model_3d" not in PartModelDocument(loaded.path).part(KEY)
    assert loaded.part(KEY)["custom_geometry_notes"] == before["custom_geometry_notes"]


def test_model_replacement_respects_brackets_and_headers_inside_toml_values(document, model_3d):
    model_3d["features"][0]["id"] = "hole[1"
    model_3d["features"][1]["id"] = "slot]2"
    lines = document.path.read_text(encoding="utf-8").splitlines(keepends=True)
    start, _ = module_manifest._part_span(lines, KEY)
    lines[start:start] = [
        'design_notes = """Do not interpret the following text as a part:\n',
        '[[parts]]\n',
        'key = "imaginary_part"\n',
        '"""\n',
        'labels = [\n',
        '  ["model_3d"]\n',
        ']\n',
        "model_3d = " + module_manifest._format_toml_value(model_3d) + "\n",
    ]
    document.path.write_text("".join(lines), encoding="utf-8")
    loaded = PartModelDocument(document.path)
    before = deepcopy(loaded.part(KEY))
    loaded.remove_model_feature(KEY, "slot]2")
    loaded.save()
    assert loaded.part(KEY)["design_notes"] == before["design_notes"]
    assert loaded.part(KEY)["labels"] == [["model_3d"]]
    assert loaded.part(KEY)["model_3d"]["features"] == [model_3d["features"][0]]
    loaded.set_model_3d(KEY, None)
    loaded.save()
    assert {field: value for field, value in before.items() if field != "model_3d"} == loaded.part(KEY)


def test_model_configuration_uses_real_manifest_editor_save_and_validation(tmp_path, model_3d):
    from temsim.column.state_layout import layout_configuration_from_state
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.optics.column import default_state

    root = tmp_path / "instruments"
    source_bytes = (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes()
    from temsim.shared_tip import copy_catalog_tree
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    _legacy_winding_inputs(root.parent / "subassemblies/projector_stack.toml")
    editor, target = ManifestEditor(root), ManifestTarget(MODULE, KEY)
    configuration = layout_configuration_from_state(default_state())
    draft = PartModelDocument(root / MODULE)
    original = deepcopy(draft.part(KEY))
    draft.set_model_3d(KEY, model_3d)
    draft.save(project_save=lambda path, updates: editor.save(target, updates, configuration))
    assert draft.part(KEY) == {**original, "model_3d": model_3d}
    field = next(field for field in editor.fields(target) if field.label == "model_3d")
    assert field.editable and field.value == model_3d
    assert not draft.dirty
    before = draft.path.read_bytes()
    invalid = deepcopy(model_3d)
    invalid["features"][0]["center_mm"] = [999.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        editor.save(target, {("parts", KEY, "model_3d"): invalid}, configuration)
    assert draft.path.read_bytes() == before
    assert (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes() == source_bytes
