"""Saved internal detail stays editable, transactional and separate from fields."""
from copy import deepcopy

import pytest

from temsim.module_manifest import validate_document
from temsim.parameter_impact import describe_parameter_impact
from temsim.parameter_semantics import describe_parameter
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.mark.parametrize("module,key,field", [
    ("gun/FEG.toml", "feg_stigmator", "gun_stigmator_insulation_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "c1_c2_pole_piece_cartridge", "mechanical_cartridge_spacer_start_local_z_mm"),
    ("column/C3_ProbeCorrector.toml", "condenser_aperture_2", "aperture_support_rod_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "condenser_stigmator", "stigmator_support_insulation_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "condenser_lens_3_housing", "mechanical_c3_end_cooling_plate_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "adapter_lens_housing", "mechanical_adapter_end_cooling_plate_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "condenser_aperture_3", "aperture_support_rod_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "beam_deflector", "deflector_support_registration_slot_sweep_deg"),
    ("column/C3_ProbeCorrector.toml", "probe_tl22_lens_housing", "mechanical_tl22_end_cooling_plate_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "probe_dph2_coils", "winding_support_end_cover_thickness_mm"),
    ("column/C3_ProbeCorrector.toml", "probe_hp2_hexapole", "hexapole_support_former_wall_thickness_mm"),
])
def test_internal_dimensions_are_visible_and_described_without_claiming_field_coupling(module, key, field):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    path = ("parts", key, field)
    part = draft.part(key)
    specs = part_dimension_specs(draft.document, key)
    assert path in {spec.path for spec in specs if spec.editable}
    meaning = describe_parameter(part, path)
    assert meaning.category == "physical" and meaning.source_kind == "estimated"
    for mode in ("ideal", "real", "custom"):
        impact = describe_parameter_impact(part, path, simulation_mode=mode)
        assert impact.status == "inactive"
        expected = {"display", "geometry"}
        if field.startswith(("mechanical_cartridge_", "mechanical_c3_end_", "mechanical_adapter_end_", "mechanical_tl22_end_")):
            expected.add("field_map_identity")
        assert set(impact.effects) == expected
        assert "Magnetic field" not in impact.affected_results


def test_cartridge_subdivision_reports_imported_map_revalidation_without_claiming_new_field_solver():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ProbeCorrector.toml")
    key = "c1_c2_pole_piece_cartridge"
    impact = describe_parameter_impact(draft.part(key),
        ("parts", key, "mechanical_cartridge_spacer_start_local_z_mm"), simulation_mode="custom",
        descriptors={"condenser_lens_1": {"source_path": "existing_map.npz"}})
    assert impact.status == "configuration_required"
    assert "field_map_identity" in impact.effects and "magnetic_field" not in impact.effects


@pytest.mark.parametrize("module,key,field,value", [
    ("gun/FEG.toml", "feg_stigmator", "gun_stigmator_insulation_thickness_mm", 20.),
    ("column/C3_ProbeCorrector.toml", "c1_c2_pole_piece_cartridge", "mechanical_cartridge_spacer_start_local_z_mm", 301.),
    ("project_and_recording_system/EnergyFilter.toml", "energy_filter_zebra", "strip_active_height_mm", 28.672),
    ("project_and_recording_system/EnergyFilter.toml", "energy_filter_zebra", "mechanical_housing_wall_mm", 40.),
    ("column/C3_ProbeCorrector.toml", "condenser_aperture_2", "aperture_support_rod_start_y_mm", 5.),
    ("column/C3_ProbeCorrector.toml", "condenser_stigmator", "stigmator_support_insulation_thickness_mm", 20.),
    ("column/C3_ProbeCorrector.toml", "condenser_lens_3_housing", "mechanical_c3_end_cooling_plate_thickness_mm", 20.),
    ("column/C3_ProbeCorrector.toml", "condenser_lens_3_excitation_coil", "mechanical_inner_diameter_mm", 160.),
    ("column/C3_ProbeCorrector.toml", "adapter_lens_housing", "mechanical_adapter_end_cooling_plate_thickness_mm", 2.),
    ("column/C3_ProbeCorrector.toml", "adapter_lens_excitation_coil", "mechanical_inner_diameter_mm", 100.),
    ("column/C3_ProbeCorrector.toml", "condenser_aperture_3", "aperture_support_rod_start_y_mm", 5.),
    ("column/C3_ProbeCorrector.toml", "beam_deflector", "deflector_support_registration_slot_sweep_deg", 90.),
    ("column/C3_ProbeCorrector.toml", "probe_tl22_lens_housing", "mechanical_tl22_end_cooling_plate_thickness_mm", 1.5),
    ("column/C3_ProbeCorrector.toml", "probe_dph2_coils", "winding_support_end_cover_thickness_mm", 2.),
    ("column/C3_ProbeCorrector.toml", "probe_hp2_hexapole", "hexapole_support_former_wall_thickness_mm", 1.),
])
def test_invalid_internal_edit_preserves_draft_history_and_source(module, key, field, value):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    before, raw = deepcopy(draft.document), draft.path.read_bytes()
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", key, field), value)
    assert draft.document == before and not draft.dirty and not draft.can_undo
    assert draft.path.read_bytes() == raw


@pytest.mark.parametrize("lens,prefix,value", (
    ("condenser_lens_3", "mechanical_c3_end_", 98.),
    ("adapter_lens", "mechanical_adapter_end_", 80.),
))
def test_valid_coil_id_edit_synchronizes_end_layers_and_parent_radial_size(lens, prefix, value):
    from temsim.cad_catalog import part_solids

    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    before, raw = deepcopy(draft.document), draft.path.read_bytes()
    coil, housing = lens + "_excitation_coil", lens + "_housing"
    rows = {p["key"]: p for p in draft.document["parts"]}
    old_shapes = {key: tuple(s.shape for s in part_solids(rows[key], rows)) for key in (coil, housing)}
    outer = rows[coil]["mechanical_outer_diameter_mm"]
    expected_parent = deepcopy(rows[lens])
    expected_parent["mechanical_coil_radial_thickness_mm"] = (outer - value) / 2.
    draft.set_dimension(("parts", coil, "mechanical_inner_diameter_mm"), value)
    assert draft.part(coil)["mechanical_outer_diameter_mm"] == outer
    assert draft.part(coil)["mechanical_inner_diameter_mm"] == value
    assert draft.part(housing)[prefix + "inner_diameter_mm"] == value
    # The only parent change is its linked physical winding thickness; no
    # excitation, optical gap or effective field parameter is recalibrated.
    assert draft.part(lens) == expected_parent
    rows = {p["key"]: p for p in draft.document["parts"]}
    for key in (coil, housing):
        assert tuple(s.shape for s in part_solids(rows[key], rows)) != old_shapes[key]
    validate_document(draft.document)
    after = deepcopy(draft.document)
    draft.undo()
    assert draft.document == before
    draft.redo()
    assert draft.document == after
    assert draft.path.read_bytes() == raw


def test_saved_camera_shape_cannot_disagree_with_the_rendered_footprint():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"project_and_recording_system/NoEnergyFilter.toml")
    draft.part("camera")["mechanical_footprint_shape"] = "disk"
    with pytest.raises(ValueError, match="must be square"):
        validate_document(draft.document)
    with pytest.raises(ValueError, match="must be square"):
        part_model_from_document(draft.document, "camera")


def test_copied_cartridge_interfaces_follow_placement_resize_and_undo():
    from temsim.cad_condenser_cartridge import cartridge_solids
    source = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ProbeCorrector.toml")
    target = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"project_and_recording_system/NoEnergyFilter.toml")
    key = target.copy_component_from(source.document, "c1_c2_pole_piece_cartridge",
                                     "copied_cartridge", 2400., include_children=False)
    first, last = "mechanical_cartridge_spacer_start_local_z_mm", "mechanical_cartridge_spacer_end_local_z_mm"
    row = target.part(key)
    assert (row[first], row[last]) == (2340., 2360.)
    assert "magnetic_lens_keys" not in row
    assert row["copy_provenance"]["source_magnetic_lens_keys"] == ["condenser_lens_1", "condenser_lens_2"]
    impact = describe_parameter_impact(row, ("parts", key, first), simulation_mode="custom")
    assert "field_map_identity" not in impact.effects
    target.place_component(key, 2700.)
    assert (target.part(key)[first], target.part(key)[last]) == (2640., 2660.)
    target.set_dimension(("parts", key, "length_mm"), 600.)
    row = target.part(key)
    assert (row[first], row[last]) == (2580., 2620.)
    assert len(cartridge_solids(row)) == 3
    target.undo()
    assert (target.part(key)[first], target.part(key)[last]) == (2640., 2660.)


@pytest.mark.parametrize("module,key", [
    ("gun/FEG.toml", "feg_stigmator"),
    ("column/C3_ProbeCorrector.toml", "c1_c2_pole_piece_cartridge"),
    ("column/C3_ProbeCorrector.toml", "condenser_aperture_2"),
    ("column/C3_ProbeCorrector.toml", "condenser_stigmator"),
    ("column/C3_ProbeCorrector.toml", "condenser_lens_3_housing"),
    ("column/C3_ProbeCorrector.toml", "adapter_lens_housing"),
    ("column/C3_ProbeCorrector.toml", "condenser_aperture_3"),
    ("column/C3_ProbeCorrector.toml", "probe_tl22_lens_housing"),
    ("column/C3_ProbeCorrector.toml", "probe_hp2_hexapole"),
    ("column/C3_ProbeCorrector.toml", "probe_dph2_coils"),
])
def test_replacement_step_material_region_keeps_priority_over_saved_detail_profile(module, key):
    from temsim.cad_catalog import is_reconstruction_region
    from temsim.part_materials import configured_region_colour
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/module)
    row = deepcopy(draft.part(key))
    row["model_3d"] = {"base": {"kind": "step"}}
    assert is_reconstruction_region(row, "step_0")
    assert configured_region_colour(row, "step_0", (.3, .4, .5, 1.)) == (.3, .4, .5, 1.)


@pytest.mark.parametrize("lens,prefix,value", (
    ("condenser_lens_3", "mechanical_c3_end_", 2.2),
    ("adapter_lens", "mechanical_adapter_end_", .6),
    ("probe_tl22_lens", "mechanical_tl22_end_", .6),
))
def test_lens_internal_plate_edit_requires_revalidating_imported_map_without_adding_a_field(lens, prefix, value):
    from dataclasses import replace
    from temsim.optics.column import default_state
    from temsim.physics.lens_field_provider import lens_geometry_binding
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ProbeCorrector.toml")
    key, field = lens + "_housing", prefix + "cooling_plate_thickness_mm"
    impact = describe_parameter_impact(draft.part(key), ("parts", key, field), simulation_mode="custom",
        descriptors={lens: {"source_path": "existing_map.npz"}})
    assert impact.status == "configuration_required"
    assert "field_map_identity" in impact.effects and "magnetic_field" not in impact.effects
    state = default_state()
    assembly = state._resolved_assembly
    before = lens_geometry_binding(state, lens).geometry_fingerprint
    updated = tuple(replace(part, data={**part.data, field: value}) if part.key == key else part
                    for part in assembly.parts)
    state._resolved_assembly = replace(assembly, parts=updated)
    assert lens_geometry_binding(state, lens).geometry_fingerprint != before


@pytest.mark.parametrize("key", ("condenser_lens_3_housing", "adapter_lens_housing", "probe_tl22_lens_housing"))
def test_lens_detail_layers_do_not_lock_the_original_housing_material(key):
    from temsim.cad_catalog import is_reconstruction_region
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT/"column/C3_ProbeCorrector.toml")
    part = draft.part(key)
    assert not is_reconstruction_region(part, "body")
    for region in ("upper_cooling_plate", "upper_coil_insulation", "lower_cooling_plate", "lower_coil_insulation"):
        assert is_reconstruction_region(part, region)
