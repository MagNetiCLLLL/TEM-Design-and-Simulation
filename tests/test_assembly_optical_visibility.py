"""Functional 3D filtering must not mistake mechanical hosts for passive parts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from temsim.assembly_optical_visibility import optical_mesh_visible


def visible(row, region="body", *, parts=None, wireframe=False):
    return optical_mesh_visible(SimpleNamespace(region=region, wireframe=wireframe),
                                row, parts_by_key=parts)


@pytest.mark.parametrize("profile", [
    "magnetic_pole_piece", "magnetic_excitation_coil", "magnetic_lens_yoke",
    "c1_c2_pole_piece_cartridge",
])
def test_lens_material_children_remain_visible_despite_mechanical_only(profile):
    row = dict(key="part", name="arbitrary renamed part", mechanical_only=True,
               mechanical_profile=profile, default_excitation_percent=0)
    assert visible(row)
    assert not visible({**row, "mechanical_profile": "magnetic_lens_housing"})


@pytest.mark.parametrize("profile", [
    "vacuum_liner", "vacuum_bore", "post_projector_detector_chamber",
    "transverse_goniometer", "side_entry_sample_holder", "branch_reference_plane",
])
def test_passive_components_never_become_optical_from_names_or_parents(profile):
    assert not visible(dict(key="support", name="Objective lens / active detector",
                            parent_key="objective_lens", mechanical_profile=profile))


def test_channel_is_not_an_extra_body_but_real_winding_host_is_kept():
    channel = dict(key="probe_dph1_deflector", layout_role="control_channel",
                   physical_host_key="probe_dph1_coils")
    host = dict(key="probe_dph1_coils", mechanical_only=True,
                mechanical_profile="deflector_coil_insert")
    for region in ("x_winding_1", "x_winding_2", "y_winding_1", "y_winding_2"):
        assert visible(host, region)
        assert not visible(channel, region)
    assert not visible(host, "housing")
    # An explicit link can identify an unprofiled native winding, not an
    # arbitrary body which happens to have a coil-like name.
    custom_host = dict(key="host")
    link = dict(key="drive", physical_host_key="host", layout_role="control_channel")
    assert visible(custom_host, "x_winding_1", parts={"drive": link})
    assert not visible(custom_host, "body", parts={"drive": link})


def test_known_region_semantics_keep_field_material_and_remove_supports():
    accelerator = dict(key="feg_accelerator", stage_centers_z_mm=[1, 2])
    assert visible(accelerator, "electrode_1")
    assert not visible(accelerator, "insulator_2")
    assert not visible(accelerator, "body")
    hp = dict(key="probe_hp1_hexapole")
    for region in ("pole_pin_1", "coil_2", "return_yoke"):
        assert visible(hp, region)
    assert not visible(hp, "housing")
    assert not visible(hp, "insulator")
    plates = dict(key="blanker", mechanical_profile="electrostatic_parallel_plates")
    assert visible(plates, "electrode_positive")
    assert not visible(plates, "housing")


def test_only_detector_receiving_regions_remain_visible():
    eds = dict(key="eds_detector_system", mechanical_only=True,
               mechanical_profile="transverse_eds_detector_array")
    assert visible(eds, "segment_1_sensor")
    assert not visible(eds, "segment_1_support")
    assert not visible(eds, "segment_1_cold_shield")
    eels = dict(key="energy_filter_zebra", mechanical_profile="zebra_eels_detector")
    assert visible(eels, "active_strip_2_scintillator")
    assert visible(eels, "active_strip_2")
    assert not visible(eels, "optical_backing")
    assert not visible(eels, "housing")
    assert not visible(eels, "alignment_active_area", wireframe=True)
    assert visible(dict(key="camera", mechanical_profile="camera_sensor_plane"))


def test_apertures_sources_and_cell_context_survive_and_reference_lenses_do_not():
    assert visible(dict(key="feg_tip", tip_definition_file="tip.toml"))
    assert visible(dict(key="feg_extractor"))
    assert visible(dict(key="objective_aperture", aperture_plate_form="perforated_strip"))
    assert visible(dict(key="energy_filter_slit", mechanical_profile="xo_energy_slit_assembly"),
                   "blade_negative")
    assert not visible(dict(key="virtual_objective", virtual=True))
    for key in ("cell_window_upstream", "specimen_cell", "cell_sample_reference"):
        assert visible(SimpleNamespace(key=key, data={"cell_context": True}),
                       wireframe=key != "cell_window_upstream")


def test_unclassified_imported_and_custom_bodies_are_not_guessed_and_inputs_unchanged():
    row = dict(key="mystery", name="lens", model_3d={"base": {"kind": "box"}})
    before = deepcopy(row)
    assert not visible(row)
    assert row == before
    assert not visible(dict(key="objective_lens", mechanical_profile="magnetic_lens_assembly",
                            model_3d={"base": {"kind": "step"}}), "step_1")
    assert not visible(dict(key="copied_pole", mechanical_profile="magnetic_pole_piece",
                            mechanical_part_role="custom_mechanical"))
    assert not visible(None)


def test_real_saved_assembly_profiles_keep_each_functional_family():
    """Audit actual saved metadata without running optics or remeshing CAD."""
    from temsim.optics.column import default_state
    assembly = default_state()._resolved_assembly
    parts = {part.key: part for part in assembly.parts}
    examples = {
        "feg_tip": "body", "feg_extractor": "body", "feg_accelerator": "electrode_1",
        "condenser_lens_1_excitation_coil": "body", "condenser_lens_1_yoke": "body",
        "condenser_lens_1_lower_pole": "body", "objective_upper_pole": "body",
        "probe_dph1_coils": "x_winding_1", "probe_hp1_hexapole": "pole_pin_1",
        "ac_deflector": "coil_1_x_winding_1", "condenser_stigmator": "coil_1",
        "objective_aperture": "body", "sample": "body", "haadf": "body",
        "camera": "body", "eds_detector_system": "segment_1_sensor",
    }
    for key, region in examples.items():
        assert visible(parts[key], region, parts=parts), key
    for key in ("condenser_lens_1_housing", "objective_lens_housing", "sample_stage",
                "sample_holder", "post_projector_detector_chamber", "camera_recording_housing"):
        assert not visible(parts[key], parts=parts), key
