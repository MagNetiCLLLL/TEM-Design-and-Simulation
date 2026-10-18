"""Shared controls follow hardware through geometry changes and snapshots."""
from copy import deepcopy
import tomllib

import pytest

from temsim import module_manifest
from temsim.optics.column import default_state
from temsim.optics.shared_deflectors import resolve_shared_deflector_parts
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.mark.parametrize("variant", ["C2", "C3", "C3_ProbeCorrector",
    "C3_ImageCorrector", "C3_ProbeCorrector_ImageCorrector"])
def test_all_column_variants_share_descan_but_keep_scan_after_corrector(variant):
    path = INSTRUMENT_CONFIG_ROOT / "column" / f"{variant}.toml"
    document = module_manifest.read_document(path)
    module_manifest.validate_document(document)
    parts = {part["key"]: part for part in document["parts"]}
    channel, host = parts["descan_deflector"], parts["image_diffraction_deflector"]
    assert channel["physical_host_key"] == host["key"]
    assert channel["interaction_centers_local_z_mm"] == host["interaction_centers_local_z_mm"]
    assert channel["local_center_z_mm"] == host["local_center_z_mm"]
    assert channel["effective_thickness_mm"] == host["effective_thickness_mm"]
    assert "parent_key" not in channel
    ac, beam = parts["ac_deflector"], parts["beam_deflector"]
    assert "physical_host_key" not in ac
    assert ac["local_start_z_mm"] > beam["local_end_z_mm"]
    if "probe_tl12_lens" in parts:
        assert ac["local_start_z_mm"] > parts["probe_tl12_lens"]["local_end_z_mm"]


@pytest.mark.parametrize("variant,centres", [
    ("C2", (455.0, 480.0)),
    ("C3", (705.0, 730.0)),
    ("C3_ProbeCorrector", (937.5, 962.5)),
    ("C3_ImageCorrector", (705.0, 730.0)),
    ("C3_ProbeCorrector_ImageCorrector", (1029.0, 1054.0)),
])
def test_relocated_ac_default_is_scoped_to_calibrated_column(variant, centres):
    document = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / "column" / f"{variant}.toml")
    parts = {part["key"]: part for part in document["parts"]}
    assert tuple(parts["ac_deflector"]["interaction_centers_local_z_mm"]) == centres


def test_ac_can_move_independently_of_downstream_descan():
    document = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    original = deepcopy(document)
    ac = next(part for part in document["parts"] if part["key"] == "ac_deflector")
    for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
                  "optical_reference_local_z_mm"):
        ac[field] += 1.0
    ac["interaction_centers_local_z_mm"] = [value + 1.0 for value in ac["interaction_centers_local_z_mm"]]
    module_manifest.validate_document(document)
    assert [part for part in document["parts"] if part["key"] != "ac_deflector"] == [
        part for part in original["parts"] if part["key"] != "ac_deflector"
    ]


@pytest.mark.parametrize("field,value,error", [
    ("interaction_centers_local_z_mm", [940.0, 960.0], "interaction centres must match"),
    ("mechanical_inter_coil_gap_mm", 9.0, "body length must equal"),
    ("mechanical_coil_length_mm", 14.0, "field thickness must fit"),
    ("effective_thickness_mm", 16.0, "field thickness must fit"),
    ("optical_reference_local_z_mm", 951.0, "optical reference must equal"),
])
def test_ac_inconsistent_declared_geometry_is_rejected(field, value, error):
    document = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    ac = next(part for part in document["parts"] if part["key"] == "ac_deflector")
    ac[field] = value
    with pytest.raises(ValueError, match=error):
        module_manifest.validate_document(document)


def test_legacy_channel_geometry_is_replaced_by_edited_host_without_changing_input():
    path = INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml"
    original = tomllib.loads(path.read_text(encoding="utf-8"))
    parts = {part["key"]: part for part in original["parts"]}
    channel, host = parts["descan_deflector"], parts["image_diffraction_deflector"]
    for field in ("layout_role", "layout_owner", "physical_host_key", "physical_host_status"):
        channel.pop(field, None)
    channel["local_center_z_mm"] = 12.
    channel["interaction_centers_local_z_mm"] = [10., 14.]
    host["interaction_centers_local_z_mm"] = [1330., 1355.]
    host["local_center_z_mm"] = 1342.5
    before = deepcopy(original)
    resolved = resolve_shared_deflector_parts(original)
    updated = next(p for p in resolved["parts"] if p["key"] == "descan_deflector")
    assert updated["local_center_z_mm"] == 1342.5
    assert updated["interaction_centers_local_z_mm"] == [1330., 1355.]
    assert original == before


@pytest.fixture(scope="module")
def base_state():
    return default_state()


def test_default_runtime_ac_centres_match_manifest_and_clear_neighbours(base_state):
    assembly = base_state._resolved_assembly
    ac = base_state.ac_deflector
    assert (ac.upper_z_mm, ac.lower_z_mm) == (1387.5, 1412.5)
    assert (ac.upper_surface_z_mm, ac.lower_surface_z_mm) == (1380.0, 1420.0)
    assert assembly.part("probe_tl12_lens").end_z_mm < ac.upper_surface_z_mm
    assert ac.lower_surface_z_mm < assembly.part("condenser_stigmator").start_z_mm
    assert ac.lower_surface_z_mm < assembly.part("mini_condenser").start_z_mm
    # The calibrated static beam waist lies in the open inter-coil gap.
    assert ac.upper_z_mm + ac.mechanical_coil_length_mm / 2 < 1400.0007
    assert ac.lower_z_mm - ac.mechanical_coil_length_mm / 2 > 1400.0007
    assert (base_state.descan_deflector.upper_z_mm,
            base_state.descan_deflector.lower_z_mm) == (1674.4, 1699.4)
    host = base_state.image_diffraction_deflector
    assert host.upper_surface_z_mm == pytest.approx(1666.9)
    assert host.lower_surface_z_mm == pytest.approx(1706.9)
    assert assembly.part("objective_lower_pole").end_z_mm + 3.0 == pytest.approx(host.upper_surface_z_mm)
    assert host.lower_surface_z_mm < assembly.part("objective_stigmator").start_z_mm
    assert base_state.selected_area_aperture.z_mm == pytest.approx(1824.4)
    assert base_state.fluorescent_screen.z_mm == pytest.approx(2714.15)


def test_bound_channel_and_deepcopy_follow_live_host_geometry(base_state):
    channel, host = deepcopy((base_state.descan_deflector, base_state.image_diffraction_deflector))
    before = channel.upper_z_mm
    host.optical_center_z_mm += 4.
    host.thickness_mm += 1.
    assert channel.upper_z_mm == host.upper_z_mm
    assert channel.upper_z_mm != before
    assert channel.lower_z_mm == host.lower_z_mm
    assert channel.effective_thickness_mm == host.thickness_mm
    assert channel._physical_host is host
    assert base_state.descan_deflector._physical_host is base_state.image_diffraction_deflector
    assert base_state.image_diffraction_deflector.thickness_mm != host.thickness_mm


def test_state_roundtrip_preserves_commands_and_rebinds_host(base_state):
    state = type(base_state).from_dict(base_state.to_dict())
    state.image_diffraction_deflector.upper_x_mrad = .31
    state.descan_deflector.kick_x_mrad = -.12
    state.descan_deflector.scan_enabled = True
    restored = type(state).from_dict(state.to_dict())
    channel, host = restored.descan_deflector, restored.image_diffraction_deflector
    assert host.upper_x_mrad == .31
    assert channel.kick_x_mrad == -.12
    assert channel.scan_enabled
    assert channel._physical_host is host
    assert (channel.upper_z_mm, channel.lower_z_mm) == (host.upper_z_mm, host.lower_z_mm)


def test_scan_cannot_be_moved_above_probe_corrector():
    document = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    parts = {part["key"]: part for part in document["parts"]}
    ac, beam = parts["ac_deflector"], parts["beam_deflector"]
    # Reproduce the tempting but physically incorrect cross-corrector merge.
    for key in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
                "length_mm", "optical_reference_local_z_mm", "interaction_centers_local_z_mm"):
        ac[key] = beam[key]
    with pytest.raises(ValueError, match="downstream of the probe corrector"):
        module_manifest._validate_objective_assembly(document["parts"])
