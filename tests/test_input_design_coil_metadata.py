"""Input-only additive migration; never reuse or execute old transport."""
from copy import deepcopy

import pytest
import tomli_w

from temsim.input_design_compatibility import additive_coil_metadata_only


def documents():
    old = {"format_version": 1, "parts": [
        {"key": "lens", "mechanical_profile": "magnetic_lens_assembly",
         "local_start_z_mm": 0., "drive_percent": 50.},
        {"key": "coil", "mechanical_profile": "magnetic_excitation_coil",
         "field_source_key": "lens", "mechanical_inner_diameter_mm": 20.,
         "mechanical_outer_diameter_mm": 30., "local_start_z_mm": 0., "local_end_z_mm": 10.},
    ]}
    new = deepcopy(old)
    new["parts"][0].update(air_core_reference_ampere_turns=3000.,
        air_core_reference_status="engineering_reference_not_oem_not_particle_qualified",
        air_core_reference_source="Explicit engineering initialization",
        air_core_reference_source_urls=["https://example.com/reference"],
        gaussian_reference_coil_length_mm=10.)
    new["parts"][1]["coil_ampere_turn_fraction"] = 1.
    return old, new


def admit(old, new):
    return additive_coil_metadata_only(tomli_w.dumps(old).encode(), tomli_w.dumps(new).encode())


def test_valid_optional_metadata_addition_is_narrow_and_does_not_mutate_inputs():
    old, new = documents()
    saved = deepcopy((old, new))
    assert admit(old, new)
    assert (old, new) == saved


def test_new_conductor_area_cannot_be_ignored_as_old_input_metadata():
    old, new = documents()
    new["parts"][1]["winding_wire_cross_section_mm2"] = .125
    assert not admit(old, new)


@pytest.mark.parametrize("change", ["drive", "geometry", "type", "unknown", "new_part", "source", "remove"])
def test_existing_physical_changes_and_unknown_additions_still_reject(change):
    old, new = documents()
    if change == "drive":
        new["parts"][0]["drive_percent"] = 51.
    elif change == "geometry":
        new["parts"][1]["mechanical_inner_diameter_mm"] = 21.
    elif change == "type":
        new["format_version"] = True
    elif change == "unknown":
        new["parts"][0]["arbitrary_new_field"] = 1.
    elif change == "new_part":
        new["parts"].append(deepcopy(new["parts"][1]))
    elif change == "source":
        new["emission_energy_ev"] = .3
    else:
        del new["parts"][0]["drive_percent"]
    assert not admit(old, new)


@pytest.mark.parametrize("value", [-1., 0., True, float("inf")])
def test_invalid_reference_metadata_is_not_a_migration(value):
    old, new = documents()
    new["parts"][0]["air_core_reference_ampere_turns"] = value
    assert not admit(old, new)


def test_already_captured_metadata_cannot_be_recalibrated():
    _, old = documents()
    new = deepcopy(old)
    new["parts"][0]["air_core_reference_ampere_turns"] = 6000.
    assert not admit(old, new)


@pytest.mark.parametrize("value", [-1., 0., True, float("inf"), 2., 1.])
def test_coil_share_without_a_complete_reference_owner_is_rejected(value):
    old, _ = documents()
    new = deepcopy(old)
    new["parts"][1]["coil_ampere_turn_fraction"] = value
    assert not admit(old, new)


def test_new_metadata_cannot_reinterpret_a_multipole_as_a_round_lens():
    old, new = documents()
    old["parts"][0]["mechanical_profile"] = "hexapole"
    new["parts"][0]["mechanical_profile"] = "hexapole"
    assert not admit(old, new)


@pytest.mark.parametrize("explicit", [False, True])
def test_both_historical_default_field_encodings_restore_exactly(explicit):
    from dataclasses import fields
    from temsim.immutable_json import thaw_json
    from temsim.instrument_snapshot import decode_instrument, encode_instrument
    from temsim.optics.column import default_state
    from temsim.optics.model import State

    graph = thaw_json(encode_instrument(default_state()))
    node = next(row for row in graph["nodes"] if row["type"] == "temsim.optics.model:State")
    names = {"round_lens_field_model": "gaussian", "gaussian_geometry_coupling": "fixed_calibration"}
    assert not names.keys() & node["attributes"].keys()
    if explicit:
        node["attributes"].update(names)
        selected = set(node["fields"]) | names.keys()
        node["fields"] = [f.name for f in fields(State) if f.name in selected]
    restored = decode_instrument(graph)
    assert restored.round_lens_field_model == "gaussian"
    assert restored.gaussian_geometry_coupling == "fixed_calibration"
    assert thaw_json(encode_instrument(restored)) == graph


@pytest.mark.parametrize("name", ["_runtime_copper_coil_provider_cache", "_runtime_nonlinear_provider_cache",
                                  "_effective_aberration_cache", "_multiplane_transfer_cache"])
def test_derived_field_and_transfer_caches_never_change_input_identity(name):
    from temsim.instrument_snapshot import encode_instrument
    from temsim.optics.column import default_state
    state = default_state()
    original = encode_instrument(state)
    setattr(state, name, {"synthetic_derived_product": object()})
    assert encode_instrument(state) == original
