"""Current filter state is explicit; no former two-carrier reconstruction."""
from temsim.energy_filter_coordinates import ENTRANCE_COORDINATE_CONVENTION
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import ENERGY_FILTER_MULTIPOLE_KEYS
from temsim.immutable_json import json_digest, thaw_json
from temsim.input_assets import InputAssetStore
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.optics.column import default_state
from temsim.optics.energy_filter import (
    energy_filter_from_dict, serialise_energy_filter, ensure_energy_filter,
)
from temsim.runtime_parameters import RuntimeTarget, editable_parameters, runtime_targets


@pytest.fixture
def filter_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    state.energy_filter.alignment_x_mrad = .23
    state.energy_filter.alignment_y_mrad = -.17
    return state


def test_current_filter_roundtrip_preserves_nondefault_state_without_reconfiguration(filter_state):
    state = filter_state
    ef = state.energy_filter
    ef.energy_slit.inserted = True
    ef.energy_slit.gap_m = 11e-6
    ef.energy_slit.centre_m = 2e-6
    ef.energy_slit.zero_loss_offset_m = -3e-6
    ef.energy_slit.requested_width_ev = 12.
    ef.energy_slit.requested_centre_loss_ev = 19.
    ef.output_detector_inserted = True
    ef.zebra_detector.inserted = False
    ef.zebra_detector.enabled = False
    ef.bias_tube.enabled = True
    ef.bias_tube.offset_ev = 20.
    ef.fast_shutter.enabled = False
    ef.fast_shutter.open = False
    ef.camera_deflector.enabled = False
    ef.camera_deflector.active_strip = 3
    ef.multi_eels_region_count = 4
    for index, element in enumerate(ef.multipoles):
        element.multipole_field.set_component(2, normal=index + .125, skew=-index - .375)
        element.calibration.normal_trim_coefficients[1] = .0125 * index
        element.calibration.skew_trim_coefficients[2] = -.0025 * index
    before = deepcopy(vars(ef.energy_slit))
    payload = serialise_energy_filter(ef)
    assert vars(ef.energy_slit) == before
    assert not {"entrance_m12", "exit_m12", "m12_frames_placed", "optical_integration_enabled"} & payload.keys()
    assert [item["key"] for item in payload["multipoles"]] == list(ENERGY_FILTER_MULTIPOLE_KEYS)
    ensure_energy_filter(state)
    assert serialise_energy_filter(ef) == payload
    restored = energy_filter_from_dict(deepcopy(payload), state.beam_voltage_kv)
    assert serialise_energy_filter(restored) == payload
    for actual, expected in zip(restored.multipoles, ef.multipoles, strict=True):
        np.testing.assert_array_equal(actual.frame.origin_m, expected.frame.origin_m)
        np.testing.assert_array_equal(actual.multipole_field.normal_coefficients, expected.multipole_field.normal_coefficients)
        np.testing.assert_array_equal(actual.multipole_field.skew_coefficients, expected.multipole_field.skew_coefficients)


@pytest.mark.parametrize("change", [
    lambda p: p.pop("multipoles"),
    lambda p: p.update(multipoles=p["multipoles"][:2]),
    lambda p: p.update(entrance_m12=p["multipoles"][2]),
    lambda p: p.update(exit_m12=p["multipoles"][3]),
    lambda p: p.update(optical_integration_enabled=True),
    lambda p: p.update(slit_width_ev=10.),
    lambda p: p.update(selected_loss_ev=0.),
    lambda p: p["multipoles"][0].update(key=ENERGY_FILTER_MULTIPOLE_KEYS[1]),
    lambda p: p["multipoles"][0].update(role="entrance"),
    lambda p: p["multipoles"][0]["field"].pop("skew_coefficients"),
    lambda p: p["multipoles"][0]["calibration"].pop("normal_trim_coefficients"),
    lambda p: p["multipoles"][0].update(enabled="false"),
    lambda p: p.update(energy_slit=None),
    lambda p: p["energy_slit"].pop("centre_m"),
    lambda p: p["energy_slit"].update(inserted="false"),
    lambda p: p.update(fast_shutter=None),
    lambda p: p["fast_shutter"].update(open="false"),
    lambda p: p["camera_deflector"].update(active_strip=2.5),
    lambda p: p["zebra_detector"].update(unknown=1),
])
def test_incomplete_or_retired_filter_records_are_rejected(change, filter_state):
    state = filter_state
    payload = serialise_energy_filter(state.energy_filter)
    change(payload)
    with pytest.raises(ValueError):
        energy_filter_from_dict(payload, state.beam_voltage_kv)


def test_installed_filter_cannot_be_disabled_as_a_transport_bypass(filter_state):
    state = filter_state
    state.energy_filter.enabled = False
    with pytest.raises(ValueError, match="physical installation"):
        ensure_energy_filter(state)
    assert state.energy_filter.enabled is False


def test_logical_dp12_and_filter_installation_expose_only_supported_controls(filter_state):
    state = filter_state
    targets = runtime_targets(state)
    virtual = [target for target in targets.values() if getattr(target.obj, "KIND", "") == "virtual_layout"]
    assert virtual
    assert "probe_dp12_scan_deflector" in {target.key for target in virtual}
    for target in virtual:
        expected = {"enabled", "alignment_x_mrad", "alignment_y_mrad"} if target.key == "probe_dp12_scan_deflector" else set()
        assert {p.name for p in editable_parameters(target)} == expected
    fields = {p.name for p in editable_parameters(targets["energy_filter"])}
    assert "enabled" not in fields
    assert "optical_integration_enabled" not in fields
    physical = state.ac_deflector
    fields = {p.name for p in editable_parameters(RuntimeTarget(physical.key, physical.name, physical))}
    assert {"kick_x_mrad", "kick_y_mrad", "upper_coil_gain", "scan_enabled"} <= fields


def test_complete_current_state_retains_filter_operating_choices(filter_state):
    from temsim.optics.model import State
    state = filter_state
    state.energy_filter.fast_shutter.open = False
    state.energy_filter.fast_shutter.enabled = False
    state.energy_filter.energy_slit.inserted = True
    state.energy_filter.zebra_detector.inserted = False
    state.energy_filter.camera_deflector.active_strip = 3
    state.energy_filter.multi_eels_region_count = 4
    payload = state.to_dict()
    restored = State.from_dict(deepcopy(payload))
    assert restored.to_dict()["energy_filter"] == payload["energy_filter"]


def test_slit_owns_window_requests_and_physical_configuration(filter_state):
    from temsim.optics.energy_filter import configure_energy_slit_from_software
    state = filter_state
    ef = state.energy_filter
    ef.slit_width_ev = 17.
    ef.selected_loss_ev = 33.
    assert ef.energy_slit.requested_width_ev == 17.
    assert ef.energy_slit.requested_centre_loss_ev == 33.
    configure_energy_slit_from_software(ef)
    assert ef.energy_slit.derived_width_ev == pytest.approx(17.)
    assert ef.energy_slit.derived_centre_loss_ev == pytest.approx(33.)
    payload = serialise_energy_filter(ef)
    assert not {"slit_width_ev", "selected_loss_ev"} & payload.keys()
    assert payload["energy_slit"]["requested_width_ev"] == 17.
    assert payload["energy_slit"]["requested_centre_loss_ev"] == 33.


@pytest.mark.parametrize("dispersion", [0.75, -2.5])
@pytest.mark.parametrize("zero_offset", [0., 19e-6])
def test_slit_inverse_energy_window_uses_displacement_units_and_offset(dispersion, zero_offset):
    from temsim.optics.energy_filter_slit import create_energy_selection_slit
    slit = create_energy_selection_slit(dispersion_um_per_ev=dispersion)
    slit.zero_loss_offset_m = zero_offset
    slit.configure_energy_window(41., 7.)
    assert slit.derived_centre_loss_ev == pytest.approx(41.)
    assert slit.derived_width_ev == pytest.approx(7.)
    assert slit.centre_m == pytest.approx(zero_offset + dispersion * 41e-6)


def test_filter_record_marks_coordinates_and_migrates_legacy_alignment_once(filter_state):
    state = filter_state
    current = serialise_energy_filter(state.energy_filter)
    assert current["entrance_coordinate_convention"] == ENTRANCE_COORDINATE_CONVENTION
    restored = energy_filter_from_dict(current, state.beam_voltage_kv)
    assert restored.alignment_x_mrad == .23
    legacy = deepcopy(current)
    legacy.pop("entrance_coordinate_convention")
    original = deepcopy(legacy)
    restored = energy_filter_from_dict(legacy, state.beam_voltage_kv)
    assert restored.alignment_x_mrad == -.23
    assert restored.alignment_y_mrad == -.17
    assert legacy == original
    reloaded = energy_filter_from_dict(serialise_energy_filter(restored), state.beam_voltage_kv)
    assert reloaded.alignment_x_mrad == -.23
    bad = {**current, "entrance_coordinate_convention": "unknown"}
    with pytest.raises(ValueError, match="coordinates"):
        energy_filter_from_dict(bad, state.beam_voltage_kv)


@pytest.fixture(params=[False, True], ids=["inline", "assets"])
def filter_graph(request, filter_state):
    lease = InputAssetStore(budget_bytes=1024**2).capture() if request.param else None
    filter_state.extra_input_array = np.arange(8192, dtype=float)
    filter_state.extra_input_alias = filter_state.extra_input_array
    try:
        yield thaw_json(encode_instrument(filter_state, asset_store=lease)), lease
    finally:
        if lease is not None:
            lease.close()


def test_complete_snapshot_keeps_column_alignment_and_input_assets(filter_graph):
    graph, assets = filter_graph
    assert graph["coordinate_conventions"] == {"energy_filter_entrance": ENTRANCE_COORDINATE_CONVENTION}
    before = json_digest(graph)
    restored = decode_instrument(graph, assets=assets)
    assert restored.energy_filter.alignment_x_mrad == .23
    assert restored.energy_filter.alignment_y_mrad == -.17
    np.testing.assert_array_equal(restored.extra_input_array, np.arange(8192, dtype=float))
    if assets is not None:
        assert restored.extra_input_array is restored.extra_input_alias
    assert json_digest(encode_instrument(restored, asset_store=assets)) == before
    assert json_digest(graph) == before


@pytest.mark.parametrize("alignment_x,installed", [(0., True), (.23, True), (.23, False)])
def test_untagged_graph_requires_unambiguous_filter_alignment(filter_graph, alignment_x, installed):
    graph, assets = filter_graph
    graph.pop("coordinate_conventions", None)
    node = next(node for node in graph["nodes"] if node["type"] == "temsim.optics.energy_filter:EnergyFilterSystem")
    node["attributes"]["alignment_x_mrad"] = alignment_x
    node["attributes"]["enabled"] = installed
    graph["nodes"][graph["root"]["ref"]]["attributes"]["energy_filter_installed"] = installed
    before = json_digest(graph)
    if alignment_x:
        with pytest.raises(ValueError, match="Unversioned Energy Filter entrance X alignment"):
            decode_instrument(graph, assets=assets)
    else:
        restored = decode_instrument(graph, assets=assets)
        assert restored.energy_filter.alignment_x_mrad == 0.
        assert restored.energy_filter.alignment_y_mrad == -.17
        tagged = thaw_json(encode_instrument(restored, asset_store=assets))
        assert tagged.pop("coordinate_conventions") == {"energy_filter_entrance": ENTRANCE_COORDINATE_CONVENTION}
        assert json_digest(tagged) == before
    assert json_digest(graph) == before


@pytest.mark.parametrize("conventions", [{}, None, {"energy_filter_entrance": "unknown"}])
def test_complete_graph_rejects_unknown_or_incomplete_coordinates(filter_graph, conventions):
    graph, assets = filter_graph
    graph["coordinate_conventions"] = conventions
    before = json_digest(graph)
    with pytest.raises(ValueError, match="working-point coordinates"):
        decode_instrument(graph, assets=assets)
    assert json_digest(graph) == before
