"""Service drawings persist independently from prescribed gas transport."""
from copy import deepcopy

import pytest

from temsim.vacuum import VacuumMap, VacuumRegion
from temsim.vacuum_pumping import (
    PUMPING_SCHEMA, default_pumping_layout, effective_pumping_layout,
    validate_pumping_layout,
)


def test_legacy_defaults_are_detached_and_match_shipped_drawing():
    config = VacuumMap.load()
    shipped = deepcopy(config.pumping)
    config.pumping = {}
    before = config.to_dict()
    effective = effective_pumping_layout(config)
    assert effective == shipped
    assert config.to_dict() == before
    assert not config.enabled
    effective["pumps"][0]["name"] = "Changed drawing"
    assert effective_pumping_layout(config) == shipped
    assert VacuumMap.from_dict({}).pumping == {}
    assert effective_pumping_layout(VacuumMap.from_dict({}))["schema"] == PUMPING_SCHEMA


def test_shared_service_connections_and_explicit_empty_drawing():
    config = VacuumMap.load()
    layout = effective_pumping_layout(config)
    assert sum(pump["kind"] == "igp" for pump in layout["pumps"]) == 4
    assert sum(pump["kind"] == "turbo" for pump in layout["pumps"]) == 1
    lines = {line["key"]: line for line in layout["lines"]}
    assert lines["projection_turbo"]["target"] == lines["airlock_turbo"]["target"] == "pump:turbo"
    assert lines["projection_turbo"]["open"]
    assert not lines["airlock_turbo"]["open"]
    assert lines["turbo_backing"]["source"] == "pump:turbo"
    assert lines["turbo_backing"]["target"] == "pump:scroll"
    assert lines["airlock_roughing"]["target"] == "pump:scroll"
    assert lines["airlock_transfer"]["target"] == "region:specimen"
    assert "airlock" not in {region.key for region in config.regions}
    empty = {"schema": PUMPING_SCHEMA, "pumps": [], "auxiliaries": [], "lines": []}
    config.pumping = deepcopy(empty)
    assert effective_pumping_layout(config) == empty


def test_defaults_do_not_invent_custom_region_assignments():
    regions = [VacuumRegion("custom", "Custom chamber", "source", "sample")]
    layout = default_pumping_layout(regions)
    assert all("region:custom" not in (line["source"], line["target"]) for line in layout["lines"])
    validate_pumping_layout(layout, {"custom"})


def test_exact_legacy_snapshot_restores_without_injecting_display_metadata():
    from temsim.immutable_json import thaw_json
    from temsim.instrument_snapshot import decode_instrument, encode_instrument
    from temsim.optics.column import default_state

    state = default_state()
    state.objective_lens.percent = 61.23456789
    state.vacuum_map.regions[-1].medium.pressure_mbar = 4.321e-7
    graph = thaw_json(encode_instrument(state))
    node = next(node for node in graph["nodes"] if node["type"] == "temsim.vacuum:VacuumMap")
    assert "pumping" not in node["fields"]
    node["attributes"].pop("pumping")
    restored = decode_instrument(graph)
    assert restored.objective_lens.percent == 61.23456789
    assert restored.vacuum_map.regions[-1].medium.pressure_mbar == 4.321e-7
    assert "pumping" not in vars(restored.vacuum_map)
    assert restored.vacuum_map.pumping == {}
    assert effective_pumping_layout(restored.vacuum_map)["pumps"]
    assert restored.vacuum_map.signature() == state.vacuum_map.signature()
    assert "pumping" not in vars(restored.vacuum_map)
    assert thaw_json(encode_instrument(restored)) == graph
    # Only this named optional attribute is display-only.
    from temsim.parameter_registry import unmapped_public_inputs
    restored.vacuum_map.unknown_transport_control = 1
    assert "instrument.vacuum_map.unknown_transport_control" in unmapped_public_inputs(restored)


@pytest.mark.parametrize("mutation, message", [
    (lambda value: value.update(schema="future"), "schema"),
    (lambda value: value["pumps"].append(deepcopy(value["pumps"][0])), "Duplicate"),
    (lambda value: value["pumps"][0].update(kind="invented"), "pump kind"),
    (lambda value: value["lines"][0].update(target="pump:missing"), "missing endpoint"),
    (lambda value: value["lines"][0].update(target=value["lines"][0]["source"]), "itself"),
    (lambda value: value["lines"][0].update(open=1), "boolean"),
    (lambda value: value["lines"][0].update(open=False), "always open"),
    (lambda value: value["lines"][0].update(role="pressure_solver"), "line role"),
    (lambda value: value["auxiliaries"][0].update(anchor=""), "nonempty"),
])
def test_invalid_drawing_rejected_without_mutation(mutation, message):
    config = VacuumMap.load()
    mutation(config.pumping)
    before = deepcopy(config.pumping)
    with pytest.raises(ValueError, match=message):
        config.validate()
    assert config.pumping == before


def _edited_drawing(config):
    layout = effective_pumping_layout(config)
    layout["pumps"][0]["name"] = "Renamed ion pump"
    layout["pumps"][-1]["kind"] = "rotary"
    next(line for line in layout["lines"] if line["key"] == "airlock_turbo")["open"] = True
    layout["lines"].append({
        "key": "shared_post_column", "source": "region:post_column",
        "target": "pump:turbo", "role": "high_vacuum",
        "valve": "New isolation", "open": False,
    })
    return validate_pumping_layout(layout, (region.key for region in config.regions))


@pytest.mark.parametrize("enabled", [False, True])
def test_drawing_roundtrips_map_state_profile_and_snapshot(tmp_path, enabled):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.profile_io import apply_profile_values, read_profile, save_profile

    state = default_state()
    state.vacuum_map.enabled = enabled
    state.vacuum_map.pumping = _edited_drawing(state.vacuum_map)
    expected = state.vacuum_map.to_dict()
    path = tmp_path / "services.toml"
    state.vacuum_map.save(path)
    assert VacuumMap.load(path).to_dict() == expected
    assert State.from_dict(state.to_dict()).vacuum_map.to_dict() == expected
    restored = capture_instrument_snapshot(state).restore()
    assert restored.vacuum_map.to_dict() == expected
    profile = tmp_path / "operating.toml"
    save_profile(profile, state, AssemblyCatalog().default_selection())
    _, values = read_profile(profile)
    apply_profile_values(restored, values)
    assert restored.vacuum_map.to_dict() == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_service_edits_preserve_calculation_and_physical_identities(enabled):
    from temsim.calculation_cache import calculation_signatures, state_model_signature
    from temsim.gui.condenser_scan_design import _physical_graph_signature
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.vacuum import resolve_regions

    state = default_state()
    state.vacuum_map.enabled = enabled
    before = capture_instrument_snapshot(state)
    signatures = calculation_signatures(state)
    identity = state_model_signature(state)
    transport_signature = state.vacuum_map.signature()
    regions = deepcopy(resolve_regions(state, include_disabled=True))
    state.vacuum_map.pumping = _edited_drawing(state.vacuum_map)
    after = capture_instrument_snapshot(state)
    assert before.digest != after.digest  # Saved working points retain the edit.
    assert before.physical_digest == after.physical_digest
    assert _physical_graph_signature(before.graph) == _physical_graph_signature(after.graph)
    assert state_model_signature(state) == identity
    assert calculation_signatures(state) == signatures
    assert state.vacuum_map.signature() == transport_signature
    assert resolve_regions(state, include_disabled=True) == regions
    assert state.vacuum_map.enabled is enabled
    # Removing all drawn hardware must also preserve physical identities.
    state.vacuum_map.pumping = {"schema": PUMPING_SCHEMA, "pumps": [], "auxiliaries": [], "lines": []}
    assert capture_instrument_snapshot(state).physical_digest == before.physical_digest
    assert calculation_signatures(state) == signatures
    # Actual gas inputs retain the previous conservative invalidation policy.
    state.vacuum_map.regions[-1].medium.pressure_mbar *= 2
    changed = capture_instrument_snapshot(state)
    assert changed.physical_digest != before.physical_digest
    assert _physical_graph_signature(changed.graph) != _physical_graph_signature(before.graph)
    assert state_model_signature(state) != identity
    assert calculation_signatures(state) != signatures
    assert state.vacuum_map.signature() != transport_signature
