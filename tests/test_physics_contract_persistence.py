"""P0 provenance survives current storage and rejects corrupt ownership."""
from dataclasses import replace

import pytest

from temsim.artifact_store import ArtifactIntegrityError, ArtifactStore, INCIDENT_SEED_CODEC
from temsim.immutable_json import json_digest, thaw_json
from temsim.particle_section_io import _pack_record_graph, _unpack_record_graph
from temsim.physics.core import build_propagation_plan
from temsim.physics.ideal_column_reference import column_plan_for_model
from temsim.physics.interaction_registry import InteractionRegistry
from temsim.physics.plan_contract import contract_for_slice, validate_plan_contract
from test_shared_chromatic_screens import fixture, plan_for
from test_shared_field_archives import storage_result


def test_plan_graph_roundtrip_keeps_immutable_scene_and_owned_residuals():
    plan = plan_for(fixture())
    arrays = {}
    graph = _pack_record_graph(plan, arrays)
    restored = _unpack_record_graph(graph, arrays, maximum_unpacked_bytes=1 << 24)
    assert restored.physics_contract == plan.physics_contract
    assert restored.signature == plan.signature
    with pytest.raises(TypeError):
        restored.physics_contract["scene"]["payload"]["settings"]["beam_voltage_kv"] = 1.


def test_prefix_reuse_checks_executed_actions_instead_of_slicing_scene_metadata():
    from temsim.physics.particle_sections import _prefix_matches
    state = fixture()
    original = plan_for(state)
    state.simulation_time_s = 2.  # No time-dependent drive in this fixture.
    same_actions = plan_for(state)
    assert same_actions.physics_contract != original.physics_contract
    last = len(original.z_mm)-1
    assert _prefix_matches(original, same_actions, last, last)
    state.lenses[0].cc_mm = 3.
    different_actions = plan_for(state)
    assert not _prefix_matches(original, different_actions, last, last)


def test_reference_and_checkpoint_slice_do_not_claim_removed_or_repeated_actions():
    state = fixture()
    state.lenses[0].cs_mm = 2.
    plan = plan_for(state)
    reference = column_plan_for_model(plan, "first_order_reference")
    original = InteractionRegistry.from_dict(plan.physics_contract["interactions"])
    assert {row.effect_family for row in original.records} >= {"Cs", "Cc"}
    kept = InteractionRegistry.from_dict(reference.physics_contract["interactions"])
    assert "Cs" not in {row.effect_family for row in kept.records}
    assert "Cc" in {row.effect_family for row in kept.records}
    assert reference.physics_contract["scene"] == plan.physics_contract["scene"]
    after = contract_for_slice(plan.physics_contract, 0., .1)
    assert not any(row.representation == "residual" for row in
                   InteractionRegistry.from_dict(after["interactions"]).records)
    assert after["scene"] == plan.physics_contract["scene"]


@pytest.mark.parametrize("damage", ("checksum", "duplicate", "scene", "mode"))
def test_plan_construction_rejects_invalid_contract(damage):
    plan = plan_for(fixture())
    contract = thaw_json(plan.physics_contract)
    if damage == "checksum":
        contract["numerical_digest"] = "changed"
    else:
        if damage == "duplicate":
            record = dict(contract["interactions"]["records"][-1])
            record["interaction_id"] += ":again"
            record["action_id"] += ":again"
            contract["interactions"]["records"].append(record)
        elif damage == "scene":
            contract["scene"]["payload"]["settings"]["beam_voltage_kv"] = 1.
        else:
            contract["mode"] = "field_native"
        contract.pop("digest")
        contract["digest"] = json_digest(contract)
    with pytest.raises(ValueError):
        replace(plan, physics_contract=contract)


def _store(tmp_path, result):
    # The storage-only gun/branch fixture does not execute source physics.
    # Supply an actual current column plan, including its matching contract.
    state = fixture()
    plan = build_propagation_plan(state, 1., 3., checkpoint_z_mm=(1., 3.))
    simulation = replace(result.simulation, incident_plan=plan)
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    manifest = result.calculation_manifest
    store.put_incident_simulation_seed(manifest, simulation)
    return store, manifest, plan


def test_current_incident_seed_preserves_plan_contract(storage_result, tmp_path):
    store, manifest, plan = _store(tmp_path, storage_result)
    loaded = store.get_incident_simulation_seed(manifest)
    assert loaded.incident_plan.physics_contract == plan.physics_contract
    validate_plan_contract(loaded.incident_plan.physics_contract)


@pytest.mark.parametrize("damage", ("missing", "checksum", "duplicate"))
def test_current_incident_seed_rejects_broken_ownership(storage_result, tmp_path, damage):
    store, manifest, _ = _store(tmp_path, storage_result)
    dependency = manifest.calculation_signatures["incident"]
    bundle = store.get_array_bundle(manifest, product_key="incident",
                                   dependency_signature=dependency, codec=INCIDENT_SEED_CODEC)
    metadata = thaw_json(bundle.metadata)
    if damage == "missing":
        metadata.pop("plan_physics_contract")
    else:
        contract = metadata["plan_physics_contract"]
        if damage == "checksum":
            contract["numerical_digest"] = "changed"
        else:
            records = contract["interactions"]["records"]
            records.append(dict(records[0]))
            contract.pop("digest")
            contract["digest"] = json_digest(contract)
    store.put_array_bundle(manifest, product_key="incident", dependency_signature=dependency,
                           codec=INCIDENT_SEED_CODEC, arrays=dict(bundle.arrays), metadata=metadata)
    with pytest.raises(ArtifactIntegrityError):
        store.get_incident_simulation_seed(manifest)
