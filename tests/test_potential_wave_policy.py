"""Numerical exponential selection is part of immutable execution history."""
from copy import deepcopy

import pytest

from temsim.immutable_json import json_digest, thaw_json
from temsim.physics.potential_wave_policy import potential_wave_operator_policy, potential_support_policy


@pytest.mark.parametrize("version,order", [(1, 4), (2, 4), (2, 8)])
def test_legacy_receipt_means_taylor_and_cannot_enable_chebyshev(version, order):
    old = dict(kinetic_order=order, reference_quadrature="cf4",
               scalar_splitting=True, quartic_scalar_splitting=False)
    if version == 2:
        old["higher_order_scalar_splitting"] = order == 8
    previous = {"potential_operator_policy": {
        "schema": f"potential-wave-operator-policy-v{version}",
        "policy": old, "digest": json_digest(old)}}
    current = potential_wave_operator_policy(old, previous)
    assert current["schema"] == "potential-wave-operator-policy-v3"
    assert current["policy"]["residual_exponential_method"] == "taylor"
    assert current["policy"]["maximum_chebyshev_group"] == 8
    for changed in (dict(residual_exponential_method="chebyshev"),
                    dict(maximum_chebyshev_group=16)):
        with pytest.raises(ValueError, match="policy changed"):
            potential_wave_operator_policy(dict(old, **changed), previous)
    corrupted = deepcopy(previous)
    corrupted["potential_operator_policy"]["policy"]["residual_exponential_method"] = "chebyshev"
    corrupted["potential_operator_policy"]["digest"] = json_digest(
        corrupted["potential_operator_policy"]["policy"])
    with pytest.raises(ValueError, match="historical"):
        potential_wave_operator_policy(old, corrupted)


@pytest.mark.parametrize("method,group", [(True, 8), ("auto", 8), ("chebyshev", True),
                                         ("chebyshev", 8.), ("chebyshev", 0), ("chebyshev", 33)])
def test_invalid_policy_rejected_before_execution(method, group):
    with pytest.raises(ValueError):
        potential_wave_operator_policy(dict(residual_exponential_method=method,
                                            maximum_chebyshev_group=group))


def test_method_and_group_cannot_change_on_resume_or_lose_receipt():
    selected = dict(residual_exponential_method="chebyshev", maximum_chebyshev_group=16)
    receipt = potential_wave_operator_policy(selected)
    assert potential_wave_operator_policy(selected, {"potential_operator_policy": receipt}) == receipt
    for changed in (dict(residual_exponential_method="taylor"), dict(maximum_chebyshev_group=8)):
        with pytest.raises(ValueError, match="policy changed"):
            potential_wave_operator_policy(dict(selected, **changed), {"potential_operator_policy": receipt})
    history = {"snapshot_digest": "executed", "modes": [{"stages": [
        {"residual_exponential_method": "chebyshev"}]}]}
    with pytest.raises(ValueError, match="explicit operator policy"):
        potential_wave_operator_policy({}, history)
    broken = thaw_json(receipt)
    broken["policy"]["maximum_chebyshev_group"] = 8
    with pytest.raises(ValueError, match="policy changed"):
        potential_wave_operator_policy(selected, {"potential_operator_policy": broken})


def test_weighted_pressure_policy_has_distinct_identity_and_cannot_reinterpret_old_receipts():
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    ledger = NumericalSupportLedger(.02, 1e-6)
    options = {"support_projection_policy": "weighted_segment_and_retry", "support_ledger": ledger}
    current = potential_support_policy(options)
    assert current["schema"] == "potential-support-policy-v3"
    assert current["trigger"] == "typed_resource_failure_only"
    assert current["allocation"] == "remaining_for_admissible_grid"
    assert current["digest"] == json_digest({key: current[key] for key in ("policy", "trigger", "allocation")})
    record = {"support_ledger": ledger.to_dict(), "snapshot_digest": "executed", "potential_support_policy": current}
    assert potential_support_policy(options, record) == current
    old = {"schema": "potential-support-policy-v1", "policy": current["policy"], "digest": json_digest(current["policy"])}
    saved = deepcopy(old)
    with pytest.raises(ValueError, match="trigger changed"):
        potential_support_policy(options, dict(record, potential_support_policy=old))
    assert old == saved
    omitted = dict(record)
    omitted.pop("potential_support_policy")
    omitted["stages"] = [{"support_projection_policy": current["policy"]}]
    with pytest.raises(ValueError, match="trigger changed"):
        potential_support_policy(options, omitted)
    tampered = thaw_json(current)
    tampered["trigger"] = "step_start_and_retry"
    tampered["digest"] = json_digest({key: tampered[key] for key in ("policy", "trigger")})
    with pytest.raises(ValueError, match="Invalid stored"):
        potential_support_policy(options, dict(record, potential_support_policy=tampered))


def _legacy_transition_fixture():
    from types import SimpleNamespace
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.unified_wave_support import prepare_unified_support, stamp_unified_support
    from test_potential_wave_resampling import _wave
    mode = SimpleNamespace(mode_id="original-mode", weight_per_reference_electron=.25)
    record = prepare_unified_support({}, mode, enabled=True, l2_limit=.02,
        probability_limit=1e-6, source_digest="original-source", allow_allocate=True)
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    _, ledger = ledger.resample_spectrum(_wave(), z_m=.5, mode_weight=.25)
    assert len(ledger.events) == 1
    record.update(support_ledger=ledger.to_dict(), snapshot_digest="unchanged-field",
                  source_checkpoint_digest="original-source", source_mode_id="original-mode",
                  source_mode_weight_per_reference=.25, unified_wave_identity="old-executor")
    record = stamp_unified_support(record)
    payload = {"policy": "weighted_segment_and_retry", "trigger": "typed_resource_failure_only"}
    record["potential_support_policy"] = {"schema": "potential-support-policy-v2", **payload, "digest": json_digest(payload)}
    shapes = {"amplitude": [32, 32], "basis_m": [2, 2], "origin_m": [2], "curvature_m1": [2, 2], "tilt_rad": [2]}
    source = {"checkpoint_digest": "complete-checkpoint", "manifest_digest": "complete-manifest",
              "source_checkpoint_digest": "original-source", "mode_id": "original-mode", "mode_weight": .25,
              "previous_executor_identity": "old-executor", "executor_identity": "new-executor",
              "array_descriptors": {key: {"file": key+".npy", "sha256": "a"*64,
                  "shape": shape, "nbytes": 16} for key, shape in shapes.items()}}
    request = {"wave_grid": {"maximum_pixels": 8192, "maximum_support_loss_fraction": 1e-6},
               "execution": {"cache_directory": "old-cache", "segment_steps": 10}}
    changed = deepcopy(request)
    changed["execution"]["cache_directory"] = "new-cache"
    return record, dict(source_reference=source, native_plane_m=.5,
        previous_implementation="frozen-old", implementation="frozen-new",
        previous_request=request, request=changed)


def test_explicit_pressure_transition_preserves_old_events_limits_and_full_provenance():
    from temsim.physics.potential_wave_policy import transition_support_policy, validate_support_policy_transition
    record, kwargs = _legacy_transition_fixture()
    saved = deepcopy(record)
    options = {"support_projection_policy": "weighted_segment_and_retry"}
    with pytest.raises(ValueError, match="explicit v2 to v3"):
        potential_support_policy(options, record)
    migrated = transition_support_policy(record, **kwargs)
    assert record == saved
    assert thaw_json(migrated["support_ledger"]) == saved["support_ledger"]
    assert thaw_json(migrated["unified_support_allocation"]) == saved["unified_support_allocation"]
    assert thaw_json(migrated["support_policy_transition"]["previous_context"]) == saved
    assert migrated["potential_support_policy"]["transition_digest"] == migrated["support_policy_transition"]["digest"]
    assert validate_support_policy_transition(migrated) == migrated["support_policy_transition"]
    assert potential_support_policy(options, migrated) == migrated["potential_support_policy"]
    with pytest.raises(ValueError, match="original context|Only an explicit"):
        transition_support_policy(migrated, **kwargs)


@pytest.mark.parametrize("broken", ["receipt", "policy_link", "prefix", "budget", "allocation", "source", "operator", "executor"])
def test_pressure_transition_resume_rejects_missing_or_changed_bound_evidence(broken):
    from temsim.physics.potential_wave_policy import transition_support_policy
    record, kwargs = _legacy_transition_fixture()
    record["potential_operator_policy"] = potential_wave_operator_policy({})
    migrated = thaw_json(transition_support_policy(record, **kwargs))
    if broken == "receipt":
        migrated.pop("support_policy_transition")
    elif broken == "policy_link":
        migrated["potential_support_policy"].pop("transition_digest")
        payload = {key: value for key, value in migrated["potential_support_policy"].items() if key not in ("schema", "digest")}
        migrated["potential_support_policy"]["digest"] = json_digest(payload)
    elif broken == "prefix":
        migrated["support_ledger"]["events"] = []
        migrated["support_ledger"]["schema"] = "controlled-numerical-support-ledger-v1"
    elif broken == "budget":
        migrated["support_ledger"]["maximum_probability"] *= 2
    elif broken == "allocation":
        migrated["unified_support_allocation"]["source_checkpoint_digest"] = "different-source"
    elif broken == "source":
        migrated["source_checkpoint_digest"] = "different-source"
    elif broken == "operator":
        migrated["potential_operator_policy"] = potential_wave_operator_policy({"kinetic_order": 8})
    else:
        migrated["unified_wave_identity"] = "different-executor"
    with pytest.raises(ValueError):
        potential_support_policy({"support_projection_policy": "weighted_segment_and_retry"}, migrated)


@pytest.mark.parametrize("broken", ["request", "native_z", "source_mode", "arrays", "old_policy", "old_digest"])
def test_transition_gate_rejects_unannounced_changes_before_admission(broken):
    from temsim.physics.potential_wave_policy import transition_support_policy
    record, kwargs = _legacy_transition_fixture()
    if broken == "request":
        kwargs["request"]["wave_grid"]["maximum_pixels"] = 16384
    elif broken == "native_z":
        kwargs["native_plane_m"] = .499
    elif broken == "source_mode":
        kwargs["source_reference"]["mode_id"] = "changed-mode"
    elif broken == "arrays":
        kwargs["source_reference"]["array_descriptors"].pop("amplitude")
    elif broken == "old_policy":
        record["potential_support_policy"]["schema"] = "potential-support-policy-v1"
    else:
        record["unified_support_ledger_digest"] = "changed"
    with pytest.raises(ValueError):
        transition_support_policy(record, **kwargs)


@pytest.mark.parametrize("with_geometry", [False, True])
def test_migrated_policy_survives_real_propagation_cache_and_next_plane(tmp_path, with_geometry):
    from dataclasses import replace
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
    from temsim.physics.potential_wave_observation import observe_potential_checkpoint
    from temsim.physics.potential_wave_policy import transition_support_policy, validate_support_policy_transition
    from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
    from temsim.physics.unified_wave_support import prepare_unified_support
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_flux import BeamState
    from test_potential_wave_observation import _mode, _snapshot
    snapshot, mode = _snapshot(), _mode()
    record = prepare_unified_support({}, mode, enabled=True, l2_limit=.02, probability_limit=1e-6,
        source_digest="real-controlled-source", allow_allocate=True)
    old_policy = {"policy": "weighted_segment_and_retry", "trigger": "typed_resource_failure_only"}
    record.update(snapshot_digest=snapshot.numerical_digest, unified_wave_identity="old-executor",
        source_checkpoint_digest="real-controlled-source", source_mode_id=mode.mode_id,
        source_mode_weight_per_reference=mode.weight_per_reference_electron,
        potential_support_policy={"schema": "potential-support-policy-v2", **old_policy, "digest": json_digest(old_policy)})
    _, kwargs = _legacy_transition_fixture()
    kwargs["native_plane_m"] = 1.
    kwargs["source_reference"].update(source_checkpoint_digest="real-controlled-source", mode_id=mode.mode_id,
                                      mode_weight=mode.weight_per_reference_electron)
    migrated = transition_support_policy(record, **kwargs)
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1000., 1e-12, migrated)
    geometry = PotentialGeometrySnapshot(1., 1.000003, ())
    def observe(checkpoint, z):
        options = dict(maximum_step_m=5e-7, support_projection_policy="weighted_segment_and_retry")
        if with_geometry:
            return observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (z,), **options)[0]
        return observe_potential_checkpoint(snapshot, checkpoint, (z,), **options)[0]
    first = observe(entrance, 1.000001)
    validate_support_policy_transition(first.record)
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("migrated-support", with_geometry)
    restored = store.put(key, first)
    restored = store.get(key)
    continued = observe(restored, 1.000002)
    validate_support_policy_transition(continued.record)
    assert continued.record["support_policy_transition"] == migrated["support_policy_transition"]
    assert continued.record["support_ledger"] == migrated["support_ledger"]
    assert continued.beam.total_weight == pytest.approx(entrance.beam.total_weight, abs=2e-13)
    broken = dict(restored.record)
    broken.pop("support_policy_transition")
    with pytest.raises(ValueError, match="transition receipt"):
        observe(replace(restored, record=broken), 1.000002)


def test_first_pressure_event_binds_exact_legacy_prefix_and_survives_transition_validation():
    from temsim.physics.potential_wave_policy import transition_support_policy, validate_support_policy_transition
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.unified_wave_support import stamp_unified_support
    from test_potential_weighted_resampling import _pressure_context, _pressure_input
    record, kwargs = _legacy_transition_fixture()
    migrated = thaw_json(transition_support_policy(record, **kwargs))
    previous = NumericalSupportLedger.from_dict(migrated["support_ledger"])
    wave = _pressure_input()
    _, ledger = previous.resample_spectrum_pressure(wave, z_m=.52, mode_weight=.25,
        admissible_shape=lambda shape: max(shape) <= 64, resource_context=_pressure_context(wave))
    assert ledger.events[-1]["pressure_allocation"]["previous_ledger_digest"] == json_digest(record["support_ledger"])
    assert ledger.to_dict()["schema"] == "controlled-numerical-support-ledger-v4"
    assert ledger.events[:len(previous.events)] == previous.events
    assert ledger.events[-1]["probability_per_reference"] > ledger.maximum_probability/100.
    migrated["support_ledger"] = ledger.to_dict()
    migrated = stamp_unified_support(migrated)
    validate_support_policy_transition(migrated)
    assert potential_support_policy({"support_projection_policy": "weighted_segment_and_retry"}, migrated) == migrated["potential_support_policy"]
    corrupted = deepcopy(migrated)
    corrupted["support_ledger"]["events"].pop(0)
    corrupted["support_ledger"]["events"][0]["event_index"] = 0
    with pytest.raises(ValueError, match="prefix|remaining|preceding ledger"):
        validate_support_policy_transition(corrupted)


@pytest.mark.parametrize("policy", ["sampling_retry", "step_start_and_retry", "step_boundaries_and_retry"])
def test_other_support_policy_identities_remain_exactly_unchanged(policy):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    receipt = potential_support_policy({"support_projection_policy": policy, "support_ledger": NumericalSupportLedger(.02, 1e-6)})
    assert thaw_json(receipt) == {"schema": "potential-support-policy-v1", "policy": policy, "digest": json_digest(policy)}


@pytest.mark.parametrize("with_geometry", [False, True])
def test_real_checkpoint_method_continuation_and_same_plane_validation(tmp_path, with_geometry):
    from dataclasses import replace
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
    from temsim.physics.potential_wave_observation import observe_potential_checkpoint
    from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_flux import BeamState
    from test_potential_wave_observation import _mode, _snapshot
    snapshot = _snapshot()
    entrance = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), 1000., 1e-12,
                                {"source_scope": "controlled numerical exponential regression"})
    geometry = PotentialGeometrySnapshot(1., 1.000003, ())

    def observe(checkpoint, z, method="chebyshev", group=8):
        options = dict(maximum_step_m=5e-7, residual_exponential_method=method,
                       maximum_chebyshev_group=group)
        if with_geometry:
            return observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (z,), **options)[0]
        return observe_potential_checkpoint(snapshot, checkpoint, (z,), **options)[0]

    first = observe(entrance, 1.000001)
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 20*1024**2)
    key = store.key("chebyshev-policy", with_geometry)
    store.put(key, first)
    restored = store.get(key)
    continued = observe(restored, 1.000002)
    assert continued.record["potential_operator_policy"] == first.record["potential_operator_policy"]
    assert continued.beam.total_weight == pytest.approx(entrance.beam.total_weight, abs=2e-13)
    for z in (1.000001, 1.000002):
        for changed in (dict(method="taylor"), dict(group=16)):
            with pytest.raises(ValueError, match="policy changed"):
                observe(restored, z, **changed)
        with pytest.raises(ValueError, match="policy changed"):
            observe(replace(restored, record=dict(restored.record, potential_operator_policy=None)), z)
