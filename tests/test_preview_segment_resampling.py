"""Explicit segment-start preview policy without source or budget renewal."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from temsim.immutable_json import json_digest, thaw_json
from temsim.optics.column import default_state
from temsim.physics.potential_wave_policy import (
    potential_support_policy, transition_preview_segment_policy,
    transition_support_policy, validate_support_policy_transition,
)
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
from test_potential_wave_policy import _legacy_transition_fixture
from test_unified_wave_runtime import _request, _runtime


def _request_preview(tmp_path, **changes):
    return _request(tmp_path, column_exploratory_preview=True,
        column_support_projection=True, column_weighted_resampling=True,
        column_preview_support_probability=.01, column_preview_support_l2=.5,
        **changes)


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_segment_flag_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="column_preview_segment_resampling"):
        replace(TipWaveRequest(), column_preview_segment_resampling=value).validate()


@pytest.mark.parametrize("changes", [dict(column_exploratory_preview=False),
    dict(column_support_projection=False), dict(column_weighted_resampling=False),
    dict(column_model="ideal_first_order")])
def test_segment_policy_requires_all_preview_guards(tmp_path, changes):
    request = replace(_request_preview(tmp_path), column_preview_segment_resampling=True, **changes)
    with pytest.raises(ValueError):
        request.validate()


def test_segment_policy_is_unified_only_and_keeps_gun_fields_and_numerics(tmp_path):
    state = default_state()
    state.simulation_mode = "unified"
    runtime = _runtime(state, 1., 1.000001)
    old_request = _request_preview(tmp_path)
    request = replace(old_request, column_preview_segment_resampling=True)
    old = UnifiedPotentialWaveAdapter(state, old_request, runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    assert TipWaveRequest().column_preview_segment_resampling is False
    assert old.support_projection_policy == "weighted_segment_and_retry"
    assert new.support_projection_policy == "preview_segment_and_retry"
    assert "column_preview_segment_resampling" not in old.numerical_profile
    assert new.numerical_profile["column_preview_segment_resampling"] is True
    assert new.identity != old.identity
    assert new.column_grid == old.column_grid
    assert new.snapshot.numerical_digest == old.snapshot.numerical_digest
    assert new.runtime.geometry.digest == old.runtime.geometry.digest
    assert new.operator_options == old.operator_options
    assert replace(request, column_preview_segment_resampling=False) == old_request
    for name in ("source", "surface", "gun", "radial_gun", "wave_grid"):
        assert getattr(request, name) == getattr(old_request, name)
    state.simulation_mode = "ideal"
    with pytest.raises(ValueError, match="only in unified"):
        UnifiedPotentialWaveAdapter(state, request)


def _preview_transition_fixture(*, linked=False):
    record, kwargs = _legacy_transition_fixture()
    request = asdict(TipWaveRequest(column_exploratory_preview=True,
        column_support_projection=True, column_weighted_resampling=True,
        column_preview_support_probability=.01, column_preview_support_l2=.5))
    kwargs.update(previous_request=deepcopy(request), request=deepcopy(request))
    if linked:
        record = thaw_json(transition_support_policy(record, **kwargs))
        kwargs["source_reference"]["previous_executor_identity"] = "new-executor"
        kwargs["source_reference"]["executor_identity"] = "preview-executor"
    else:
        record["potential_support_policy"] = thaw_json(potential_support_policy({
            "support_projection_policy": "weighted_segment_and_retry",
            "support_ledger": NumericalSupportLedger.from_dict(record["support_ledger"])}))
    kwargs["request"]["column_preview_segment_resampling"] = True
    kwargs["request"]["execution"]["cache_directory"] = "independent-preview-cache"
    return record, kwargs


@pytest.mark.parametrize("linked", [False, True])
def test_explicit_preview_transition_preserves_the_entire_original_context(linked):
    record, kwargs = _preview_transition_fixture(linked=linked)
    original = deepcopy(record)
    options = {"support_projection_policy": "preview_segment_and_retry"}
    with pytest.raises(ValueError, match="policy changed"):
        potential_support_policy(options, record)
    migrated = transition_preview_segment_policy(record, **kwargs)
    assert record == original
    assert thaw_json(migrated["support_policy_transition"]["previous_context"]) == original
    for key, value in original.items():
        if key not in ("potential_support_policy", "support_policy_transition", "unified_wave_identity"):
            assert json_digest(migrated[key]) == json_digest(value)
    policy = migrated["potential_support_policy"]
    assert policy["schema"] == "potential-support-policy-v4"
    assert policy["voluntary"] == "first_residual_at_segment_start_once"
    assert policy["voluntary_allocation"] == "original_one_percent_event_limit"
    assert policy["allocation"] == "remaining_for_admissible_grid"
    assert validate_support_policy_transition(migrated) == migrated["support_policy_transition"]
    assert potential_support_policy(options, migrated) == policy
    with pytest.raises(ValueError, match="resource-only v3"):
        transition_preview_segment_policy(migrated, **kwargs)


@pytest.mark.parametrize("broken", ["new_flag", "old_flag", "preview", "pixels", "probability", "l2", "step", "source", "arrays", "native_z"])
def test_preview_transition_rejects_every_unannounced_change(broken):
    record, kwargs = _preview_transition_fixture()
    if broken == "new_flag":
        kwargs["request"]["column_preview_segment_resampling"] = False
    elif broken == "old_flag":
        kwargs["previous_request"]["column_preview_segment_resampling"] = True
    elif broken == "preview":
        kwargs["request"]["column_exploratory_preview"] = False
    elif broken in ("probability", "l2"):
        kwargs["request"]["column_preview_support_" + broken] *= .5
    elif broken == "pixels":
        kwargs["request"]["column_preview_pixels"] = 1024
    elif broken == "step":
        kwargs["request"]["column_step_mm"] *= 2
    elif broken == "source":
        kwargs["source_reference"]["mode_id"] = "different-mode"
    elif broken == "arrays":
        kwargs["source_reference"]["array_descriptors"].pop("curvature_m1")
    else:
        kwargs["native_plane_m"] = .499
    with pytest.raises(ValueError):
        transition_preview_segment_policy(record, **kwargs)


@pytest.mark.parametrize("broken", ["receipt", "prefix", "budget", "source", "ancestor"])
def test_preview_restoration_fails_closed_if_any_original_evidence_is_lost(broken):
    record, kwargs = _preview_transition_fixture(linked=True)
    migrated = thaw_json(transition_preview_segment_policy(record, **kwargs))
    if broken == "receipt":
        migrated.pop("support_policy_transition")
    elif broken == "prefix":
        migrated["support_ledger"]["events"] = []
        migrated["support_ledger"]["schema"] = "controlled-numerical-support-ledger-v1"
    elif broken == "budget":
        migrated["support_ledger"]["maximum_probability"] *= 2
    elif broken == "source":
        migrated["source_mode_id"] = "different-mode"
    else:
        migrated["support_policy_transition"]["previous_context"].pop("support_policy_transition")
    with pytest.raises(ValueError):
        validate_support_policy_transition(migrated)


def test_fresh_preview_policy_has_a_distinct_receipt_without_reinterpreting_v3():
    ledger = NumericalSupportLedger(.02, 1e-6)
    old = potential_support_policy(dict(support_projection_policy="weighted_segment_and_retry", support_ledger=ledger))
    new = potential_support_policy(dict(support_projection_policy="preview_segment_and_retry", support_ledger=ledger))
    assert old["schema"] == "potential-support-policy-v3"
    assert old["trigger"] == "typed_resource_failure_only"
    assert new["schema"] == "potential-support-policy-v4"
    assert old != new
    historical = {"snapshot_digest": "executed", "support_ledger": ledger.to_dict()}
    with pytest.raises(ValueError):
        potential_support_policy(dict(support_projection_policy="preview_segment_and_retry"), historical)


@pytest.mark.parametrize("broken", ["kinetic", "electric", "geometry", "series"])
def test_preview_transition_cannot_rewrite_accepted_phase_or_geometry_history(broken):
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.electric_wave_reference import ElectricWaveHistory, ElectrostaticWaveReference
    record, kwargs = _preview_transition_fixture()
    kinetic = KineticPhaseLedger(.01).require("unchanged-field", 200000., 4, .49).append(.49, .5, (.001,))
    electric = ElectricWaveHistory(ElectrostaticWaveReference("unchanged-field", .49, 200000., 0.),
        .01, ({"lower_z_m": .49, "upper_z_m": .5, "forward_error_bound": .001},))
    record.update(kinetic_phase_ledger=kinetic.to_dict(), electric_wave_history=electric.to_dict(),
        potential_geometry={"last_z_m": .5, "entry_z_m": .49, "geometry_digest": "same-geometry",
            "source_scene_geometry_digest": "same-scene", "policy": {"maximum_step_m": .001},
            "normalization_repair": False, "physical_losses": [], "axial_resource_retries": [],
            "output_weight_per_reference": kwargs["source_reference"]["mode_weight"],
            "accepted_steps": 10, "sampling_refinements": 2, "series_error_log_bound": 1e-12})
    migrated = thaw_json(transition_preview_segment_policy(record, **kwargs))
    if broken == "kinetic":
        migrated["kinetic_phase_ledger"] = KineticPhaseLedger(.01).require(
            "unchanged-field", 200000., 4, .49).append(.49, .5, (.0001,)).to_dict()
    elif broken == "electric":
        migrated["electric_wave_history"] = replace(electric, events=(
            {"lower_z_m": .49, "upper_z_m": .5, "forward_error_bound": .0001},)).to_dict()
    elif broken == "geometry":
        migrated["potential_geometry"]["geometry_digest"] = "changed-geometry"
    else:
        migrated["potential_geometry"]["series_error_log_bound"] = 0.
    with pytest.raises(ValueError, match="original phase|geometry|cumulative propagation"):
        validate_support_policy_transition(migrated)


@pytest.mark.parametrize("count", [2, 9])
def test_actual_segments_receive_exact_native_start_once_per_mode_and_reuse_cache(tmp_path, monkeypatch, count):
    import temsim.physics.unified_wave_runtime as runtime_module
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from test_unified_weighted_resampling import _modes
    state = default_state()
    state.simulation_mode = "unified"
    request = _request_preview(tmp_path, column_preview_segment_resampling=True)
    runtime = _runtime(state, 1., 1.0000008)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    source = _modes(count)
    source_digest = source.digest
    store = ExecutedWaveStore(tmp_path/"segments", "controlled-preview-segments", 2**28)
    original_observe = runtime_module.observe_potential_geometry_checkpoint
    calls, completed = [], []
    def observe(snapshot, geometry, entrance, stops, **options):
        calls.append((entrance.beam.modes[0].mode_id, options["segment_start_m"], tuple(stops)))
        assert options["support_projection_policy"] == "preview_segment_and_retry"
        return original_observe(snapshot, geometry, entrance, stops, **options)
    monkeypatch.setattr(runtime_module, "observe_potential_geometry_checkpoint", observe)
    output = adapter.column(source, 1000.0006, store=store, checkpoint_callback=completed.append)
    assert source.digest == source_digest
    assert len(completed) >= 2
    assert len(calls) == count*len(completed)
    starts = [1., *[item.record["unified_executed_plane_z_m"] for item in completed[:-1]]]
    for index, start in enumerate(starts):
        assert [row[1] for row in calls[count*index:count*(index+1)]] == [start]*count
        assert [row[0] for row in calls[count*index:count*(index+1)]] == [mode.mode_id for mode in source.beam.modes]
    assert [mode.mode_id for mode in output.beam.modes] == [mode.mode_id for mode in source.beam.modes]
    summary = output.record["unified_support_summary"]
    assert source.beam.total_weight-output.beam.total_weight == pytest.approx(
        summary["used_probability_per_reference"], abs=2e-12)
    for before, after in zip(source.beam.modes, output.beam.modes, strict=True):
        context = output.record["unified_mode_contexts"][after.mode_id]
        allocation = context["unified_support_allocation"]
        assert allocation["input_weight_per_reference"] == before.weight_per_reference_electron
        assert allocation["maximum_probability_per_reference"] == before.weight_per_reference_electron*.01
        assert context["potential_support_policy"]["schema"] == "potential-support-policy-v4"
        assert np.isfinite(after.plane.amplitude).all()
    files = sorted(str(path.relative_to(store.root)) for path in store.root.rglob("*.npy"))
    def no_recompute(*args, **kwargs):
        raise AssertionError("A matching completed mode segment must be reused")
    monkeypatch.setattr(runtime_module, "observe_potential_geometry_checkpoint", no_recompute)
    reused = adapter.column(source, 1000.0006, store=store)
    assert reused.digest == output.digest
    assert sorted(str(path.relative_to(store.root)) for path in store.root.rglob("*.npy")) == files


def test_actual_nine_mode_v3_checkpoint_can_explicitly_continue_under_v4_without_budget_reset(tmp_path):
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from test_unified_weighted_resampling import _modes
    state = default_state()
    state.simulation_mode = "unified"
    runtime = _runtime(state, 1., 1.0000008)
    old_request = _request_preview(tmp_path)
    request = replace(old_request, column_preview_segment_resampling=True)
    old = UnifiedPotentialWaveAdapter(state, old_request, runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    source = _modes(9)
    complete = old.column(source, 1000.0002)
    store = ExecutedWaveStore(tmp_path/"transition", "original-complete-preview", 2**28)
    saved = store.put(store.key("original-complete"), complete)
    saved_digest = saved.digest
    record = thaw_json(saved.record)
    native = record["unified_executed_plane_z_m"]
    for mode, row in zip(saved.beam.modes, saved.beam.modes.rows, strict=True):
        previous = record["unified_mode_contexts"][mode.mode_id]
        source_reference = dict(checkpoint_digest=saved.digest, manifest_digest=saved.record["storage"]["manifest_digest"],
            source_checkpoint_digest=previous["unified_support_allocation"]["source_checkpoint_digest"],
            mode_id=mode.mode_id, mode_weight=mode.weight_per_reference_electron,
            array_descriptors=thaw_json(row["arrays"]), previous_executor_identity=old.identity,
            executor_identity=new.identity)
        record["unified_mode_contexts"][mode.mode_id] = transition_preview_segment_policy(previous,
            source_reference=source_reference, native_plane_m=native,
            previous_implementation="controlled-old-implementation", implementation="controlled-new-implementation",
            previous_request=asdict(old_request), request=asdict(request))
    record["unified_wave_identity"] = new.identity
    migrated = replace(saved, record=record)
    assert saved.digest == saved_digest
    for before, after in zip(saved.beam.modes, migrated.beam.modes, strict=True):
        assert before.weight_per_reference_electron == after.weight_per_reference_electron
        assert np.array_equal(before.plane.amplitude, after.plane.amplitude)
    output = new.column(migrated, 1000.0006)
    assert len(output.beam.modes) == 9
    for mode_id, previous in saved.record["unified_mode_contexts"].items():
        current = output.record["unified_mode_contexts"][mode_id]
        validate_support_policy_transition(current)
        old_ledger = NumericalSupportLedger.from_dict(previous["support_ledger"])
        ledger = NumericalSupportLedger.from_dict(current["support_ledger"])
        assert ledger.maximum_probability == old_ledger.maximum_probability
        assert ledger.maximum_l2 == old_ledger.maximum_l2
        assert ledger.events[:len(old_ledger.events)] == old_ledger.events
        assert current["unified_support_allocation"] == previous["unified_support_allocation"]
        assert current["support_policy_transition"]["previous_context"] == previous
    restored = store.put(store.key("continued"), output)
    assert new.column(restored, restored.plane_z_mm) is restored


def _published_weight_context(record, weight):
    record["potential_geometry"] = {"last_z_m": .5, "entry_z_m": .49,
        "output_weight_per_reference": weight, "geometry_digest": "same-geometry",
        "source_scene_geometry_digest": "same-scene", "policy": {"maximum_step_m": .001},
        "normalization_repair": False, "physical_losses": [], "axial_resource_retries": [],
        "accepted_steps": 10, "sampling_refinements": 0, "series_error_log_bound": 1e-12}


def test_preview_transition_preserves_exact_published_roundoff_above_source_allocation():
    record, kwargs = _preview_transition_fixture()
    original_weight = record["unified_support_allocation"]["input_weight_per_reference"]
    published = float(np.nextafter(original_weight, np.inf))
    kwargs["source_reference"]["mode_weight"] = published
    _published_weight_context(record, published)
    migrated = transition_preview_segment_policy(record, **kwargs)
    assert migrated["potential_geometry"]["output_weight_per_reference"] == published
    assert migrated["support_policy_transition"]["source_reference"]["mode_weight"] == published
    assert json_digest(migrated["support_ledger"]) == json_digest(record["support_ledger"])
    assert migrated["unified_support_allocation"]["input_weight_per_reference"] == original_weight


@pytest.mark.parametrize("broken", ["outside_lossless_tolerance", "invented_weight", "legacy_v2"])
def test_roundoff_admission_cannot_change_published_weight_or_extend_existing_limits(broken):
    from temsim.physics.wave_flux import FLUX_ATOL, FLUX_RTOL
    record, kwargs = (_legacy_transition_fixture() if broken == "legacy_v2" else _preview_transition_fixture())
    original_weight = record["unified_support_allocation"]["input_weight_per_reference"]
    published = float(np.nextafter(original_weight, np.inf))
    if broken == "outside_lossless_tolerance":
        published = original_weight*(1+2*FLUX_RTOL)+2*FLUX_ATOL
    kwargs["source_reference"]["mode_weight"] = published
    _published_weight_context(record, original_weight if broken == "invented_weight" else published)
    transition = transition_support_policy if broken == "legacy_v2" else transition_preview_segment_policy
    with pytest.raises(ValueError, match="published mode weight|source mode"):
        transition(record, **kwargs)


def test_previous_request_missing_new_flag_means_exact_false_default():
    record, kwargs = _preview_transition_fixture()
    kwargs["previous_request"].pop("column_preview_segment_resampling")
    migrated = transition_preview_segment_policy(record, **kwargs)
    assert migrated["potential_support_policy"]["schema"] == "potential-support-policy-v4"

