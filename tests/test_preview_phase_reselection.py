"""Explicit estimated representation changes retain original wave budgets."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from temsim.immutable_json import json_digest, thaw_json
from temsim.optics.column import default_state
from temsim.physics.potential_wave_policy import (
    potential_support_policy, representation_estimate_history,
    transition_preview_phase_policy, transition_preview_segment_policy,
    transition_support_policy, validate_support_policy_transition,
)
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
from test_preview_segment_resampling import _request_preview
from test_potential_wave_policy import _legacy_transition_fixture
from test_unified_wave_runtime import _runtime


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_phase_flag_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="column_preview_phase_reselection"):
        replace(TipWaveRequest(), column_preview_phase_reselection=value).validate()


@pytest.mark.parametrize("changes", [dict(column_exploratory_preview=False),
    dict(column_support_projection=False), dict(column_weighted_resampling=False),
    dict(column_preview_segment_resampling=False), dict(column_model="ideal_first_order")])
def test_phase_preparation_requires_every_explicit_preview_guard(tmp_path, changes):
    request = _request_preview(tmp_path, column_preview_segment_resampling=True,
        column_preview_phase_reselection=True)
    with pytest.raises(ValueError):
        replace(request, **changes).validate()


@pytest.mark.parametrize("kind", ["probability", "l2"])
@pytest.mark.parametrize("override", [False, True])
def test_zero_effective_budget_is_rejected_early_only_for_phase_preparation(tmp_path, kind, override):
    request = _request_preview(tmp_path, column_preview_segment_resampling=True,
        column_preview_phase_reselection=True)
    field = "column_preview_support_"+kind
    if override:
        request = replace(request, **{field: 0.})
    else:
        grid_field = "maximum_support_loss_fraction" if kind == "probability" else "maximum_accumulated_support_l2"
        request = replace(request, **{field: None}, wave_grid=replace(request.wave_grid, **{grid_field: 0.}))
    with pytest.raises(ValueError, match="positive effective"):
        request.validate()
    assert replace(request, column_preview_phase_reselection=False).validate()


def test_phase_profile_changes_only_explicit_column_representation_identity(tmp_path):
    state = default_state()
    state.simulation_mode = "unified"
    runtime = _runtime(state, 1., 1.000001)
    old_request = _request_preview(tmp_path, column_preview_segment_resampling=True)
    request = replace(old_request, column_preview_phase_reselection=True)
    old = UnifiedPotentialWaveAdapter(state, old_request, runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    assert TipWaveRequest().column_preview_phase_reselection is False
    assert old.support_projection_policy == "preview_segment_and_retry"
    assert new.support_projection_policy == "preview_phase_segment_and_retry"
    assert "column_preview_phase_reselection" not in old.numerical_profile
    assert new.numerical_profile["phase_preparation_backend"] == "CPU"
    assert new.numerical_profile["phase_preparation_maximum_pixels"] == 4096
    assert new.column_grid.maximum_pixels == 2048
    assert new.column_grid.maximum_working_bytes == old.column_grid.maximum_working_bytes
    assert new.column_grid == old.column_grid
    assert new.operator_options == old.operator_options
    assert new.snapshot.numerical_digest == old.snapshot.numerical_digest
    assert new.runtime.geometry.digest == old.runtime.geometry.digest
    assert new.identity != old.identity
    assert new.numerical_profile["accuracy_qualified"] is False
    for key in ("source", "gun", "surface", "radial_gun", "wave_grid"):
        assert getattr(old_request, key) == getattr(request, key)
    assert replace(request, column_preview_phase_reselection=False) == old_request
    state.simulation_mode = "ideal"
    with pytest.raises(ValueError, match="only in unified"):
        UnifiedPotentialWaveAdapter(state, request)


def _phase_transition_fixture(*, previous_version=3, linked=False):
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    record, kwargs = _legacy_transition_fixture()
    source_digest = json_digest({"controlled_fixture": "original executed source"})
    record["source_checkpoint_digest"] = source_digest
    allocation = record["unified_support_allocation"]
    allocation["source_checkpoint_digest"] = source_digest
    allocation["digest"] = json_digest({key: value for key, value in allocation.items() if key != "digest"})
    kwargs["source_reference"]["source_checkpoint_digest"] = source_digest
    request = asdict(TipWaveRequest(column_exploratory_preview=True,
        column_support_projection=True, column_weighted_resampling=True,
        column_preview_support_probability=allocation["probability_fraction_limit"],
        column_preview_support_l2=allocation["relative_l2_limit"]))
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
    if previous_version == 4:
        record = thaw_json(transition_preview_segment_policy(record, **kwargs))
        kwargs["previous_request"] = deepcopy(kwargs["request"])
        kwargs["previous_implementation"] = kwargs["implementation"]
        kwargs["source_reference"]["previous_executor_identity"] = kwargs["source_reference"]["executor_identity"]
    kwargs["implementation"] = "explicit-phase-implementation"
    kwargs["source_reference"]["executor_identity"] = "explicit-phase-executor"
    kwargs["request"]["column_preview_phase_reselection"] = True
    kwargs["request"]["execution"]["cache_directory"] = "independent-phase-cache"
    return record, kwargs


@pytest.mark.parametrize("previous_version", [3, 4])
@pytest.mark.parametrize("linked", [False, True])
def test_phase_transition_retains_full_prior_context_and_zero_new_spend(previous_version, linked):
    record, kwargs = _phase_transition_fixture(previous_version=previous_version, linked=linked)
    original = deepcopy(record)
    with pytest.raises(ValueError, match="policy changed"):
        potential_support_policy({"support_projection_policy": "preview_phase_segment_and_retry"}, record)
    migrated = transition_preview_phase_policy(record, **kwargs)
    assert record == original
    assert thaw_json(migrated["support_policy_transition"]["previous_context"]) == original
    for key, value in original.items():
        if key not in ("potential_support_policy", "support_policy_transition", "unified_wave_identity"):
            assert json_digest(migrated[key]) == json_digest(value)
    policy = migrated["potential_support_policy"]
    assert policy["schema"] == "potential-support-policy-v5"
    assert policy["representation_error"] == "estimated_not_certified"
    assert policy["representation_estimator"] == "two_dense_grids_shifted_relative_5percent_absolute_1e-7_norm_charged_v2"
    assert policy["phase_preparation"]["maximum_pixels"] == "min(65536, 2 * propagation_maximum_pixels)"
    assert representation_estimate_history(migrated).used_l2_estimate == 0.
    assert validate_support_policy_transition(migrated) == migrated["support_policy_transition"]
    restored = json.loads(json.dumps(thaw_json(migrated)))
    assert validate_support_policy_transition(restored) == restored["support_policy_transition"]
    assert representation_estimate_history(restored).to_dict() == representation_estimate_history(migrated).to_dict()


@pytest.mark.parametrize("broken", ["new_flag", "old_flag", "segment", "preview", "pixels",
    "probability", "l2", "step", "source", "arrays", "native_z"])
def test_phase_transition_rejects_unannounced_changes(broken):
    record, kwargs = _phase_transition_fixture()
    if broken == "new_flag":
        kwargs["request"]["column_preview_phase_reselection"] = False
    elif broken == "old_flag":
        kwargs["previous_request"]["column_preview_phase_reselection"] = True
    elif broken == "segment":
        kwargs["request"]["column_preview_segment_resampling"] = False
    elif broken == "preview":
        kwargs["request"]["column_exploratory_preview"] = False
    elif broken == "pixels":
        kwargs["request"]["column_preview_pixels"] = 1024
    elif broken in ("probability", "l2"):
        kwargs["request"]["column_preview_support_"+broken] *= .5
    elif broken == "step":
        kwargs["request"]["column_step_mm"] *= 2
    elif broken == "source":
        kwargs["source_reference"]["mode_id"] = "another-mode"
    elif broken == "arrays":
        kwargs["source_reference"]["array_descriptors"].pop("basis_m")
    else:
        kwargs["native_plane_m"] = .499
    with pytest.raises(ValueError):
        transition_preview_phase_policy(record, **kwargs)


@pytest.mark.parametrize("broken", ["history", "receipt", "strict_prefix", "allocation", "original_context"])
def test_phase_restore_rejects_lost_or_changed_original_evidence(broken):
    record, kwargs = _phase_transition_fixture(previous_version=4, linked=True)
    migrated = thaw_json(transition_preview_phase_policy(record, **kwargs))
    if broken == "history":
        migrated.pop("representation_estimates")
    elif broken == "receipt":
        migrated.pop("support_policy_transition")
    elif broken == "strict_prefix":
        migrated["support_ledger"]["events"] = []
    elif broken == "allocation":
        migrated["unified_support_allocation"]["maximum_l2_per_reference"] *= 2
    else:
        migrated["support_policy_transition"]["previous_context"]["source_mode_id"] = "changed"
    with pytest.raises(ValueError):
        validate_support_policy_transition(migrated)


def test_old_request_missing_phase_flag_is_false_and_v5_cannot_be_migrated_twice():
    record, kwargs = _phase_transition_fixture()
    kwargs["previous_request"].pop("column_preview_phase_reselection")
    migrated = transition_preview_phase_policy(record, **kwargs)
    with pytest.raises(ValueError, match="v3 or v4"):
        transition_preview_phase_policy(migrated, **kwargs)


def test_old_v3_to_v4_transition_accepts_only_the_new_false_default():
    from test_preview_segment_resampling import _preview_transition_fixture
    record, kwargs = _preview_transition_fixture()
    kwargs["previous_request"].pop("column_preview_phase_reselection")
    migrated = transition_preview_segment_policy(record, **kwargs)
    assert migrated["potential_support_policy"]["schema"] == "potential-support-policy-v4"
    assert "representation_estimates" not in migrated
    kwargs["request"]["column_preview_phase_reselection"] = True
    with pytest.raises(ValueError, match="cannot change other"):
        transition_preview_segment_policy(record, **kwargs)


def test_material_context_keeps_representation_history_without_issuing_budget():
    from temsim.physics.potential_material_phase_budget import kinetic_context
    record, kwargs = _phase_transition_fixture()
    migrated = transition_preview_phase_policy(record, **kwargs)
    result = kinetic_context(migrated)
    assert result["representation_estimates"] == thaw_json(migrated["representation_estimates"])
    assert result["support_policy_transition"] == thaw_json(migrated["support_policy_transition"])


def test_inherited_material_estimates_remain_read_only_and_cannot_be_reinitialized():
    from types import SimpleNamespace
    from temsim.physics.potential_support_lineage import capture_inherited_support
    record, kwargs = _phase_transition_fixture()
    migrated = thaw_json(transition_preview_phase_policy(record, **kwargs))
    mode = SimpleNamespace(mode_id=kwargs["source_reference"]["mode_id"], weight_per_reference_electron=.25)
    checkpoint = SimpleNamespace(record=migrated, beam=SimpleNamespace(modes=(mode,)),
        plane_z_mm=500., digest=json_digest({"fixture": "complete parent"}))
    prefix = capture_inherited_support(checkpoint, migrated["snapshot_digest"])
    inherited = {**migrated, "inherited_potential_support": prefix, "support_prefix_digest": prefix["digest"]}
    inherited.pop("support_ledger")
    assert representation_estimate_history(inherited).to_dict() == migrated["representation_estimates"]
    assert potential_support_policy({}, inherited) == migrated["potential_support_policy"]
    with pytest.raises(ValueError, match="disabled after material branching"):
        potential_support_policy({"support_projection_policy": "preview_phase_segment_and_retry"}, inherited)
    inherited.pop("representation_estimates")
    with pytest.raises(ValueError, match="representation estimate history"):
        representation_estimate_history(inherited, initialize=True)


def test_representation_history_cannot_be_borrowed_from_another_source_or_mode():
    record, kwargs = _phase_transition_fixture()
    migrated = thaw_json(transition_preview_phase_policy(record, **kwargs))
    for name, changed in (("source_mode_id", "another-mode"),
                          ("source_checkpoint_digest", json_digest({"other": "source"}))):
        broken = deepcopy(migrated)
        broken["representation_estimates"]["source_binding"][name] = changed
        with pytest.raises(ValueError, match="another source"):
            representation_estimate_history(broken)


@pytest.mark.parametrize("previous_version", [3, 4])
def test_actual_nine_mode_transition_continues_and_reuses_complete_cache(tmp_path, monkeypatch, previous_version):
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from test_unified_weighted_resampling import _modes
    import temsim.physics.unified_wave_runtime as runtime_module
    state = default_state()
    state.simulation_mode = "unified"
    runtime = _runtime(state, 1., 1.0000008)
    old_request = _request_preview(tmp_path, column_preview_segment_resampling=previous_version == 4)
    request = replace(old_request, column_preview_segment_resampling=True, column_preview_phase_reselection=True)
    old = UnifiedPotentialWaveAdapter(state, old_request, runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    source = _modes(9)
    complete = old.column(source, 1000.0002)
    store = ExecutedWaveStore(tmp_path/f"phase-v{previous_version}", "controlled-complete-phase-transition", 2**28)
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
        record["unified_mode_contexts"][mode.mode_id] = transition_preview_phase_policy(previous,
            source_reference=source_reference, native_plane_m=native,
            previous_implementation="controlled-old-implementation", implementation="controlled-phase-implementation",
            previous_request=asdict(old_request), request=asdict(request))
    record["unified_wave_identity"] = new.identity
    migrated = replace(saved, record=record)
    assert saved.digest == saved_digest
    for before, after in zip(saved.beam.modes, migrated.beam.modes, strict=True):
        assert before.weight_per_reference_electron == after.weight_per_reference_electron
        for key in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_array_equal(getattr(before.plane, key), getattr(after.plane, key))
    original_observe = runtime_module.observe_potential_geometry_checkpoint
    calls = []
    def observe(snapshot, geometry, entrance, stops, **options):
        assert options["support_projection_policy"] == "preview_phase_segment_and_retry"
        assert options["representation_estimates"] == entrance.record["representation_estimates"]
        assert options["segment_start_m"] <= stops[0]
        calls.append((entrance.beam.modes[0].mode_id, options["segment_start_m"]))
        return original_observe(snapshot, geometry, entrance, stops, **options)
    monkeypatch.setattr(runtime_module, "observe_potential_geometry_checkpoint", observe)
    completed = []
    output = new.column(migrated, 1000.0006, store=store, checkpoint_callback=completed.append)
    assert len(output.beam.modes) == 9
    assert len(calls) == 9*len(completed)
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
        representation_estimate_history(current).require_prefix(
            representation_estimate_history(migrated.record["unified_mode_contexts"][mode_id]))
    profile = output.record["unified_numerical_profile"]
    assert profile["representation_estimates_certified"] is False
    assert profile["accuracy_qualified"] is False
    files = sorted(str(path) for path in store.root.rglob("*.npy"))
    def no_recompute(*args, **kwargs):
        raise AssertionError("Completed phase-mode segments must be reused without further execution")
    monkeypatch.setattr(runtime_module, "observe_potential_geometry_checkpoint", no_recompute)
    reused = new.column(migrated, 1000.0006, store=store)
    assert reused.digest == output.digest
    assert sorted(str(path) for path in store.root.rglob("*.npy")) == files
    assert new.column(reused, reused.plane_z_mm) is reused
    broken = thaw_json(reused.record)
    next(iter(broken["unified_mode_contexts"].values())).pop("representation_estimates")
    with pytest.raises(ValueError, match="representation estimate history"):
        new.column(replace(reused, record=broken), reused.plane_z_mm)


def test_v5_streamed_recording_stop_preserves_estimates_and_original_source(tmp_path):
    from types import SimpleNamespace
    from temsim.detector.wave_readout import _recording_stop_streamed
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_flux import BeamState
    from test_unified_wave_runtime import _resolved_support_checkpoint
    state = default_state()
    state.simulation_mode = "unified"
    source = _resolved_support_checkpoint()
    first_mode = source.beam.modes[0]
    second = replace(first_mode, mode_id="outside-disk", weight_per_reference_electron=.13,
        plane=first_mode.plane._replace_metadata(origin_m=np.array((8e-6, 0.))))
    source = replace(source, beam=BeamState((first_mode, second), source.beam.reference_plane))
    request = _request_preview(tmp_path, column_preview_segment_resampling=True,
        column_preview_phase_reselection=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state, 1., 1.000004))
    first = adapter.column(source, 1000.0001)
    detector = SimpleNamespace(key="inserted-disk", geometry="disk", outer_width_mm=.006,
        inner_diameter_mm=0., readout_enabled=False)
    store = ExecutedWaveStore(tmp_path/"streamed-v5", "streamed-estimate-history", 2**28)
    stopped = _recording_stop_streamed(first, detector, store, verify=lambda: None, cancelled=lambda: False)
    assert adapter.column(stopped, stopped.plane_z_mm) is stopped
    output = adapter.column(stopped, 1000.0002)
    assert output.record["unified_source_checkpoint_digest"] == source.digest
    assert [row["detector"] for row in output.record["unified_recording_stops"]] == [detector.key]
    for mode_id, context in output.record["unified_mode_contexts"].items():
        previous = first.record["unified_mode_contexts"][mode_id]
        assert context["unified_support_allocation"] == previous["unified_support_allocation"]
        representation_estimate_history(context).require_prefix(representation_estimate_history(previous))
        assert len(context["external_recording_losses"]) == 1


def test_actual_nonempty_two_mode_estimates_survive_resume_cache_and_read_only_material(tmp_path, monkeypatch):
    from temsim.physics.potential_representation_estimates import require_combined_support_allowance
    from temsim.physics.potential_support_lineage import capture_inherited_support
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_flux import BeamState
    from test_potential_geometry import _checkpoint
    from test_preview_phase_carrier import _phase_case
    import temsim.physics.unified_wave_runtime as runtime_module
    state = default_state()
    state.simulation_mode = "unified"
    _, wave, _, _ = _phase_case()
    source = _checkpoint(wave)
    first_mode = replace(source.beam.modes[0], energy_kev=200.)
    second_mode = replace(first_mode, mode_id="second-real-phase-mode", weight_per_reference_electron=.13,
        plane=replace(wave, amplitude=wave.amplitude*np.exp(.29j)))
    source = replace(source, plane_z_mm=1., beam=BeamState((first_mode, second_mode), source.beam.reference_plane))
    request = _request_preview(tmp_path, column_preview_segment_resampling=True,
        column_preview_phase_reselection=True, column_kinetic_order=8)
    request = replace(request, column_step_mm=.001,
        execution=replace(request.execution, segment_steps=1),
        wave_grid=replace(request.wave_grid, maximum_pixels=256))
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state, .001, .001004))
    store = ExecutedWaveStore(tmp_path/"nonempty-phase", "controlled-real-nonempty-estimates", 2**28)
    original_digest = source.digest
    first = adapter.column(source, 1.001, store=store)
    assert source.digest == original_digest
    for context in first.record["unified_mode_contexts"].values():
        history = representation_estimate_history(context)
        assert history.events and history.used_l2_estimate > 0.
        assert all(event["receipt"]["certified"] is False for event in history.events)
    cached = store.get(first.record["storage"]["key"])
    assert cached.digest == first.digest
    output = adapter.column(cached, 1.002, store=store)
    for mode_id, context in output.record["unified_mode_contexts"].items():
        previous = first.record["unified_mode_contexts"][mode_id]
        history = representation_estimate_history(context)
        history.require_prefix(representation_estimate_history(previous))
        strict = NumericalSupportLedger.from_dict(context["support_ledger"])
        old_strict = NumericalSupportLedger.from_dict(previous["support_ledger"])
        assert strict.events[:len(old_strict.events)] == old_strict.events
        assert strict.maximum_l2 == old_strict.maximum_l2
        assert strict.maximum_probability == old_strict.maximum_probability
        remaining = require_combined_support_allowance(strict, history)
        assert remaining < strict.maximum_l2-strict.used_l2
        assert remaining >= 0.
        assert context["unified_support_allocation"] == previous["unified_support_allocation"]
    profile = output.record["unified_numerical_profile"]
    assert profile["estimated_representation_l2_per_reference"] > 0.
    assert profile["combined_projection_and_estimated_l2_per_reference"] > profile["complex_l2_projection_bound_per_reference"]
    # Retiring one real accepted mode to a material prefix preserves its
    # nonempty estimates but grants no active descendant projection budget.
    entrance = adapter._entrance(output, output.beam.modes[0])
    prefix = capture_inherited_support(entrance, adapter.snapshot.numerical_digest)
    inherited = {**thaw_json(entrance.record), "inherited_potential_support": prefix,
                 "support_prefix_digest": prefix["digest"]}
    inherited.pop("support_ledger")
    assert representation_estimate_history(inherited).to_dict() == representation_estimate_history(entrance.record).to_dict()
    assert potential_support_policy({}, inherited)["schema"] == "potential-support-policy-v5"
    with pytest.raises(ValueError, match="disabled after material branching"):
        potential_support_policy({"support_projection_policy": "preview_phase_segment_and_retry"}, inherited)
    files = sorted(str(path) for path in store.root.rglob("*.npy"))
    def no_recompute(*args, **kwargs):
        raise AssertionError("A nonempty accepted estimate history must be reused with its mode segment")
    monkeypatch.setattr(runtime_module, "observe_potential_geometry_checkpoint", no_recompute)
    reused = adapter.column(cached, 1.002, store=store)
    assert reused.digest == output.digest
    assert sorted(str(path) for path in store.root.rglob("*.npy")) == files
    assert adapter.column(reused, reused.plane_z_mm) is reused

