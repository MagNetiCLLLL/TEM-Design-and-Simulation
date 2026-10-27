"""Bounded unified diagnostic routing with an explicit synthetic gun fixture."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.beam_path_audit import incident_checkpoints
from temsim.physics import particle_sections, unified_field_particle
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from test_particle_sections import fixture
from test_unified_field_particle import controlled


def forbid_legacy_paths(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unified audit must not reset, recalibrate, or repeat legacy transport")
    monkeypatch.setattr("temsim.physics.core.propagate", forbidden)
    monkeypatch.setattr("temsim.physics.aperture_clipping.clip_segment", forbidden)
    monkeypatch.setattr("temsim.physics.column_wall.clip_column_wall", forbidden)
    monkeypatch.setattr("temsim.physics.unified_scan_calibration.calibrate_unified_scan", forbidden)


@pytest.mark.parametrize("gun_stopped", [False, True])
def test_audit_preserves_source_identity_and_upstream_survivors_at_real_contacts(fixture, monkeypatch, gun_stopped):
    state, gun, calls = fixture
    state.simulation_mode = "unified"
    # The shared drift fixture reuses its zero array for several channels.
    # Give energy its own storage before applying distinct physical offsets.
    emitted = replace(gun.exit_bundle, energy_offset_ev=np.array((.3, -.1, .2)))
    gun = replace(gun, exit_bundle=emitted)
    prepared_sources = []
    def source(candidate):
        from temsim.vacuum import bind_gun_environment
        bind_gun_environment(candidate)
        prepared_sources.append(candidate)
        return gun
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", source)
    emitted.weight[:] = (.2, .3, .5)
    emitted.flight_time_s[:] = (1e-8, 1.1e-8, 1.2e-8)
    gun.flight_time_s[-1] = emitted.flight_time_s
    if gun_stopped:
        emitted.alive[-1] = False
        emitted.flight_time_s[-1] = gun.flight_time_s[-1, -1] = np.nan
        gun.blocked_z_mm[-1] = 449.5
        gun = replace(gun, blocked_key=("", "", "gun_aperture"))
    aperture = PosedWaveAperture("test_iris", 450.5, 450.6, .00005, CoordinateRegistration())
    monkeypatch.setattr(unified_field_particle, "get_unified_field_runtime", lambda current:
        replace(controlled(current, pieces=(aperture,)), metadata={
            "source_contract": "Explicit synthetic gun and real bounded column contact"}))
    execute = unified_field_particle.execute_unified_plan
    captured = []
    def record(*args, **kwargs):
        captured.append((args, kwargs))
        return execute(*args, **kwargs)
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", record)
    forbid_legacy_paths(monkeypatch)
    original = capture_instrument_snapshot(state).digest
    planes = (450., 450.25, 450.5, 451.)
    actual_gun, cp, masks = incident_checkpoints(state, planes, step_mm=.05)
    assert actual_gun is gun
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[0] is not state and args[0].step_mm == .05
    assert args[0] is not prepared_sources[0]
    assert hasattr(prepared_sources[0].electron_gun, "_vacuum_regions")
    expected_column = capture_instrument_snapshot(state).restore()
    expected_column.step_mm = .05
    from temsim import input_io
    with input_io.input_scope(expected_column, inherit=False):
        particle_sections._events(expected_column)
        expected_runtime = controlled(expected_column, pieces=(aperture,))
    assert args[1].unified_runtime.snapshot.numerical_digest == expected_runtime.snapshot.numerical_digest
    assert args[1].unified_runtime.geometry.digest == expected_runtime.geometry.digest
    assert kwargs["history_policy"] == "checkpoints" and "unified_resume" not in kwargs
    for actual, expected in zip(args[2:6], (emitted.x_m, emitted.tx_rad, emitted.y_m, emitted.ty_rad)):
        np.testing.assert_array_equal(actual, np.where(emitted.alive, expected, np.nan))
    np.testing.assert_array_equal(args[6], emitted.energy_offset_ev)
    np.testing.assert_array_equal(kwargs["initial_time_s"], emitted.flight_time_s)
    np.testing.assert_array_equal(emitted.weight, [.2, .3, .5])
    np.testing.assert_array_equal(kwargs["initial_kinetic_energy_ev"],
        state.electron_gun.nominal_exit_energy_ev + emitted.energy_offset_ev)
    np.testing.assert_array_equal(cp.z_mm, planes)
    np.testing.assert_array_equal(cp.flight_time_s[0], emitted.flight_time_s)
    np.testing.assert_array_equal(masks[:2], np.broadcast_to(emitted.alive, (2, 3)))
    np.testing.assert_array_equal(masks[2:], [[True, False, False]] * 2)
    # The real kernel retains the last contact coordinate after absorption;
    # masks, rather than a NaN convention, own whether that ray still exists.
    assert np.isfinite(cp.x_m[1, 1])
    assert cp.x_m[-1, 1] == cp.x_m[2, 1]
    assert cp.unified_state is None  # Observations do not invent a restart receipt.
    assert capture_instrument_snapshot(state).digest == original


def test_gun_exit_only_audit_preserves_raw_boundary_and_excludes_zero_current(fixture, monkeypatch):
    state, gun, _ = fixture
    state.simulation_mode = "unified"
    gun.exit_bundle.weight[-1] = 0.
    monkeypatch.setattr(unified_field_particle, "get_unified_field_runtime", controlled)
    forbid_legacy_paths(monkeypatch)
    actual, cp, masks = incident_checkpoints(state, [450.])
    assert actual is gun
    np.testing.assert_array_equal(cp.x_m[0], gun.exit_bundle.x_m)
    np.testing.assert_array_equal(cp.flight_time_s[0], gun.exit_bundle.flight_time_s)
    np.testing.assert_array_equal(masks, [[True, True, False]])


def test_missing_source_clock_rejects_before_column_execution(fixture, monkeypatch):
    state, _, _ = fixture
    state.simulation_mode = "unified"
    monkeypatch.setattr("temsim.physics.simulation._validate_gun_flight_times", lambda _: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing executed source clocks must not reach column transport")
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", forbidden)
    with pytest.raises(ValueError, match="executed gun flight-time"):
        incident_checkpoints(state, [451.])


@pytest.mark.parametrize("step", [True, float("nan"), float("inf"), 0., -.05])
def test_invalid_unified_audit_step_is_not_silently_replaced(fixture, step):
    state, _, calls = fixture
    state.simulation_mode = "unified"
    with pytest.raises(ValueError):
        incident_checkpoints(state, [451.], step_mm=step)
    assert not calls


def test_legacy_incident_observation_keeps_its_existing_route(fixture, monkeypatch):
    state, gun, _ = fixture
    state.simulation_mode = "ideal"
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy audit must not enter the unified route")
    monkeypatch.setattr("temsim.optics.beam_path_audit._unified_incident_checkpoints", forbidden)
    actual, cp, masks = incident_checkpoints(state, [450., 451.])
    assert actual is gun and masks.all()
    np.testing.assert_allclose(cp.x_m[-1], gun.exit_bundle.x_m + .001 * gun.exit_bundle.tx_rad,
                               rtol=1e-14, atol=1e-22)


def _bounded_batch_case(fixture, monkeypatch, scenario="contact"):
    """Synthetic full gun; real bounded field integration and physical stops."""
    state, original, _ = fixture
    state.simulation_mode = "unified"
    state.simulation_time_s = .21e-9
    emitted = replace(original.exit_bundle, energy_offset_ev=np.array((.3, -.1, .2)),
        weight=np.array((.2, .3, .5)), flight_time_s=np.array((1e-8, 1.1e-8, 1.2e-8)))
    gun = replace(original, exit_bundle=emitted,
                  flight_time_s=np.vstack((emitted.flight_time_s-1e-10, emitted.flight_time_s)))
    gun_calls = []
    def source(candidate):
        gun_calls.append(candidate.electron_gun.emitter.ray_count)
        return gun
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", source)
    aperture = PosedWaveAperture("batch_iris", 450.5, 450.6, .00005, CoordinateRegistration())
    def runtime(candidate):
        value = controlled(candidate, pieces=(aperture,) if scenario == "contact" else ())
        if scenario == "dynamic":
            from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
            from test_dynamic_potential import coil
            value = replace(value, snapshot=DynamicPotentialSnapshot(value.snapshot, (coil(),)),
                            metadata={"emission_time_s": candidate.simulation_time_s})
        return value
    monkeypatch.setattr(unified_field_particle, "get_unified_field_runtime", runtime)
    forbid_legacy_paths(monkeypatch)
    return state, gun, gun_calls


def _limit_to_batch(monkeypatch, state, plan, count, observation_count=2):
    budget = unified_field_particle.unified_history_memory_bytes(
        state, plan, count, history_policy="checkpoints")
    # Earlier slices stay resident while the final bounded batch executes.
    budget += 49 * observation_count * (3-count)
    monkeypatch.setattr(unified_field_particle, "MAX_UNIFIED_HISTORY_BYTES", budget)


@pytest.mark.parametrize("scenario", ["drift", "contact", "dynamic"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_real_batched_observations_are_bitwise_equal_with_original_measure(
        fixture, monkeypatch, scenario, batch_size):
    import weakref
    state, gun, gun_calls = _bounded_batch_case(fixture, monkeypatch, scenario)
    before = capture_instrument_snapshot(state).digest
    execute = unified_field_particle.execute_unified_plan
    calls, references = [], []
    def record(*args, **kwargs):
        if references:
            assert references[-1]() is None  # Previous immutable checkpoints were released.
        result = execute(*args, **kwargs)
        calls.append({"count": len(args[2]), "x": args[2].copy(),
                      "energy": kwargs["initial_kinetic_energy_ev"].copy(),
                      "plan": args[1]})
        references.append(weakref.ref(result[-1].x_m))
        return result
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", record)
    planes = (450., 450.25, 450.5, 451.)
    _, baseline, baseline_mask = incident_checkpoints(state, planes, step_mm=.05)
    assert len(calls) == 1 and calls[0]["count"] == 3
    _limit_to_batch(monkeypatch, state, calls[0]["plan"], batch_size, len(planes))
    calls.clear()
    actual_gun, actual, masks = incident_checkpoints(state, planes, step_mm=.05)
    assert actual_gun is gun and len(gun_calls) == 2  # Exactly once per complete audit.
    assert all(value == state.electron_gun.emitter.ray_count for value in gun_calls)
    assert [row["count"] for row in calls] == ([1, 1, 1] if batch_size == 1 else [2, 1])
    assert all(row["plan"] is calls[0]["plan"] for row in calls)
    np.testing.assert_array_equal(np.concatenate([row["x"] for row in calls]), gun.exit_bundle.x_m)
    np.testing.assert_array_equal(np.concatenate([row["energy"] for row in calls]),
        state.electron_gun.nominal_exit_energy_ev+gun.exit_bundle.energy_offset_ev)
    np.testing.assert_array_equal(gun.exit_bundle.weight, [.2, .3, .5])
    np.testing.assert_array_equal(gun.exit_bundle.ray_id, [1, 3, 8])
    for key in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(baseline, key))
    np.testing.assert_array_equal(masks, baseline_mask)
    assert actual.unified_state is None and all(ref() is None for ref in references)
    assert capture_instrument_snapshot(state).digest == before


def test_batches_use_full_population_checkpoint_grid_even_when_policy_depends_on_count(fixture, monkeypatch):
    from temsim.physics import simulation
    state, _, _ = _bounded_batch_case(fixture, monkeypatch, "drift")
    counts, plans = [], []
    def checkpoint_policy(start, stop, count):
        counts.append(count)
        return (450.37 if count == 3 else 450.63, stop)
    monkeypatch.setattr(simulation, "_column_checkpoint_planes", checkpoint_policy)
    execute = unified_field_particle.execute_unified_plan
    def record(*args, **kwargs):
        result = execute(*args, **kwargs)
        plans.append(args[1])
        return result
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", record)
    _, baseline, original_mask = incident_checkpoints(state, [450., 451.], step_mm=.05)
    _limit_to_batch(monkeypatch, state, plans[0], 1)
    _, actual, actual_mask = incident_checkpoints(state, [450., 451.], step_mm=.05)
    assert set(counts) == {3} and len(plans) == 4
    for plan in plans:
        np.testing.assert_array_equal(plan.z_mm, plans[0].z_mm)
        assert 450.37 in plan.z_mm and 450.63 not in plan.z_mm
    np.testing.assert_array_equal(actual.x_m, baseline.x_m)
    np.testing.assert_array_equal(actual_mask, original_mask)


@pytest.mark.parametrize("failure", ["cancel", "second_batch", "mesh"])
def test_incomplete_batched_audit_publishes_no_observations_and_keeps_source(fixture, monkeypatch, failure):
    state, gun, calls = _bounded_batch_case(fixture, monkeypatch, "drift")
    execute = unified_field_particle.execute_unified_plan
    plans = []
    def learn(*args, **kwargs):
        result = execute(*args, **kwargs)
        plans.append(args[1])
        return result
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", learn)
    incident_checkpoints(state, [450., 451.], step_mm=.05)
    _limit_to_batch(monkeypatch, state, plans[0], 1)
    cancelled = [False]
    state._tuning_cancelled = lambda: cancelled[0]
    before = capture_instrument_snapshot(state).digest
    source_weights = gun.exit_bundle.weight.copy()
    iterations = []
    def fail(*args, **kwargs):
        iterations.append(len(args[2]))
        if failure == "second_batch" and len(iterations) == 2:
            raise ValueError("Controlled second batch failure")
        result = execute(*args, **kwargs)
        if failure == "cancel":
            cancelled[0] = True
        elif failure == "mesh":
            changed = np.asarray(result[-1].z_mm).copy()
            changed[1] += 1e-4
            result = (*result[:-1], replace(result[-1], z_mm=changed))
        return result
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", fail)
    with pytest.raises(RuntimeError if failure == "cancel" else ValueError):
        incident_checkpoints(state, [450., 451.], step_mm=.05)
    assert iterations == ([1, 1] if failure == "second_batch" else [1])
    assert len(calls) == 2
    np.testing.assert_array_equal(gun.exit_bundle.weight, source_weights)
    assert capture_instrument_snapshot(state).digest == before


def test_audit_rejects_a_single_ray_exceeding_safe_history_budget(fixture, monkeypatch):
    state, _, calls = _bounded_batch_case(fixture, monkeypatch)
    monkeypatch.setattr(unified_field_particle, "MAX_UNIFIED_HISTORY_BYTES", 1)
    def forbidden(*args, **kwargs):
        pytest.fail("An infeasible history budget must fail before column allocation")
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", forbidden)
    with pytest.raises(ValueError, match="memory budget|One unified audit ray"):
        incident_checkpoints(state, [451.], step_mm=.05)
    assert len(calls) == 1


def test_audit_reserves_three_simultaneous_observation_copies_before_execution(fixture, monkeypatch):
    state, gun, _ = _bounded_batch_case(fixture, monkeypatch, "drift")
    # A large synthetic source needs no actual propagation for this admission
    # test. The budget admits two output copies but cannot hold publication's
    # batch arrays, concatenated arguments and immutable checkpoint copies.
    population, planes = 1200, np.linspace(450., 451., 21)
    original = gun.exit_bundle
    fields = ("x_m", "tx_rad", "y_m", "ty_rad", "energy_offset_ev", "weight", "alive", "flight_time_s")
    emitted = replace(original, **{key: np.resize(getattr(original, key), population) for key in fields},
                      ray_id=np.arange(population))
    gun = replace(gun, exit_bundle=emitted,
        **{key: np.vstack((getattr(emitted, key), getattr(emitted, key)))
           for key in ("x_m", "tx_rad", "y_m", "ty_rad")},
        flight_time_s=np.vstack((emitted.flight_time_s-1e-10, emitted.flight_time_s)),
        blocked_z_mm=np.full(population, np.nan), blocked_key=("",)*population)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", lambda _: gun)
    monkeypatch.setattr(unified_field_particle, "MAX_UNIFIED_HISTORY_BYTES", 120*len(planes)*population)
    def forbidden(*args, **kwargs):
        pytest.fail("Unpublishable observation arrays must be rejected before integration")
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", forbidden)
    with pytest.raises(ValueError, match="observation assembly"):
        incident_checkpoints(state, planes, step_mm=.05)


@pytest.mark.parametrize("count", [True, 2, 3.0])
def test_checkpoint_population_override_cannot_shrink_or_coerce_source(fixture, count):
    state, gun, _ = fixture
    e = gun.exit_bundle
    initial = (e.x_m, e.tx_rad, e.y_m, e.ty_rad, e.flight_time_s)
    with pytest.raises(ValueError, match="Checkpoint population"):
        particle_sections._trace_segment(state, "incident", 450., 451., initial,
            e.energy_offset_ev, e.weight, (e.ray_id, np.zeros(3)),
            (e.alive, gun.blocked_z_mm, list(gun.blocked_key)), (), None, float("inf"),
            initial_kicks=True, checkpoint_ray_count=count)


def test_batches_preserve_zero_weight_and_gun_stopped_source_columns(fixture, monkeypatch):
    state, gun, gun_calls = _bounded_batch_case(fixture, monkeypatch, "drift")
    emitted = gun.exit_bundle
    emitted.weight[:] = (.4, .6, 0.)
    emitted.alive[1] = False
    emitted.flight_time_s[1] = gun.flight_time_s[-1, 1] = np.nan
    gun.blocked_z_mm[1] = 449.5
    gun = replace(gun, blocked_key=("", "gun_aperture", ""))
    def source(candidate):
        gun_calls.append(candidate.electron_gun.emitter.ray_count)
        return gun
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", source)
    execute = unified_field_particle.execute_unified_plan
    plans, source_columns = [], []
    def record(*args, **kwargs):
        source_columns.append(args[2].copy())
        result = execute(*args, **kwargs)
        plans.append(args[1])
        return result
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", record)
    _, baseline, baseline_mask = incident_checkpoints(state, [450., 451.], step_mm=.05)
    _limit_to_batch(monkeypatch, state, plans[0], 1)
    source_columns.clear()
    _, actual, actual_mask = incident_checkpoints(state, [450., 451.], step_mm=.05)
    np.testing.assert_array_equal(np.concatenate(source_columns),
        np.where(emitted.alive, emitted.x_m, np.nan))
    np.testing.assert_array_equal(emitted.weight, [.4, .6, 0.])
    np.testing.assert_array_equal(actual_mask, [[True, False, False], [True, False, False]])
    np.testing.assert_array_equal(actual_mask, baseline_mask)
    for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(baseline, key))
    assert len(gun_calls) == 2 and actual.x_m.shape == (2, 3)


def test_batched_progress_is_monotone_across_each_full_column_slice(fixture, monkeypatch):
    from temsim.physics.transport_progress import transport_progress
    state, _, _ = _bounded_batch_case(fixture, monkeypatch, "drift")
    execute = unified_field_particle.execute_unified_plan
    plans = []
    def record(*args, **kwargs):
        result = execute(*args, **kwargs)
        plans.append(args[1])
        return result
    monkeypatch.setattr(unified_field_particle, "execute_unified_plan", record)
    incident_checkpoints(state, [450., 451.], step_mm=.05)
    _limit_to_batch(monkeypatch, state, plans[0], 2)
    labels = []
    with transport_progress(labels.append):
        incident_checkpoints(state, [450., 451.], step_mm=.05)
    values = [label.local_progress for label in labels if getattr(label, "local_progress", None) is not None]
    fractions = np.array([value.completed / value.total for value in values])
    assert len(fractions) > 2 and fractions[0] == 0. and fractions[-1] == 1.
    assert np.all(np.diff(fractions) >= 0.)
    assert any("1/2" in str(label) and label.local_progress.completed == 1_000_000
               and label.local_progress.total == 2_000_000 for label in labels)

