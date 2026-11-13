"""Mock transport and exact-array growth proofs; no physical source trials."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.base import (
    GunEqualTimeHistory, GunExitBundle, GunPlaneArrival, GunTraceResult,
    native_history_for_ray,
)
from temsim.optics.electron_gun.emitter import ColdFieldEmitter, EmissionSampling, HALTON_PREFIX_SAMPLING_SCHEMA
from temsim.physics import incremental_particles as incremental


def emitter(count=9):
    result = ColdFieldEmitter(0., 1., 1., ray_count=count)
    result.sampling = EmissionSampling(schema=HALTON_PREFIX_SAMPLING_SCHEMA, seed=0)
    return result


def state(count=9):
    source = emitter(count)
    gun = SimpleNamespace(emitter=source, type_key="cold_feg", ray_count=count,
                          emitted_current_a=1e-9)
    return SimpleNamespace(electron_gun=gun, vacuum_map=SimpleNamespace(enabled=False),
                           nanopulser=SimpleNamespace(enabled=False))


def metadata(count):
    return {"schema": incremental.SCHEMA, "implementation": "f"*64,
        "source_parameters_digest": "a"*64, "gun_dependency_digest": "b"*64,
        "sampling": {"schema": HALTON_PREFIX_SAMPLING_SCHEMA, "table_sha": "c"*64},
        "emission_epoch_s": 0., "original_count": count, "reference_current_a": 1e-9,
        "column_input_digest": "d"*64}


def synthetic_trace(gun, bundle):
    """Deliberately population-dependent mock steps expose accidental retraces."""
    n = len(bundle.ray_id)
    x = bundle.x_m + n*1e-13
    zero = np.zeros(n)
    alive = bundle.ray_id % 4 != 0
    clock = np.where(alive, (1.+n*.01)*1e-8, np.nan)
    values = dict(x_m=x, y_m=bundle.y_m.copy(), tx_rad=bundle.tx_rad.copy(), ty_rad=bundle.ty_rad.copy())
    histories = {key: np.vstack((getattr(bundle, key), value)) for key, value in values.items()}
    native = GunEqualTimeHistory(np.array([0., (1.+n*.01)*1e-8]),
        np.vstack((zero, np.ones(n))), **histories,
        alive=np.vstack((np.ones(n, bool), alive)), completed=np.vstack((np.zeros(n, bool), alive)))
    stopped = np.where(alive, np.nan, .5)
    labels = tuple("" if value else "physical-stop" for value in alive)
    phase = incremental.capture_emission_phase(gun, bundle)
    report = dict(transmitted_particle_count=int(alive.sum()), minimum_exit_kinetic_energy_ev=float(n),
        maximum_exit_kinetic_energy_ev=float(n+1), minimum_expected_exit_kinetic_energy_ev=float(n),
        maximum_expected_exit_kinetic_energy_ev=float(n+1), maximum_exit_energy_error_ev=n*1e-9,
        surface_mesh_displacement_m=0., launch_boundary={"original_count": n}, model="synthetic")
    trace = GunTraceResult(np.array([0., 1.]), **histories,
        exit_bundle=GunExitBundle(**values, energy_offset_ev=bundle.energy_offset_ev.copy(),
            weight=bundle.weight.copy(), ray_id=bundle.ray_id.copy(), alive=alive, flight_time_s=clock),
        blocked_z_mm=stopped, blocked_key=labels, emitted_current_a=gun.emitted_current_a,
        dpa_transmitted_current_a=gun.emitted_current_a*float(bundle.weight[alive].sum()),
        c1_transmitted_current_a=gun.emitted_current_a*float(bundle.weight[alive].sum()),
        equal_time_history=native,
        plane_arrivals=(GunPlaneArrival("exit", "Exit", 1., clock, x, zero, alive.copy(), alive.copy()),),
        emission_reference={key: phase[key] for key in ("ray_id", "position_m", "direction", "normal")},
        flight_time_s=np.vstack((zero, clock)), emission_phase=phase,
        electrostatic_model_report=report)
    return trace


def setup_extension(monkeypatch, old_count=9, new_count=14):
    old_state = state(old_count)
    old = replace(synthetic_trace(old_state.electron_gun, old_state.electron_gun.emitter.emit()),
                  incremental_source=metadata(old_count))
    new_state = state(new_count)
    new_state.electron_gun.emit = lambda: new_state.electron_gun.emitter.emit()
    calls = []
    def trace(gun, *, cancelled=None, emitted_bundle=None):
        calls.append(emitted_bundle.ray_id.copy())
        return synthetic_trace(gun, emitted_bundle)
    monkeypatch.setattr("temsim.optics.electron_gun.tracing._trace_feg_to_exit", trace)
    monkeypatch.setattr("temsim.physics.grounded_particle_step.gun_step_workers", nullcontext)
    return old, new_state, calls


def test_only_new_original_ids_enter_transport_and_old_phase_is_exact(monkeypatch):
    old, current, calls = setup_extension(monkeypatch)
    original = deepcopy(old)
    result = incremental.extend_source_trace(current, old, metadata(14))
    np.testing.assert_array_equal(calls, [np.arange(9, 14)])
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s"):
        np.testing.assert_array_equal(getattr(result, name)[:, :9], getattr(old, name))
        np.testing.assert_array_equal(getattr(original, name), getattr(old, name))
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "ray_id", "alive", "flight_time_s"):
        np.testing.assert_array_equal(getattr(result.exit_bundle, name)[:9], getattr(old.exit_bundle, name))
    np.testing.assert_array_equal(result.blocked_z_mm[:9], old.blocked_z_mm)
    assert result.blocked_key[:9] == old.blocked_key
    assert result.equal_time_history is None
    assert result.transport_partitions[0] is old
    history, local = native_history_for_ray(result, 12)
    assert history is result.transport_partitions[1].equal_time_history and local == 3
    assert result.equal_time_front_at_plane("exit").x_m.shape == (14,)


def test_new_current_measure_preserves_losses_and_old_weights(monkeypatch):
    old, current, _ = setup_extension(monkeypatch)
    result = incremental.extend_source_trace(current, old, metadata(14))
    np.testing.assert_array_equal(old.exit_bundle.weight, np.full(9, 1/9))
    np.testing.assert_array_equal(result.exit_bundle.weight, np.full(14, 1/14))
    expected = 1e-9*np.count_nonzero(result.exit_bundle.alive)/14
    assert result.c1_transmitted_current_a == pytest.approx(expected)
    assert result.emitted_current_a == old.emitted_current_a == 1e-9
    assert result.incremental_source["old_measure_ratio_in_new_result"] == 9/14
    assert result.incremental_source["transported_original_count"] == 5
    report = result.electrostatic_model_report
    assert report["transmitted_particle_count"] == np.count_nonzero(result.exit_bundle.alive)
    assert report["minimum_exit_kinetic_energy_ev"] == 5
    assert report["maximum_exit_kinetic_energy_ev"] == 10
    assert report["maximum_exit_energy_error_ev"] == old.electrostatic_model_report["maximum_exit_energy_error_ev"]
    assert len(report["native_partition_reports"]) == 2


@pytest.mark.parametrize("field", ["implementation", "source_parameters_digest", "gun_dependency_digest",
    "sampling", "emission_epoch_s", "reference_current_a"])
def test_changed_dependencies_never_enter_append_transport(monkeypatch, field):
    old, current, calls = setup_extension(monkeypatch)
    changed = metadata(14)
    changed[field] = "different"
    assert incremental.extend_source_trace(current, old, changed) is None
    assert calls == []


@pytest.mark.parametrize("field", ["energy_offset_ev", "position_m", "direction", "normal", "emission_time_s", "ray_id"])
def test_prefix_mutation_fails_before_transport(monkeypatch, field):
    old, current, calls = setup_extension(monkeypatch)
    altered = dict(old.emission_phase)
    altered[field] = altered[field].copy()
    altered[field].flat[0] += 1
    old = replace(old, emission_phase=altered)
    with pytest.raises(ValueError, match="prefix changed"):
        incremental.extend_source_trace(current, old, metadata(14))
    assert calls == []


def test_arbitrary_historical_measure_is_not_relabelled(monkeypatch):
    old, current, calls = setup_extension(monkeypatch)
    phase = {**old.emission_phase, "weight": np.full(9, .5/9)}
    with pytest.raises(ValueError, match="1/N quadrature"):
        incremental.extend_source_trace(current, replace(old, emission_phase=phase), metadata(14))
    assert calls == []


def test_repeated_growth_flattens_native_lineage_without_rewriting_partitions(monkeypatch):
    old, current, calls = setup_extension(monkeypatch)
    middle = incremental.extend_source_trace(current, old, metadata(14))
    current.electron_gun.emitter.ray_count = current.electron_gun.ray_count = 18
    result = incremental.extend_source_trace(current, middle, metadata(18))
    assert [len(part.exit_bundle.ray_id) for part in result.transport_partitions] == [9, 5, 4]
    assert result.transport_partitions[0] is old
    np.testing.assert_array_equal(result.x_m[:, :14], middle.x_m)
    np.testing.assert_array_equal(calls[-1], np.arange(14, 18))


def test_historical_missing_launch_arrays_do_not_become_a_seed(monkeypatch):
    old, current, calls = setup_extension(monkeypatch)
    with pytest.raises(ValueError, match="complete original launch"):
        incremental.extend_source_trace(current, replace(old, emission_phase=None), metadata(14))
    assert calls == []


def test_only_source_count_is_removed_from_complete_column_inputs(monkeypatch):
    from temsim.optics.column import default_state
    s = default_state()
    original = incremental.column_input_digest(s)
    s.electron_gun.emitter.ray_count *= 2
    assert incremental.column_input_digest(s) == original
    s.step_mm *= .5
    assert incremental.column_input_digest(s) != original


def test_metadata_removes_only_documented_count_leaves_without_field_execution(monkeypatch):
    import json
    from temsim.optics.column import default_state
    s = default_state()
    s.electron_gun.emitter.sampling = EmissionSampling(schema=HALTON_PREFIX_SAMPLING_SCHEMA)
    for component in s.deflectors:
        if hasattr(component, "scan_enabled"):
            component.scan_enabled = False
    s.ac_deflector.scan_enabled = False
    s.descan_deflector.scan_enabled = False
    monkeypatch.setattr(incremental, "column_input_digest", lambda _s: "d"*64)
    s.electron_gun._instrument_magnetic_identity = "e"*64
    cache = {"requested_count": s.electron_gun.ray_count,
             "components": s.electron_gun.to_dict()["components"],
             "executed_components": s.electron_gun.to_dict()["components"],
             "unchanged_solver_and_boundary": {"field": "exact"}}
    monkeypatch.setattr(s.electron_gun, "_cache_key", lambda _n: json.dumps(cache))
    first = incremental.source_metadata(s)
    s.electron_gun.emitter.ray_count *= 2
    cache["requested_count"] = s.electron_gun.ray_count
    for owner in ("components", "executed_components"):
        cache[owner][s.electron_gun.emitter.key]["ray_count"] = s.electron_gun.ray_count
    second = incremental.source_metadata(s)
    assert first["gun_dependency_digest"] == second["gun_dependency_digest"]
    assert first["source_parameters_digest"] == second["source_parameters_digest"]
    cache["unchanged_solver_and_boundary"]["field"] = "changed"
    assert incremental.source_metadata(s)["gun_dependency_digest"] != second["gun_dependency_digest"]


def test_legacy_growth_rejected_but_explicit_different_source_is_fresh(monkeypatch):
    from temsim.optics.column import default_state
    s = default_state()
    s.electron_gun.emitter.sampling = None
    trace = SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=np.arange(9)),
        incremental_source={"source_parameters_digest": incremental.json_digest(incremental._source_payload(s.electron_gun))})
    previous = SimpleNamespace(simulation=SimpleNamespace(gun_trace=trace))
    s.electron_gun.emitter.ray_count += 1
    with pytest.raises(ValueError, match="historical batch-dependent source"):
        incremental.validate_particle_count_growth(previous, s)
    s.electron_gun.emitter.energy_spread_fwhm_ev += .1
    incremental.validate_particle_count_growth(previous, s)


@pytest.mark.parametrize("setting", ["dynamic", "medium", "collective"])
def test_same_source_unsupported_transport_growth_is_not_silently_recomputed(monkeypatch, setting):
    s = state(14)
    gun = s.electron_gun
    gun.to_dict = lambda: {"components": {gun.emitter.key: {"ray_count": gun.ray_count, "sampling": HALTON_PREFIX_SAMPLING_SCHEMA}}}
    if setting == "dynamic":
        s.deflectors = (SimpleNamespace(enabled=True, scan_enabled=True),)
    elif setting == "medium":
        s.vacuum_map.enabled = True
    else:
        s.collective_fields_enabled = True
    record = {"schema": incremental.NON_NESTED_SCHEMA, "column_input_digest": "same-inputs",
        "source_parameters_digest": incremental.json_digest(incremental._source_payload(gun))}
    previous = SimpleNamespace(gun_trace=SimpleNamespace(
        exit_bundle=SimpleNamespace(ray_id=np.arange(9)), incremental_source=record))
    monkeypatch.setattr(incremental, "column_input_digest", lambda _s: "same-inputs")
    with pytest.raises(ValueError, match="source prefix is stable, but append-only transport reuse is unavailable"):
        incremental.validate_particle_count_growth(previous, s)
    # A changed field/geometry/solver is a different, explicitly edited input.
    monkeypatch.setattr(incremental, "column_input_digest", lambda _s: "different-inputs")
    incremental.validate_particle_count_growth(previous, s)


def test_native_prefix_is_used_for_candidate_routing_without_emission_or_fields(monkeypatch):
    s = state(14)
    gun = s.electron_gun
    gun.to_dict = lambda: {"components": {gun.emitter.key: {"ray_count": gun.ray_count, "sampling": HALTON_PREFIX_SAMPLING_SCHEMA}}}
    old = synthetic_trace(gun, gun.emitter.emit(9))
    record = {**metadata(9), "source_parameters_digest": incremental.json_digest(incremental._source_payload(gun))}
    old = replace(old, incremental_source=record)
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "f"*64)
    monkeypatch.setattr(gun.emitter, "emit", lambda *_: pytest.fail("Candidate routing must not emit"))
    assert incremental.can_extend_particle_result(SimpleNamespace(gun_trace=old), s)
    s.vacuum_map.enabled = True
    assert not incremental.can_extend_particle_result(SimpleNamespace(gun_trace=old), s)


def test_section_appends_only_tail_and_retains_original_precision(monkeypatch):
    from temsim.physics import particle_sections as sections
    from temsim.physics.simulation import Branch
    from temsim.physics.core import PropagationCheckpoints
    n, total = 9, 14
    s = state(total)
    phase = s.electron_gun.emitter.emit()
    initial = tuple(getattr(phase, name) for name in ("x_m", "tx_rad", "y_m", "ty_rad")) + (np.zeros(total),)
    energy, kinetic, weights = phase.energy_offset_ev, 300000+phase.energy_offset_ev, phase.weight
    identity = (phase.ray_id, np.zeros(total))
    survival = (np.ones(total, bool), np.full(total, np.nan), [""]*total)
    z = np.array([0., 1.])
    plan = SimpleNamespace(z_mm=z, checkpoint_index=np.array([0, 1]), unified_runtime=None)

    def segment(initial, energy, weights, identity, survival):
        x, tx, y, ty, clock = [np.vstack((value, value)) for value in initial]
        clock[1] += 1e-8
        energies = np.vstack((300000+energy, 300000+energy))
        branch = Branch("incident", (1., 1., 1.), z, x, y, tx, ty, survival[0].copy(),
            survival[1].copy(), list(survival[2]), 1., energy.copy(), weights.copy(),
            source_ray_id=identity[0].copy(), source_azimuth_rad=identity[1].copy(),
            flight_time_s=clock, kinetic_energy_ev=energies)
        cp = PropagationCheckpoints(z, x, tx, y, ty, clock, energies)
        return sections.ParticleSectionSegment("incident", branch, plan, cp, "unused")

    old_weights = np.full(n, 1/n)
    old = segment(tuple(value[:n] for value in initial), energy[:n], old_weights,
                  tuple(value[:n] for value in identity), tuple(value[:n] for value in survival))
    old_dep = sections._digest_arrays(*(value[:n] for value in initial), energy[:n], kinetic[:n], old_weights,
        survival[0][:n], survival[1][:n], *(value[:n] for value in identity),
        np.asarray(survival[2][:n], dtype=str), np.asarray(""))
    old = replace(old, initial_dependency_signature=old_dep)
    calls = []
    def transport(_s, _name, _start, _stop, initial, energy, weights, identity, survival, *args, **kwargs):
        calls.append((identity[0].copy(), kwargs["checkpoint_ray_count"]))
        return segment(initial, energy, weights, identity, survival), 0., False
    monkeypatch.setattr(sections, "_trace_segment", transport)
    monkeypatch.setattr(sections, "_prefix_matches", lambda *args, **kwargs: True)
    result = incremental.extend_column_segment(s, old, plan, name="incident", start=0., stop=1.,
        initial=initial, energy=energy, weights=weights, identity=identity, survival=survival,
        events=(), boundary=float("inf"), initial_kicks=True, save_z_mm=(), dependency_context="",
        kinetic0=kinetic, preserve_existing_mesh=False, dependency="full-population")
    assert result.initial_dependency_signature == "full-population"
    np.testing.assert_array_equal(calls[0][0], np.arange(n, total))
    assert calls[0][1] == total
    for name in ("x", "tx", "y", "ty", "flight_time_s", "kinetic_energy_ev"):
        np.testing.assert_array_equal(getattr(result.branch, name)[:, :n], getattr(old.branch, name))
    np.testing.assert_array_equal(result.branch.ray_weight, weights)
    np.testing.assert_array_equal(old.branch.ray_weight, old_weights)
    np.testing.assert_array_equal(result.checkpoints.flight_time_s[:, :n], old.checkpoints.flight_time_s)
    kwargs = dict(name="incident", start=0., stop=1., initial=initial, energy=energy,
        weights=weights, identity=identity, survival=survival, events=(), boundary=float("inf"),
        initial_kicks=True, save_z_mm=(), dependency_context="", kinetic0=kinetic,
        preserve_existing_mesh=False, dependency="full-population")
    for incompatible in (replace(old, initial_dependency_signature="changed-launch"),
                         replace(old, plan=SimpleNamespace(z_mm=np.array([0., 2.])))):
        assert incremental.extend_column_segment(s, incompatible, plan, **kwargs) is None
    monkeypatch.setattr(sections, "_prefix_matches", lambda *args, **kwargs: False)
    assert incremental.extend_column_segment(s, old, plan, **kwargs) is None
    assert len(calls) == 1


@pytest.mark.parametrize("section", [False, True])
def test_official_whole_and_preview_pass_prior_source_without_starting_transport(monkeypatch, section):
    from temsim.optics.column import default_state
    from temsim.physics import simulation, particle_sections
    s = default_state()
    s.simulation_mode = "custom"
    s.electron_gun.emitter.sampling = EmissionSampling(schema=HALTON_PREFIX_SAMPLING_SCHEMA)
    s.electron_gun.emitter.ray_count = 14
    s.sample.inserted = False
    current = synthetic_trace(s.electron_gun, s.electron_gun.emitter.emit())
    old = replace(current, incremental_source={"schema": incremental.SCHEMA})
    previous = SimpleNamespace(gun_trace=old, section_checkpoint=None)
    seen = []
    def source(_s, *, previous_trace=None):
        seen.append(previous_trace)
        return current
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", source)
    monkeypatch.setattr("temsim.physics.instrument_magnetic.active_column_events", lambda _s: ())
    monkeypatch.setattr(particle_sections, "gun_dependency_signature", lambda _s: "new")
    class StopBeforeColumn(Exception):
        pass
    def stop(*args, **kwargs):
        raise StopBeforeColumn()
    if section:
        monkeypatch.setattr(particle_sections, "_trace_segment", stop)
        execute = lambda: particle_sections.run_particle_section(s,
            observation_stop_z_mm=s.electron_gun.exit_plane_z_mm+1.,
            existing_simulation=previous, resolved_layout=object())
    else:
        monkeypatch.setattr(simulation, "build_propagation_plan", stop)
        execute = lambda: simulation.run(s, existing_simulation=previous, resolved_layout=object())
    with pytest.raises(StopBeforeColumn):
        execute()
    assert seen == [old]


def test_public_count_override_growth_rejects_before_fields_or_emission(monkeypatch):
    from temsim.optics.electron_gun.source import trace_source_to_exit
    s = state(9)
    s.electron_gun.emitter.sampling = None
    previous = SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=np.arange(9)))
    monkeypatch.setattr("temsim.optics.electron_gun.source_policy.require_physical_gun_source", lambda _gun: None)
    monkeypatch.setattr("temsim.vacuum.bind_gun_environment", lambda _s: pytest.fail("No environment preparation"))
    monkeypatch.setattr("temsim.physics.gun_field_environment.instrument_gun_field_context", lambda _s: pytest.fail("No field capture"))
    monkeypatch.setattr(incremental, "source_metadata", lambda _s: pytest.fail("No expensive preparation"))
    with pytest.raises(ValueError, match="requested count in the captured input state"):
        trace_source_to_exit(s, count=14, previous_trace=previous)


def test_unified_population_merge_preserves_internal_state_and_checks_old_boxes(monkeypatch):
    from temsim.physics.core import PropagationCheckpoints
    from temsim.physics import unified_particle_restart as restart
    from temsim.physics import unified_spatial_dependencies as dependencies
    z = np.array([0., 1.])
    def checkpoint(n, digest, spatial):
        phase = np.full((2, n), 1e-9)
        internal = restart.UnifiedParticleCheckpointState(
            np.full(n, 300000.), np.zeros(n), np.zeros((2, n)), np.ones((2, n), bool),
            np.full(n, np.nan), ("",)*n, digest, "f"*64, 0.,
            np.tile(np.array([[0., 0., 0.], [1., 1., 1.]]), (2, 1, 1)), (spatial,)*2)
        return PropagationCheckpoints(z, phase, phase, phase, phase, np.zeros((2, n)),
            np.full((2, n), 300000.), internal)
    before, tail = checkpoint(2, "a"*64, "c"*64), checkpoint(1, "b"*64, "c"*64)
    plan = SimpleNamespace(unified_runtime=SimpleNamespace(digest="b"*64))
    monkeypatch.setattr(restart, "validate_unified_checkpoint", lambda cp, _plan: cp.unified_state)
    monkeypatch.setattr(dependencies, "spatial_runtime_identity", lambda runtime, box: "c"*64)
    result = incremental.merge_population_checkpoints(before, tail, object(), plan)
    state = result.unified_state
    assert state.runtime_digest == "b"*64
    for name in ("reference_energy_ev", "reference_phi_v", "blocked_z_mm"):
        np.testing.assert_array_equal(getattr(state, name)[:2], getattr(before.unified_state, name))
    for name in ("induction_work_ev", "active"):
        np.testing.assert_array_equal(getattr(state, name)[:, :2], getattr(before.unified_state, name))
    monkeypatch.setattr(dependencies, "spatial_runtime_identity", lambda runtime, box: "d"*64)
    with pytest.raises(ValueError, match="consumed changed fields or physical boundaries"):
        incremental.merge_population_checkpoints(before, tail, object(), plan)
