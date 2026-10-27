"""Exact prepared transport reuse without merging material probability branches."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.specimen.projected_trajectory_reuse import ProjectedTrajectoryReuse
from test_unified_projected_downstream import fixture, build


def request(state, simulation, elastic):
    terminal = elastic.terminal_electrons
    return dict(state=state, start=450., stop=452.,
        phase=tuple(np.array(value) for value in (simulation.incident.x[0, :2],
            simulation.incident.tx[0, :2], simulation.incident.y[0, :2], simulation.incident.ty[0, :2])),
        events=(), energy_offset=terminal.kinetic_energy_ev-state.beam_voltage_kv*1000,
        save_z_mm=(450., 451.137, 452.),
        initial_time_s=simulation.incident.flight_time_s[0, :2].copy(),
        kinetic_energy_ev=terminal.kinetic_energy_ev.copy(), weights=np.array([.4, .6]),
        source_ids=np.array([7, 13]), source_azimuths=np.array([.1, .2]),
        absolute_probability=.5, recording_interceptions=False)


def compact(branch, key):
    from temsim.detector.projected_response import _compact_response
    return _compact_response(branch, (450., 452.), key)


def assert_branch_equal(left, right):
    for key in ("z", "x", "tx", "y", "ty", "flight_time_s", "kinetic_energy_ev",
                "alive", "blocked_z", "energy_offset_ev", "ray_weight", "source_ray_id",
                "source_azimuth_rad", "interaction_kick_x_rad", "interaction_kick_y_rad"):
        np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
    assert left.blocked_key == right.blocked_key
    assert left.weight == right.weight


@pytest.mark.parametrize("dynamic", [False, True])
def test_exact_repeat_keeps_original_nodes_energy_clock_and_detached_stops(monkeypatch, dynamic):
    state, simulation, elastic, _ = fixture(monkeypatch, dynamic=dynamic, aperture=True)
    kwargs = request(state, simulation, elastic)
    cache = ProjectedTrajectoryReuse()
    original, hit = cache.propagate(**kwargs)
    assert not hit
    cached, hit = cache.propagate(**kwargs)
    assert hit and len(cached[0]) < len(original[0])
    rows = np.searchsorted(original[0], cached[0])
    for before, after in zip(original[:7], cached[:7]):
        np.testing.assert_array_equal(before[rows], after)
    np.testing.assert_array_equal(cached[7], [True, False])
    assert cached[-1] == ("", "bounded_iris")
    assert cached[8][1] == 450.5
    assert not np.shares_memory(cached[1], original[1])
    cached[1][0, 0] = 123.
    cached[8][1] = 0.
    again, hit = cache.propagate(**kwargs)
    assert hit and again[1][0, 0] != 123. and again[8][1] == 450.5


@pytest.mark.parametrize("changed", ["x", "tx", "y", "ty", "energy_offset", "initial_time_s",
    "kinetic_energy_ev", "weights", "source_ids", "source_azimuths", "absolute_probability",
    "recording_interceptions", "grid", "instrument"])
def test_prepared_input_change_cannot_reuse_transport(monkeypatch, changed):
    state, simulation, elastic, _ = fixture(monkeypatch, dynamic=True)
    kwargs = request(state, simulation, elastic)
    cache = ProjectedTrajectoryReuse()
    cache.propagate(**kwargs)
    if changed in ("x", "tx", "y", "ty"):
        index = ("x", "tx", "y", "ty").index(changed)
        kwargs["phase"][index][0] += 1e-8
    elif changed in ("energy_offset", "kinetic_energy_ev"):
        kwargs[changed][0] += .1
    elif changed == "initial_time_s":
        kwargs[changed][0] += 1e-12
    elif changed == "weights":
        kwargs[changed] = np.array([.3, .7])
    elif changed == "source_ids":
        kwargs[changed] = kwargs[changed][::-1].copy()
    elif changed == "source_azimuths":
        kwargs[changed][0] += .01
    elif changed == "absolute_probability":
        kwargs[changed] = .4
    elif changed == "recording_interceptions":
        kwargs[changed] = True
    elif changed == "grid":
        state.step_mm = .04
    else:
        state.objective_lens.percent += .1
    _, hit = cache.propagate(**kwargs)
    assert not hit


@pytest.mark.parametrize("aperture_start", [450., 450.5])
def test_direct_vacuum_share_only_transport_and_keep_separate_ledgers(monkeypatch, aperture_start):
    state, simulation, elastic, _ = fixture(monkeypatch, dynamic=True,
        aperture=True, aperture_start=aperture_start)
    cache = ProjectedTrajectoryReuse()
    direct = compact(build(state, simulation, elastic, _projected_trajectory_reuse=cache), "direct")
    vacuum_elastic = replace(elastic, terminal_electrons=replace(elastic.terminal_electrons,
        material_path_nm=np.zeros(2)))
    vacuum = compact(build(state, simulation, vacuum_elastic, _projected_trajectory_reuse=cache), "vacuum")
    fresh = compact(build(state, simulation, vacuum_elastic), "vacuum")
    assert direct.metrics["prepared_trajectory_reuse_misses"] == 1
    assert vacuum.metrics["prepared_trajectory_reuse_hits"] == 1
    assert vacuum.branches[0].name == "specimen_primary:vacuum_miss"
    assert direct.branches[0].name == "specimen_primary:zero_loss"
    assert direct.branches[0].interaction_kind == "sample_region_primary"
    assert vacuum.branches[0].interaction_kind == "vacuum"
    assert direct.metrics["source_probability_conserved"]
    assert vacuum.metrics["source_probability_conserved"]
    assert_branch_equal(vacuum.branches[0], fresh.branches[0])
    for key in ("tracked_downstream_source_probability", "inelastic_absorbed_source_probability",
                "exit_plane_source_probability", "numerical_domain_truncated_conditional_probability"):
        assert vacuum.metrics[key] == fresh.metrics[key]
    np.testing.assert_array_equal(elastic.terminal_electrons.material_path_nm, 5.)
    np.testing.assert_array_equal(vacuum_elastic.terminal_electrons.material_path_nm, 0.)


def test_absorption_probability_is_not_aliased_with_vacuum(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch)
    import temsim.specimen.downstream_transport as downstream
    marker = object()
    actual = downstream._inelastic_specs
    monkeypatch.setattr(downstream, "_inelastic_specs", lambda distribution, count:
        ((("zero_loss", 0., 0., .8, "real_zero_loss", 0.),), .8, .2)
        if distribution is marker else actual(distribution, count))
    cache = ProjectedTrajectoryReuse()
    direct = build(state, simulation, elastic, inelastic_distribution=marker,
                   _projected_trajectory_reuse=cache)
    vacuum = build(state, simulation, replace(elastic, terminal_electrons=replace(
        elastic.terminal_electrons, material_path_nm=np.zeros(2))), _projected_trajectory_reuse=cache)
    assert direct.branches[0].weight == pytest.approx(.4)
    assert direct.metrics["inelastic_absorbed_source_probability"] == pytest.approx(.1)
    assert vacuum.branches[0].weight == pytest.approx(.5)
    assert vacuum.metrics["inelastic_absorbed_source_probability"] == 0.
    assert vacuum.metrics["prepared_trajectory_reuse_hits"] == 0


def test_failed_or_cancelled_transport_is_not_published(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch)
    kwargs = request(state, simulation, elastic)
    import temsim.physics.core as core
    import temsim.physics.optical_tuning as tuning
    execute = core.execute_propagation_plan
    cache = ProjectedTrajectoryReuse()
    def failed(*args, **options):
        raise RuntimeError("failed transport")
    monkeypatch.setattr(core, "execute_propagation_plan", failed)
    with pytest.raises(RuntimeError, match="failed transport"):
        cache.propagate(**kwargs)
    assert cache._stored is None
    monkeypatch.setattr(core, "execute_propagation_plan", execute)
    original_check = tuning.check_tuning_cancelled
    cancelled = []
    def check(current):
        if cancelled:
            raise RuntimeError("cancelled transport")
        original_check(current)
    def finish_then_cancel(*args, **options):
        result = execute(*args, **options)
        cancelled.append(True)
        return result
    monkeypatch.setattr(core, "execute_propagation_plan", finish_then_cancel)
    monkeypatch.setattr(tuning, "check_tuning_cancelled", check)
    with pytest.raises(RuntimeError, match="cancelled transport"):
        cache.propagate(**kwargs)
    assert cache._stored is None


def test_projected_response_builder_uses_request_local_transport_cache(monkeypatch):
    state, simulation, _, _ = fixture(monkeypatch, dynamic=True)
    simulation.incident.energy_offset_ev = simulation.incident.kinetic_energy_ev[-1]-state.beam_voltage_kv*1000
    import temsim.detector.projected_response as response
    import temsim.physics.core as core
    monkeypatch.setattr(response, "_response_identity", lambda *args: None)
    execute, calls = core.execute_propagation_plan, []
    def tracked(*args, **kwargs):
        calls.append(True)
        return execute(*args, **kwargs)
    monkeypatch.setattr(core, "execute_propagation_plan", tracked)
    result = response.build_projected_responses(state, simulation,
        SimpleNamespace(elements=()), stop_z_mm=452.)
    assert len(calls) == 1
    assert result["vacuum"].metrics["prepared_trajectory_reuse_hits"] == 1
    assert result["direct"].metrics["prepared_trajectory_reuse_misses"] == 1
    assert_branch_equal(result["direct"].branches[0], result["vacuum"].branches[0])


def test_field_domain_rejection_is_not_hidden_or_cached(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch)
    directions = np.tile(np.array([1., 0., .001])/np.sqrt(1.+1e-6), (2, 1))
    elastic = replace(elastic, terminal_electrons=replace(elastic.terminal_electrons,
        direction=directions))
    cache = ProjectedTrajectoryReuse()
    with pytest.raises(ValueError, match="outside captured electric potential domain"):
        build(state, simulation, elastic, _projected_trajectory_reuse=cache)
    assert cache._stored is None
    with pytest.raises(ValueError, match="outside captured electric potential domain"):
        build(state, simulation, elastic)


def test_storage_limit_does_not_reduce_propagation_accuracy(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch)
    import temsim.specimen.projected_trajectory_reuse as reuse
    monkeypatch.setattr(reuse, "MAX_RETAINED_BYTES", 1)
    cache = ProjectedTrajectoryReuse()
    original, _ = cache.propagate(**request(state, simulation, elastic))
    repeated, hit = cache.propagate(**request(state, simulation, elastic))
    assert not hit and cache._stored is None
    for left, right in zip(original[:-1], repeated[:-1]):
        np.testing.assert_array_equal(left, right)
