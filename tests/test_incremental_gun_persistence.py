"""Synthetic numeric histories only; no electron source or transport execution."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.gun_trace_extensions_io import pack_gun_extensions, unpack_gun_extensions
from temsim.optics.electron_gun.base import (
    GunEqualTimeHistory, GunExitBundle, GunTraceResult,
)
from temsim.particle_section_io import _data_classes, _pack, _unpack


def native_trace(first_id, count, time):
    ids = np.arange(first_id, first_id + count, dtype=np.int64)
    z = np.array([0., 1.])
    phase = np.arange(2 * count, dtype=float).reshape(2, count) * 1e-9
    launch = {"x_m": phase[0].copy(), "y_m": phase[0].copy(),
              "tx_rad": np.zeros(count), "ty_rad": np.zeros(count),
              "energy_offset_ev": np.full(count, .125), "ray_id": ids.copy(),
              "weight": np.full(count, 1. / count)}
    times = np.asarray(time, dtype=float)
    shape = (len(times), count)
    history_phase = np.broadcast_to(ids, shape).copy() * 1e-9
    history = GunEqualTimeHistory(times, np.broadcast_to(times[:, None], shape).copy(),
        history_phase, history_phase.copy(), np.zeros(shape), np.zeros(shape),
        np.ones(shape, dtype=bool), np.zeros(shape, dtype=bool))
    exit_bundle = GunExitBundle(phase[-1], phase[-1].copy(), np.zeros(count), np.zeros(count),
        launch["energy_offset_ev"], launch["weight"], ids,
        np.ones(count, dtype=bool), np.ones(count))
    return GunTraceResult(z, phase, phase.copy(), np.zeros_like(phase), np.zeros_like(phase),
        exit_bundle, np.full(count, np.nan), tuple("" for _ in ids), 1e-6, None, None,
        equal_time_history=history, flight_time_s=np.broadcast_to(z[:, None], phase.shape).copy(),
        emission_reference={"ray_id": ids.copy(),
            "position_m": np.column_stack((phase[0], phase[0], np.zeros(count))),
            "direction": np.tile([0., 0., 1.], (count, 1)),
            "normal": np.tile([0., 0., -1.], (count, 1))},
        emission_phase=launch,
        incremental_source={"schema": "synthetic-io-only", "count": count})


def combined_trace(old, tail):
    from temsim.physics.incremental_particles import merge_gun_traces
    phase = {key: np.concatenate((old.emission_phase[key], tail.emission_phase[key]))
             for key in old.emission_phase}
    phase["weight"] = np.full(len(phase["ray_id"]), 1. / len(phase["ray_id"]))
    return merge_gun_traces(old, tail, phase,
                           {"schema": "synthetic-io-only", "count": len(phase["ray_id"])})


def test_extensions_retain_original_partition_meshes_and_values():
    old = native_trace(0, 3, [0., .5, 1.])
    tail = native_trace(3, 2, [0., .25, .75, 1.])
    merged = combined_trace(old, tail)
    arrays = {"existing_emission_x": merged.emission_phase["x_m"]}
    record = pack_gun_extensions(merged, arrays)
    restored = unpack_gun_extensions(record, arrays)
    assert restored["emission_phase"]["x_m"] is arrays["existing_emission_x"]
    before = tuple(part.equal_time_history.time_s.copy() for part in (old, tail))
    for expected, part in zip(before, restored["transport_partitions"]):
        np.testing.assert_array_equal(part.equal_time_history.time_s, expected)
    combined = replace(merged, **restored)
    front = combined.equal_time_front_at_time(.5)
    np.testing.assert_array_equal(front.x_m[:3], old.equal_time_history.sample(.5).x_m)
    np.testing.assert_array_equal(front.x_m[3:], tail.equal_time_history.sample(.5).x_m)
    for part, expected in zip((old, tail), before):
        np.testing.assert_array_equal(part.equal_time_history.time_s, expected)


def test_incident_artifact_roundtrip_retains_all_native_partitions(tmp_path):
    from temsim.artifact_store import ArtifactStore
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.calculation_manifest import capture_calculation_manifest
    from temsim.optics.column import default_state
    from temsim.physics.core import PropagationCheckpoints, build_propagation_plan
    from temsim.physics.simulation import Branch, Simulation, validate_incident_flight_times
    state = default_state()
    state.simulation_mode = "ideal_thin"
    state.electron_gun.emitter.ray_count = 9
    state.sample.inserted = False
    start, stop = state.electron_gun.exit_plane_z_mm, state.electron_gun.exit_plane_z_mm + .1
    plan = build_propagation_plan(state, start, stop, (),
        save_z_mm=(start, stop), checkpoint_z_mm=(start, stop))
    z = np.asarray(plan.z_mm)
    shape = (len(z), 9)
    zeros = np.zeros(shape)
    clocks = np.broadcast_to(np.linspace(1., 2., len(z))[:, None], shape).copy()
    energy = np.full(shape, 300000.125)
    gun = combined_trace(native_trace(0, 4, [0., .5, 1.]),
                         native_trace(4, 5, [0., .25, .75, 1.]))
    cp = PropagationCheckpoints(z, zeros, zeros.copy(), zeros.copy(), zeros.copy(),
                                flight_time_s=clocks, kinetic_energy_ev=energy)
    branch = Branch("Synthetic incident", (1., 0., 0.), z, zeros, zeros.copy(),
        zeros.copy(), zeros.copy(), np.ones(9, dtype=bool), np.full(9, np.nan),
        [""] * 9, 1., gun.exit_bundle.energy_offset_ev, ray_weight=gun.exit_bundle.weight,
        source_ray_id=gun.exit_bundle.ray_id, source_azimuth_rad=np.zeros(9),
        flight_time_s=clocks, kinetic_energy_ev=energy)
    simulation = Simulation(branch, {}, {}, gun_trace=gun, incident_plan=plan, incident_checkpoints=cp)
    assert validate_incident_flight_times(simulation)
    manifest = capture_calculation_manifest(state, selection=AssemblyCatalog().default_selection(),
                                             ray_count=9, step_mm=state.step_mm)
    store = ArtifactStore(tmp_path / "cache", quota_bytes=32 * 1024**2)
    store.put_incident_simulation_seed(manifest, simulation)
    restored = store.get_incident_simulation_seed(manifest)
    assert restored is not None
    assert validate_incident_flight_times(restored)
    actual = restored.gun_trace
    assert actual.equal_time_history is None and len(actual.transport_partitions) == 2
    for before, after in zip(gun.transport_partitions, actual.transport_partitions):
        np.testing.assert_array_equal(before.equal_time_history.time_s, after.equal_time_history.time_s)
        np.testing.assert_array_equal(before.exit_bundle.energy_offset_ev, after.exit_bundle.energy_offset_ev)
        np.testing.assert_array_equal(before.emission_phase["weight"], after.emission_phase["weight"])
    np.testing.assert_array_equal(actual.equal_time_front_at_time(.5).x_m,
                                  gun.equal_time_front_at_time(.5).x_m)


def test_section_graph_keeps_prefix_extensions_and_historical_defaults():
    trace = native_trace(0, 3, [0., 1.])
    arrays = {}
    graph = _pack(trace, arrays, {}, _data_classes())
    restored = _unpack(graph, arrays, _data_classes())
    np.testing.assert_array_equal(restored.emission_phase["energy_offset_ev"],
                                  trace.emission_phase["energy_offset_ev"])
    historical = deepcopy(graph)
    for name in ("emission_phase", "incremental_source", "transport_partitions"):
        historical["fields"].pop(name)
    legacy = _unpack(historical, arrays, _data_classes())
    assert legacy.emission_phase is None and legacy.incremental_source is None
    assert legacy.transport_partitions == ()


def test_absent_extensions_do_not_create_a_reusable_historical_seed():
    trace = replace(native_trace(0, 3, [0., 1.]), emission_phase=None, incremental_source=None)
    assert pack_gun_extensions(trace, {}) is None
    assert unpack_gun_extensions(None, {}) == {}


@pytest.mark.parametrize("record", [{}, {"schema": "unknown", "graph": {}},
                                    {"schema": "gun-prefix-transport-extensions-v1", "graph": {"type": "os.system"}}])
def test_unknown_extension_graphs_are_rejected(record):
    with pytest.raises((ValueError, KeyError)):
        unpack_gun_extensions(record, {})


def test_wrong_partition_type_is_rejected():
    trace = replace(native_trace(0, 3, [0., 1.]), transport_partitions=("not-a-trace",))
    arrays = {}
    graph = pack_gun_extensions(trace, arrays)
    with pytest.raises(ValueError, match="partitions"):
        unpack_gun_extensions(graph, arrays)
