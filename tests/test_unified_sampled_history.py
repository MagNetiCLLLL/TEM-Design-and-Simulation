"""Exact history retention on small mathematical field/operator fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import unified_field_particle as particle


def test_sampled_indices_keep_explicit_nodes_and_restart_anchoring():
    z = np.unique(np.r_[np.linspace(0., 2., 41), .73, .731, 1.11])
    cp = np.searchsorted(z, [0., .73, 2.])
    required = np.searchsorted(z, [0., .731, 1.11, 2.])
    plan = SimpleNamespace(z_mm=z, checkpoint_index=cp, save_index=required)
    state = SimpleNamespace(history_step_mm=.3, recording_planes=())
    selected = particle._history_indices(state, plan, history_policy="sampled")
    assert len(selected) < len(z)
    assert set(cp) | set(required) <= set(selected)
    start = int(cp[1])
    np.testing.assert_array_equal(
        particle._history_indices(state, plan, start_index=start, history_policy="sampled"),
        selected[selected >= start])
    np.testing.assert_array_equal(plan.z_mm, z)


@pytest.mark.parametrize("spacing", [0., -1., np.inf, np.nan, True])
def test_sampled_history_rejects_invalid_display_spacing(spacing):
    state = SimpleNamespace(history_step_mm=spacing, recording_planes=())
    plan = SimpleNamespace(z_mm=np.linspace(0., 1., 11), checkpoint_index=np.array([0, 10]))
    with pytest.raises(ValueError, match="display history step"):
        particle._history_indices(state, plan, history_policy="sampled")


def test_tiny_display_spacing_never_allocates_a_larger_target_grid():
    state = SimpleNamespace(history_step_mm=1e-300, recording_planes=())
    plan = SimpleNamespace(z_mm=np.linspace(0., 1000., 101), checkpoint_index=np.array([0, 100]))
    np.testing.assert_array_equal(particle._history_indices(state, plan, history_policy="sampled"),
                                  np.arange(101))


def test_display_spacing_finer_than_mesh_does_not_reduce_memory_budget():
    state = SimpleNamespace(history_step_mm=.001, recording_planes=())
    plan = SimpleNamespace(z_mm=np.linspace(0., 1., 11), checkpoint_index=np.array([0, 10]))
    assert particle.unified_history_memory_bytes(state, plan, 2000, history_policy="sampled") == (
        particle.unified_history_memory_bytes(state, plan, 2000, history_policy="full"))


def test_production_budget_fallback_preserves_exact_receiver_and_reports_policy(monkeypatch):
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.core import execute_propagation_plan
    from temsim.physics.recording_clipping import clip_recording_planes
    from test_unified_field_particle import controlled
    from test_unified_field_runtime import parked_state

    state = parked_state()
    state.step_mm, state.history_step_mm = .05, .3
    receiver = state.recording_planes[0]
    receiver.z_mm = 450.73
    receiver.geometry, receiver.inner_diameter_mm, receiver.outer_width_mm = "disk", 0., .02
    runtime = controlled(state)
    plan = particle.build_unified_plan(state, 450., 452., save_z_mm=(451.13,),
        checkpoint_z_mm=(450., 450.5, 452.), _captured_runtime=runtime)
    original_mesh = plan.z_mm.copy()
    np.testing.assert_array_equal(plan.save_index, np.arange(len(plan.z_mm)))
    phase = (np.array([0., 2e-5]), np.zeros(2), np.zeros(2), np.zeros(2))
    clock = np.array([1e-9, 2e-9])
    full_energy = []
    full = execute_propagation_plan(state, plan, *phase, initial_time_s=clock,
        return_flight_times=True, energy_output=full_energy)
    np.testing.assert_array_equal(full[0], plan.z_mm)
    assert particle.select_unified_history_policy(state, plan, 2) == "full"
    budget = particle.unified_history_memory_bytes(state, plan, 2, history_policy="sampled")
    assert budget < particle.unified_history_memory_bytes(state, plan, 2)
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", budget)
    bounded_energy, receipts = [], []
    with backend_receipts(receipts):
        bounded = execute_propagation_plan(state, plan, *phase, initial_time_s=clock,
            return_flight_times=True, energy_output=bounded_energy)
    assert len(bounded[0]) < len(full[0])
    assert {450., 450.5, 451.13, 452., receiver.z_mm} <= set(bounded[0])
    predecessor = plan.z_mm[np.searchsorted(plan.z_mm, receiver.z_mm)-1]
    assert predecessor in bounded[0]
    selected = np.searchsorted(full[0], bounded[0])
    for actual, reference in zip(bounded[1:6], full[1:6]):
        np.testing.assert_array_equal(actual, reference[selected])
    np.testing.assert_array_equal(bounded_energy[0], full_energy[0][selected])
    np.testing.assert_array_equal(plan.z_mm, original_mesh)
    for name in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        np.testing.assert_array_equal(getattr(bounded[-1], name), getattr(full[-1], name))
    stopped = [clip_recording_planes(state, r[0], r[1], r[3], np.ones(2, bool),
        np.full(2, np.nan), ["", ""]) for r in (full, bounded)]
    for actual, reference in zip(stopped[1], stopped[0]):
        np.testing.assert_array_equal(actual, reference)
    assert receipts[-1]["history_policy"] == "sampled"
    assert receipts[-1]["requested_history_policy"] == "auto"
    assert receipts[-1]["retained_history_nodes"] == len(bounded[0])
    assert receipts[-1]["integration_nodes"] == len(plan.z_mm)
    assert receipts[-1]["history_memory_bytes"] == budget
    assert receipts[-1]["retained_history_maximum_gap_mm"] == np.max(np.diff(bounded[0]))
    assert "markers remain sampling dependent" in receipts[-1]["history_scope"]
    with pytest.raises(ValueError, match="history requires"):
        particle.execute_unified_plan(state, plan, *phase, initial_time_s=clock)
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", budget-1)
    receipts = []
    with backend_receipts(receipts):
        batched = execute_propagation_plan(state, plan, *phase, initial_time_s=clock,
            return_flight_times=True)
    for actual, expected in zip(batched[:-1], bounded[:-1]):
        np.testing.assert_array_equal(actual, expected)
    assert receipts[-1]["source_batch_count"] == 2
    assert receipts[-1]["maximum_source_batch_particles"] == 1
    assert receipts[-1]["source_particles"] == 2
    assert receipts[-1]["history_memory_bytes"] <= budget-1
    assert receipts[-1]["assembled_history_bytes"] >= sum(values.nbytes for values in batched[:-1])


def test_section_fallback_and_resume_keep_source_axis_weights_and_stops(monkeypatch):
    """Synthetic entrance coordinates exercise section wiring, never gun emission."""
    from temsim.physics.particle_sections import _trace_segment
    from temsim.physics.lens_field_provider import CoordinateRegistration
    from temsim.physics.posed_wave_aperture import PosedWaveAperture
    from test_unified_field_particle import controlled
    from test_unified_field_runtime import parked_state

    state = parked_state()
    state.step_mm, state.history_step_mm = .05, .3
    iris = PosedWaveAperture("iris", 451., 451.1, .015, CoordinateRegistration())
    runtime = controlled(state, pieces=(iris,))
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda _: runtime)
    initial = (np.array([0., 1e-6, 3e-5]), np.zeros(3), np.zeros(3),
               np.zeros(3), np.array([1e-8, 1.1e-8, 1.2e-8]))
    energy, weight = np.array([0., 1., -1.]), np.array([.2, .3, .5])
    identity = (np.array([17, 4, 91]), np.array([.2, .4, .6]))
    survival = (np.ones(3, bool), np.full(3, np.nan), ["", "", ""])

    def section(previous=None, boundary=452.):
        return _trace_segment(state, "incident", 450., 452., initial, energy, weight,
            identity, survival, (), previous, boundary, initial_kicks=True,
            save_z_mm=(450.5, 451., 451.13))

    full, _, _ = section()
    budget = particle.unified_history_memory_bytes(state, full.plan, 3, history_policy="sampled")
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", budget)
    bounded, _, _ = section()
    assert len(bounded.branch.z) < len(full.branch.z)
    resumed, start, reused = section(bounded, boundary=450.8)
    assert reused and start == 450.5
    two_particle_budget = particle.unified_history_memory_bytes(state, full.plan, 2,
        history_policy="sampled")
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", two_particle_budget)
    batched, _, _ = section()
    batched_resume, batch_start, batch_reused = section(batched, boundary=450.8)
    assert batch_reused and batch_start == 450.5
    for candidate in (bounded, resumed, batched, batched_resume):
        rows = np.searchsorted(full.branch.z, candidate.branch.z)
        for key in ("x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev"):
            np.testing.assert_array_equal(getattr(candidate.branch, key), getattr(full.branch, key)[rows])
        for key in ("alive", "blocked_z", "blocked_key", "ray_weight", "source_ray_id", "source_azimuth_rad"):
            np.testing.assert_array_equal(getattr(candidate.branch, key), getattr(full.branch, key))
    np.testing.assert_array_equal(bounded.branch.source_ray_id, identity[0])
    np.testing.assert_array_equal(bounded.branch.ray_weight, weight)
    assert bounded.branch.blocked_key == ["", "", "iris"]


def test_batched_dynamic_resume_preserves_induction_energy_clocks_and_bounds(monkeypatch):
    from dataclasses import replace
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.potential_snapshot import PotentialContribution
    from temsim.physics.unified_particle_restart import validate_unified_checkpoint
    from test_dynamic_potential import coil
    from test_unified_field_particle import controlled
    from test_unified_field_runtime import parked_state
    from test_unified_particle_prefix import execute, plan_for

    state = parked_state()
    state.simulation_time_s = .21e-9
    radial, axial = np.array([0., .02]), np.array([.449, .453])
    voltage = np.broadcast_to(200000.+2e6*(axial-.45), (2, 2)).copy()
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, radial, axial, voltage))
    runtime = controlled(state, (PotentialContribution("electric", ("electric",), electric, "electric"),))
    runtime = replace(runtime, snapshot=DynamicPotentialSnapshot(runtime.snapshot, (coil(),)),
                      metadata={"emission_time_s": state.simulation_time_s})
    plan = plan_for(state, runtime, monkeypatch)
    full, full_energy = execute(state, plan)
    cap = particle.unified_history_memory_bytes(state, plan, 1, history_policy="sampled")
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", cap)
    batched, batched_energy = execute(state, plan)
    selected = np.searchsorted(full[0], batched[0])
    for actual, expected in zip(batched[1:6], full[1:6]):
        np.testing.assert_array_equal(actual, expected[selected])
    np.testing.assert_array_equal(batched_energy, full_energy[selected])
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        np.testing.assert_array_equal(getattr(batched[-1], name), getattr(full[-1], name))
    actual, expected = validate_unified_checkpoint(batched[-1]), validate_unified_checkpoint(full[-1])
    for name in ("reference_energy_ev", "reference_phi_v", "induction_work_ev", "active",
                 "blocked_z_mm", "swept_bounds_m", "spatial_dependency_identities"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    resumed, resumed_energy = execute(state, plan, previous=batched[-1], row=2)
    start = np.searchsorted(batched[0], resumed[0][0])
    for actual, expected in zip(resumed[1:6], batched[1:6]):
        np.testing.assert_array_equal(actual, expected[start:])
    np.testing.assert_array_equal(resumed_energy, batched_energy[start:])


@pytest.mark.parametrize("failure", ["memory", "cancel", "execution"])
def test_batch_failure_never_publishes_partial_energy_history(monkeypatch, failure):
    from temsim.physics.unified_particle_batches import execute_unified_batches
    from temsim.physics import wave_execution

    state = SimpleNamespace(history_step_mm=.3, recording_planes=(),
                            _tuning_cancelled=lambda: failure == "cancel")
    plan = SimpleNamespace(z_mm=np.linspace(0., 1., 11),
        checkpoint_index=np.array([0, 10]), save_index=np.array([0, 10]))
    cap = particle.unified_history_memory_bytes(state, plan, 1, history_policy="sampled")
    monkeypatch.setattr(particle, "MAX_UNIFIED_HISTORY_BYTES", cap)
    admitted = []

    def admit(additional_bytes):
        admitted.append(additional_bytes)
        if failure == "memory":
            raise MemoryError("Insufficient output and bounded execution memory")

    monkeypatch.setattr(wave_execution, "check_available_memory", admit)
    calls = []

    def execute(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("Execution failed")

    monkeypatch.setattr(particle, "execute_unified_plan", execute)
    output = []
    expected = MemoryError if failure == "memory" else RuntimeError
    with pytest.raises(expected):
        execute_unified_batches(state, plan, (np.zeros(3),)*4, None,
            start_index=0, include_initial_plane_kicks=True,
            defer_nonfinite_until_clipping=False, initial_time_s=None,
            return_flight_times=False, initial_kinetic_energy_ev=None,
            energy_output=output, unified_resume=None, history_policy="sampled")
    assert output == []
    assert bool(admitted) == (failure != "cancel")
    assert bool(calls) == (failure == "execution")
    if admitted:
        assert admitted[0] > cap
