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
    with pytest.raises(ValueError, match="sampled history"):
        execute_propagation_plan(state, plan, *phase, initial_time_s=clock)


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
    for candidate in (bounded, resumed):
        rows = np.searchsorted(full.branch.z, candidate.branch.z)
        for key in ("x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev"):
            np.testing.assert_array_equal(getattr(candidate.branch, key), getattr(full.branch, key)[rows])
        for key in ("alive", "blocked_z", "blocked_key", "ray_weight", "source_ray_id", "source_azimuth_rad"):
            np.testing.assert_array_equal(getattr(candidate.branch, key), getattr(full.branch, key))
    np.testing.assert_array_equal(bounded.branch.source_ray_id, identity[0])
    np.testing.assert_array_equal(bounded.branch.ray_weight, weight)
    assert bounded.branch.blocked_key == ["", "", "iris"]
