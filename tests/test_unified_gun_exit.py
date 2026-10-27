"""Exact source-boundary observations and unchanged positive-length transport."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics import particle_sections as sections
from temsim.physics import unified_field_particle as particle
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.interaction_registry import InteractionRegistry
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from test_particle_sections import fixture, trace
from test_unified_field_particle import controlled
from test_unified_field_runtime import parked_state


def _runtime(current, pieces=()):
    return replace(controlled(current, pieces=pieces), metadata={
        "source_contract": "Synthetic executed gun fixture; bounded real column propagation"})


@pytest.mark.parametrize("stopped", [False, True])
def test_gun_exit_is_an_exact_observation_without_field_or_boundary_action(fixture, monkeypatch, stopped):
    state, gun, calls = fixture
    state.simulation_mode = "unified"
    emitted = gun.exit_bundle
    emitted.weight[:] = (.5, .3, .2)
    emitted.energy_offset_ev[:] = (.3, -.1, .2)
    emitted.flight_time_s[:] = (1e-8, 1.1e-8, 1.2e-8)
    gun.flight_time_s[-1] = emitted.flight_time_s
    if stopped:
        emitted.alive[-1] = False
        emitted.flight_time_s[-1] = gun.flight_time_s[-1, -1] = np.nan
        gun.blocked_z_mm[-1] = 449.5
        gun = replace(gun, blocked_key=("", "", "gun_aperture"))
        monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", lambda _state: gun)
    aperture = PosedWaveAperture("column_entrance", 450., 450.1, .00005, CoordinateRegistration())
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current: _runtime(current, (aperture,)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Observing the executed gun exit must not propagate or repeat column contacts")

    monkeypatch.setattr(sections, "execute_propagation_plan", forbidden)
    monkeypatch.setattr("temsim.physics.recording_clipping.clip_recording_planes", forbidden)
    result = trace(state, 450.)
    segment = result.section_checkpoint.segments[0]
    branch, cp = segment.branch, segment.checkpoints
    for output, source in (("x", "x_m"), ("y", "y_m"), ("tx", "tx_rad"), ("ty", "ty_rad"),
                           ("flight_time_s", "flight_time_s")):
        np.testing.assert_array_equal(getattr(branch, output)[-1], getattr(emitted, source))
        np.testing.assert_array_equal(getattr(cp, source)[-1], getattr(emitted, source))
    np.testing.assert_array_equal(branch.alive, emitted.alive)
    np.testing.assert_array_equal(branch.blocked_z, gun.blocked_z_mm)
    assert tuple(branch.blocked_key) == gun.blocked_key
    np.testing.assert_array_equal(branch.source_ray_id, emitted.ray_id)
    np.testing.assert_array_equal(branch.ray_weight, emitted.weight)
    np.testing.assert_array_equal(branch.energy_offset_ev, emitted.energy_offset_ev)
    np.testing.assert_array_equal(cp.kinetic_energy_ev[-1], state.beam_voltage_kv*1000 + emitted.energy_offset_ev)
    assert cp.unified_state is None  # No invented reference potential or executed field prefix.
    assert not segment.plan.step_m.size
    assert not InteractionRegistry.from_dict(segment.plan.physics_contract["interactions"]).records
    assert "observation" in segment.plan.physics_contract["scope"]
    assert sections.validate_section_checkpoint(result.section_checkpoint) is result.section_checkpoint
    repeated = trace(state, 450., result)
    assert repeated.metrics["section_gun_reused"]
    assert not repeated.metrics["section_reused_prefix"]
    np.testing.assert_array_equal(repeated.incident.x, result.incident.x)


@pytest.mark.parametrize("through_archive", [False, True])
def test_extending_gun_exit_reuses_source_and_applies_column_entrance_once(fixture, monkeypatch, through_archive):
    state, gun, calls = fixture
    state.simulation_mode = "unified"
    aperture = PosedWaveAperture("column_entrance", 450., 450.1, .00005, CoordinateRegistration())
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current: _runtime(current, (aperture,)))
    initial = trace(state, 450.)
    np.testing.assert_array_equal(initial.incident.alive, [True, True, True])
    if through_archive:
        from temsim.particle_section_io import _pack, _unpack, _data_classes
        arrays = {}
        checkpoint = initial.section_checkpoint
        restored = _unpack(_pack(checkpoint, arrays, {}, _data_classes()), arrays, _data_classes())
        initial = replace(initial, section_checkpoint=restored)
        assert restored.segments[0].checkpoints.unified_state is None
        assert sections.validate_section_checkpoint(restored) is restored
    resumed = trace(state, 451., initial)
    assert len(calls) == 1 and resumed.metrics["section_gun_reused"]
    assert not resumed.metrics["section_reused_prefix"]
    np.testing.assert_array_equal(resumed.incident.alive, [True, False, False])
    np.testing.assert_array_equal(resumed.incident.blocked_z[1:], [450., 450.])
    assert resumed.incident.blocked_key[1:] == ["column_entrance", "column_entrance"]
    fresh = trace(state, 451.)
    for field in ("x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev", "alive", "blocked_z", "ray_weight"):
        np.testing.assert_array_equal(getattr(resumed.incident, field), getattr(fresh.incident, field))
    assert sections.validate_section_checkpoint(resumed.section_checkpoint) is resumed.section_checkpoint


@pytest.mark.parametrize("interval", [(451., 451.), (451., 450.), (449., 450.), (450., 453.)])
def test_observation_exception_does_not_admit_other_degenerate_or_invalid_intervals(monkeypatch, interval):
    state = parked_state()
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current: _runtime(current))
    with pytest.raises(ValueError, match="outside"):
        build_propagation_plan(state, *interval)


def test_gun_exit_plan_cannot_fabricate_a_source_by_executing_the_column_kernel(monkeypatch):
    state = parked_state()
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current: _runtime(current))
    plan = build_propagation_plan(state, 450., 450.)
    with pytest.raises(ValueError, match="executed gun checkpoint"):
        execute_propagation_plan(state, plan, *(np.zeros(1) for _ in range(4)))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, False, np.bool_(True), 0., -.1])
@pytest.mark.parametrize("field", ["state", "maximum"])
def test_step_limits_are_validated_independently_before_minimum_or_runtime_capture(monkeypatch, field, value):
    state = parked_state()
    state.step_mm = value if field == "state" else .1

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid numerics must be rejected before field capture")

    monkeypatch.setattr(particle, "get_unified_field_runtime", forbidden)
    with pytest.raises(ValueError, match="positive and finite"):
        build_propagation_plan(state, 450., 450.2, maximum_step_mm=value if field == "maximum" else .05)


@pytest.mark.parametrize("maximum", [None, .2, .04])
def test_valid_step_cap_retains_the_existing_uniform_mesh(monkeypatch, maximum):
    state = parked_state()
    state.step_mm = .1
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current: _runtime(current))
    plan = build_propagation_plan(state, 450., 450.2, maximum_step_mm=maximum)
    effective = .1 if maximum is None else min(.1, maximum)
    count = max(1, int(np.ceil((450.2-450.)/effective)))
    np.testing.assert_array_equal(plan.z_mm, np.linspace(450., 450.2, count+1))


def test_finite_subnormal_step_reports_the_node_budget_instead_of_overflow(monkeypatch):
    state = parked_state()
    state.step_mm = .1
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current: _runtime(current))
    with pytest.raises(ValueError, match="node budget"):
        build_propagation_plan(state, 450., 450.2, maximum_step_mm=5e-324)
