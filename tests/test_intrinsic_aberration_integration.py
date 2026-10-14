"""Ownership, actual execution and persistence of lens-local intrinsic terms."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.intrinsic_aberrations import validated_coefficients
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from test_magnetic_lens_aberration import _RoundLens, _state


def fixture(row=None):
    lens = _RoundLens(peak_t=0., focal_mm=10.)
    lens.cs_mm = 0.
    state = _state(lens)
    state.lens_intrinsic_aberrations = {} if row is None else {lens.key: row}
    return state


def plan_for(state):
    return build_propagation_plan(state, -.1, .1, (),
        save_z_mm=(-.1, 0., .1), checkpoint_z_mm=(0., .1))


def test_zero_default_does_not_add_screen_or_change_particle_results():
    state = fixture()
    before = deepcopy(vars(state.lenses[0]))
    base = plan_for(state)
    state.lens_intrinsic_aberrations[state.lenses[0].key] = validated_coefficients()
    zero = plan_for(state)
    assert base.signature == zero.signature
    assert not zero.intrinsic_aberration_kicks
    rays = [execute_propagation_plan(state, plan, np.array([1e-5]), np.zeros(1),
        np.array([2e-5]), np.zeros(1)) for plan in (base, zero)]
    for left, right in zip(rays[0][:5], rays[1][:5]):
        np.testing.assert_array_equal(left, right)
    assert vars(state.lenses[0]) == before


def test_a1_and_legacy_cs_execute_once_with_independent_analytic_kicks():
    state = fixture({"a1_mm": .001})
    state.lenses[0].cs_mm = .5
    plan = plan_for(state)
    assert len(plan.intrinsic_aberration_kicks) == 1
    assert np.count_nonzero(plan.cs_kick_m3) == 1
    x, y = np.array([1e-5]), np.array([2e-5])
    trace = execute_propagation_plan(state, plan, x, np.zeros(1), y, np.zeros(1))
    a1_m, f_m, cs_m = 1e-6, .01, .0005
    common = cs_m*(x*x+y*y)/f_m**4
    np.testing.assert_allclose(trace[2][-1], -a1_m*x/f_m**2-common*x, rtol=1e-10)
    np.testing.assert_allclose(trace[4][-1], a1_m*y/f_m**2-common*y, rtol=1e-10)
    # Exact checkpoints are after the screen; a resume must not execute it again.
    checkpoint = trace[-1]
    node = int(np.flatnonzero(plan.z_mm == 0.)[0])
    resumed = execute_propagation_plan(state, plan, checkpoint.x_m[0], checkpoint.tx_rad[0],
        checkpoint.y_m[0], checkpoint.ty_rad[0], start_index=node,
        include_initial_plane_kicks=False)
    for name in ("x_m", "tx_rad", "y_m", "ty_rad"):
        np.testing.assert_allclose(getattr(trace[-1], name)[-1],
            getattr(resumed[-1], name)[-1], rtol=1e-13, atol=1e-16)


def test_first_order_planes_include_intrinsic_c1_and_a1_focus():
    from temsim.physics.first_order import trace_transverse_transfer
    state = fixture({"c1_mm": .003, "a1_mm": .001})
    transfer = trace_transverse_transfer(state, -.1, .1, maximum_step_mm=.05)
    # Independent thin screen between equal drifts, in (x,y,theta_x,theta_y).
    drift = np.eye(4); drift[:2, 2:] = np.eye(2)*1e-4
    screen = np.eye(4)
    screen[2:, :2] = -np.diag([.004e-3/.01**2, .002e-3/.01**2])
    np.testing.assert_allclose(transfer.matrix, drift @ screen @ drift, rtol=1e-12, atol=1e-14)


def test_profile_persistence_independent_snapshot_and_scoped_cache_identity():
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    from temsim.calculation_cache import calculation_signatures
    state = default_state()
    upstream = next(lens for lens in state.lenses if lens.z_mm < state.sample.z_mm)
    downstream = next(lens for lens in state.lenses if lens.z_mm > state.sample.z_mm+1.)
    strengths = [(lens.key, lens.percent, lens.cs_mm) for lens in state.lenses]
    original = calculation_signatures(state)
    state.lens_intrinsic_aberrations[downstream.key] = {"a1_mm": .002}
    post = calculation_signatures(state)
    assert original["incident"] == post["incident"]
    assert original["column"] != post["column"]
    state.lens_intrinsic_aberrations[upstream.key] = {"a1_mm": .003}
    assert calculation_signatures(state)["incident"] != post["incident"]
    payload = state.to_dict()
    restored = State.from_dict(payload)
    assert restored.lens_intrinsic_aberrations[upstream.key]["a1_mm"] == .003
    assert [(lens.key, lens.percent, lens.cs_mm) for lens in restored.lenses] == strengths
    payload["lens_intrinsic_aberrations"][upstream.key]["a1_mm"] = 9.
    assert state.lens_intrinsic_aberrations[upstream.key]["a1_mm"] == .003
    assert restored.lens_intrinsic_aberrations[upstream.key]["a1_mm"] == .003


def test_plan_archive_retains_frozen_screen_and_its_effect():
    from temsim.particle_section_io import _pack_record_graph, _unpack_record_graph
    state = fixture({"b2_mm": .04, "b2_azimuth_deg": 31.})
    plan = plan_for(state)
    arrays = {}
    graph = _pack_record_graph(plan, arrays)
    restored = _unpack_record_graph(graph, arrays, maximum_unpacked_bytes=1<<24)
    assert restored.intrinsic_aberration_kicks == plan.intrinsic_aberration_kicks
    changed = plan_for(fixture({"b2_mm": .08}))
    assert changed.signature != plan.signature
    outputs = [execute_propagation_plan(state, candidate, np.array([1e-5]),
        np.zeros(1), np.array([2e-5]), np.zeros(1)) for candidate in (plan, restored)]
    for left, right in zip(outputs[0][:5], outputs[1][:5]):
        np.testing.assert_array_equal(left, right)


def test_previous_inputs_add_only_zero_intrinsics_without_changing_optics():
    from temsim.optics.column import default_state
    from temsim.optics.model import State
    current = default_state().to_dict()
    previous = deepcopy(current)
    previous["schema_version"] = 78
    del previous["lens_intrinsic_aberrations"]
    del previous["nanoprobe_range"]
    restored = State.from_dict(previous).to_dict()
    assert restored["lens_intrinsic_aberrations"] == {}
    for key in ("lenses", "corrector_elements", "electron_gun", "stigmators"):
        assert restored[key] == current[key]
    assert previous["schema_version"] == 78 and "lens_intrinsic_aberrations" not in previous
    previous["lens_intrinsic_aberrations"] = {"objective_lens": {"a1_mm": .1}}
    with pytest.raises(ValueError, match="schema 78"):
        State.from_dict(previous)


@pytest.mark.parametrize("tilted_electric", [False, True])
def test_full_column_wave_uses_intrinsic_screen_and_resumes_once_on_gpu(tmp_path, tilted_electric):
    from test_wave_device import _cuda
    from test_tilted_spherical_column import (spherical_fixture, propagate,
        common_complex_field, field_points, action_count)
    from test_tilted_column_wave import _prepared, _centroid
    from temsim.physics.column_wave import _propagate_column_segmented
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_grid import WaveGridNumerics
    _cuda()
    state, original, source = spherical_fixture(angle_mrad=.1 if tilted_electric else 0.,
        origin_m=(20e-6, 5e-6))
    state.lens_intrinsic_aberrations = {state.lenses[0].key: {"a1_mm": .0001}}
    prepared = _prepared(state)
    if tilted_electric:
        from temsim.optics.column import default_state
        from temsim.physics.instrument_electric import capture_instrument_electric_field
        electric = capture_instrument_electric_field(default_state())
        def with_electric(value):
            return (replace(value[0], electric_field=electric,
                electric_field_identity=electric.numerical_identity), *value[1:])
        original, prepared = with_electric(original), with_electric(prepared)
    assert len(prepared[0].intrinsic_aberration_kicks) == 1
    baseline, cpu = propagate(state, original, source), propagate(state, prepared, source)
    gpu = propagate(state, prepared, source, backend="Require GPU")
    assert gpu.record["modes"][0]["compute_backend"] == "cupy"
    expected_actions = 2 if tilted_electric else 1
    assert action_count(gpu.record) == expected_actions
    points = field_points(cpu.beam.modes[0].plane)
    reference = common_complex_field(cpu.beam.modes[0], points)
    np.testing.assert_allclose(common_complex_field(gpu.beam.modes[0], points), reference,
        rtol=2e-7, atol=2e-9*np.max(abs(reference)))
    origin = source.beam.modes[0].plane.origin_m
    rays = [execute_propagation_plan(state, candidate[0], np.array([origin[0]]),
        np.zeros(1), np.array([origin[1]]), np.zeros(1)) for candidate in (original, prepared)]
    ray_delta = np.array([rays[1][1][-1, 0]-rays[0][1][-1, 0],
                          rays[1][3][-1, 0]-rays[0][3][-1, 0]])
    wave_delta = _centroid(cpu.beam.modes[0].plane)-_centroid(baseline.beam.modes[0].plane)
    assert np.linalg.norm(ray_delta) > 1e-12
    np.testing.assert_allclose(wave_delta, ray_delta, rtol=.02, atol=5e-13)
    state.acceleration_enabled = True
    state.acceleration_backend = "Require GPU"
    gpu_ray = execute_propagation_plan(state, prepared[0], np.array([origin[0]]),
        np.zeros(1), np.array([origin[1]]), np.zeros(1))
    assert state.active_backend.startswith("CUDA GPU")
    for name in ("x_m", "y_m", "tx_rad", "ty_rad"):
        np.testing.assert_allclose(getattr(gpu_ray[-1], name), getattr(rays[1][-1], name),
            rtol=8e-11, atol=2e-19)
    node = int(np.flatnonzero(prepared[0].z_mm == prepared[0].intrinsic_aberration_kicks[0].z_mm)[0])
    store = ExecutedWaveStore(tmp_path, "intrinsic-objective", 1<<27)
    options = WaveGridNumerics(compute_backend="Require GPU", maximum_pixels=512,
        maximum_working_bytes=512*1024**2, maximum_device_working_bytes=512*1024**2)
    segmented, hit = _propagate_column_segmented(state, source, float(prepared[0].z_mm[-1]),
        store=store, segment_steps=node, _prepared=prepared, grid_numerics=options)
    assert not hit and action_count(segmented.record) == expected_actions
    np.testing.assert_allclose(common_complex_field(segmented.beam.modes[0], points), reference,
        rtol=3e-7, atol=3e-9*np.max(abs(reference)))
    assert segmented.beam.total_weight == pytest.approx(source.beam.total_weight, abs=2e-12)
