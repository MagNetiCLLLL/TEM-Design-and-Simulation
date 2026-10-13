"""Real-CUDA parity of event boundaries, native actions and resumed clocks."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.intrinsic_particle_gpu import intrinsic_gpu_transport, _slice_inputs
from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick
from temsim.physics.electrostatic_column_transport import _momentum_speed


@pytest.fixture(autouse=True)
def cuda_hardware():
    from temsim.physics.compute_backend import cuda_capability
    from test_wave_device import _cuda
    _cuda()
    if not cuda_capability().available:
        pytest.skip("No CUDA hardware")


def inputs_fixture(*, timed=True):
    z = np.linspace(0., 10., 9)
    count = 7
    energy = np.linspace(280000., 320000., count)
    momentum = np.array([_momentum_speed(value)[0] for value in energy])
    speed = np.array([_momentum_speed(value)[1] for value in energy])
    stages = [np.zeros(2*len(z)-1) for _ in range(5)]
    # Constant quadrupole + hexapole force and overlapping native actions
    # exercise segmented integration rather than just a final image shift.
    stages[0][:] = 30.; stages[1][:] = -20.
    stages[2][:] = 2e5; stages[3][:] = 3e4
    actions = [np.zeros(len(z)) for _ in range(5)]
    for node in (0, 4, 8):
        actions[0][node] = 2e7
        actions[1][node] = 2.
        actions[2][node] = .03
        actions[3][node] = 2e-5
        actions[4][node] = -1e-5
    x = np.linspace(-2e-5, 2e-5, count)
    y = np.linspace(1e-5, -1e-5, count)
    coords = (x, np.full(count, 2e-4), y, np.full(count, -1e-4))
    save = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    checkpoints = np.array([0, 4, 8], dtype=np.int64)
    forcing = np.zeros(3*(len(z)-1))
    inputs = (*stages, 1./momentum, *actions[:3], np.diff(z)*1e-3,
              *coords, *actions[3:], save, checkpoints, np.zeros_like(stages[0]),
              forcing, forcing.copy(), np.array([_momentum_speed(300000.)[0]]))
    kicks = {node: (FrozenIntrinsicAberrationKick("lens", z[node]*1e-3, .01,
                 (("c1_mm", .002), ("a1_mm", .001), ("b2_mm", .03))),)
             for node in (0, 4, 8)}
    times = np.linspace(1e-9, 2e-9, count) if timed else None
    return inputs, z, energy, times, 1./speed, kicks


def compare_arrays(actual, expected, *, electric=False):
    assert len(actual) == len(expected)
    for index, (got, wanted) in enumerate(zip(actual, expected)):
        # Existing compiled magnetic display history is float32. Checkpoints,
        # flight times and electric history retain float64 throughout.
        tolerance = 1e-6 if index < 4 and not electric else 2e-10
        np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=2e-18,
                                   equal_nan=True, err_msg=f"transport output {index}")


@pytest.mark.parametrize("timed", [False, True])
def test_native_actions_and_intrinsics_execute_once_on_gpu(timed):
    from temsim.physics.vector_field_transport import vector_map_rk4
    inputs, z, energy, times, invspeed, kicks = inputs_fixture(timed=timed)
    originals = tuple(value.copy() for value in inputs)
    expected = vector_map_rk4(*inputs, z_mm=z, mapped_fields=(),
        posed_spherical_kicks=kicks, initial_time_s=times,
        inverse_speed=invspeed if timed else None)
    actual = intrinsic_gpu_transport(inputs, z_mm=z, initial_kinetic_energy_ev=energy,
        posed_spherical_kicks=kicks, initial_time_s=times)
    assert len(actual) == (10 if timed else 8)
    compare_arrays(actual, expected)
    for value, original in zip(inputs, originals):
        np.testing.assert_array_equal(value, original)
    retained = tuple(value.copy() for value in actual)
    intrinsic_gpu_transport(inputs, z_mm=z, initial_kinetic_energy_ev=energy,
        posed_spherical_kicks=kicks, initial_time_s=times)
    for value, original in zip(actual, retained):
        np.testing.assert_array_equal(value, original)


def test_gpu_events_preserve_electric_energy_and_clock_then_resume():
    from test_electrostatic_column_transport import fixture
    from temsim.physics.electrostatic_column_transport import electrostatic_column_rk4
    inputs, z, energy, times, _speed, kicks = inputs_fixture()
    _state, plan = fixture(gradient=2e6, radial=2e7)
    electric = plan.electric_field
    expected, _, _ = electrostatic_column_rk4(inputs, z_mm=z, electric_field=electric,
        initial_kinetic_energy_ev=energy, initial_time_s=times,
        posed_spherical_kicks=kicks)
    actual = intrinsic_gpu_transport(inputs, z_mm=z, electric_field=electric,
        initial_kinetic_energy_ev=energy, initial_time_s=times,
        posed_spherical_kicks=kicks)
    assert len(actual) == 12
    compare_arrays(actual, expected, electric=True)
    node = 4
    resumed_inputs = _slice_inputs(inputs, node, len(z)-1,
        tuple(actual[i][1] for i in range(4, 8)),
        np.array([0, 2, 4]), np.array([0, 4]), suppress_start=True)
    resumed = intrinsic_gpu_transport(resumed_inputs, z_mm=z[node:], electric_field=electric,
        initial_kinetic_energy_ev=actual[11][1], initial_time_s=actual[9][1],
        posed_spherical_kicks={4: kicks[8]})
    for index in (4, 5, 6, 7, 9, 11):
        np.testing.assert_allclose(resumed[index][-1], actual[index][-1], rtol=1e-12, atol=1e-22)


def test_empty_saved_history_and_event_only_requests():
    inputs, z, energy, times, _speed, kicks = inputs_fixture()
    one = list(_slice_inputs(inputs, 4, 4, inputs[10:14], np.array([], dtype=np.int64),
                            np.array([0], dtype=np.int64)))
    actual = intrinsic_gpu_transport(one, z_mm=z[4:5],
        initial_kinetic_energy_ev=energy, initial_time_s=times,
        posed_spherical_kicks={0: kicks[4]})
    from temsim.physics.vector_field_transport import vector_map_rk4
    expected = vector_map_rk4(*one, z_mm=z[4:5], mapped_fields=(),
        posed_spherical_kicks={0: kicks[4]}, initial_time_s=times, inverse_speed=_speed)
    compare_arrays(actual, expected)
    assert actual[0].shape == (0, 7)


def test_cancel_keeps_input_arrays_unchanged():
    inputs, z, energy, times, _speed, kicks = inputs_fixture()
    originals = tuple(value.copy() for value in inputs)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise InterruptedError("cancel at next event")
    with pytest.raises(InterruptedError, match="next event"):
        intrinsic_gpu_transport(inputs, z_mm=z, initial_kinetic_energy_ev=energy,
            initial_time_s=times, posed_spherical_kicks=kicks, cancel_check=cancel)
    for value, original in zip(inputs, originals):
        np.testing.assert_array_equal(value, original)


def test_posed_cs_then_tilted_intrinsic_uses_real_fast_cuda_spans(monkeypatch):
    from temsim.physics import ray_integrator, particle_gpu_reference
    from temsim.physics.lens_field_provider import CoordinateRegistration
    from temsim.physics.posed_aberrations import FrozenLensAberrationKick
    from temsim.physics.vector_field_transport import vector_map_rk4
    inputs, z, energy, times, invspeed, kicks = inputs_fixture()
    angle = .002
    co, si = np.cos(angle), np.sin(angle)
    pose = CoordinateRegistration((2e-6, -1e-6, .005),
                                  ((co, 0., si), (0., 1., 0.), (-si, 0., co)))
    kicks = {4: (FrozenLensAberrationKick("lens", 0., 3e7, pose),
                 replace(kicks[4][0], local_z_m=0., registration=pose))}
    expected = vector_map_rk4(*inputs, z_mm=z, mapped_fields=(),
        posed_spherical_kicks=kicks, initial_time_s=times, inverse_speed=invspeed)
    real_fast = ray_integrator.cuda_rk4
    real_reference = particle_gpu_reference.gpu_reference_transport
    span_sizes, event_sizes = [], []
    def fast(*args, **kwargs):
        span_sizes.append(len(args[9]))
        return real_fast(*args, **kwargs)
    def reference(args, **kwargs):
        event_sizes.append(len(args[9]))
        return real_reference(args, **kwargs)
    monkeypatch.setattr(ray_integrator, "cuda_rk4", fast)
    monkeypatch.setattr(particle_gpu_reference, "gpu_reference_transport", reference)
    actual = intrinsic_gpu_transport(inputs, z_mm=z, initial_kinetic_energy_ev=energy,
        initial_time_s=times, posed_spherical_kicks=kicks)
    compare_arrays(actual, expected)
    assert span_sizes == [4, 4]
    assert event_sizes == [0]
    # The legacy geometric Cs screen and the tilted canonical screen are
    # noncommuting. CPU agreement must preserve their production order.
    reversed_order = vector_map_rk4(*inputs, z_mm=z, mapped_fields=(),
        posed_spherical_kicks={4: tuple(reversed(kicks[4]))},
        initial_time_s=times, inverse_speed=invspeed)
    assert np.max(np.abs(expected[5]-reversed_order[5])) > 1e-15


def test_core_require_gpu_records_complete_intrinsic_execution():
    from test_intrinsic_aberration_integration import fixture, plan_for
    from temsim.physics.core import execute_propagation_plan
    from temsim.physics.backend_execution import backend_receipts
    state = fixture({"c1_mm": .003, "a1_mm": .002, "a1_azimuth_deg": 31.})
    state.lenses[0].cs_mm = .5
    plan = plan_for(state)
    source = (np.array([1e-5, -2e-5]), np.zeros(2),
              np.array([2e-5, 1e-5]), np.zeros(2))
    expected = execute_propagation_plan(state, plan, *source,
        initial_time_s=np.array([1e-9, 2e-9]), return_flight_times=True)
    state.acceleration_enabled = True
    state.acceleration_backend = "Require GPU"
    rows = []
    with backend_receipts(rows):
        actual = execute_propagation_plan(state, plan, *source,
            initial_time_s=np.array([1e-9, 2e-9]), return_flight_times=True)
    assert rows and all(row["actual"] == "CUDA GPU" and row["outcome"] == "returned"
                        and not row["reason"] for row in rows)
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
        np.testing.assert_allclose(getattr(actual[-1], name), getattr(expected[-1], name),
                                   rtol=2e-12, atol=1e-20)
