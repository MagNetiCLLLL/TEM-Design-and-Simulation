"""Production backend policy and full-column intrinsic CUDA parity."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics import core
from temsim.physics.backend_execution import backend_receipts
from temsim.physics.compute_backend import GPUExecutionError
from test_intrinsic_aberration_integration import fixture, plan_for


def _sources(count=33):
    rng = np.random.default_rng(3412)
    return (rng.uniform(-2e-5, 2e-5, count), rng.uniform(-2e-4, 2e-4, count),
            rng.uniform(-2e-5, 2e-5, count), rng.uniform(-2e-4, 2e-4, count))


def _all_terms():
    return dict(c1_mm=.0003, a1_mm=.001, a1_azimuth_deg=31.,
                b2_mm=.01, b2_azimuth_deg=47., a2_mm=.02, a2_azimuth_deg=-19.,
                s3_mm=.3, s3_azimuth_deg=17., a3_mm=.6, a3_azimuth_deg=73.,
                c5_mm=200., a5_mm=300., a5_azimuth_deg=-8.)


def _run(state, plan, sources, backend, *, initial_energy=None, **kwargs):
    state.acceleration_enabled = backend != "CPU"
    state.acceleration_backend = backend
    count = len(sources[0])
    if initial_energy is None:
        initial_energy = np.linspace(290000., 310000., count)
    rows, energy = [], []
    with backend_receipts(rows):
        result = core.execute_propagation_plan(state, plan, *sources,
            initial_kinetic_energy_ev=initial_energy,
            initial_time_s=np.linspace(0., 1e-12, count), return_flight_times=True,
            energy_output=energy, **kwargs)
    return result, energy[0], rows


def _same(reference, actual):
    # Plotted trajectories deliberately retain the existing float32 CUDA ABI;
    # exact checkpoints, energy and time must agree at float64 accuracy.
    for left, right in zip(reference[0][1:5], actual[0][1:5]):
        np.testing.assert_allclose(right, left, rtol=2e-6, atol=2e-12)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        left, right = (getattr(value[0][-1], name) for value in (reference, actual))
        np.testing.assert_allclose(right, left, rtol=8e-11, atol=2e-20)
    np.testing.assert_allclose(actual[1], reference[1], rtol=8e-13)


@pytest.mark.parametrize("backend", ["Auto", "Require GPU"])
def test_top_level_backend_runs_sparse_intrinsic_events_on_real_cuda(backend):
    from test_wave_device import _cuda
    _cuda()
    state = fixture(_all_terms())
    state.lenses[0].cs_mm = .5
    plan = plan_for(state)
    original = plan.intrinsic_aberration_kicks[0]
    # Include consecutive events and coincident old native actions. This checks
    # the actual production segmented path, including a saved event node.
    screens = tuple(replace(original, local_z_m=float(z)*1e-3)
                    for z in plan.z_mm[1:-1])
    magnetic = np.full_like(plan.magnetic_t, .01)
    mid = np.full_like(plan.midpoint_magnetic_t, .01)
    thin = plan.thin_power_m1.copy(); thin[1] = .8
    rotation = plan.thin_rotation_rad.copy(); rotation[1] = .02
    kick = plan.kick_x_rad.copy(); kick[1] = 1e-5
    plan = replace(plan, intrinsic_aberration_kicks=screens,
        magnetic_t=magnetic, midpoint_magnetic_t=mid,
        thin_power_m1=thin, thin_rotation_rad=rotation, kick_x_rad=kick)
    source = _sources(3000)
    snapshots = tuple(a.copy() for a in source)
    lens_before = deepcopy(vars(state.lenses[0]))
    definition_before = deepcopy(state.lens_intrinsic_aberrations)
    expected = _run(state, plan, source, "CPU")
    actual = _run(state, plan, source, backend)
    _same(expected, actual)
    assert state.active_backend.startswith("CUDA GPU")
    assert actual[2][-1]["actual"] == "CUDA GPU" and not actual[2][-1]["retried"]
    assert state._last_ray_device_receipt is None
    for old, new in zip(snapshots, source):
        np.testing.assert_array_equal(new, old)
    assert vars(state.lenses[0]) == lens_before
    assert state.lens_intrinsic_aberrations == definition_before
    # Selecting CPU after a GPU run must obey the same top-level setting.
    cpu_again = _run(state, plan, source, "CPU")
    _same(expected, cpu_again)
    assert cpu_again[2][-1]["actual"] == "CPU"


def test_electric_intrinsic_gpu_preserves_energy_time_and_after_screen_resume():
    from test_wave_device import _cuda
    from test_electrostatic_column_transport import fixture as electric_fixture
    from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick
    _cuda()
    state, plan = electric_fixture(radial=2e7, step=.2)
    screen = FrozenIntrinsicAberrationKick("electric-test-lens", .004, .01,
                                          tuple(_all_terms().items()))
    plan = replace(plan, intrinsic_aberration_kicks=(screen,))
    source = _sources()
    expected = _run(state, plan, source, "CPU")
    actual = _run(state, plan, source, "Require GPU")
    _same(expected, actual)
    checkpoint = actual[0][-1]
    row = int(np.flatnonzero(checkpoint.z_mm == 4.)[0])
    resumed = core.execute_propagation_plan(state, plan,
        *(getattr(checkpoint, name)[row] for name in ("x_m", "tx_rad", "y_m", "ty_rad")),
        start_index=int(np.flatnonzero(plan.z_mm == 4.)[0]), include_initial_plane_kicks=False,
        initial_kinetic_energy_ev=checkpoint.kinetic_energy_ev[row],
        initial_time_s=checkpoint.flight_time_s[row], return_flight_times=True)[-1]
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        np.testing.assert_allclose(getattr(resumed, name)[-1], getattr(checkpoint, name)[-1],
                                   rtol=8e-11, atol=2e-20)


@pytest.mark.parametrize("backend", ["Auto", "Require GPU"])
def test_only_declared_gpu_resource_failure_can_retry_unchanged_cpu_input(monkeypatch, backend):
    from temsim.physics import intrinsic_particle_gpu
    monkeypatch.setattr(core, "choose_ray_backend", lambda *a, **k: ("CUDA GPU", None))
    state = fixture(_all_terms()); plan = plan_for(state); source = _sources()
    # Reference is captured before simulating GPU unavailability.
    monkeypatch.setattr(core, "choose_ray_backend", lambda *a, **k: ("CPU", None))
    expected = _run(state, plan, source, "CPU")
    monkeypatch.setattr(core, "choose_ray_backend", lambda *a, **k: ("CUDA GPU", None))
    def fail(*args, **kwargs):
        raise GPUExecutionError("out_of_memory", "injected intrinsic allocation failure")
    monkeypatch.setattr(intrinsic_particle_gpu, "intrinsic_gpu_transport", fail)
    if backend == "Require GPU":
        with pytest.raises(GPUExecutionError, match="allocation failure"):
            _run(state, plan, source, backend)
    else:
        actual = _run(state, plan, source, backend)
        _same(expected, actual)
        assert actual[2][-1]["retried"]
        assert actual[2][-1]["actual"] == "CPU"
        assert any(row["actual"] == "CUDA GPU" and row["outcome"] == "raised"
                   for row in actual[2])
        assert "allocation failure" in actual[2][-1]["reason"]


def test_intrinsic_physics_errors_are_not_relabelled_as_gpu_failures(monkeypatch):
    from temsim.physics import intrinsic_particle_gpu
    monkeypatch.setattr(core, "choose_ray_backend", lambda *a, **k: ("CUDA GPU", None))
    def invalid(*args, **kwargs):
        raise ValueError("intrinsic inclination outside physical domain")
    monkeypatch.setattr(intrinsic_particle_gpu, "intrinsic_gpu_transport", invalid)
    state = fixture(_all_terms())
    with pytest.raises(ValueError, match="outside physical domain"):
        _run(state, plan_for(state), _sources(), "Auto")
