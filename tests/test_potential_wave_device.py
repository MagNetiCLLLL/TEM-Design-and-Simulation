"""Residual device parity, unchanged-input retries, and retained admission."""
from contextlib import contextmanager

import numpy as np
import pytest

from temsim.physics import potential_wave_device as execution
from temsim.physics import potential_wave_carrier as carrier
from temsim.physics.compute_backend import GPUExecutionError, cupy_capability
from temsim.physics.wave_device import WaveDevice
from test_potential_wave_carrier import _wave, _round, _multipole, _snapshot, ENERGY_EV


def _arguments():
    wave = _wave(n=64)
    coordinate = np.linspace(-1., 1., 64)
    x, y = np.meshgrid(coordinate, coordinate)
    shape = wave.amplitude.shape
    return (wave.amplitude, np.zeros((2, *shape)), .01*(x*x+y*y),
            np.zeros((2, *shape)), np.zeros((2, *shape)), np.zeros(shape), 1.), {
        "phase": .001*(x*x+y*y), "scalar_splitting": True, "kinetic_order": 6,
        "quartic_scalar_splitting": False, "higher_order_scalar_splitting": True,
        "count": 2, "degree": 3, "k_distance": .1, "boundary_limit": 1e-9,
        "envelope_limit": 1e-10, "cancelled": lambda: False,
    }


def test_cpu_execution_does_not_probe_cuda_and_leaves_input_unchanged(monkeypatch):
    import temsim.physics.wave_device as device
    monkeypatch.setattr(device, "cupy_module", lambda: pytest.fail("CPU must not load CUDA"))
    args, options = _arguments()
    original = args[0].copy()
    output, record = execution.execute_residual_exponential(*args, **options)
    np.testing.assert_array_equal(args[0], original)
    np.testing.assert_allclose(output, original*np.exp(-2j*options["phase"]), rtol=3e-16, atol=1e-18)
    assert record["compute_backend"] == "numpy" and record["fallback_reason"] is None


def test_resource_retry_uses_original_input_after_completed_substep(monkeypatch):
    args, options = _arguments()
    expected, _ = execution.execute_residual_exponential(*args, **options)
    original = args[0].copy()
    attempted = []
    # This observes resource policy with NumPy; it does not claim GPU execution.
    @contextmanager
    def scopes(policy, **kwargs):
        attempted.append(policy)
        yield WaveDevice()
    monkeypatch.setattr(execution, "device_scope", scopes)
    operation = carrier._apply_residual
    calls = [0]
    def interrupted(*a, **k):
        calls[0] += 1
        if calls[0] == 4:
            raise GPUExecutionError("out_of_memory", "injected after one accepted substep")
        return operation(*a, **k)
    monkeypatch.setattr(carrier, "_apply_residual", interrupted)
    output, record = execution.execute_residual_exponential(*args, **options, requested="Auto")
    assert attempted == ["Auto", "CPU"]
    np.testing.assert_array_equal(args[0], original)
    np.testing.assert_array_equal(output, expected)
    assert record["retry_input"] == "unchanged host residual entrance"
    assert "out_of_memory" in record["fallback_reason"]


def test_gpu_resource_retry_replans_with_cpu_host_budget_and_cpu_workspace_base(monkeypatch):
    from temsim.physics import potential_wave_chebyshev as planner
    from test_potential_wave_chebyshev import _input
    args, options, _ = _input()
    original = args[0].copy()
    # The host's unchanged Taylor kernel fits, while the larger CPU
    # Chebyshev recurrence does not. GPU planning has a separate budget.
    options.update(maximum_host_working_bytes=640*args[0].size,
        maximum_device_working_bytes=1024**2, maximum_chebyshev_group=4)
    expected, _ = execution.execute_residual_exponential(*args,
        **{**options, "exponential_method": "taylor"})
    gpu_plan = planner.plan_chebyshev_exponential(options["k_distance"],
        options["generator_norm_bound"], options["count"], options["degree"],
        options["series_error_budget"], cells=args[0].size,
        maximum_working_bytes=options["maximum_device_working_bytes"], maximum_group=4,
        base_bytes_per_cell=384)
    policies, plans, applications = [], [], [0]
    # This isolates dispatch and planning with NumPy; actual bounded CUDA
    # allocations have independent device qualification tests.
    @contextmanager
    def scopes(policy, **kwargs):
        policies.append(policy)
        yield WaveDevice(backend="cupy" if policy == "Auto" else "numpy",
            maximum_working_bytes=options["maximum_device_working_bytes"])
    plan = planner.plan_chebyshev_exponential
    def capture_plan(*args, **kwargs):
        plans.append((kwargs["base_bytes_per_cell"], kwargs["maximum_working_bytes"]))
        return plan(*args, **kwargs)
    apply = carrier._apply_residual
    def fail_after_first_group(*args, **kwargs):
        applications[0] += 1
        if applications[0] == gpu_plan.groups[0].degree+2:
            raise GPUExecutionError("out_of_memory", "injected after a GPU-planned group")
        return apply(*args, **kwargs)
    monkeypatch.setattr(execution, "device_scope", scopes)
    monkeypatch.setattr(planner, "plan_chebyshev_exponential", capture_plan)
    monkeypatch.setattr(carrier, "_apply_residual", fail_after_first_group)
    actual, record = execution.execute_residual_exponential(*args, **options, requested="Auto")
    assert policies == ["Auto", "CPU"]
    assert plans == [(384, options["maximum_device_working_bytes"]),
                     (640, options["maximum_host_working_bytes"])]
    assert record["exponential_method"] == "taylor"
    assert record["retry_input"] == "unchanged host residual entrance"
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(args[0], original)


@pytest.mark.parametrize("error", [ValueError("physical admission"), InterruptedError("cancelled"),
    GPUExecutionError("invalid_configuration", "invalid kernel"),
    GPUExecutionError("out_of_memory", "no memory")])
def test_required_gpu_and_nonresource_failures_never_retry(monkeypatch, error):
    args, options = _arguments()
    policies = []
    @contextmanager
    def failing(policy, **kwargs):
        policies.append(policy)
        raise error
        yield
    monkeypatch.setattr(execution, "device_scope", failing)
    with pytest.raises(type(error), match=str(error)):
        execution.execute_residual_exponential(*args, **options, requested="Require GPU")
    assert policies == ["Require GPU"]


def test_device_cleanup_error_cannot_hide_original_cancellation(monkeypatch):
    args, options = _arguments()
    def stop():
        return True
    options["cancelled"] = stop
    def failed_cleanup():
        raise RuntimeError("cleanup failed")
    monkeypatch.setattr(execution, "release_device_memory", failed_cleanup)
    with pytest.raises(InterruptedError, match="cancelled") as caught:
        execution.execute_residual_exponential(*args, **options)
    assert any("cleanup also failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("failure", ["boundary", "spectrum"])
def test_host_admission_is_not_bypassed_by_device_selection(failure, monkeypatch):
    args, options = _arguments()
    amplitude = args[0].copy()
    if failure == "boundary":
        amplitude[0, 0] = 1.
    else:
        amplitude *= np.exp(2j*np.pi*.46875*np.arange(64)[None, :])
    @contextmanager
    def scope(*a, **k):
        yield WaveDevice()
    monkeypatch.setattr(execution, "device_scope", scope)
    with pytest.raises(ValueError, match="boundary|Nyquist"):
        execution.execute_residual_exponential(amplitude, *args[1:], **options, requested="Auto")


@pytest.mark.parametrize("order,split", [(2, False), (4, False), (4, True), (6, False), (6, True), (8, True)])
def test_real_gpu_complete_carrier_matches_cpu_and_returns_host(order, split):
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    source = _wave(n=64, curvature=.2, tilt=(2e-7, -1e-7))
    snapshot = _snapshot(_round(), _multipole())
    options = {"energy_ev": ENERGY_EV, "kinetic_order": order,
               "scalar_splitting": split,
               "quartic_scalar_splitting": split and order == 4,
               "higher_order_scalar_splitting": split and order in (6, 8)}
    cpu, a = carrier.propagate_carrier_potential(snapshot, source, (0., 1e-6, 2e-6), **options)
    gpu, b = carrier.propagate_carrier_potential(snapshot, source, (0., 1e-6, 2e-6),
        residual_compute_backend="Require GPU", **options)
    assert type(gpu.amplitude) is np.ndarray
    assert b["residual_backends"] == ["cupy"]
    assert np.linalg.norm(gpu.amplitude-cpu.amplitude) < 2e-13
    for name in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(cpu, name), getattr(gpu, name))
    assert gpu.probability == pytest.approx(cpu.probability, rel=2e-13)
    assert b["kinetic_remainder_phase_bound"] == a["kinetic_remainder_phase_bound"]
    assert b["series_amplitude_error_bound"] == a["series_amplitude_error_bound"]
    assert all(row["boundary_check"] == "each bounded exponential substep" for row in b["residual_steps"])


@pytest.mark.parametrize("order,split", [(2, False), (4, False), (4, True), (6, False), (6, True), (8, True)])
def test_real_gpu_mixed_electric_operator_keeps_full_grid_cross_terms(order, split):
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    from test_potential_wave_carrier_fft import _case
    from temsim.physics.wave_device import device_scope, release_device_memory
    amplitude, _, velocity, scalar, mechanical, frequency, defect, _ = _case()
    kwargs = {"quartic_scalar_splitting": split and order == 4,
              "higher_order_scalar_splitting": split and order in (6, 8)}
    expected = carrier._apply_residual(amplitude, velocity, scalar, mechanical, frequency,
                                       order, defect, .873, **kwargs)
    with device_scope("Require GPU", required_bytes=640*amplitude.size) as session:
        cp = session.xp
        device_values = [cp.asarray(value) for value in
                         (amplitude, velocity, scalar, mechanical, frequency, defect)]
        actual_device = carrier._apply_residual(*device_values[:5], order, device_values[5], .873, **kwargs)
        actual = cp.asnumpy(actual_device)
        del actual_device, device_values
    release_device_memory()
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 3e-14
