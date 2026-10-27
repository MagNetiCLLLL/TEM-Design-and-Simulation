"""Complete Hermitian exponential, original checks, and transactional resources."""
from contextlib import contextmanager
from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.physics import potential_wave_carrier as carrier
from temsim.physics import potential_wave_device as device
from temsim.physics.compute_backend import GPUExecutionError, cupy_capability
from temsim.physics.potential_wave_chebyshev import (
    ChebyshevPlanUnavailable, chebyshev_working_bytes, plan_chebyshev_exponential,
)
from temsim.physics.wave_device import WaveDevice
from temsim.physics.wave_magnetic_residual import _series_plan
from test_potential_wave_carrier_fft import _case, _dense


def _input(order=8, split=True, kappa=16.25):
    amplitude, _, velocity, scalar, mechanical, frequency, defect, derivatives = _case((5, 7), .28)
    amplitude /= np.linalg.norm(amplitude)
    beta = .83
    dense = _dense(velocity, scalar, mechanical, defect, derivatives, order, beta)
    u = np.sum(mechanical**2, axis=0)+defect
    scalar_part = scalar.copy()
    for n, c in ((2, 1/8), (3, 1/16), (4, 5/128)):
        if 2*n <= order:
            scalar_part += c*beta**(2*n-1)*u**n
    if split:
        dense -= np.diag(scalar_part.ravel())
    norm = 1.2*float(np.max(abs(np.linalg.eigvalsh(dense))))
    distance = kappa/norm
    count, degree, bound = _series_plan(abs(distance)*norm, 1e-12)
    args = (amplitude, velocity, scalar_part if split else scalar, mechanical, frequency, defect, beta)
    options = dict(phase=scalar_part*distance*.5 if split else scalar*distance,
        scalar_splitting=split, kinetic_order=order, quartic_scalar_splitting=split and order == 4,
        higher_order_scalar_splitting=split and order >= 6, count=count, degree=degree,
        k_distance=distance, boundary_limit=.99, envelope_limit=.99, cancelled=lambda: False,
        exponential_method="chebyshev", generator_norm_bound=norm, series_error_budget=bound,
        maximum_host_working_bytes=2**26, maximum_device_working_bytes=2**26)
    return args, options, dense


@pytest.mark.parametrize("order", [4, 6, 8])
@pytest.mark.parametrize("split", [False, True])
def test_same_complete_mixed_operator_matches_independent_dense_exponential(order, split):
    args, options, dense = _input(order, split)
    original = args[0].copy()
    expected_input = original*np.exp(-1j*options["phase"]) if split else original
    expected = (expm(-1j*options["k_distance"]*dense)@expected_input.ravel()).reshape(original.shape)
    if split:
        expected *= np.exp(-1j*options["phase"])
    actual, record = device.execute_residual_exponential(*args, **options)
    assert np.linalg.norm(actual-expected) < 3e-13
    assert abs(np.vdot(actual, actual).real-1.) < 3e-13
    np.testing.assert_array_equal(args[0], original)
    assert record["exponential_method"] == "chebyshev"
    assert record["actual_operator_applications"] < options["count"]*options["degree"]
    assert record["old_taylor_planned_applications"] == options["count"]*options["degree"]
    assert record["original_flux_tolerance_unchanged"]
    certificate = record["chebyshev_certificate"]
    assert certificate["series_error_bound"] <= options["series_error_budget"]
    assert certificate["original_substep_checks"] == options["count"]
    assert certificate["coefficient_time_roundoff_bound"] == 0.


def test_plan_uses_exact_product_time_and_contains_both_negative_and_positive_spectrum():
    norm, distance = .073813471168, 491.3783731
    count, degree, bound = _series_plan(abs(distance)*norm, 1e-12)
    plan = plan_chebyshev_exponential(distance, norm, count, degree, bound,
        cells=31, maximum_working_bytes=2**26)
    values = np.r_[-1., np.nextafter(-1., 0.), np.linspace(-.99, .99, 27), np.nextafter(1., 0.), 1.]
    first = plan.groups[0]
    for j, coefficients in enumerate(first.coefficients, 1):
        # Independent scalar polynomial evaluation at spectral endpoints.
        from numpy.polynomial.chebyshev import chebval
        exact_time = Fraction(distance)*Fraction(norm)*j/count
        assert np.max(abs(chebval(values, coefficients)-np.exp(-1j*float(exact_time)*values))) < 2e-14
    assert "exact Fraction" in plan.to_metadata()["time_definition"]
    assert plan.to_metadata()["recurrence_roundoff_amplification"] > plan.applications


def test_backend_workspace_base_changes_only_allocation_not_rational_certificate():
    args, options, _ = _input()
    call = dict(cells=args[0].size, maximum_working_bytes=2**26, maximum_group=8)
    values = (options["k_distance"], options["generator_norm_bound"], options["count"],
              options["degree"], options["series_error_budget"])
    cpu = plan_chebyshev_exponential(*values, **call)
    gpu = plan_chebyshev_exponential(*values, **call, base_bytes_per_cell=384)
    assert cpu.base_bytes_per_cell == 640 and gpu.base_bytes_per_cell == 384
    assert cpu.groups == gpu.groups
    assert cpu.working_bytes-gpu.working_bytes == (640-384)*args[0].size
    assert cpu.error_bound == gpu.error_bound
    for plan in (cpu, gpu):
        assert plan.to_metadata()["planning_base_bytes_per_cell"] == plan.base_bytes_per_cell


def test_device_group_reduction_obeys_its_own_exact_byte_boundary():
    args, options, _ = _input()
    cells = 10000
    values = (options["k_distance"], options["generator_norm_bound"], options["count"],
              options["degree"], options["series_error_budget"])
    required = chebyshev_working_bytes(cells, 3, base_bytes_per_cell=384)
    plan = plan_chebyshev_exponential(*values, cells=cells, maximum_working_bytes=required,
                                    base_bytes_per_cell=384)
    assert max(group.count for group in plan.groups) == 3
    assert plan.working_bytes == required
    smaller = plan_chebyshev_exponential(*values, cells=cells, maximum_working_bytes=required-1,
                                       base_bytes_per_cell=384)
    assert max(group.count for group in smaller.groups) == 2
    with pytest.raises(ChebyshevPlanUnavailable, match="workspace"):
        plan_chebyshev_exponential(*values, cells=cells, maximum_working_bytes=required)


@pytest.mark.parametrize("base", [None, True, 0, 383, 384., "384"])
def test_invalid_or_unqualified_backend_workspace_base_is_rejected(base):
    with pytest.raises(ValueError):
        chebyshev_working_bytes(100, 1, base_bytes_per_cell=base)
    args, options, _ = _input()
    with pytest.raises(ValueError):
        plan_chebyshev_exponential(options["k_distance"], options["generator_norm_bound"],
            options["count"], options["degree"], options["series_error_budget"],
            cells=args[0].size, maximum_working_bytes=2**26, base_bytes_per_cell=base)


def test_complete_original_check_schedule_matches_independent_exponential_at_every_time(monkeypatch):
    args, options, dense = _input(split=False)
    observed = []
    boundary = carrier._check_boundary
    def inspect(value, limit):
        observed.append(np.array(value, copy=True))
        return boundary(value, limit)
    monkeypatch.setattr(carrier, "_check_boundary", inspect)
    device.execute_residual_exponential(*args, **options)
    assert len(observed) == options["count"]+1
    for index, value in enumerate(observed):
        expected = expm(-1j*options["k_distance"]*index/options["count"]*dense)@args[0].ravel()
        assert np.linalg.norm(value.ravel()-expected) < 3e-13


@pytest.mark.parametrize("distance", [12.345, -12.345])
def test_signed_time_and_finite_unitary_gauge_keep_the_raw_complex_phase(monkeypatch, distance):
    args, options, dense = _input(split=False, kappa=abs(distance))
    options["k_distance"] = math.copysign(options["k_distance"], distance)
    expected, _ = device.execute_residual_exponential(*args, **options)
    phase = np.exp(1j*np.linspace(-.9, 1.1, args[0].size).reshape(args[0].shape))
    apply = carrier._apply_residual
    def transformed(value, *a, **k):
        return phase*apply(np.conj(phase)*value, *a, **k)
    monkeypatch.setattr(carrier, "_apply_residual", transformed)
    actual, _ = device.execute_residual_exponential(phase*args[0], *args[1:], **options)
    assert np.linalg.norm(actual-phase*expected) < 3e-13


@pytest.mark.parametrize("kind", ["boundary", "spectrum"])
def test_every_original_substep_remains_checked_even_when_final_state_returns(monkeypatch, kind):
    size = 8
    if kind == "boundary":
        initial = np.zeros((size, size), complex); initial[4, 4] = 1.
        other = np.zeros_like(initial); other[0, 0] = 1.
    else:
        initial = np.ones((size, size), complex)/size
        other = (-1.)**np.indices(initial.shape)[1]/size
    matrix = np.outer(initial.ravel(), other.ravel().conj())+np.outer(other.ravel(), initial.ravel().conj())
    monkeypatch.setattr(carrier, "_apply_residual", lambda value, *a, **k:(matrix@value.ravel()).reshape(value.shape))
    count, degree, bound = _series_plan(2*np.pi, 1e-12)
    zero = np.zeros_like(initial.real)
    original = initial.copy()
    with pytest.raises(ValueError, match="boundary|Nyquist"):
        device.execute_residual_exponential(initial, np.zeros((2, size, size)), zero,
            np.zeros((2, size, size)), np.zeros((2, size, size)), zero, 1.,
            phase=zero, scalar_splitting=False, kinetic_order=2,
            quartic_scalar_splitting=False, higher_order_scalar_splitting=False,
            count=count, degree=degree, k_distance=2*np.pi,
            boundary_limit=.01 if kind == "boundary" else .99, envelope_limit=.001,
            cancelled=lambda: False, exponential_method="chebyshev", generator_norm_bound=1.,
            series_error_budget=bound)
    np.testing.assert_array_equal(initial, original)
    assert np.linalg.norm(expm(-2j*np.pi*matrix)@initial.ravel()-initial.ravel()) < 3e-15


def test_group_workspace_reduces_group_before_allocation_and_preserves_original_method(monkeypatch):
    args, options, _ = _input()
    needed = chebyshev_working_bytes(args[0].size, 3)
    options["maximum_host_working_bytes"] = needed
    _, record = device.execute_residual_exponential(*args, **options)
    assert max(record["chebyshev_certificate"]["group_sizes"]) <= 3
    assert record["chebyshev_certificate"]["estimated_working_bytes"] <= needed
    # The original 640-byte/cell Taylor baseline fits but no new recurrence
    # or output buffer fits. Keep Require GPU as the requested backend.
    options["maximum_host_working_bytes"] = 640*args[0].size
    calls = []
    @contextmanager
    def scope(policy, **kwargs):
        calls.append(policy)
        yield WaveDevice()
    monkeypatch.setattr(device, "device_scope", scope)
    actual, record = device.execute_residual_exponential(*args, **options, requested="Require GPU")
    expected, _ = device.execute_residual_exponential(*args, **{**options, "exponential_method":"taylor"})
    np.testing.assert_array_equal(actual, expected)
    assert calls == ["Require GPU", "CPU"]
    assert record["exponential_method"] == "taylor"
    assert "workspace" in record["exponential_method_fallback"]


def test_auto_resource_retry_after_completed_group_uses_entire_original_input(monkeypatch):
    args, options, _ = _input()
    expected, reference = device.execute_residual_exponential(*args, **options)
    first_group_calls = reference["chebyshev_certificate"]["group_degrees"][0]
    calls, policies = [0], []
    apply = carrier._apply_residual
    @contextmanager
    def scopes(policy, **kwargs):
        policies.append(policy)
        yield WaveDevice()
    def fail(value, *a, **k):
        calls[0] += 1
        if calls[0] == first_group_calls+2:
            raise GPUExecutionError("out_of_memory", "injected after a completed group")
        return apply(value, *a, **k)
    monkeypatch.setattr(device, "device_scope", scopes)
    monkeypatch.setattr(carrier, "_apply_residual", fail)
    actual, record = device.execute_residual_exponential(*args, **options, requested="Auto")
    np.testing.assert_array_equal(actual, expected)
    assert policies == ["Auto", "CPU"]
    assert record["retry_input"] == "unchanged host residual entrance"


@pytest.mark.parametrize("error", [InterruptedError("cancelled"), ValueError("physical gate"),
    GPUExecutionError("invalid_configuration", "invalid kernel"), GPUExecutionError("out_of_memory", "no memory")])
def test_required_gpu_does_not_publish_or_retry_failed_group(monkeypatch, error):
    args, options, _ = _input()
    original = args[0].copy(); policies = []
    @contextmanager
    def scope(policy, **kwargs):
        policies.append(policy)
        yield WaveDevice()
    def fail(*a, **k): raise error
    monkeypatch.setattr(device, "device_scope", scope)
    monkeypatch.setattr(carrier, "_apply_residual", fail)
    with pytest.raises(type(error), match=str(error)):
        device.execute_residual_exponential(*args, **options, requested="Require GPU")
    assert policies == ["Require GPU"]
    np.testing.assert_array_equal(args[0], original)


def test_cancellation_during_certificate_does_not_touch_input():
    args, options, _ = _input()
    original = args[0].copy()
    options["cancelled"] = lambda: True
    with pytest.raises(InterruptedError, match="cancelled"):
        device.execute_residual_exponential(*args, **options)
    np.testing.assert_array_equal(args[0], original)


@pytest.mark.parametrize("updates", [dict(maximum_chebyshev_group=True), dict(maximum_chebyshev_group=0),
    dict(maximum_chebyshev_group=33), dict(maximum_host_working_bytes=True), dict(exponential_method="unknown"),
    dict(generator_norm_bound=float("nan")), dict(generator_norm_bound=None), dict(generator_norm_bound=-1.),
    dict(series_error_budget=0.), dict(count=10**9), dict(degree=True)])
def test_invalid_explicit_options_are_not_silent_taylor_fallback(updates):
    args, options, _ = _input()
    with pytest.raises(ValueError):
        device.execute_residual_exponential(*args, **{**options, **updates})


def test_zero_generator_keeps_old_no_action_without_dividing_by_zero():
    args, options, _ = _input(split=False)
    options.update(count=0, degree=0, generator_norm_bound=0., series_error_budget=0.)
    actual, record = device.execute_residual_exponential(*args, **options)
    np.testing.assert_array_equal(actual, args[0])
    assert record["actual_operator_applications"] == 0


@pytest.mark.parametrize("order", [4, 6, 8])
def test_real_small_gpu_chebyshev_matches_cpu_complete_mixed_operator(order):
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    args, options, _ = _input(order)
    cpu, a = device.execute_residual_exponential(*args, **options)
    gpu, b = device.execute_residual_exponential(*args, **options, requested="Require GPU")
    assert type(gpu) is np.ndarray and b["compute_backend"] == "cupy"
    assert np.linalg.norm(gpu-cpu) < 3e-13
    # The certified polynomial and every original check time are identical;
    # allocation plans belong to their separately bounded execution devices.
    cpu, gpu = a["chebyshev_certificate"], b["chebyshev_certificate"]
    workspace = {"estimated_working_bytes", "planning_base_bytes_per_cell", "workspace"}
    assert {key:value for key,value in cpu.items() if key not in workspace} == {
        key:value for key,value in gpu.items() if key not in workspace}
