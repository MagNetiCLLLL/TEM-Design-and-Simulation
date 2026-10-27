"""Declared exponential allowances and achieved certificates have distinct roles."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.immutable_json import thaw_json
from temsim.physics import potential_wave_carrier as carrier
from temsim.physics import potential_wave_device as device
from temsim.physics.potential_wave_chebyshev import (
    ChebyshevPlanUnavailable, plan_chebyshev_exponential,
)
from temsim.physics.potential_geometry import observe_potential_geometry_checkpoint
from temsim.physics.wave_magnetic_residual import _series_plan
from test_potential_wave_carrier_fft import _case, _dense
from test_potential_wave_carrier_chebyshev import _periodic_residual
from test_potential_wave_carrier import _round, _snapshot, _wave
from test_potential_geometry import _checkpoint, _geometry


# Scalars captured from a genuine frozen-source-20 residual at 502.3833869 mm.
# They are planning evidence, not a replacement for its original complex wave.
_K_DISTANCE = 3191461422163.9116*.0008468520220588083
_NORM_BOUND = 1.4995638811125571e-7
_DECLARED_ERROR = 3.337808257528363e-14
_TAYLOR_BOUND = 8.758865964312184e-15


@pytest.mark.parametrize("base", [384, 640])
def test_observed_roundoff_fallback_fits_original_declared_error_and_all_406_check_times(base):
    count, degree, bound = _series_plan(_K_DISTANCE*_NORM_BOUND, _DECLARED_ERROR)
    assert (count, degree) == (406, 18)
    assert bound == _TAYLOR_BOUND
    options = dict(cells=1936**2, maximum_working_bytes=24*1024**3,
                   maximum_group=8, base_bytes_per_cell=base)
    with pytest.raises(ChebyshevPlanUnavailable, match="coefficient rounding"):
        plan_chebyshev_exponential(_K_DISTANCE, _NORM_BOUND, count, degree, bound, **options)
    plan = plan_chebyshev_exponential(_K_DISTANCE, _NORM_BOUND, count, degree, _DECLARED_ERROR, **options)
    assert plan.old_applications == 7308 and plan.applications == 1628
    assert sum(group.count for group in plan.groups) == 406
    assert plan.error_bound > bound  # A valid alternative need not match Taylor's overshoot.
    assert plan.error_bound <= _DECLARED_ERROR
    assert plan.working_bytes <= options["maximum_working_bytes"]


def test_complete_mixed_h8_preserves_all_checks_and_raw_complex_dense_reference(monkeypatch):
    amplitude, _, velocity, scalar, mechanical, frequency, defect, derivatives = _case((9, 11), .28)
    amplitude /= np.linalg.norm(amplitude)
    beta = .83
    dense = _dense(velocity, scalar, mechanical, defect, derivatives, 8, beta)
    multiplication = np.sum(mechanical**2, axis=0)+defect
    scalar_part = scalar+beta**3*multiplication**2/8+beta**5*multiplication**3/16+5*beta**7*multiplication**4/128
    dense -= np.diag(scalar_part.ravel())
    norm = 1.2*float(np.max(abs(np.linalg.eigvalsh(dense))))
    distance = _K_DISTANCE*_NORM_BOUND/norm
    count, degree, _ = _series_plan(abs(distance)*norm, _DECLARED_ERROR)
    args = (amplitude, velocity, scalar_part, mechanical, frequency, defect, beta)
    options = dict(phase=scalar_part*distance*.5, scalar_splitting=True, kinetic_order=8,
        quartic_scalar_splitting=False, higher_order_scalar_splitting=True, count=count, degree=degree,
        k_distance=distance, boundary_limit=.99, envelope_limit=.99, cancelled=lambda: False,
        exponential_method="chebyshev", generator_norm_bound=norm, series_error_budget=_DECLARED_ERROR,
        maximum_host_working_bytes=2**26, maximum_device_working_bytes=2**26)
    original = args[0].copy()
    phase = np.exp(-1j*options["phase"])
    expected = (expm(-1j*options["k_distance"]*dense)@(original*phase).ravel()).reshape(original.shape)*phase
    counts = {"boundary": 0, "spectrum": 0}
    check_boundary, check_spectrum = carrier._check_boundary, carrier._check_envelope_sampling
    def boundary(*values):
        counts["boundary"] += 1
        return check_boundary(*values)
    def spectrum(*values):
        counts["spectrum"] += 1
        return check_spectrum(*values)
    monkeypatch.setattr(carrier, "_check_boundary", boundary)
    monkeypatch.setattr(carrier, "_check_envelope_sampling", spectrum)
    output, record = device.execute_residual_exponential(*args, **options)
    assert record["exponential_method"] == "chebyshev"
    assert count == 406 and counts == {"boundary": 407, "spectrum": 409}
    assert record["actual_operator_applications"] < 7308
    # This is an independent full noncommuting dense exponential, with no
    # phase fit, normalization, position alignment, or removed scalar screen.
    assert np.linalg.norm(output-expected) < 2e-12
    assert abs(float(np.vdot(output, output).real)-1.) < 2e-12
    np.testing.assert_array_equal(args[0], original)
    assert record["original_flux_tolerance_unchanged"]


def _geometry_case():
    wave = _wave(n=64, pitch=1e-11, width=4e-11)
    checkpoint = replace(_checkpoint(wave), plane_z_mm=1e-3)
    snapshot, geometry = _snapshot(_round()), _geometry(start=1e-6, stop=1.04e-6)
    options = dict(maximum_step_m=1.1e-8, exponential_error=1e-10, kinetic_order=8,
        scalar_splitting=True, higher_order_scalar_splitting=True,
        residual_exponential_method="chebyshev")
    return snapshot, geometry, checkpoint, options


def test_real_round_lens_geometry_appends_actual_certificates_and_preserves_old_history(monkeypatch):
    snapshot, geometry, checkpoint, options = _geometry_case()
    residual, rows = carrier._residual, []
    def observe(*args, **kwargs):
        output, record = residual(*args, **kwargs)
        rows.append(record)
        return output, record
    monkeypatch.setattr(carrier, "_residual", observe)
    first, second = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (1.01e-6, 1.02e-6), **options)
    assert len(rows) == 4 and all(row["residual_execution"]["exponential_method"] == "chebyshev" for row in rows)
    assert any(row["series_error_bound"] != row["original_taylor_series_error_bound"] for row in rows)
    for row in rows:
        certificate = row["residual_execution"]["chebyshev_certificate"]
        assert row["series_error_bound"] == certificate["series_error_bound"]
        assert certificate["series_error_budget"] == row["declared_series_error_budget"]
        assert row["series_error_bound"] <= min(row["declared_series_error_budget"], 1e-12)
    prefix = first.record["potential_geometry"]["series_error_log_bound"]
    assert prefix == pytest.approx(math.fsum(math.log1p(row["series_error_bound"]) for row in rows[:2]), rel=3e-15)
    original_flux_allowance = math.expm1(math.fsum(math.log1p(
        2*row["original_taylor_series_error_bound"]+row["original_taylor_series_error_bound"]**2
        +row["roundoff_relative_allowance"]) for row in rows[:2]))
    actual = second.record["potential_geometry"]["series_error_log_bound"]
    assert actual == pytest.approx(math.fsum(math.log1p(row["series_error_bound"]) for row in rows), rel=3e-15)
    old_only = math.fsum(math.log1p(row["original_taylor_series_error_bound"]) for row in rows)
    assert abs(actual-old_only) > 1e-18
    rows.clear()
    resumed, = observe_potential_geometry_checkpoint(snapshot, geometry, second, (1.03e-6,), **options)
    assert resumed.record["potential_geometry"]["series_error_log_bound"] == pytest.approx(
        actual+math.fsum(math.log1p(row["series_error_bound"]) for row in rows), rel=3e-15)
    # New per-residual diagnostics are optional to a stored checkpoint. Its
    # accepted accumulated bound is not recalculated under the new executor.
    legacy = thaw_json(second.record)
    def remove_new_diagnostics(value):
        if isinstance(value, dict):
            value.pop("declared_series_error_budget", None)
            value.pop("original_taylor_series_error_bound", None)
            for item in value.values():
                remove_new_diagnostics(item)
        elif isinstance(value, list):
            for item in value:
                remove_new_diagnostics(item)
    remove_new_diagnostics(legacy)
    legacy_resumed, = observe_potential_geometry_checkpoint(snapshot, geometry,
        replace(second, record=legacy), (1.03e-6,), **options)
    assert legacy_resumed.record["potential_geometry"] == resumed.record["potential_geometry"]
    np.testing.assert_array_equal(legacy_resumed.beam.modes[0].plane.amplitude, resumed.beam.modes[0].plane.amplitude)
    # The true weight includes floating-point flux changes; the executor does
    # not normalize them away. Compare against the unchanged original bounds.
    source_weight = checkpoint.beam.modes[0].weight_per_reference_electron
    assert abs(first.beam.modes[0].weight_per_reference_electron-source_weight) <= source_weight*original_flux_allowance


def test_unavailable_chebyshev_keeps_the_original_taylor_field_and_bound(monkeypatch):
    import temsim.physics.potential_wave_chebyshev as planner
    args, options = _periodic_residual(monkeypatch)
    original = args[1].amplitude.copy()
    expected, old = carrier._residual(*args, **options)
    def unavailable(*args, **kwargs):
        raise ChebyshevPlanUnavailable("controlled coefficient-rounding refusal")
    monkeypatch.setattr(planner, "plan_chebyshev_exponential", unavailable)
    output, record = carrier._residual(*args, **options, residual_exponential_method="chebyshev")
    np.testing.assert_array_equal(output.amplitude, expected.amplitude)
    np.testing.assert_array_equal(args[1].amplitude, original)
    assert record["series_error_bound"] == record["original_taylor_series_error_bound"] == old["series_error_bound"]
    assert record["declared_series_error_budget"] == args[7]
    assert record["residual_execution"]["exponential_method"] == "taylor"


@pytest.mark.parametrize("invalid", [float("nan"), -1., 1.])
def test_invalid_selected_certificate_cannot_replace_the_declared_allowance(monkeypatch, invalid):
    args, options = _periodic_residual(monkeypatch)
    execute = device.execute_residual_exponential
    def invalid_certificate(*args, **kwargs):
        output, evidence = execute(*args, **kwargs)
        evidence["chebyshev_certificate"]["series_error_bound"] = invalid
        return output, evidence
    monkeypatch.setattr(device, "execute_residual_exponential", invalid_certificate)
    with pytest.raises(ValueError, match="declared series allowance"):
        carrier._residual(*args, **options, residual_exponential_method="chebyshev")


def test_cancelled_chebyshev_does_not_publish_a_field_or_touch_the_original(monkeypatch):
    args, options = _periodic_residual(monkeypatch)
    original = args[1].amplitude.copy()
    check, calls = carrier._check_boundary, [0]
    def boundary(*values):
        calls[0] += 1
        return check(*values)
    monkeypatch.setattr(carrier, "_check_boundary", boundary)
    args = (*args[:-1], lambda: calls[0] >= 2)
    with pytest.raises(InterruptedError):
        carrier._residual(*args, **options, residual_exponential_method="chebyshev")
    assert calls[0] == 2
    np.testing.assert_array_equal(args[1].amplitude, original)


def test_real_gpu_declared_allowance_matches_cpu_actual_h8_field_and_certificate(monkeypatch):
    from temsim.physics.compute_backend import cupy_capability
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    args, options = _periodic_residual(monkeypatch)
    _, baseline = carrier._residual(*args, **options)
    changed = list(args)
    changed[3] = _K_DISTANCE*_NORM_BOUND/baseline["generator_norm_bound"]
    changed[7] = _DECLARED_ERROR
    cpu, a = carrier._residual(*changed, **options, residual_exponential_method="chebyshev")
    gpu, b = carrier._residual(*changed, **options, residual_exponential_method="chebyshev",
        residual_compute_backend="Require GPU")
    assert b["residual_execution"]["compute_backend"] == "cupy"
    assert b["residual_execution"]["exponential_method"] == "chebyshev"
    assert b["operator_applications"] == 7308 and b["actual_operator_applications"] == 1628
    assert np.linalg.norm(gpu.amplitude-cpu.amplitude) < 2e-12
    for key in ("series_error_bound", "original_taylor_series_error_bound", "declared_series_error_budget",
                "operator_applications", "actual_operator_applications", "roundoff_relative_allowance"):
        assert a[key] == b[key]
    assert b["series_error_bound"] <= _DECLARED_ERROR
    assert b["residual_execution"]["chebyshev_certificate"]["original_substep_checks"] == 406
