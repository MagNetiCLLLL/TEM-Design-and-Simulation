"""Whole optical-step resource retries preserve the original physical history."""
from dataclasses import replace
import weakref

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint, _bounded_upper
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.potential_wave_carrier import ScalarPhaseSamplingError
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_flux import BeamState
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, WaveGridBudgetError
from test_retarded_wave import _source, _mode
from test_potential_geometry import _snapshot, _checkpoint, _geometry


def _pressure(*, scalar=True):
    error = WaveGridBudgetError("controlled unresolved whole-step sampling resource pressure")
    error.__cause__ = (ScalarPhaseSamplingError if scalar else WaveSamplingError)("controlled phase gate", 3.)
    return error


def _dynamic_case():
    snapshot, mode = _source(offset=1e-4), _mode()
    start, stop = .45, .45002
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), start*1000., 1e-12, {})
    geometry = PotentialGeometrySnapshot(start, .45004, ())
    options = dict(maximum_step_m=stop-start, emission_time_s=.1, kinetic_order=2,
        reference_quadrature="cf4", support_projection_policy="weighted_segment_and_retry",
        support_ledger=NumericalSupportLedger(.01, 1e-7), kinetic_phase_ledger=KineticPhaseLedger(.01),
        grid_numerics=WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=128*1024**2))
    return snapshot, geometry, checkpoint, start, stop, options


def test_whole_step_retry_matches_two_complete_halves_and_preserves_all_histories(monkeypatch):
    import temsim.physics.potential_wave_observation as observation
    import temsim.physics.potential_geometry as geometry_module
    snapshot, geometry, checkpoint, start, stop, options = _dynamic_case()
    middle = _bounded_upper(start, stop, (stop-start)/2.)
    original, original_mask = observation.observe_potential_mode, geometry_module._mask_checkpoint
    calls, failed_arrays, masks = [], [], []
    original_amplitude = checkpoint.beam.modes[0].plane.amplitude.copy()
    original_record = thaw_json(checkpoint.record)

    def fail_after_complete_optical_step(source, mode, lower, targets, **kwargs):
        assert all(value() is None for value in failed_arrays)
        calls.append((mode, lower, targets[0], kwargs["support_ledger"], kwargs["kinetic_phase_ledger"],
            kwargs.get("electric_wave_history"), kwargs["exponential_error"], kwargs["kinetic_remainder_phase_limit"]))
        output = original(source, mode, lower, targets, **kwargs)
        if len(calls) == 1:
            # Simulate a late scalar resource refusal after full private
            # CF4/residual/retarded execution, so all provisional history exists.
            failed_arrays.append(weakref.ref(output[0].mode.plane.amplitude))
            raise _pressure()
        return output

    def mask(value, captured, lower, upper, **kwargs):
        masks.append((lower, upper))
        return original_mask(value, captured, lower, upper, **kwargs)

    monkeypatch.setattr(observation, "observe_potential_mode", fail_after_complete_optical_step)
    monkeypatch.setattr(geometry_module, "_mask_checkpoint", mask)
    actual, = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (stop,), **options)
    assert [(row[1], row[2]) for row in calls] == [(start, stop), (start, middle), (middle, stop)]
    assert calls[0][0] is calls[1][0]
    assert calls[0][3:6] == calls[1][3:6]
    assert calls[1][6] == calls[0][6]*(middle-start)/(stop-start)
    assert calls[1][7] == calls[0][7]*(middle-start)/(stop-start)
    assert (start, stop) not in masks
    assert all(value() is None for value in failed_arrays)
    monkeypatch.setattr(observation, "observe_potential_mode", original)
    _, expected = observe_potential_geometry_checkpoint(snapshot, geometry, checkpoint, (middle, stop), **options)
    left, right = actual.beam.modes[0], expected.beam.modes[0]
    for field in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(left.plane, field), getattr(right.plane, field))
    assert left.weight_per_reference_electron == right.weight_per_reference_electron
    assert left.energy_kev == right.energy_kev and left.axial_reference == right.axial_reference
    for key in ("support_ledger", "kinetic_phase_ledger", "electric_wave_history"):
        assert actual.record[key] == expected.record[key]
    clock = actual.record["electric_wave_history"]["dynamic_state"]
    assert clock["emission_time_s"] == .1 and clock["axial_work_ev"] != 0.
    assert clock["flight_time_s"] == left.axial_reference.flight_time_s
    details = actual.record["potential_geometry"]
    assert details["accepted_steps"] == expected.record["potential_geometry"]["accepted_steps"] == 2
    assert details["physical_losses"] == expected.record["potential_geometry"]["physical_losses"]
    assert details["series_error_log_bound"] == expected.record["potential_geometry"]["series_error_log_bound"]
    assert details["kinetic_phase_bound"] == expected.record["potential_geometry"]["kinetic_phase_bound"]
    retries = details["axial_resource_retries"]
    assert len(retries) == 1 and retries[0]["failed_upper_z_m"] == stop
    assert retries[0]["retry_upper_z_m"] == middle and retries[0]["failed_candidate_committed"] is False
    np.testing.assert_array_equal(checkpoint.beam.modes[0].plane.amplitude, original_amplitude)
    assert thaw_json(checkpoint.record) == original_record
    continuation = {key: value for key, value in options.items() if key not in ("support_ledger", "kinetic_phase_ledger")}
    resumed, = observe_potential_geometry_checkpoint(snapshot, geometry, actual, (.45003,), **continuation)
    assert resumed.record["potential_geometry"]["axial_resource_retries"] == retries
    for key in ("support_ledger", "kinetic_phase_ledger", "electric_wave_history"):
        assert resumed.record[key]["events"][:len(actual.record[key]["events"])] == actual.record[key]["events"]
    # Old v1 geometry records did not carry this optional diagnostic list.
    legacy_record = thaw_json(actual.record)
    legacy_record["potential_geometry"].pop("axial_resource_retries")
    legacy, = observe_potential_geometry_checkpoint(snapshot, geometry,
        replace(actual, record=legacy_record), (.45003,), **continuation)
    assert legacy.record["potential_geometry"]["axial_resource_retries"] == ()
    np.testing.assert_array_equal(legacy.beam.modes[0].plane.amplitude, resumed.beam.modes[0].plane.amplitude)


@pytest.mark.parametrize("case", ["ordinary_budget", "nonweighted", "automatic_disabled", "domain",
    "cancelled", "cancel_at_resource_failure"])
def test_whole_step_retry_is_limited_to_typed_weighted_scalar_pressure(monkeypatch, case):
    import temsim.physics.potential_wave_observation as observation
    calls, cancelled = [], [False]
    initial = _checkpoint()
    options = dict(maximum_step_m=1e-6, support_ledger=NumericalSupportLedger(.01,1e-7),
        support_projection_policy="weighted_segment_and_retry",
        grid_numerics=WaveGridNumerics(automatic_refinement=case != "automatic_disabled"),
        cancelled=lambda: cancelled[0])
    if case == "nonweighted":
        options["support_projection_policy"] = "sampling_retry"
    error = (_pressure(scalar=case != "ordinary_budget") if case not in ("domain", "cancelled")
        else ValueError("controlled physical domain failure") if case == "domain"
        else InterruptedError("controlled optical cancellation"))
    def fail(*args, **kwargs):
        calls.append(args)
        if case == "cancel_at_resource_failure":
            cancelled[0] = True
        raise error
    monkeypatch.setattr(observation, "observe_potential_mode", fail)
    expected = InterruptedError if case == "cancel_at_resource_failure" else type(error)
    with pytest.raises(expected):
        observe_potential_geometry_checkpoint(_snapshot(), _geometry(), initial, (1.000001,), **options)
    assert len(calls) == 1 and options["support_ledger"].events == ()
    assert "potential_geometry" not in initial.record


@pytest.mark.parametrize("limit", ["representable", "attempts"])
def test_whole_step_bisection_is_bounded_without_publishing_failed_history(monkeypatch, limit):
    import temsim.physics.potential_wave_observation as observation
    start = 1.
    stop = float(np.nextafter(start, np.inf)) if limit == "representable" else 1.000004
    initial = _checkpoint()
    calls, errors = [], []
    def fail(snapshot, mode, lower, targets, **kwargs):
        calls.append((mode, lower, targets[0], kwargs["support_ledger"]))
        error = _pressure()
        errors.append(error)
        raise error
    monkeypatch.setattr(observation, "observe_potential_mode", fail)
    ledger = NumericalSupportLedger(.01,1e-7)
    with pytest.raises(WaveGridBudgetError) as caught:
        observe_potential_geometry_checkpoint(_snapshot(), _geometry(), initial, (stop,),
            maximum_step_m=stop-start, support_ledger=ledger,
            support_projection_policy="weighted_segment_and_retry")
    assert len(calls) == (1 if limit == "representable" else 12)
    assert all(row[0] is calls[0][0] and row[1] == start and row[3] is ledger for row in calls)
    assert all(a[2] > b[2] > start for a,b in zip(calls,calls[1:]))
    assert isinstance(caught.value.__cause__, ScalarPhaseSamplingError)
    assert ("cannot be bisected" if limit == "representable" else "12 bounded") in caught.value.__notes__[-1]
    assert ledger.events == () and "potential_geometry" not in initial.record


def test_cancellation_after_successful_refined_candidate_prevents_step_commit(monkeypatch):
    import temsim.physics.potential_wave_observation as observation
    import temsim.physics.potential_geometry as geometry_module
    original, original_limit = observation.observe_potential_mode, geometry_module._geometry_step_limit
    state = {"calls":0, "cancelled":False}
    def observe(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise _pressure()
        return original(*args, **kwargs)
    def limit(*args, include_zero=False, **kwargs):
        value = original_limit(*args, include_zero=include_zero, **kwargs)
        if include_zero:
            state["cancelled"] = True
        return value
    monkeypatch.setattr(observation,"observe_potential_mode",observe)
    monkeypatch.setattr(geometry_module,"_geometry_step_limit",limit)
    initial, ledger = _checkpoint(), NumericalSupportLedger(.01,1e-7)
    with pytest.raises(InterruptedError,match="before step commit"):
        observe_potential_geometry_checkpoint(_snapshot(),_geometry(),initial,(1.000001,),
            maximum_step_m=1e-6, support_ledger=ledger,
            support_projection_policy="weighted_segment_and_retry",cancelled=lambda:state["cancelled"])
    assert state["calls"] == 2 and ledger.events == ()
    assert "potential_geometry" not in initial.record
