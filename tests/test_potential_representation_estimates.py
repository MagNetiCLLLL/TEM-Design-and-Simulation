"""Exploratory estimates preserve complex phase and the original allocation."""
import copy
from dataclasses import replace
import math
import tracemalloc

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.cache_memory import retained_memory_inventory
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave_support import NumericalSupportLedger
import temsim.physics.potential_representation_estimates as estimates
from temsim.physics.potential_representation_estimates import (
    RepresentationEstimateLedger, RepresentationBudgetError,
    estimate_carrier_representational_change, require_combined_support_allowance)
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, WaveGridBudgetError


WAVELENGTH = 2e-12
BINDING = {"source_checkpoint_digest": "a"*64, "source_mode_id": "energy:4",
           "support_allocation_digest": "b"*64}


def _wave(shape=(64, 64), *, sigma=4.):
    y, x = np.indices(shape, dtype=float)
    x -= shape[1]//2
    y -= shape[0]//2
    amplitude = np.exp(-(x*x+y*y)/(4*sigma*sigma))*(1+.13j*x/sigma)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.array([[1e-7, .2e-7], [-.1e-7, 1.3e-7]]),
                     np.array([.003, -.007]), np.array([[.2, .03], [.03, -.1]]),
                     np.array([2e-7, -1e-7]))


def _estimate(a, b, **kwargs):
    return estimate_carrier_representational_change(a, b, WAVELENGTH,
        **{"z_m": .541, "mode_weight": .27,
           "absolute_quadrature_relative_tolerance": 1e-7,
           "numerics": WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=64*1024**2), **kwargs})


def _piston_receipt(theta=.001):
    wave = _wave()
    changed = replace(wave, amplitude=wave.amplitude*np.exp(1j*theta))
    return _estimate(wave, changed)


@pytest.mark.parametrize("shape", [(64, 64), (63, 65), (63, 64), (64, 65)])
def test_identical_fields_and_parity_preserve_norm_and_explicit_estimate_scope(shape):
    wave = _wave(shape)
    before = wave.amplitude.tobytes()
    receipt = _estimate(wave, wave)
    assert receipt["maximum_probe_l2"] == 0.
    assert receipt["estimated_l2_per_reference"] == math.fsum((receipt["roundoff_floor_l2"],
        receipt["absolute_convergence_tolerance_l2"]))
    assert receipt["probability_loss_charged"] == 0.
    assert not receipt["certified"] and "not a certified" in receipt["error_scope"]
    assert not receipt["phase_alignment_applied"] and not receipt["normalization_applied"]
    assert wave.amplitude.tobytes() == before
    assert receipt["probes"][2]["shift_cells_xy"] == [.5, .5]
    assert all(all(p > n for p, n in zip(row["shape"], shape)) for row in receipt["probes"])


@pytest.mark.parametrize("theta", [.001, .25, 2.7])
def test_global_piston_is_measured_without_phase_alignment_or_norm_repair(theta):
    receipt = _piston_receipt(theta)
    expected = math.sqrt(.27)*2*abs(math.sin(theta/2))
    assert receipt["maximum_probe_l2"] == pytest.approx(expected, rel=2e-13)
    assert receipt["estimated_l2_per_reference"] >= receipt["maximum_probe_l2"]
    assert receipt["norm_roundoff_per_reference"] < 1e-15


def test_known_quadratic_tilt_rechart_keeps_complete_relative_carrier():
    wave = _wave(sigma=3.)
    coordinates = wave.coordinates_m()-wave.origin_m[:, None, None]
    delta_q = np.array([[.12, -.037], [-.037, .09]])
    delta_tilt = np.array([3e-7, -4e-7])
    action = (.5*np.einsum("iyx,ij,jyx->yx", coordinates, delta_q, coordinates)
              +np.einsum("i,iyx->yx", delta_tilt, coordinates))
    changed = replace(wave, amplitude=wave.amplitude*np.exp(-2j*np.pi*action/WAVELENGTH),
                      curvature_m1=wave.curvature_m1+delta_q, tilt_rad=wave.tilt_rad+delta_tilt)
    receipt = _estimate(wave, changed)
    assert receipt["estimated_l2_per_reference"] < 1e-7
    wrong = replace(changed, amplitude=wave.amplitude)
    uncompensated = _estimate(wave, wrong)
    assert uncompensated["maximum_probe_l2"] > .1


@pytest.mark.parametrize("shape", [(31, 33), (32, 32), (32, 33)])
def test_half_cell_fourier_probe_matches_direct_interpolation(shape):
    rng = np.random.default_rng(27)
    values = rng.normal(size=shape)+1j*rng.normal(size=shape)
    actual = estimates._shift_envelope(values, lambda: False)
    fy, fx = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    coefficient = np.fft.fft2(values)/(shape[0]*shape[1])
    expected = (np.exp(2j*np.pi*np.outer(np.arange(shape[0])+.5, fy))@coefficient
                @np.exp(2j*np.pi*np.outer(fx, np.arange(shape[1])+.5)))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=5e-14)


def test_roundtrip_immutable_history_binds_source_and_original_support_prefix():
    strict = NumericalSupportLedger(.02, 1e-6)
    empty = RepresentationEstimateLedger.for_support(strict, source_binding=BINDING)
    first = empty.append(_piston_receipt(), strict)
    second_receipt = _piston_receipt(.0003)
    second_receipt["z_m"] = .6
    second = first.append(second_receipt, strict)
    restored = RepresentationEstimateLedger.from_dict(second.to_dict())
    assert restored == second
    assert restored.require_prefix(first) is restored
    assert restored.require_source_binding(BINDING) is restored
    assert strict.events == () and empty.events == ()
    assert restored.used_l2_estimate == math.fsum(
        row["receipt"]["estimated_l2_per_reference"] for row in restored.events)
    assert require_combined_support_allowance(strict, restored) == strict.maximum_l2-restored.used_l2_estimate
    with pytest.raises(TypeError):
        restored.events[0]["receipt"]["mode_weight"] = 3.
    detached = restored.to_dict()
    detached["events"].clear()
    assert len(restored.events) == 2


def test_strict_events_can_continue_without_reinterpreting_prior_history():
    wave = _wave((128, 128))
    strict = NumericalSupportLedger(.02, 1e-6)
    _, accepted = strict.compact(wave, wavelength_m=WAVELENGTH, z_m=.5, mode_weight=.27)
    assert accepted.events
    history = RepresentationEstimateLedger.for_support(accepted, source_binding=BINDING)
    history = history.append(_piston_receipt(), accepted)
    _, downstream = accepted.compact(wave, wavelength_m=WAVELENGTH, z_m=.55, mode_weight=.27)
    assert downstream.events[:len(accepted.events)] == accepted.events
    remaining = require_combined_support_allowance(downstream, history)
    assert remaining == downstream.maximum_l2-math.fsum((downstream.used_l2, history.used_l2_estimate))
    with pytest.raises(ValueError, match="missing strict|prefix"):
        require_combined_support_allowance(strict, history)
    altered = replace(accepted, events=({**dict(accepted.events[0]), "z_m": .499},))
    with pytest.raises(ValueError, match="prefix"):
        require_combined_support_allowance(altered, history)


def test_l2_estimate_reservation_never_issues_new_probability_or_l2_budget():
    strict = NumericalSupportLedger(.001, 1e-6)
    empty = RepresentationEstimateLedger.for_support(strict)
    receipt = _piston_receipt(.002)
    with pytest.raises(RepresentationBudgetError):
        empty.append(receipt, strict)
    assert empty.events == () and strict.used_probability == 0.
    with pytest.raises(ValueError, match="allocation"):
        require_combined_support_allowance(replace(strict, maximum_l2=.1), empty)


@pytest.mark.parametrize("mutation", [
    lambda record: record.update(certified=True),
    lambda record: record.update(schema="other"),
    lambda record: record.update(strict_anchor_digest="x"*64),
    lambda record: record["events"][0].update(event_index=2),
    lambda record: record["events"][0].update(previous_estimates_digest="0"*64),
    lambda record: record["events"][0]["receipt"].update(estimated_l2_per_reference=0.),
    lambda record: record["events"][0]["receipt"].update(probability_loss_charged=1e-9),
    lambda record: record["events"][0]["receipt"].update(phase_alignment_applied=True),
    lambda record: record["events"][0]["receipt"]["probes"][2].update(shift_cells_xy=[0., 0.]),
    lambda record: record["events"][0]["receipt"].update(mode_weight=True),
])
def test_tampered_histories_are_rejected(mutation):
    strict = NumericalSupportLedger(.02, 1e-6)
    history = RepresentationEstimateLedger.for_support(strict).append(_piston_receipt(), strict)
    record = history.to_dict()
    mutation(record)
    with pytest.raises(ValueError):
        RepresentationEstimateLedger.from_dict(record)


def test_wrong_source_or_shortened_accepted_estimate_prefix_is_rejected():
    strict = NumericalSupportLedger(.02, 1e-6)
    empty = RepresentationEstimateLedger.for_support(strict, source_binding=BINDING)
    history = empty.append(_piston_receipt(), strict)
    with pytest.raises(ValueError, match="source"):
        history.require_source_binding({**BINDING, "source_mode_id": "energy:5"})
    with pytest.raises(ValueError, match="prefix"):
        empty.require_prefix(history)
    with pytest.raises(ValueError, match="source"):
        history.require_source_binding(None)


@pytest.mark.parametrize("change", ["origin", "period", "norm", "wavelength", "backend"])
def test_invalid_physics_is_not_disguised_as_optional_numerical_decline(change):
    wave = _wave()
    changed, kwargs = wave, {}
    if change == "origin":
        changed = replace(wave, origin_m=wave.origin_m+[1e-10, 0.])
    elif change == "period":
        changed = replace(wave, basis_m=wave.basis_m*1.001)
    elif change == "norm":
        changed = replace(wave, amplitude=wave.amplitude*.99)
    elif change == "wavelength":
        with pytest.raises(ValueError) as error:
            estimate_carrier_representational_change(wave, wave, 0., z_m=.5, mode_weight=.27,
                                                     numerics=WaveGridNumerics())
        assert not isinstance(error.value, WaveSamplingError)
        return
    else:
        kwargs["numerics"] = WaveGridNumerics(compute_backend="gpu")
    with pytest.raises(ValueError) as error:
        _estimate(wave, changed, **kwargs)
    assert not isinstance(error.value, (WaveSamplingError, WaveGridBudgetError))


def test_insufficient_probe_ceiling_and_memory_decline_before_fft(monkeypatch):
    wave = _wave()
    called = []
    monkeypatch.setattr(estimates, "refine_plane_wave", lambda *args, **kwargs: called.append(True))
    with pytest.raises(WaveSamplingError, match="two strictly denser"):
        _estimate(wave, wave, numerics=WaveGridNumerics(maximum_pixels=64))
    with pytest.raises(WaveGridBudgetError):
        _estimate(wave, wave, numerics=WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=1000))
    assert called == []


def test_unconverged_probe_is_declined_without_repair(monkeypatch):
    wave = _wave()
    values = iter([.01, .02, .03])
    monkeypatch.setattr(estimates, "_difference", lambda *args: next(values))
    with pytest.raises(WaveSamplingError, match="locally converge"):
        _estimate(wave, wave)


@pytest.mark.parametrize("stop", [1, 10, 30])
def test_cancellation_never_returns_receipt_or_mutates_sources(stop):
    wave = _wave()
    before = wave.amplitude.tobytes()
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= stop
    with pytest.raises(InterruptedError):
        _estimate(wave, wave, cancelled=cancelled)
    assert wave.amplitude.tobytes() == before


def test_probe_preflight_covers_allocations_and_unique_source_owners(monkeypatch):
    wave = _wave((128, 128))
    changed = replace(wave, amplitude=wave.amplitude*np.exp(.001j))
    numerics = WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=64*1024**2)
    records = []
    original_check = WaveGridNumerics.check_working_memory
    def check(self, shape, *, working_bytes, retained_bytes=0):
        records.append((shape, working_bytes, retained_bytes))
        return original_check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", check)
    tracemalloc.start()
    receipt = _estimate(wave, changed, numerics=numerics, retained_bytes=12345)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert receipt
    assert records[0][2] == 12345+sum(retained_memory_inventory(wave.amplitude, changed.amplitude).values())
    assert peak <= max(work for _, work, _ in records)


def test_actual_preparation_ceiling_allows_independent_3300_probes():
    first, second = estimates._probe_shapes((3300, 3300), 4096)
    assert first == (3696, 3696) and second == (4096, 4096)


def test_native_plane_can_be_negative_without_changing_forward_history_order():
    wave = _wave()
    receipt = _estimate(wave, wave, z_m=-.5)
    strict = NumericalSupportLedger(.02, 1e-6)
    history = RepresentationEstimateLedger.for_support(strict).append(receipt, strict)
    assert history.events[0]["receipt"]["z_m"] == -.5
    receipt["z_m"] = -.6
    with pytest.raises(ValueError, match="backwards"):
        history.append(receipt, strict)


def test_nonfinite_phase_difference_cannot_be_reported_as_zero_cost():
    wave = _wave()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(WaveSamplingError, match="finite numerical range"):
            estimates._difference(wave.amplitude, wave.amplitude, wave.basis_m,
                np.full((2, 2), 1e308), np.zeros(2), 1e-100, 0., lambda: False)


def test_mixed_absolute_criterion_is_fully_charged_even_for_near_zero_change(monkeypatch):
    wave = _wave()
    # Relative convergence fails, but every observation is below the declared
    # small absolute tolerance. This is an estimated preview, not a bound.
    values = iter([1e-8, 2e-8, 4e-8])
    monkeypatch.setattr(estimates, "_difference", lambda *args: next(values))
    receipt = _estimate(wave, wave)
    absolute = 1e-7*math.sqrt(.27)*math.sqrt(wave.probability)
    assert receipt["schema"].endswith("v2")
    assert receipt["probe_spread_l2"] > .05*receipt["maximum_probe_l2"]
    assert receipt["absolute_convergence_tolerance_l2"] == absolute
    assert receipt["estimated_l2_per_reference"] == math.fsum((receipt["maximum_probe_l2"],
        receipt["probe_spread_l2"], receipt["roundoff_floor_l2"], absolute))
    strict = NumericalSupportLedger(absolute, 1e-6)
    with pytest.raises(RepresentationBudgetError):
        RepresentationEstimateLedger.for_support(strict).append(receipt, strict)


@pytest.mark.parametrize("zero", ["weight", "probability"])
def test_zero_reference_amplitude_has_no_invented_estimated_loss(zero):
    wave = _wave()
    kwargs = {}
    if zero == "weight":
        kwargs["mode_weight"] = 0.
    else:
        wave = replace(wave, amplitude=np.zeros_like(wave.amplitude))
    receipt = _estimate(wave, wave, **kwargs)
    assert receipt["estimated_l2_per_reference"] == 0.
    assert receipt["absolute_convergence_tolerance_l2"] == 0.
    assert receipt["probability_loss_charged"] == 0.


@pytest.mark.parametrize("field", ["absolute_tolerance_relative_amplitude", "absolute_convergence_tolerance_l2",
                                    "estimated_l2_per_reference"])
def test_absolute_convergence_tolerance_and_charge_cannot_be_tampered(field):
    strict = NumericalSupportLedger(.02, 1e-6)
    empty = RepresentationEstimateLedger.for_support(strict)
    receipt = _piston_receipt()
    receipt[field] *= .5
    with pytest.raises(ValueError):
        empty.append(receipt, strict)


def test_legacy_v1_receipt_is_replayed_without_new_tolerance_or_repriced_cost():
    strict = NumericalSupportLedger(.02, 1e-6)
    empty = RepresentationEstimateLedger.for_support(strict)
    legacy = _piston_receipt()
    legacy["schema"] = "common-si-carrier-representation-estimate-v1"
    legacy.pop("absolute_tolerance_relative_amplitude")
    legacy.pop("absolute_convergence_tolerance_l2")
    legacy["estimated_l2_per_reference"] = math.fsum((legacy["maximum_probe_l2"],
        legacy["probe_spread_l2"], legacy["roundoff_floor_l2"]))
    first = empty.append(legacy, strict)
    prefix = first.to_dict()
    second = first.append(_piston_receipt(), strict)
    assert second.require_prefix(first) is second
    assert second.events[0]["receipt"] == first.events[0]["receipt"]
    assert RepresentationEstimateLedger.from_dict(prefix).to_dict() == prefix
    assert second.used_l2_estimate == math.fsum((legacy["estimated_l2_per_reference"],
        second.events[1]["receipt"]["estimated_l2_per_reference"]))


def test_v1_cannot_borrow_new_absolute_convergence_permission(monkeypatch):
    wave = _wave()
    values = iter([1e-8, 2e-8, 4e-8])
    monkeypatch.setattr(estimates, "_difference", lambda *args: next(values))
    receipt = _estimate(wave, wave)
    receipt["schema"] = "common-si-carrier-representation-estimate-v1"
    receipt.pop("absolute_tolerance_relative_amplitude")
    receipt.pop("absolute_convergence_tolerance_l2")
    receipt["estimated_l2_per_reference"] = math.fsum((receipt["maximum_probe_l2"],
        receipt["probe_spread_l2"], receipt["roundoff_floor_l2"]))
    with pytest.raises(ValueError, match="convergence"):
        RepresentationEstimateLedger.for_support(NumericalSupportLedger(.02, 1e-6)).append(
            receipt, NumericalSupportLedger(.02, 1e-6))


def test_default_estimator_keeps_v1_semantics_and_preview_requires_explicit_rule():
    wave = _wave()
    receipt = _estimate(wave, wave, absolute_quadrature_relative_tolerance=0.)
    assert receipt["schema"].endswith("v1")
    assert "absolute_convergence_tolerance_l2" not in receipt
    assert receipt["estimated_l2_per_reference"] == receipt["roundoff_floor_l2"]
    for invalid in (True, 1e-5, -.01, float("nan")):
        with pytest.raises(ValueError):
            _estimate(wave, wave, absolute_quadrature_relative_tolerance=invalid)


def test_actual_broad_tail_rechart_still_fails_unresolved_mixed_quadrature():
    wave = _wave(sigma=8.)
    coordinates = wave.coordinates_m()-wave.origin_m[:, None, None]
    delta_q = np.array([[.12, -.037], [-.037, .09]])
    delta_tilt = np.array([3e-7, -4e-7])
    action = (.5*np.einsum("iyx,ij,jyx->yx", coordinates, delta_q, coordinates)
              +np.einsum("i,iyx->yx", delta_tilt, coordinates))
    changed = replace(wave, amplitude=wave.amplitude*np.exp(-2j*np.pi*action/WAVELENGTH),
                      curvature_m1=wave.curvature_m1+delta_q, tilt_rad=wave.tilt_rad+delta_tilt)
    before = wave.amplitude.tobytes()
    with pytest.raises(WaveSamplingError, match="locally converge"):
        _estimate(wave, changed)
    assert wave.amplitude.tobytes() == before


def test_zero_weight_absolute_tolerance_requires_numeric_zero_not_boolean():
    wave = _wave()
    receipt = _estimate(wave, wave, mode_weight=0.)
    receipt["absolute_convergence_tolerance_l2"] = False
    strict = NumericalSupportLedger(.02, 1e-6)
    with pytest.raises(ValueError, match="Absolute representation tolerance"):
        RepresentationEstimateLedger.for_support(strict).append(receipt, strict)
