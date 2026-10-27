"""Remaining spatial allocations are explicit, bounded input transactions."""
from dataclasses import replace
import copy
import math
import weakref

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.physics.potential_wave_support import (
    NumericalSupportLedger, SPATIAL_PRESSURE_SHARES,
)
from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError, WaveSamplingError
from test_potential_wave_support import _wave, _edge_wave, _ledger
from test_potential_weighted_resampling import _pressure_context, _pressure_ledger


def _context(wave, numerics=None):
    result = _pressure_context(wave, numerics)
    result["candidate_family"] = "centered_quintic_crop_remaining_share_v1"
    return result


def _crop(ledger=None, wave=None, share=1., **options):
    wave = _wave() if wave is None else wave
    ledger = NumericalSupportLedger(.02, 1e-6) if ledger is None else ledger
    return ledger.compact_pressure(wave, wavelength_m=2e-12, z_m=.503, mode_weight=.25,
        allocation_fraction=share, resource_context=_context(wave, options.get("numerics")), **options)


@pytest.mark.parametrize("shape", [(128, 128), (127, 129)])
@pytest.mark.parametrize("share", SPATIAL_PRESSURE_SHARES)
def test_spatial_remaining_receipt_matches_direct_complex_difference_and_original_weight(shape, share):
    wave, original = _wave(shape), NumericalSupportLedger(.02, 1e-6)
    output, pending = _crop(original, wave, share)
    assert output is not wave and len(pending.events) == 1 and original.events == ()
    event = pending.events[0]
    (y0, y1), (x0, x1) = event["crop_indices_yx"]
    embedded = np.zeros_like(wave.amplitude)
    embedded[y0:y1, x0:x1] = output.amplitude
    assert math.sqrt(.25)*np.linalg.norm(embedded-wave.amplitude) == pytest.approx(pending.used_l2, rel=3e-14)
    assert .25*(wave.probability-output.probability) == pytest.approx(pending.used_probability, abs=2e-16)
    assert pending.used_probability <= share*original.maximum_probability
    assert pending.used_l2 <= share*original.maximum_l2
    assert event["pressure_allocation"]["previous_ledger_digest"] == json_digest(original.to_dict())
    assert event["pressure_allocation"]["allocation_fraction"] == share
    assert not event["physical_absorption"] and not event["probability_repair"]
    assert pending.to_dict()["schema"].endswith("v6")
    assert NumericalSupportLedger.from_dict(pending.to_dict()) == pending
    np.testing.assert_array_equal(output.coordinates_m(), wave.coordinates_m()[:, y0:y1, x0:x1])
    for key in ("origin_m", "basis_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, key), getattr(wave, key))
    (a, b), (c, d) = event["core_indices_yx"]
    np.testing.assert_array_equal(output.amplitude[a-y0:b-y0+1, c-x0:d-x0+1], wave.amplitude[a:b+1, c:d+1])
    if share == 1.:
        assert pending.used_probability > original.maximum_probability/100.


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_old_prefix_and_pressure_digests_remain_exact_across_v6_forward_extension(version):
    if version == 1:
        _, _, old = _ledger()
    elif version in (2, 5):
        _, old = NumericalSupportLedger(.02, 1e-6).taper_boundary(_edge_wave(tail=1e-9),
            z_m=.4, mode_weight=.25, transition_cells=2 if version == 2 else 8)
    elif version == 3:
        from test_potential_weighted_resampling import _ledger as spectral
        _, old = spectral()
    else:
        _, _, old = _pressure_ledger()
    previous = copy.deepcopy(old.to_dict())
    assert previous["schema"].endswith(f"v{version}")
    _, pending = _crop(old)
    assert pending.to_dict()["events"][:-1] == previous["events"]
    assert pending.events[-1]["pressure_allocation"]["previous_ledger_digest"] == json_digest(previous)
    assert NumericalSupportLedger.from_dict(previous).to_dict() == previous
    assert NumericalSupportLedger.from_dict(pending.to_dict()) == pending
    if version != 1:
        _, _, later = _pressure_ledger(pending)
        assert later.events[-1]["pressure_allocation"]["previous_ledger_digest"] == json_digest(pending.to_dict())
        assert later.events[:-1] == pending.events
        assert NumericalSupportLedger.from_dict(later.to_dict()) == later
    assert old.to_dict() == previous


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema="controlled-numerical-support-ledger-v5"),
    lambda d: d.update(maximum_probability=d["maximum_probability"]*2),
    lambda d: d["events"].append(copy.deepcopy(d["events"][-1])),
    lambda d: d["events"][-1].update(mode_weight=1.),
    lambda d: d["events"][-1].update(transition_cells=8),
    lambda d: d["events"][-1]["core_indices_yx"][0].__setitem__(0, 0),
    lambda d: d["events"][-1]["pressure_allocation"].update(schema="remaining-support-pressure-allocation-v1"),
    lambda d: d["events"][-1]["pressure_allocation"].update(previous_ledger_digest="0"*64),
    lambda d: d["events"][-1]["pressure_allocation"].update(remaining_probability_per_reference=1.),
    lambda d: d["events"][-1]["pressure_allocation"].update(remaining_l2_per_reference=1.),
    lambda d: d["events"][-1]["pressure_allocation"].update(allocation_fraction=.01),
    lambda d: d["events"][-1]["pressure_allocation"].update(allocation_fraction=True),
    lambda d: d["events"][-1]["pressure_allocation"]["resource_context"].update(candidate_family="fixed_aspect_even_fft_friendly_v1"),
    lambda d: d["events"][-1]["pressure_allocation"]["resource_context"].update(growth_guard_scale=2.),
    lambda d: d["events"][-1]["pressure_allocation"]["resource_context"].update(previous_shape=[True, 60]),
    lambda d: d["events"][-1]["pressure_allocation"]["resource_context"].update(input_shape=[64, 64]),
])
def test_v6_rejects_reinterpreted_history_reset_share_domain_or_duplicate_spending(mutation):
    _, ledger = _crop()
    record = ledger.to_dict()
    mutation(record)
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(record)


@pytest.mark.parametrize("share", [True, 0., -.1, .03, 2., float("nan")])
def test_only_fixed_positive_shares_can_allocate_remaining_errors(share):
    with pytest.raises(ValueError):
        _crop(share=share)


def test_pressure_api_resource_cancel_underflow_and_default_compact_remain_explicit(monkeypatch):
    import temsim.physics.potential_wave_support as module
    wave, ledger = _wave(), NumericalSupportLedger(.02, 1e-6)
    a, ordinary = ledger.compact(wave, wavelength_m=2e-12, z_m=.503, mode_weight=.25)
    assert ordinary.to_dict()["schema"].endswith("v1")
    assert ordinary.used_probability <= ledger.maximum_probability/100.
    assert ordinary.used_l2 <= ledger.maximum_l2/100.
    context = _context(wave)
    context["maximum_pixels"] += 1
    with pytest.raises(ValueError, match="actual numerical"):
        ledger.compact_pressure(wave, wavelength_m=2e-12, z_m=0., allocation_fraction=1., resource_context=context)
    with pytest.raises(InterruptedError):
        _crop(cancelled=lambda: True)
    with pytest.raises(WaveGridBudgetError):
        _crop(numerics=WaveGridNumerics(maximum_working_bytes=1))
    output, same = _crop(wave=replace(wave, amplitude=wave.amplitude*1e-160))
    assert same.events == () and output.amplitude.shape == wave.amplitude.shape
    original = module.compact_controlled_support
    cancelled = [False]
    def compute(*args, **options):
        result = original(*args, **options)
        cancelled[0] = True
        return result
    monkeypatch.setattr(module, "compact_controlled_support", compute)
    with pytest.raises(InterruptedError):
        _crop(ledger, cancelled=lambda: cancelled[0])
    assert ledger.events == ()


@pytest.mark.parametrize("outcome", ["success", "cancel", "domain", "physical_failure"])
def test_spatial_search_retries_same_original_after_alias_growth_and_publishes_only_complete_operator(monkeypatch, outcome):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.wave_grid import WaveResourceTransaction
    from test_wave_resource_preparation import _narrow_source
    from test_potential_wave_carrier import _snapshot, _round, _multipole, ENERGY_EV
    source, original = _narrow_source(), NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    real_crop, real_apply, real_validate = NumericalSupportLedger.compact_pressure, carrier.apply_resolved_operator, carrier._validate_wave
    real_residual, real_proposal = carrier._residual, carrier.WaveResourceProposal
    state = {"shares": [], "raw": None, "failed": [], "expanded": False, "commits": 0,
             "residuals": 0, "accepted": None, "entrance": None}
    def crop(self, wave, **options):
        assert self is original and wave is source and all(ref() is None for ref in state["failed"])
        state["shares"].append(options["allocation_fraction"])
        assert options["resource_context"]["growth_guard_scale"] == 1.
        assert options["resource_context"]["previous_shape"] == [96, 96]
        output, pending = real_crop(self, wave, **options)
        assert output is not wave and output.amplitude.shape[0] < 96
        state["raw"] = weakref.ref(output.amplitude)
        if len(state["shares"]) == 1:
            state["failed"].append(state["raw"])
        state["accepted"] = pending
        return output, pending
    def validate(wave, *args):
        if state["raw"] is not None and state["raw"]() is wave.amplitude:
            if len(state["shares"]) == 1:
                raise WaveSamplingError("controlled first crop needs lossless admission beyond prior shape", 2.)
            if outcome == "cancel":
                raise InterruptedError("controlled next-window cancellation")
            if outcome == "domain":
                raise ValueError("controlled cropped field domain failure")
        if len(state["shares"]) == 1 and wave.amplitude.shape[0] > 96:
            state["expanded"] = True
        return real_validate(wave, *args)
    def apply(wave, operator, **options):
        if getattr(options.get("operator_working_bytes"), "__name__", "") == "residual_host_working_bytes":
            prepare = options["prepare_resource"]
            options["prepare_resource"] = WaveResourceTransaction(lambda candidate, error, pressure:
                prepare(candidate, error, replace(pressure, previous_shape=(96, 96))))
        return real_apply(wave, operator, **options)
    def residual(snapshot, wave, *args, **options):
        state["residuals"] += 1
        if state["residuals"] == 1:
            raise carrier.ScalarPhaseSamplingError("controlled original resource refusal", 2.)
        if state["residuals"] == 2:
            assert state["commits"] == 0
            state["entrance"] = wave
        if outcome == "physical_failure":
            raise ValueError("controlled physical failure after input admission")
        return real_residual(snapshot, wave, *args, **options)
    def proposal(wave, commit):
        def publish():
            state["commits"] += 1
            commit()
        return real_proposal(wave, publish)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", crop)
    monkeypatch.setattr(carrier, "apply_resolved_operator", apply)
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(carrier, "WaveResourceProposal", proposal)
    options = dict(energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric,
        support_ledger=original, support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry")
    snapshot = _snapshot(_round(), _multipole())
    if outcome == "success":
        output, record = carrier.propagate_carrier_potential(snapshot, source, (0., 1e-9), **options)
        accepted = NumericalSupportLedger.from_dict(record["support_ledger"])
        assert accepted == state["accepted"] and state["commits"] == 1
        assert accepted.to_dict()["schema"].endswith("v6")
        assert len(accepted.events) == 1 and accepted.events[0]["pressure_allocation"]["allocation_fraction"] == .02
        assert state["entrance"].amplitude.shape[0] < 96
        assert any("after lossless refinement" in row.get("reason", "") for row in record["canonical_sampling"])
        monkeypatch.setattr(carrier, "apply_resolved_operator", real_apply)
        monkeypatch.setattr(carrier, "_validate_wave", real_validate)
        monkeypatch.setattr(carrier, "_residual", real_residual)
        expected, _ = carrier.propagate_carrier_potential(snapshot, state["entrance"], (0., 1e-9),
            energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric)
        np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=0., atol=2e-14)
        assert output.probability == pytest.approx(source.probability-accepted.used_probability/.25, abs=2e-14)
    else:
        with pytest.raises(InterruptedError if outcome == "cancel" else ValueError):
            carrier.propagate_carrier_potential(snapshot, source, (0., 1e-9), **options)
        assert state["commits"] == 0
    assert state["shares"] == [.01, .02] and state["expanded"]
    assert original.events == () and all(ref() is None for ref in state["failed"])
