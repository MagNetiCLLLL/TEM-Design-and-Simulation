"""Versioned boundary windows preserve old receipts and pending transactions."""
from dataclasses import replace
import copy
import math
import weakref

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, WaveGridBudgetError
from test_potential_wave_support import _edge_wave, _ledger
from test_potential_weighted_resampling import _pressure_ledger


@pytest.mark.parametrize("width", [2, 4, 8, 16])
@pytest.mark.parametrize("shape", [(64, 67), (65, 66)])
def test_declared_quintic_window_preserves_phase_and_direct_original_weight_cost(width, shape):
    wave = _edge_wave(shape)
    original = wave.amplitude.tobytes()
    ledger = NumericalSupportLedger(.01, 1e-6)
    output, pending = ledger.taper_boundary(wave, z_m=.5, mode_weight=.25, transition_cells=width)
    axes = []
    for size in shape:
        distance = np.minimum(np.arange(size), np.arange(size)[::-1])
        t = np.minimum(distance/width, 1.)
        axes.append(6*t**5-15*t**4+10*t**3)
    window = axes[0][:, None]*axes[1][None, :]
    expected = wave.amplitude*window
    np.testing.assert_allclose(output.amplitude, expected, rtol=0., atol=1e-20)
    event = pending.events[-1]
    assert event["transition_cells"] == width
    assert event["core_indices_yx"] == tuple((width, size-1-width) for size in shape)
    assert output.amplitude[width:-width, width:-width].tobytes() == wave.amplitude[width:-width, width:-width].tobytes()
    assert event["absolute_l2_error"] == pytest.approx(np.linalg.norm(output.amplitude-wave.amplitude), rel=3e-15)
    loss = float(np.sum(abs(wave.amplitude)**2*(1-window)*(1+window)))
    assert event["absolute_probability_loss"] == pytest.approx(loss, rel=3e-15)
    assert event["l2_per_reference"] == .5*event["absolute_l2_error"]
    assert event["probability_per_reference"] == .25*event["absolute_probability_loss"]
    assert event["l2_per_reference"] <= ledger.maximum_l2/100
    assert event["probability_per_reference"] <= ledger.maximum_probability/100
    assert pending.to_dict()["schema"].endswith("v2" if width == 2 else "v5")
    assert NumericalSupportLedger.from_dict(pending.to_dict()) == pending
    assert not event["physical_absorption"] and not event["probability_repair"]
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, key), getattr(wave, key))
    assert wave.amplitude.tobytes() == original and ledger.events == ()


def test_historical_default_and_explicit_two_cells_keep_identical_v2_payload():
    wave, original = _edge_wave(), NumericalSupportLedger(.01, 1e-6)
    a, first = original.taper_boundary(wave, z_m=.5, mode_weight=.25)
    b, second = original.taper_boundary(wave, z_m=.5, mode_weight=.25, transition_cells=2)
    assert a.amplitude.tobytes() == b.amplitude.tobytes()
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("prefix_kind", ["v1", "v2", "v3", "v4"])
def test_v5_forward_extension_preserves_old_prefix_and_future_pressure_digest(prefix_kind):
    if prefix_kind == "v1":
        _, _, prefix = _ledger()
    elif prefix_kind == "v2":
        _, prefix = NumericalSupportLedger(.02, 1e-6).taper_boundary(_edge_wave(tail=1e-9), z_m=.4, mode_weight=.25)
    elif prefix_kind == "v3":
        from test_potential_weighted_resampling import _ledger as spectral_ledger
        _, prefix = spectral_ledger()
    else:
        _, _, prefix = _pressure_ledger()
    previous = copy.deepcopy(prefix.to_dict())
    assert previous["schema"].endswith(prefix_kind)
    _, wider = prefix.taper_boundary(_edge_wave(tail=1e-12), z_m=.6, mode_weight=.25, transition_cells=8)
    assert wider.to_dict()["schema"].endswith("v5")
    assert wider.to_dict()["events"][:-1] == previous["events"]
    assert prefix.to_dict() == previous
    assert NumericalSupportLedger.from_dict(previous).to_dict() == previous
    if prefix_kind != "v1":
        _, _, later = _pressure_ledger(wider)
        assert later.events[-1]["pressure_allocation"]["previous_ledger_digest"] == json_digest(wider.to_dict())
        assert later.events[:-1] == wider.events
        assert NumericalSupportLedger.from_dict(later.to_dict()) == later
        if prefix_kind == "v4":
            assert later.events[0]["pressure_allocation"] == prefix.events[0]["pressure_allocation"]


@pytest.mark.parametrize("width", [True, 4., np.int64(8), 0, 1, 3, 32, -2, None])
def test_boundary_width_must_be_an_explicit_supported_integer(width):
    with pytest.raises(ValueError, match="explicit integer"):
        NumericalSupportLedger(.01, 1e-6).taper_boundary(_edge_wave(), z_m=0., transition_cells=width)


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(transition_cells=2),
    lambda e: e.update(transition_cells=4),
    lambda e: e.update(transition_cells=True),
    lambda e: e.update(transition_cells=32),
    lambda e: e.update(core_indices_yx=[[7, 56], [8, 58]]),
    lambda e: e.update(core_indices_yx=[[8, 56], [8, 58]]),
    lambda e: e.update(from_shape=[16, 16], to_shape=[16, 16], crop_indices_yx=[[0, 16], [0, 16]], core_indices_yx=[[8, 8], [8, 8]]),
    lambda e: e.update(mode_weight=1.),
    lambda e: e.update(l2_per_reference=0.),
    lambda e: e.update(probability_per_reference=0.),
    lambda e: e.update(origin_shift_m=[1e-30, 0.]),
])
def test_v5_reader_rejects_forged_width_core_size_weight_or_cost(mutation):
    _, ledger = NumericalSupportLedger(.01, 1e-6).taper_boundary(_edge_wave(), z_m=.5, mode_weight=.25, transition_cells=8)
    data = ledger.to_dict()
    mutation(data["events"][-1])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


def test_old_schema_cannot_reinterpret_wide_window_and_v5_cannot_relabel_old_history():
    _, old = NumericalSupportLedger(.01, 1e-6).taper_boundary(_edge_wave(), z_m=.5)
    _, new = NumericalSupportLedger(.01, 1e-6).taper_boundary(_edge_wave(), z_m=.5, transition_cells=8)
    for data, version in ((new.to_dict(), "v4"), (old.to_dict(), "v5")):
        data["schema"] = "controlled-numerical-support-ledger-"+version
        with pytest.raises(ValueError, match="schema"):
            NumericalSupportLedger.from_dict(data)


@pytest.mark.parametrize("width", [4, 8, 16])
def test_wider_windows_cannot_hide_underflow_or_exceed_remaining_budget(width):
    ledger = NumericalSupportLedger(.01, 1e-6)
    for wave, weight in ((_edge_wave(tail=1e-170), .25), (_edge_wave(), 1e-320)):
        output, pending = ledger.taper_boundary(wave, z_m=0., mode_weight=weight, transition_cells=width)
        assert output is wave and pending is ledger
    wave = _edge_wave()
    _, measure = ledger.taper_boundary(wave, z_m=0., transition_cells=width)
    limited = NumericalSupportLedger(100.5*measure.used_l2, 100.5*measure.used_probability)
    for index in range(100):
        _, limited = limited.taper_boundary(wave, z_m=index*.001, transition_cells=width)
    output, same = limited.taper_boundary(wave, z_m=.2, transition_cells=width)
    assert output is wave and same is limited and len(limited.events) == 100
    too_small = replace(wave, amplitude=wave.amplitude[:2*width, :2*width])
    output, same = ledger.taper_boundary(too_small, z_m=0., transition_cells=width)
    assert output is too_small and same is ledger


@pytest.mark.parametrize("outcome", ["success", "cancel_before_next", "cancel_after", "domain", "physical_failure"])
def test_window_trials_share_raw_spectral_input_and_commit_only_successful_complete_operator(monkeypatch, outcome):
    import temsim.physics.potential_wave_carrier as carrier
    from test_wave_resource_preparation import _narrow_source
    from test_potential_wave_carrier import _snapshot, _round, _multipole, ENERGY_EV
    source, weight = _narrow_source(), .25
    initial = NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    real_spectral, real_taper = NumericalSupportLedger.resample_spectrum_pressure, NumericalSupportLedger.taper_boundary
    real_validate, real_residual, real_forecast = carrier._validate_wave, carrier._residual, carrier._pressure_resampling_constraints
    real_proposal = carrier.WaveResourceProposal
    state = {"raw": None, "pending": None, "last": None, "failed": [], "widths": [],
             "width": 0, "cancelled": False, "commits": 0, "residuals": 0, "accepted": None}
    def spectral(self, wave, **options):
        assert self is initial and wave is source
        output, pending = real_spectral(self, wave, **options)
        state.update(raw=weakref.ref(output.amplitude), pending=pending)
        return output, pending
    def taper(self, wave, **options):
        assert self is state["pending"] and state["raw"]() is wave.amplitude
        assert options["mode_weight"] == weight and all(ref() is None for ref in state["failed"])
        width = options["transition_cells"]
        state["widths"].append(width)
        state["width"] = width
        output, pending = real_taper(self, wave, **options)
        assert output is not wave and len(pending.events) == 2
        state["last"] = weakref.ref(output.amplitude)
        if width in (2, 4):
            state["failed"].append(state["last"])
        if outcome == "cancel_after" and width == 4:
            state["cancelled"] = True
        state["accepted"] = pending
        return output, pending
    def validate(wave, *args):
        if state["raw"] is not None and state["raw"]() is wave.amplitude:
            raise carrier.OccupiedCarrierBoundaryError("controlled raw spectral boundary")
        if state["last"] is not None and state["last"]() is wave.amplitude:
            if state["width"] == 2:
                if outcome == "cancel_before_next":
                    state["cancelled"] = True
                raise carrier.OccupiedCarrierBoundaryError("controlled two-cell boundary")
            if outcome == "domain" and state["width"] == 4:
                raise ValueError("controlled window field domain")
        return real_validate(wave, *args)
    def forecast(*args, **kwargs):
        fits, context = real_forecast(*args, **kwargs)
        # The four-cell window passes sampling but lacks the same reserved
        # operator growth. This isolates continuing to eight without spending
        # or forcing an outer re-projection of a different raw input.
        return lambda shape: state["width"] != 4 and fits(shape), context
    def residual(snapshot, wave, *args, **options):
        state["residuals"] += 1
        if state["residuals"] == 1:
            raise WaveSamplingError("controlled original resource pressure", 2.)
        assert state["widths"] == [2, 4, 8]
        if state["residuals"] == 2:
            assert state["commits"] == 0 and initial.events == ()
        if outcome == "physical_failure":
            raise ValueError("controlled physical failure after window admission")
        return real_residual(snapshot, wave, *args, **options)
    def proposal(wave, commit):
        def publish():
            state["commits"] += 1
            commit()
        return real_proposal(wave, publish)
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", lambda self, wave, **options: (wave, self))
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_pressure_resampling_constraints", forecast)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(carrier, "WaveResourceProposal", proposal)
    options = dict(energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric,
        support_ledger=initial, support_mode_weight=weight, support_projection_policy="weighted_segment_and_retry",
        cancelled=lambda: state["cancelled"])
    if outcome == "success":
        output, record = carrier.propagate_carrier_potential(_snapshot(_round(), _multipole()), source, (0., 1e-9), **options)
        accepted = NumericalSupportLedger.from_dict(record["support_ledger"])
        assert accepted == state["accepted"] and state["commits"] == 1
        assert accepted.to_dict()["schema"].endswith("v5")
        assert [row["kind"] for row in accepted.events] == ["numerical_spectral_projection", "numerical_boundary_taper"]
        assert accepted.events[-1]["transition_cells"] == 8
        assert output.probability == pytest.approx(source.probability-accepted.used_probability/weight, abs=2e-13)
    else:
        error = InterruptedError if outcome.startswith("cancel") else ValueError
        with pytest.raises(error):
            carrier.propagate_carrier_potential(_snapshot(_round(), _multipole()), source, (0., 1e-9), **options)
        assert state["commits"] == 0
        assert state["raw"]() is None and all(ref() is None for ref in state["failed"])
    assert initial.events == ()
