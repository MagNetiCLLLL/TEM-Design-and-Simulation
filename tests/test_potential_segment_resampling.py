"""Uncommitted preview preparation preserves complex fields and old budgets."""
from dataclasses import replace
import gc
import math
import weakref

import numpy as np
import pytest

from temsim.immutable_json import json_digest, thaw_json
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_segment_resampling import prepare_segment_resampling
from temsim.physics.potential_wave_carrier import OccupiedCarrierBoundaryError, _validate_wave
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, WaveGridBudgetError, apply_resolved_operator, refine_plane_wave
from test_potential_weighted_resampling import _input, _pressure_ledger


def _admit(numeric):
    def admit(candidate, *, retained_bytes):
        def validate(wave):
            _validate_wave(wave, numeric, 1e-9, 1e-10)
            return wave
        result, _ = apply_resolved_operator(candidate, validate, numerics=numeric,
            retained_bytes=retained_bytes,
            operator_working_bytes=lambda shape: 256*math.prod(shape))
        return result
    return admit


def test_real_complex_candidate_loss_and_carriers_match_an_independent_common_grid():
    wave = replace(_input(), origin_m=np.array((2e-8, -3e-8)),
        curvature_m1=np.array(((12., 3.), (3., -7.))),
        tilt_rad=np.array((.002, -.003)))
    original_bytes = wave.amplitude.tobytes()
    original = NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=2**26)
    weight = .27
    result = prepare_segment_resampling(wave, original, z_m=.506, mode_weight=weight,
        numerics=numeric, admit=_admit(numeric))
    assert result.wave is not wave
    assert result.ledger is not original
    assert result.diagnostics["status"] == "candidate"
    assert not result.diagnostics["committed"] and not result.diagnostics["physical_operator_executed"]
    assert result.wave.amplitude.size <= .95*wave.amplitude.size
    restored = refine_plane_wave(result.wave, wave.amplitude.shape)
    np.testing.assert_allclose(restored.basis_m, wave.basis_m, rtol=3e-15, atol=0.)
    for key in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result.wave, key), getattr(wave, key))
    measured_l2 = math.sqrt(weight)*float(np.linalg.norm(restored.amplitude-wave.amplitude))
    yy, xx = np.indices(wave.amplitude.shape)-np.asarray(wave.amplitude.shape)[:, None, None]//2
    coordinates = np.stack((xx, yy), axis=-1)@wave.basis_m.T
    action = coordinates@wave.tilt_rad+.5*np.einsum(
        "...i,ij,...j->...", coordinates, wave.curvature_m1, coordinates)
    carrier = np.exp(2j*np.pi/2.51e-12*action)
    physical_l2 = math.sqrt(weight)*float(np.linalg.norm(
        restored.amplitude*carrier-wave.amplitude*carrier))
    assert physical_l2 == pytest.approx(measured_l2, rel=2e-10, abs=2e-16)
    assert measured_l2 > 0
    assert measured_l2 <= result.ledger.used_l2*(1+1e-8)+2e-15
    assert weight*(wave.probability-result.wave.probability) == pytest.approx(
        result.ledger.used_probability, rel=2e-7, abs=3e-16)
    assert result.diagnostics["weighted_complex_l2_cost"] == result.ledger.used_l2
    for event in result.ledger.events:
        assert event["mode_weight"] == weight
        assert event["l2_per_reference"] <= original.maximum_l2/100
        assert event["probability_per_reference"] <= original.maximum_probability/100
        assert not event["physical_absorption"] and not event["probability_repair"]
    assert wave.amplitude.tobytes() == original_bytes and original.events == ()
    assert NumericalSupportLedger.from_dict(result.ledger.to_dict()) == result.ledger
    with pytest.raises(TypeError):
        result.diagnostics["committed"] = True


def test_existing_pressure_prefix_keeps_its_exact_schema_digest_and_spent_allowance():
    _, _, original = _pressure_ledger()
    payload = original.to_dict()
    digest = json_digest(payload)
    wave = _input()
    result = prepare_segment_resampling(wave, original, z_m=.53, mode_weight=.25, admit=lambda wave, **_: wave)
    assert result.wave is not wave
    assert thaw_json(result.ledger.events[:len(original.events)]) == payload["events"]
    assert original.to_dict() == payload and json_digest(original.to_dict()) == digest
    assert result.ledger.maximum_l2 == original.maximum_l2
    assert result.ledger.maximum_probability == original.maximum_probability
    assert result.ledger.used_l2 > original.used_l2
    assert result.ledger.used_probability > original.used_probability
    assert NumericalSupportLedger.from_dict(result.ledger.to_dict()) == result.ledger


def test_each_window_starts_from_same_raw_spectrum_and_rejected_arrays_are_released(monkeypatch):
    wave, ledger = _input(), NumericalSupportLedger(.5, .01)
    raw_info, rejected, widths, admitted_retained = {}, [], [], []
    real_spectrum = NumericalSupportLedger.resample_spectrum
    real_taper = NumericalSupportLedger.taper_boundary
    def spectral(self, candidate, **options):
        raw, pending = real_spectrum(self, candidate, **options)
        raw_info.update(wave=weakref.ref(raw), ledger=pending)
        return raw, pending
    def taper(self, raw, **options):
        gc.collect()
        assert all(ref() is None for ref in rejected)
        assert raw is raw_info["wave"]() and self is raw_info["ledger"]
        widths.append(options["transition_cells"])
        assert options["mode_weight"] == .25
        return real_taper(self, raw, **options)
    def admit(candidate, *, retained_bytes):
        admitted_retained.append(retained_bytes)
        if candidate is raw_info["wave"]():
            raise OccupiedCarrierBoundaryError("controlled numerical edge")
        if widths[-1] == 2:
            rejected.append(weakref.ref(candidate.amplitude))
            failed_workspace = np.ones((100, 100))
            rejected.append(weakref.ref(failed_workspace))
            raise WaveSamplingError("controlled window spectrum")
        if widths[-1] == 4:
            rejected.append(weakref.ref(candidate.amplitude))
            return refine_plane_wave(candidate, wave.amplitude.shape)
        return candidate
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum", spectral)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    result = prepare_segment_resampling(wave, ledger, z_m=.506, mode_weight=.25,
        retained_bytes=123, admit=admit)
    assert widths == [2, 4, 8]
    assert result.diagnostics["selected_transition_cells"] == 8
    assert len(result.ledger.events) == 2 and ledger.events == ()
    assert result.ledger.events[-1]["transition_cells"] == 8
    assert admitted_retained[0] >= 123+wave.amplitude.nbytes
    assert all(value > admitted_retained[0] for value in admitted_retained[1:])
    assert result.diagnostics["attempts"][2]["status"] == "insufficient saving after admission"


@pytest.mark.parametrize("failure", [WaveSamplingError("numerical alias"), WaveGridBudgetError("declared resource")])
def test_raw_numerical_rejection_keeps_identical_input_without_unnecessary_windows(monkeypatch, failure):
    wave, ledger = _input(), NumericalSupportLedger(.02, 1e-6)
    def reject(*_args, **_kwargs):
        raise failure
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", lambda *_a, **_k: pytest.fail("No occupied boundary was reported"))
    result = prepare_segment_resampling(wave, ledger, z_m=.506, admit=reject)
    assert result.wave is wave and result.ledger is ledger
    assert result.diagnostics["status"] == "unchanged"
    assert result.diagnostics["attempts"][0]["error_type"] == type(failure).__name__


@pytest.mark.parametrize("failure", [ValueError("physical field domain"), RuntimeError("invalid callback"), InterruptedError("cancelled")])
def test_domain_invalid_and_cancellation_errors_are_not_swallowed(failure):
    wave, ledger = _input(), NumericalSupportLedger(.5, .01)
    def reject(*_args, **_kwargs):
        raise failure
    with pytest.raises(type(failure), match=str(failure)) as caught:
        prepare_segment_resampling(wave, ledger, z_m=.506, admit=reject)
    assert caught.value is failure and ledger.events == ()


@pytest.mark.parametrize("when", ["before", "after_admission", "after_rejected_window"])
def test_cancellation_discards_pending_receipts_without_another_window(monkeypatch, when):
    wave, ledger = _input(), NumericalSupportLedger(.5, .01)
    flag, widths = [when == "before"], []
    real_taper = NumericalSupportLedger.taper_boundary
    def taper(self, raw, **options):
        widths.append(options["transition_cells"])
        return real_taper(self, raw, **options)
    def admit(candidate, **_options):
        if when == "after_admission":
            flag[0] = True
            return candidate
        if not widths:
            raise OccupiedCarrierBoundaryError("controlled edge")
        flag[0] = True
        raise WaveSamplingError("controlled sampling rejection")
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    with pytest.raises(InterruptedError, match="cancelled"):
        prepare_segment_resampling(wave, ledger, z_m=.506, admit=admit, cancelled=lambda: flag[0])
    assert widths == ([2] if when == "after_rejected_window" else [])
    assert ledger.events == ()


@pytest.mark.parametrize("change", ["type", "period", "origin", "curvature", "tilt", "norm", "coarser"])
def test_admission_contract_rejects_changed_physical_wave(change):
    wave, ledger = _input(), NumericalSupportLedger(.02, 1e-6)
    def admit(candidate, **_):
        if change == "type":
            return None
        if change == "period":
            return replace(candidate, basis_m=candidate.basis_m*1.01)
        if change == "origin":
            return replace(candidate, origin_m=candidate.origin_m+1e-15)
        if change == "curvature":
            return replace(candidate, curvature_m1=candidate.curvature_m1+np.eye(2))
        if change == "tilt":
            return replace(candidate, tilt_rad=candidate.tilt_rad+1e-9)
        if change == "norm":
            return replace(candidate, amplitude=candidate.amplitude*.9)
        return replace(candidate, amplitude=candidate.amplitude[::2, ::2], basis_m=candidate.basis_m*2)
    with pytest.raises(ValueError):
        prepare_segment_resampling(wave, ledger, z_m=.506, admit=admit)
    assert ledger.events == ()


def test_five_percent_saving_applies_after_lossless_admission_and_each_axis():
    from test_potential_wave_carrier import _wave
    source = _wave(n=128, pitch=1e-7, width=8e-7)
    ledger = NumericalSupportLedger(.5, .01)
    for shape in ((128, 128), (126, 126), (130, 90)):
        result = prepare_segment_resampling(source, ledger, z_m=.506,
            admit=lambda candidate, **_: refine_plane_wave(candidate, shape))
        assert result.wave is source and result.ledger is ledger
        assert result.diagnostics["attempts"][0]["status"] == "insufficient saving after admission"


def test_zero_allowance_or_zero_weight_is_an_identity_noop():
    wave = _input()
    for ledger, weight in ((NumericalSupportLedger(0., 1e-6), .25),
                           (NumericalSupportLedger(.02, 0.), .25),
                           (NumericalSupportLedger(.02, 1e-6), 0.)):
        result = prepare_segment_resampling(wave, ledger, z_m=.506, mode_weight=weight,
            admit=lambda *_a, **_k: pytest.fail("No projection was affordable"))
        assert result.wave is wave and result.ledger is ledger


def test_spectral_memory_failure_is_a_bounded_noop_with_primitive_diagnostics():
    wave, ledger = _input(), NumericalSupportLedger(.02, 1e-6)
    result = prepare_segment_resampling(wave, ledger, z_m=.506,
        numerics=WaveGridNumerics(maximum_working_bytes=2**20),
        admit=lambda *_a, **_k: pytest.fail("Preparation was not affordable"))
    assert result.wave is wave and result.ledger is ledger
    assert result.diagnostics["attempts"][0]["error_type"] == "WaveMemoryBudgetError"
    assert isinstance(json_digest(result.diagnostics), str)


def test_all_four_rejected_windows_are_bounded_and_leave_previous_history_untouched(monkeypatch):
    _, _, ledger = _pressure_ledger()
    wave, payload, widths = _input(), ledger.to_dict(), []
    real_taper = NumericalSupportLedger.taper_boundary
    def taper(self, raw, **options):
        widths.append(options["transition_cells"])
        return real_taper(self, raw, **options)
    def reject(candidate, **_):
        raise OccupiedCarrierBoundaryError("Every proposed edge remains occupied")
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    result = prepare_segment_resampling(wave, ledger, z_m=.53, admit=reject)
    assert widths == [2, 4, 8, 16]
    assert result.wave is wave and result.ledger is ledger
    assert ledger.to_dict() == payload
    assert len(result.diagnostics["attempts"]) == 5
    assert result.diagnostics["status"] == "unchanged"


def test_window_memory_preflight_refuses_before_any_window_allocation(monkeypatch):
    wave, ledger = _input(), NumericalSupportLedger(.5, .01)
    real_check = WaveGridNumerics.check_working_memory
    checks = []
    def memory(self, shape, **options):
        checks.append((shape, options))
        if len(checks) > 1:
            raise WaveGridBudgetError("Explicit window workspace refusal")
        return real_check(self, shape, **options)
    def reject(candidate, **_):
        raise OccupiedCarrierBoundaryError("Try the edge family")
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", memory)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary",
        lambda *_a, **_k: pytest.fail("No window allocation is permitted"))
    result = prepare_segment_resampling(wave, ledger, z_m=.506, admit=reject)
    assert result.wave is wave and result.ledger is ledger
    assert len(checks) == 5 and len(result.diagnostics["attempts"]) == 5
    assert all(row["error_type"] == "WaveGridBudgetError" for row in result.diagnostics["attempts"][1:])
