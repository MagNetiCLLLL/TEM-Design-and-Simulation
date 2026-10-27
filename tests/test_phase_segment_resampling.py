"""Phase-prepared private candidates reserve errors without rewriting history."""
from dataclasses import replace
import gc
import math
import weakref

import numpy as np
import pytest

from temsim.physics.potential_segment_resampling import prepare_segment_resampling
from temsim.physics.potential_wave_carrier import OccupiedCarrierBoundaryError
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import WaveGridNumerics
from test_potential_weighted_resampling import _input


def _phase_input():
    from test_potential_wave_carrier import _wave
    wave = _wave(n=128, pitch=1e-7, width=5e-7)
    y, x = (np.indices(wave.amplitude.shape)-64)*1e-7
    amplitude = wave.amplitude*np.exp(1j*np.pi/2.51e-12*(.5*x*x+.3*y*y+.08*x*y))
    return replace(wave, amplitude=amplitude,
        curvature_m1=np.array(((12., 3.), (3., -7.))), tilt_rad=np.array((.002, -.003)),
        origin_m=np.array((2e-8, -3e-8)))


def _phase_options():
    from temsim.physics.potential_representation_estimates import RepresentationEstimateLedger
    from test_potential_segment_resampling import _admit
    ledger = NumericalSupportLedger(.02, 1e-6)
    history = RepresentationEstimateLedger.for_support(ledger)
    numeric = WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=2**28)
    return ledger, history, numeric, _admit(numeric)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True, "0.1"])
def test_representation_reservation_requires_an_explicit_finite_nonnegative_number(value):
    with pytest.raises(ValueError, match="Reserved representation"):
        prepare_segment_resampling(_input(), NumericalSupportLedger(.02, 1e-6),
            z_m=.5, reserved_l2=value, admit=lambda wave, **_: wave)


def test_reserved_l2_rejects_spectral_candidate_before_admission_without_spending():
    wave, ledger = _input(), NumericalSupportLedger(.02, 1e-6)
    reference = prepare_segment_resampling(wave, ledger, z_m=.5,
        mode_weight=.25, admit=lambda wave, **_: wave)
    assert reference.wave is not wave and reference.ledger.used_l2 > 0
    result = prepare_segment_resampling(wave, ledger, z_m=.5,
        mode_weight=.25, reserved_l2=ledger.maximum_l2-reference.ledger.used_l2/2,
        admit=lambda *_a, **_k: pytest.fail("Combined allowance already exceeded"))
    assert result.wave is wave and result.ledger is ledger
    assert "combined" in result.diagnostics["reason"]
    assert ledger.events == ()


def test_reserved_l2_is_checked_again_after_each_pending_window(monkeypatch):
    wave, ledger = _input(), NumericalSupportLedger(.5, .01)
    raw, spectral = ledger.resample_spectrum(wave, z_m=.5, mode_weight=.25)
    _, windowed = spectral.taper_boundary(raw, z_m=.5, mode_weight=.25, transition_cells=2)
    window_cost = windowed.used_l2-spectral.used_l2
    assert window_cost > 0
    reservation = ledger.maximum_l2-spectral.used_l2-window_cost/2
    real_taper = NumericalSupportLedger.taper_boundary
    seen = []
    def taper(self, current, **kwargs):
        seen.append(kwargs["transition_cells"])
        assert len(self.events) == 1
        return real_taper(self, current, **kwargs)
    calls = []
    def admit(candidate, **_):
        calls.append(candidate.amplitude.shape)
        raise OccupiedCarrierBoundaryError("Explicit raw edge")
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    result = prepare_segment_resampling(wave, ledger, z_m=.5, mode_weight=.25,
        reserved_l2=reservation, admit=admit)
    assert seen == [2, 4, 8, 16]
    assert len(calls) == 1  # Every added window exceeds the reserved combined limit.
    assert result.wave is wave and result.ledger is ledger and ledger.events == ()
    assert result.diagnostics["attempts"][1]["status"] == "combined support and representation allowance exceeded"
    assert all(row["status"] in ("combined support and representation allowance exceeded", "no affordable edge window")
               for row in result.diagnostics["attempts"][1:])


def test_real_phase_candidate_preserves_original_fields_and_two_histories():
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    from temsim.physics.potential_representation_estimates import require_combined_support_allowance, RepresentationEstimateLedger
    from temsim.physics.wave_grid import refine_plane_wave
    wave = _phase_input()
    ledger, history, numeric, admit = _phase_options()
    original = wave.amplitude.tobytes()
    result = prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
        z_m=.541, mode_weight=.25, numerics=numeric, admit=admit)
    assert result.wave is not wave, result.diagnostics
    assert result.ledger is not ledger and result.representation_estimates is not history
    assert result.wave.amplitude.size <= .95*wave.amplitude.size
    assert max(result.wave.amplitude.shape) <= numeric.maximum_pixels
    assert result.diagnostics["preparation_maximum_pixels"] == 512
    assert result.diagnostics["absolute_quadrature_relative_tolerance"] == 1e-7
    assert not result.diagnostics["certified_representation_error"]
    assert not result.diagnostics["committed"]
    assert require_combined_support_allowance(result.ledger, result.representation_estimates) >= 0
    assert result.representation_estimates.used_l2_estimate > 0
    assert result.ledger.maximum_l2 == ledger.maximum_l2 and result.ledger.maximum_probability == ledger.maximum_probability
    for key in ("origin_m", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result.wave, key), getattr(wave, key))
    assert np.linalg.norm(result.wave.curvature_m1-wave.curvature_m1) > .5
    # Common physical coordinates restore all known Q and tilt. Their shared
    # analytical phase cancels algebraically, never by data fitting.
    a = refine_plane_wave(wave, (512, 512))
    b = refine_plane_wave(result.wave, (512, 512))
    yy, xx = np.indices(a.amplitude.shape)-256
    x = a.basis_m[0, 0]*xx+a.basis_m[0, 1]*yy
    y = a.basis_m[1, 0]*xx+a.basis_m[1, 1]*yy
    delta = b.curvature_m1-a.curvature_m1
    relative_phase = np.exp(1j*np.pi/2.51e-12*(delta[0, 0]*x*x+
        (delta[0, 1]+delta[1, 0])*x*y+delta[1, 1]*y*y))
    measured = .5*float(np.linalg.norm(b.amplitude*relative_phase-a.amplitude))
    estimate_and_projection = result.ledger.used_l2+result.representation_estimates.used_l2_estimate
    assert 0 < measured <= estimate_and_projection+2e-12
    assert .25*(wave.probability-result.wave.probability) == pytest.approx(
        result.ledger.used_probability, rel=1e-5, abs=2e-15)
    assert wave.amplitude.tobytes() == original and ledger.events == () and history.events == ()
    assert RepresentationEstimateLedger.from_dict(result.representation_estimates.to_dict()).to_dict() == result.representation_estimates.to_dict()


def test_no_phase_change_is_an_identity_noop_without_projection(monkeypatch):
    from temsim.physics import multiplane_wave
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    wave = _phase_input()
    ledger, history, numeric, admit = _phase_options()
    monkeypatch.setattr(multiplane_wave, "reselect_phase_carrier", lambda value, *_a, **_k: (value, []))
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum", lambda *_a, **_k: pytest.fail("No changed phase carrier"))
    result = prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
        z_m=.541, numerics=numeric, admit=admit)
    assert result.wave is wave and result.ledger is ledger and result.representation_estimates is history


@pytest.mark.parametrize("kind", ["sampling", "memory", "domain", "cancel"])
def test_phase_estimation_failures_do_not_spend_or_swallow_real_errors(monkeypatch, kind):
    from temsim.physics import potential_representation_estimates as estimates
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    from temsim.physics.wave_grid import WaveSamplingError, WaveGridBudgetError
    wave = _phase_input()
    ledger, history, numeric, admit = _phase_options()
    error = {"sampling": WaveSamplingError("Unresolved representation estimate"),
        "memory": WaveGridBudgetError("Declared preparation memory"),
        "domain": ValueError("Invalid physical domain"), "cancel": InterruptedError("Cancelled")}[kind]
    def fail(*_a, **_k):
        raise error
    monkeypatch.setattr(estimates, "estimate_carrier_representational_change", fail)
    if kind in ("domain", "cancel"):
        with pytest.raises(type(error), match=str(error)) as caught:
            prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
                z_m=.541, numerics=numeric, admit=admit)
        assert caught.value is error
    else:
        result = prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
            z_m=.541, numerics=numeric, admit=admit)
        assert result.wave is wave and result.ledger is ledger and result.representation_estimates is history
    assert ledger.events == () and history.events == ()


def test_admitted_phase_candidate_that_regrows_is_not_committed():
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    from temsim.physics.wave_grid import refine_plane_wave
    wave = _phase_input()
    ledger, history, numeric, _ = _phase_options()
    result = prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
        z_m=.541, numerics=numeric,
        admit=lambda candidate, **_: refine_plane_wave(candidate, wave.amplitude.shape))
    assert result.wave is wave and result.ledger is ledger and result.representation_estimates is history
    assert ledger.events == () and history.events == ()


def test_fine_preparation_and_windows_use_original_size_limit_and_unique_retained_buffers(monkeypatch):
    from temsim.physics import multiplane_wave
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    from temsim.physics.wave_grid import refine_plane_wave
    wave = _phase_input()
    ledger, history, numeric, _ = _phase_options()
    real_reselect, real_taper = multiplane_wave.reselect_phase_carrier, NumericalSupportLedger.taper_boundary
    seen, prepared, released = [], {}, []
    def fine(value, *args, **options):
        assert options["grid_numerics"].maximum_pixels == 2*numeric.maximum_pixels
        assert options["grid_numerics"].maximum_working_bytes == numeric.maximum_working_bytes
        assert options["grid_numerics"].compute_backend == "CPU"
        result, rows = real_reselect(value, *args, **options)
        result = refine_plane_wave(result, (256, 256), numerics=options["grid_numerics"])
        prepared["wave"] = weakref.ref(result)
        return result, rows
    def taper(self, value, **options):
        gc.collect()
        assert all(ref() is None for ref in released)
        seen.append(options["transition_cells"])
        return real_taper(self, value, **options)
    def admit(candidate, *, retained_bytes):
        assert retained_bytes >= 123+wave.amplitude.nbytes+prepared["wave"]().amplitude.nbytes
        if not seen:
            raise OccupiedCarrierBoundaryError("Explicit raw spectrum edge")
        if seen[-1] == 2:
            grown = refine_plane_wave(candidate, wave.amplitude.shape)
            released.append(weakref.ref(grown.amplitude))
            return grown  # Fits preparation cap but saves nothing against ORIGINAL.
        return candidate
    monkeypatch.setattr(multiplane_wave, "reselect_phase_carrier", fine)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    result = prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
        z_m=.541, mode_weight=.25, numerics=replace(numeric, compute_backend="Require GPU"),
        retained_bytes=123, admit=admit)
    assert result.wave is not wave, result.diagnostics
    assert seen == [2, 4]
    assert result.ledger.events[-1]["transition_cells"] == 4
    assert len(result.representation_estimates.events) == 1
    assert len(result.ledger.events) == 2  # Rejected width 2 spent nothing.
    assert ledger.events == () and history.events == ()


def test_old_pressure_prefix_is_not_rewritten_by_new_representation_history():
    from test_potential_weighted_resampling import _pressure_ledger
    from temsim.immutable_json import thaw_json, json_digest
    from temsim.physics.potential_representation_estimates import RepresentationEstimateLedger
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    wave = _phase_input()
    _, _, ledger = _pressure_ledger()
    before = ledger.to_dict()
    digest = json_digest(before)
    binding = {"source_checkpoint_digest": "a"*64, "source_mode_id": "actual-E4",
               "support_allocation_digest": "b"*64}
    history = RepresentationEstimateLedger.for_support(ledger, source_binding=binding)
    _, _, numeric, admit = _phase_options()
    result = prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
        z_m=.541, mode_weight=.25, numerics=numeric, admit=admit)
    assert result.wave is not wave, result.diagnostics
    assert thaw_json(result.ledger.events[:len(ledger.events)]) == before["events"]
    assert json_digest(ledger.to_dict()) == digest and history.events == ()
    assert dict(result.representation_estimates.source_binding) == binding


def test_cancellation_after_admission_leaves_both_original_histories():
    from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
    wave = _phase_input()
    ledger, history, numeric, real_admit = _phase_options()
    flag = [False]
    def admit(candidate, **options):
        result = real_admit(candidate, **options)
        flag[0] = True
        return result
    with pytest.raises(InterruptedError, match="cancelled"):
        prepare_phase_segment_resampling(wave, ledger, history, wavelength_m=2.51e-12,
            z_m=.541, mode_weight=.25, numerics=numeric, admit=admit, cancelled=lambda: flag[0])
    assert flag[0] and ledger.events == () and history.events == ()
