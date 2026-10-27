"""Preferred canonical chart resources remain separate from alternate sampling."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import (
    CanonicalChartResourceError, CanonicalChartResourceHint,
    WaveGridBudgetError, WaveGridNumerics, WaveResourceProposal,
    WaveResourceTransaction, WaveSamplingError, apply_resolved_operator,
)


def _split_wave(frequency=.24e9):
    n, pitch, sigma, centre = 256, 1e-9, 4e-9, 98e-9
    axis = (np.arange(n)-n//2)*pitch
    y, x = np.meshgrid(axis, axis, indexing="ij")
    amplitude = np.exp(-((x-centre)**2+y*y)/(4*sigma*sigma))*np.cos(2*np.pi*frequency*x)*np.exp(.37j)
    amplitude /= np.linalg.norm(amplitude)
    wave = PlaneWave(amplitude, np.eye(2)*pitch, np.zeros(2))
    matrix = np.eye(4)
    matrix[:2, 2:] = np.eye(2)*14e-6
    return wave, matrix, 2e-12


def _numeric():
    return WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=512*1024**2,
        angular_domain_padding="minimum_full_band")


def test_real_canonical_alternative_retains_288_domain_beside_11_fold_collins_sampling():
    source, matrix, wavelength = _split_wave()
    original = source.amplitude.copy()
    with pytest.raises(WaveSamplingError) as caught:
        propagate_plane_wave(source, matrix, np.zeros(4), wavelength, grid_numerics=_numeric())
    hint = caught.value.canonical_chart_resource_hint
    assert type(hint) is CanonicalChartResourceHint
    assert hint.input_shape == (256, 256) and hint.required_shape == (288, 288)
    assert hint.preferred_chart == "angular_spectrum_domain" and hint.alternate_chart == "collins"
    assert hint.alternate_required_scale == caught.value.required_scale == pytest.approx(10.9375)
    assert isinstance(caught.value.__cause__, WaveSamplingError)
    # A sufficient physical domain resolves precisely the same canonical map.
    records = []
    output = propagate_plane_wave(source, matrix, np.zeros(4), wavelength,
        grid_numerics=replace(_numeric(), maximum_pixels=512), sampling_records=records)
    assert output.amplitude.shape == (288, 288)
    assert output.probability == pytest.approx(source.probability, abs=2e-14)
    assert records[-1]["output_boundary_probability"] < 1e-12
    assert records[-1]["full_spectral_band_checked"] is True
    np.testing.assert_array_equal(source.amplitude, original)


def test_real_weighted_canonical_retries_full_operator_from_budgeted_original_wave():
    from temsim.physics.potential_wave_carrier import _check_boundary, _check_envelope_sampling
    source, matrix, wavelength = _split_wave(frequency=.12e9)
    original = source.amplitude.copy()
    weight, initial = .4, NumericalSupportLedger(.02, 1e-4)
    state = {"ledger": initial, "calls": [], "commits": 0, "projected": None}
    numeric = _numeric()

    def operator(wave):
        result = propagate_plane_wave(wave, matrix, np.zeros(4), wavelength, grid_numerics=numeric)
        _check_boundary(result.amplitude, 1e-9)
        _check_envelope_sampling(result.amplitude, 1e-10)
        return result

    def prepare(wave, error, pressure):
        assert wave is source and state["ledger"] is initial
        assert type(error) is CanonicalChartResourceError
        assert pressure.required_shape == error.canonical_chart_resource_hint.required_shape
        fits, context = _pressure_resampling_constraints(wave, error, numerics=numeric,
            requested="CPU", retained_bytes=0, stage="canonical retry", pressure=pressure)
        assert context["growth_guard_scale"] == 1.25
        assert context["source_failure_shape"] == [288, 288]
        projected, pending = initial.resample_spectrum_pressure(wave, z_m=.5,
            mode_weight=weight, numerics=numeric, admissible_shape=fits, resource_context=context)
        assert projected is not wave
        _check_boundary(projected.amplitude, 1e-9)
        _check_envelope_sampling(projected.amplitude, 1e-10)
        state["calls"].append((pressure, context))
        state["projected"] = projected
        def commit():
            state["commits"] += 1
            state["ledger"] = pending
        return WaveResourceProposal(projected, commit)

    output, rows = apply_resolved_operator(source, operator, numerics=numeric,
        prepare_resource=WaveResourceTransaction(prepare))
    expected = operator(state["projected"])
    np.testing.assert_array_equal(output.amplitude, expected.amplitude)
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, key), getattr(expected, key))
    accepted = state["ledger"]
    assert max(output.amplitude.shape) <= numeric.maximum_pixels
    assert len(state["calls"]) == 1 and state["commits"] == 1 and len(accepted.events) == 1
    assert accepted.events[0]["mode_weight"] == weight
    assert output.probability == pytest.approx(source.probability-accepted.used_probability/weight, abs=2e-14)
    assert accepted.used_probability <= initial.maximum_probability and accepted.used_l2 <= initial.maximum_l2
    assert rows[0]["resource_stage"] == "physical operator"
    assert initial.events == ()
    np.testing.assert_array_equal(source.amplitude, original)


def _hint(shape, required=None):
    required = required or tuple(math.ceil(n*1.125) for n in shape)
    return CanonicalChartResourceHint(tuple(shape), tuple(required),
        "angular_spectrum_domain", "declared_domain_grid_budget", "collins",
        "independent Collins chirp", 11.)


@pytest.mark.parametrize("hint_kind", ["unknown", "wrong_input", "none", "plain_callback"])
def test_unknown_stale_or_nontransactional_hint_preserves_original_sampling_failure(hint_kind):
    source, _, _ = _split_wave()
    calls = []
    def operator(wave):
        error = WaveSamplingError("unchanged ordinary Collins failure", 11.)
        if hint_kind == "unknown":
            error.canonical_chart_resource_hint = {"input_shape": wave.amplitude.shape, "required_shape": (288, 288)}
        elif hint_kind == "wrong_input":
            error.canonical_chart_resource_hint = _hint((128, 128))
        elif hint_kind == "plain_callback":
            error.canonical_chart_resource_hint = _hint(wave.amplitude.shape)
        raise error
    def prepare(wave, error, pressure=None):
        calls.append((type(error), pressure))
        return wave
    callback = prepare if hint_kind == "plain_callback" else WaveResourceTransaction(prepare)
    with pytest.raises(WaveGridBudgetError) as caught:
        apply_resolved_operator(source, operator, numerics=_numeric(), prepare_resource=callback)
    assert type(caught.value) is WaveGridBudgetError
    assert isinstance(caught.value.__context__, WaveSamplingError)
    assert caught.value.__context__.required_scale == 11.
    assert all(error_type is not CanonicalChartResourceError for error_type, _ in calls)


def test_each_resource_attempt_uses_matching_preferred_shape_not_collins_or_previous_source():
    n = 128
    yy, xx = np.meshgrid(np.arange(n)-n//2, np.arange(n)-n//2, indexing="ij")
    value = np.cos(np.pi*xx/n)**8*np.cos(np.pi*yy/n)**8*np.exp(.4j)
    source = PlaneWave(value/np.linalg.norm(value), np.eye(2)*1e-9, np.zeros(2))
    numeric = replace(_numeric(), maximum_pixels=n)
    initial = NumericalSupportLedger(.02, 1e-6)
    state = {"operators": 0, "pressures": [], "ledger": initial, "commits": 0}
    def operator(wave):
        state["operators"] += 1
        if state["operators"] <= 2:
            error = WaveSamplingError("alternate sampling only", 11.)
            growth = 1.125 if state["operators"] == 1 else 1.5
            error.canonical_chart_resource_hint = _hint(wave.amplitude.shape,
                tuple(math.ceil(n*growth) for n in wave.amplitude.shape))
            raise error
        return wave
    def prepare(wave, error, pressure):
        assert wave is source and state["ledger"] is initial
        hint = error.canonical_chart_resource_hint
        assert pressure.required_shape == hint.required_shape
        fits, context = _pressure_resampling_constraints(wave, error, numerics=numeric,
            requested="CPU", retained_bytes=0, stage="canonical retry", pressure=pressure)
        state["pressures"].append((pressure, context))
        projected, pending = initial.resample_spectrum_pressure(source, z_m=.5,
            numerics=numeric, admissible_shape=fits, resource_context=context)
        def commit():
            state["ledger"] = pending
            state["commits"] += 1
        return WaveResourceProposal(projected, commit)
    output, _ = apply_resolved_operator(source, operator, numerics=numeric,
        prepare_resource=WaveResourceTransaction(prepare))
    assert len(state["pressures"]) == 2 and state["commits"] == 1
    assert [context["growth_guard_scale"] for _, context in state["pressures"]] == [1.25, 1.5]
    assert len(state["ledger"].events) == 1 and initial.events == ()
    assert output.amplitude.shape[0] < state["pressures"][1][0].previous_shape[0]


@pytest.mark.parametrize("disabled", ["cancelled", "automatic_refinement"])
def test_canonical_hint_never_bypasses_cancellation_or_disabled_refinement(disabled):
    source, matrix, wavelength = _split_wave()
    calls = []
    numeric = replace(_numeric(), automatic_refinement=disabled != "automatic_refinement")
    with pytest.raises(InterruptedError if disabled == "cancelled" else WaveSamplingError):
        apply_resolved_operator(source,
            lambda wave: propagate_plane_wave(wave, matrix, np.zeros(4), wavelength, grid_numerics=numeric),
            numerics=numeric, cancelled=lambda: disabled == "cancelled",
            prepare_resource=WaveResourceTransaction(lambda *args: calls.append(args)))
    assert calls == []


@pytest.mark.parametrize("change", [
    {"input_shape": [256, 256]}, {"required_shape": (256, 256)},
    {"required_shape": (288, True)}, {"preferred_chart": "arbitrary"},
    {"reason": "physical domain failure"}, {"alternate_required_scale": float("inf")},
])
def test_canonical_resource_hint_rejects_untyped_or_unbounded_metadata(change):
    values = dict(input_shape=(256, 256), required_shape=(288, 288),
        preferred_chart="angular_spectrum_domain", reason="declared_domain_grid_budget",
        alternate_chart="collins", alternate_reason="sampling", alternate_required_scale=11.)
    with pytest.raises(ValueError):
        CanonicalChartResourceHint(**(values | change))
