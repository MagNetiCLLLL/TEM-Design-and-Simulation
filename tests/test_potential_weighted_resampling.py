"""Weighted Fourier ledger and genuine carrier comparisons on bounded cases."""
from dataclasses import replace
import copy
import math

import numpy as np
import pytest

from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.potential_wave_policy import potential_support_policy
from temsim.physics.wave_grid import WaveGridNumerics, refine_plane_wave


def _input():
    from test_potential_wave_carrier import _wave
    wave = _wave(n=128, pitch=1e-7, width=8e-7)
    axis = np.arange(128)-64
    return replace(wave, amplitude=wave.amplitude*(
        1.+5e-5*np.exp(2j*np.pi*.25*axis[:, None])
        +5e-5*np.exp(2j*np.pi*.25*axis[None, :])))


def _ledger():
    original = NumericalSupportLedger(.02, 1e-6)
    output, ledger = original.resample_spectrum(_input(), z_m=.519, mode_weight=.25)
    assert ledger.events and original.events == ()
    return output, ledger


def _pressure_input():
    from test_potential_wave_carrier import _wave
    wave = _wave(n=128, pitch=1e-7, width=8e-7)
    axis = np.arange(128)-64
    return replace(wave, amplitude=wave.amplitude*(
        1.+5e-4*np.exp(2j*np.pi*.25*axis[:, None])
        +5e-4*np.exp(2j*np.pi*.25*axis[None, :])))


def _pressure_context(wave, numerics=None, retained_bytes=0):
    numerics = WaveGridNumerics() if numerics is None else numerics
    return {"schema": "wave-resource-pressure-v1", "stage": "residual retry",
        "failure_kind": "WaveMemoryBudgetError", "input_shape": list(wave.amplitude.shape),
        "maximum_host_working_bytes": numerics.maximum_working_bytes,
        "maximum_device_working_bytes": numerics.maximum_device_working_bytes,
        "maximum_pixels": numerics.maximum_pixels, "retained_host_bytes": retained_bytes,
        "operator_memory_policy": "carrier-residual-host-512-plus-64MiB/device-448",
        "requested_backend": "CPU", "growth_guard_scale": 1.,
        "candidate_family": "fixed_aspect_even_fft_friendly_v1"}


def _pressure_ledger(previous=None):
    previous = NumericalSupportLedger(.02, 1e-6) if previous is None else previous
    wave = _pressure_input()
    output, ledger = previous.resample_spectrum_pressure(wave, z_m=.52,
        mode_weight=.25, admissible_shape=lambda shape: max(shape) <= 64,
        resource_context=_pressure_context(wave))
    assert output is not wave and len(ledger.events) == len(previous.events)+1
    return wave, output, ledger


def test_pressure_projection_spends_remaining_budget_without_renormalization():
    from temsim.immutable_json import json_digest
    original = NumericalSupportLedger(.02, 1e-6)
    wave, output, ledger = _pressure_ledger(original)
    assert original.events == ()
    assert output.amplitude.shape == (64, 64)
    assert ledger.to_dict()["schema"] == "controlled-numerical-support-ledger-v4"
    event = ledger.events[-1]
    assert ledger.maximum_l2 == original.maximum_l2
    assert ledger.maximum_probability == original.maximum_probability
    assert ledger.used_l2 > original.maximum_l2/100.
    assert ledger.used_probability > original.maximum_probability/100.
    assert ledger.used_l2 <= original.maximum_l2
    assert ledger.used_probability <= original.maximum_probability
    allocation = event["pressure_allocation"]
    assert allocation["previous_ledger_digest"] == json_digest(original.to_dict())
    assert allocation["remaining_l2_per_reference"] == original.maximum_l2
    assert allocation["remaining_probability_per_reference"] == original.maximum_probability
    assert event["physical_absorption"] is event["probability_repair"] is False
    assert output.probability < wave.probability
    assert .25*(wave.probability-output.probability) == pytest.approx(ledger.used_probability, rel=1e-8)
    restored = refine_plane_wave(output, wave.amplitude.shape)
    np.testing.assert_allclose(restored.basis_m, wave.basis_m, rtol=2e-15)
    assert .5*np.linalg.norm(restored.amplitude-wave.amplitude) == pytest.approx(ledger.used_l2, rel=1e-10)
    for key in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, key), getattr(wave, key))
    assert NumericalSupportLedger.from_dict(ledger.to_dict()) == ledger
    with pytest.raises(TypeError):
        allocation["remaining_l2_per_reference"] = original.maximum_l2*2.


def test_pressure_forward_transition_preserves_legacy_prefix_and_chains_exact_remaining():
    from temsim.immutable_json import json_digest
    _, old = _ledger()
    old_payload = old.to_dict()
    _, _, first = _pressure_ledger(old)
    _, _, second = _pressure_ledger(first)
    assert old.to_dict() == old_payload
    assert first.to_dict()["events"][:-1] == old_payload["events"]
    assert second.to_dict()["events"][:-1] == first.to_dict()["events"]
    for before, after in ((old, first), (first, second)):
        allocation = after.events[-1]["pressure_allocation"]
        assert allocation["previous_ledger_digest"] == json_digest(before.to_dict())
        assert allocation["remaining_l2_per_reference"] == before.maximum_l2-before.used_l2
        assert allocation["remaining_probability_per_reference"] == before.maximum_probability-before.used_probability
        assert after.maximum_l2 == old.maximum_l2
        assert after.maximum_probability == old.maximum_probability
        assert after.used_l2 > before.used_l2
        assert after.used_probability > before.used_probability
    assert NumericalSupportLedger.from_dict(second.to_dict()) == second


def test_repeated_pressure_attempts_cannot_reissue_an_exhausted_probability_allowance():
    wave = _pressure_input()
    ledger = NumericalSupportLedger(.02, 1e-6)
    for _ in range(12):
        output, updated = ledger.resample_spectrum_pressure(wave, z_m=.52,
            mode_weight=.25, admissible_shape=lambda shape: max(shape) <= 64,
            resource_context=_pressure_context(wave))
        if updated is ledger:
            assert output is wave
            break
        ledger = updated
    else:
        pytest.fail("Repeated pressure attempts renewed the original source allowance")
    assert ledger.events and ledger.used_probability <= ledger.maximum_probability
    assert ledger.used_l2 <= ledger.maximum_l2
    assert NumericalSupportLedger.from_dict(ledger.to_dict()) == ledger


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(schema="remaining-support-pressure-allocation-v0"),
    lambda a: a.update(previous_ledger_digest="0"*64),
    lambda a: a.update(remaining_l2_per_reference=.04),
    lambda a: a.update(remaining_l2_per_reference=.01),
    lambda a: a.update(remaining_probability_per_reference=2e-6),
    lambda a: a.update(remaining_probability_per_reference=5e-7),
    lambda a: a.update(remaining_probability_per_reference=True),
    lambda a: a.update(remaining_probability_per_reference=float("nan")),
    lambda a: a.pop("previous_ledger_digest"),
    lambda a: a.update(unrecognized=True),
])
def test_pressure_ledger_rejects_forged_remaining_or_previous_identity(mutation):
    _, _, ledger = _pressure_ledger()
    data = ledger.to_dict()
    mutation(data["events"][-1]["pressure_allocation"])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(schema="wave-resource-pressure-v0"),
    lambda c: c.update(stage=""),
    lambda c: c.update(stage=" residual retry"),
    lambda c: c.update(failure_kind="ValueError"),
    lambda c: c.update(input_shape=[64, 64]),
    lambda c: c.update(input_shape=[True, 128]),
    lambda c: c.update(source_failure_shape=[0, 128]),
    lambda c: c.update(maximum_host_working_bytes=0),
    lambda c: c.update(maximum_device_working_bytes=True),
    lambda c: c.update(maximum_pixels=128.),
    lambda c: c.update(retained_host_bytes=-1),
    lambda c: c.update(operator_memory_policy="unbounded"),
    lambda c: c.update(requested_backend="cpu"),
    lambda c: c.update(growth_guard_scale=.9),
    lambda c: c.update(candidate_family="arbitrary"),
    lambda c: c.update(unrecognized=True),
    lambda c: c.pop("input_shape"),
])
def test_pressure_ledger_rejects_unreproducible_resource_context(mutation):
    _, _, ledger = _pressure_ledger()
    data = ledger.to_dict()
    mutation(data["events"][-1]["pressure_allocation"]["resource_context"])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


def test_pressure_event_cannot_be_replayed_or_laundered_as_a_legacy_allocation():
    _, _, ledger = _pressure_ledger()
    replay = ledger.to_dict()
    duplicated = copy.deepcopy(replay["events"][0])
    duplicated["event_index"] = 1
    replay["events"].append(duplicated)
    with pytest.raises(ValueError, match="preceding ledger"):
        NumericalSupportLedger.from_dict(replay)
    legacy = ledger.to_dict()
    legacy["schema"] = "controlled-numerical-support-ledger-v3"
    legacy["events"][0].pop("pressure_allocation")
    with pytest.raises(ValueError, match="individual allocation"):
        NumericalSupportLedger.from_dict(legacy)
    # Retaining v4 elsewhere must not relax an old event's per-attempt cap.
    _, _, twice = _pressure_ledger(ledger)
    bad_prefix = twice.to_dict()
    bad_prefix["events"][0].pop("pressure_allocation")
    with pytest.raises(ValueError, match="individual allocation"):
        NumericalSupportLedger.from_dict(bad_prefix)


def test_new_version_cannot_reinterpret_or_reset_a_historical_ledger():
    _, old = _ledger()
    data = old.to_dict()
    data["schema"] = "controlled-numerical-support-ledger-v4"
    with pytest.raises(ValueError, match="projection kinds"):
        NumericalSupportLedger.from_dict(data)
    _, _, ledger = _pressure_ledger(old)
    data = ledger.to_dict()
    data["events"].pop(0)
    data["events"][0]["event_index"] = 0
    with pytest.raises(ValueError, match="preceding ledger"):
        NumericalSupportLedger.from_dict(data)
    data = ledger.to_dict()
    data["maximum_probability"] *= 2.
    with pytest.raises(ValueError, match="preceding ledger"):
        NumericalSupportLedger.from_dict(data)


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(to_shape=[64, 60]),
    lambda e: e.update(retained_frequency_intervals_yx=[[-23, 23], [-24, 24]]),
    lambda e: e.update(guard_nyquist_fraction=.77),
])
def test_pressure_receipt_cannot_claim_a_different_target_family_or_an_extra_filter(mutation):
    _, _, ledger = _pressure_ledger()
    data = ledger.to_dict()
    mutation(data["events"][-1])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


@pytest.mark.parametrize("kind", ["spatial", "boundary"])
def test_pressure_forward_transition_keeps_v1_and_v2_receipts_unchanged(kind):
    from temsim.immutable_json import json_digest
    from test_potential_wave_support import _wave
    original = NumericalSupportLedger(.02, 1e-6)
    if kind == "spatial":
        _, before = original.compact(_wave(), wavelength_m=2e-12, z_m=.4, mode_weight=.25)
        version = "v1"
    else:
        _, before = original.taper_boundary(_wave(shape=(64, 64)), z_m=.4, mode_weight=.25)
        version = "v2"
    assert before.events and before.to_dict()["schema"].endswith(version)
    _, _, after = _pressure_ledger(before)
    assert after.to_dict()["events"][:-1] == before.to_dict()["events"]
    assert after.events[-1]["pressure_allocation"]["previous_ledger_digest"] == json_digest(before.to_dict())
    assert NumericalSupportLedger.from_dict(after.to_dict()) == after


def test_pressure_history_remains_nonspendable_after_material_branching():
    from temsim.physics.potential_support_lineage import capture_inherited_support, common_support_summary
    from temsim.physics.potential_wave_observation import observe_potential_checkpoint
    from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
    from temsim.physics.wave_flux import BeamState, WaveMode
    from test_potential_wave_carrier import _snapshot
    _, plane, ledger = _pressure_ledger()
    norm = plane.probability
    mode = WaveMode(replace(plane, amplitude=plane.amplitude/math.sqrt(norm)),
                    .25*norm, TIP_REFERENCE, "pressure-projected-parent", 300.)
    snapshot = _snapshot()
    parent = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 520., 1e-12,
        {"snapshot_digest": snapshot.numerical_digest, "support_ledger": ledger.to_dict()})
    prefix = capture_inherited_support(parent, snapshot.numerical_digest)
    child = replace(parent, record={"snapshot_digest": snapshot.numerical_digest,
        "inherited_potential_support": prefix, "support_prefix_digest": prefix["digest"]})
    summary = common_support_summary(child.record)
    assert summary["accounting_multiplicity"] == 1
    assert summary["probability_removed_per_reference"] == ledger.used_probability
    assert summary["prefix_l2_per_reference"] == ledger.used_l2
    assert not summary["remaining_budget_reissued"]
    with pytest.raises(ValueError, match="projection is disabled"):
        observe_potential_checkpoint(snapshot, child, (.520001,), support_ledger=ledger,
            support_projection_policy="weighted_segment_and_retry")


@pytest.mark.parametrize("field", ["maximum_host_working_bytes", "maximum_device_working_bytes",
    "maximum_pixels", "retained_host_bytes"])
def test_pressure_api_records_actual_resource_limits_before_any_proposal(field):
    wave = _pressure_input()
    ledger = NumericalSupportLedger(.02, 1e-6)
    context = _pressure_context(wave)
    context[field] += 1
    def unexpected(_shape):
        pytest.fail("An invalid resource declaration reached target selection")
    with pytest.raises(ValueError, match="actual numerical resource"):
        ledger.resample_spectrum_pressure(wave, z_m=.52,
            admissible_shape=unexpected, resource_context=context)
    assert ledger.events == ()


@pytest.mark.parametrize("failure", [ValueError("Unknown admission domain"), InterruptedError("Cancelled target")])
def test_failed_pressure_selection_does_not_publish_or_spend_a_ledger(failure):
    wave = _pressure_input()
    ledger = NumericalSupportLedger(.02, 1e-6)
    def fail(_shape):
        raise failure
    with pytest.raises(type(failure), match=str(failure)):
        ledger.resample_spectrum_pressure(wave, z_m=.52,
            admissible_shape=fail, resource_context=_pressure_context(wave))
    assert ledger.events == ()


@pytest.mark.parametrize("limits,weight", [((0., 1e-6), .25), ((.02, 0.), .25), ((.02, 1e-6), 0.)])
def test_disabled_pressure_projection_cannot_create_an_event(limits, weight):
    wave = _pressure_input()
    ledger = NumericalSupportLedger(*limits)
    output, updated = ledger.resample_spectrum_pressure(wave, z_m=.52, mode_weight=weight,
        admissible_shape=lambda shape: max(shape) <= 64, resource_context=_pressure_context(wave))
    assert output is wave and updated is ledger


def test_spectral_ledger_roundtrip_is_immutable_and_weights_both_costs():
    output, ledger = _ledger()
    event = ledger.events[0]
    assert ledger.to_dict()["schema"] == "controlled-numerical-support-ledger-v3"
    assert ledger.used_l2 == .5*event["absolute_l2_error"]
    assert ledger.used_probability == .25*event["absolute_probability_loss"]
    assert ledger.used_probability <= ledger.maximum_probability/100.
    assert event["absolute_l2_error"]**2 == pytest.approx(event["absolute_probability_loss"], rel=2e-15)
    assert output.probability < event["input_probability"]
    restored = NumericalSupportLedger.from_dict(ledger.to_dict())
    assert restored == ledger
    with pytest.raises(TypeError):
        restored.events[0]["mode_weight"] = 1.


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(absolute_probability_loss=0.),
    lambda e: e.update(absolute_l2_error=0.),
    lambda e: e.update(probability_per_reference=0.),
    lambda e: e.update(l2_per_reference=0.),
    lambda e: e.update(mode_weight=.5),
    lambda e: e.update(physical_absorption=True),
    lambda e: e.update(probability_repair=True),
    lambda e: e.update(guard_nyquist_fraction=.8),
    lambda e: e.update(guard_nyquist_fraction=True),
    lambda e: e.update(origin_shift_m=[1e-15, 0.]),
    lambda e: e.update(carrier_constant_action_m=1e-15),
    lambda e: e.update(event_index=1),
    lambda e: e.update(error_scope="global propagated error bound"),
    lambda e: e.update(to_shape=[1, 1]),
    lambda e: e.update(to_shape=[33, 33]),
    lambda e: e.update(retained_frequency_intervals_yx=[[-64, 63], [-64, 63]]),
    lambda e: e.update(extra="unrecognized"),
    lambda e: e.pop("guard_nyquist_fraction"),
])
def test_spectral_ledger_rejects_tampered_receipts(mutation):
    _, ledger = _ledger()
    data = copy.deepcopy(ledger.to_dict())
    mutation(data["events"][0])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


def test_unchanged_axis_cannot_claim_an_unrecorded_frequency_filter():
    from test_potential_wave_resampling import _wave
    wave = _wave()
    spectrum = np.zeros(wave.amplitude.shape, complex)
    spectrum[:, 80] = 1./math.sqrt(spectrum.shape[0])
    field = replace(wave, amplitude=np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(spectrum), norm="ortho")))
    _, ledger = NumericalSupportLedger(.02, 1e-6).resample_spectrum(field, z_m=0.)
    data = ledger.to_dict()
    assert data["events"][0]["to_shape"][0] == field.amplitude.shape[0]
    data["events"][0]["retained_frequency_intervals_yx"][0] = [-10, 10]
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


def test_projection_does_not_keep_spending_after_same_spectrum_is_compressed():
    output, ledger = _ledger()
    result, restored = ledger.resample_spectrum(output, z_m=.52, mode_weight=.25)
    assert result is output
    assert restored is ledger


def test_mixed_spatial_and_spectral_history_never_reissues_source_budget():
    from test_potential_wave_support import _wave
    original = NumericalSupportLedger(.02, 1e-6)
    cropped, spatial = original.compact(_wave(), wavelength_m=2e-12, z_m=.4, mode_weight=.25)
    assert spatial.events
    refined = refine_plane_wave(cropped, (256, 256))
    _, combined = spatial.resample_spectrum(refined, z_m=.5, mode_weight=.25)
    assert combined.events[:len(spatial.events)] == spatial.events
    assert len(combined.events) > len(spatial.events)
    assert combined.maximum_l2 == original.maximum_l2
    assert combined.maximum_probability == original.maximum_probability
    assert combined.used_probability >= spatial.used_probability
    assert combined.used_l2 >= spatial.used_l2
    assert NumericalSupportLedger.from_dict(combined.to_dict()) == combined


def test_weighted_policy_cannot_silently_replace_historical_projection_policy():
    _, ledger = _ledger()
    previous = {"support_ledger": ledger.to_dict(), "snapshot_digest": "executed"}
    old = potential_support_policy({"support_projection_policy": "step_boundaries_and_retry",
        "support_ledger": ledger}, None)
    previous["potential_support_policy"] = old
    with pytest.raises(ValueError, match="policy changed"):
        potential_support_policy({"support_projection_policy": "weighted_segment_and_retry"}, previous)
    current = potential_support_policy({"support_projection_policy": "weighted_segment_and_retry",
        "support_ledger": ledger})
    assert current["digest"] != old["digest"]


def _common_lattice_error(reference, actual):
    ratio = np.linalg.solve(reference.basis_m, actual.basis_m)
    np.testing.assert_allclose(ratio, np.diag(np.diag(ratio)), atol=2e-15)
    shape = tuple(round(n*factor) for n, factor in zip(actual.amplitude.shape, np.diag(ratio)[::-1]))
    refined = refine_plane_wave(actual, shape)
    np.testing.assert_allclose(refined.basis_m, reference.basis_m, rtol=2e-14, atol=1e-22)
    for field in ("curvature_m1", "tilt_rad", "origin_m"):
        np.testing.assert_allclose(getattr(refined, field), getattr(reference, field), atol=2e-15)
    common = np.maximum(refined.amplitude.shape, reference.amplitude.shape)
    def pad(amplitude):
        sizes = np.array(amplitude.shape)
        lower = common//2-sizes//2
        return np.pad(amplitude, tuple((int(a), int(n-b-a)) for a, n, b in zip(lower, common, sizes)))
    # No phase fit, recentering, or normalization is made at comparison time.
    return np.linalg.norm(pad(refined.amplitude)-pad(reference.amplitude))


def test_genuine_round_lens_propagation_converges_on_finer_resource_admitted_grids():
    from test_potential_wave_carrier import _snapshot, _round
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    wave = _input()
    snapshot = _snapshot(_round(.002))
    options = dict(energy_ev=200000., kinetic_order=4,
        grid_numerics=WaveGridNumerics(angular_domain_padding="minimum_full_band"))
    baseline, _ = propagate_carrier_potential(snapshot, wave, (0., 1e-5), **options)
    errors = []
    for maximum_shape, probability_budget in ((64, 1e-6), (96, 1e-8), (120, 1e-10)):
        # A fixed resource target keeps the same guarded Fourier coefficients
        # regardless of a surplus allowance. Convergence therefore increases
        # the feasible lattice while tightening its actual loss tolerance.
        prepared, ledger = NumericalSupportLedger(.02, probability_budget).resample_spectrum_pressure(
            wave, z_m=0., mode_weight=.25,
            admissible_shape=lambda shape: max(shape) <= maximum_shape,
            resource_context=_pressure_context(wave, options["grid_numerics"]),
            numerics=options["grid_numerics"])
        actual, _ = propagate_carrier_potential(snapshot, prepared, (0., 1e-5), **options)
        assert any(event["kind"] == "numerical_spectral_projection" for event in ledger.events)
        assert prepared.amplitude.shape == (maximum_shape, maximum_shape)
        error = _common_lattice_error(baseline, actual)
        errors.append(error)
        assert actual.probability < baseline.probability
        assert actual.amplitude.size < baseline.amplitude.size
        assert ledger.used_probability <= probability_budget
    assert errors[0] < 1e-4
    assert errors[1] < errors[0]/5
    assert errors[2] < errors[1]/10


def test_fixed_resource_target_is_budget_invariant_until_actual_loss_exceeds_the_allowance():
    wave = _pressure_input()
    common = dict(z_m=.52, mode_weight=.25, admissible_shape=lambda shape: max(shape) <= 64,
                  resource_context=_pressure_context(wave))
    first, receipt = NumericalSupportLedger(.02, 1e-6).resample_spectrum_pressure(wave, **common)
    assert receipt.events and receipt.used_probability > 0. and receipt.used_l2 > 0.
    for limits in ((.01, 5e-7), (.04, 2e-6)):
        candidate, ledger = NumericalSupportLedger(*limits).resample_spectrum_pressure(wave, **common)
        np.testing.assert_array_equal(candidate.amplitude, first.amplitude)
        assert ledger.used_probability == receipt.used_probability
        assert ledger.used_l2 == receipt.used_l2
    for limits in ((receipt.used_l2/2., 1e-6), (.02, receipt.used_probability/2.)):
        original = NumericalSupportLedger(*limits)
        candidate, ledger = original.resample_spectrum_pressure(wave, **common)
        assert candidate is wave and ledger is original


@pytest.mark.parametrize("rejection", ["boundary", "sampling"])
def test_rejected_candidate_spends_no_spectral_budget_and_reuses_original(rejection, monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot
    from temsim.physics.wave_grid import WaveSamplingError, WaveMemoryBudgetError
    validate = carrier._validate_wave
    def reject_small(wave, *args):
        if min(wave.amplitude.shape) < 128:
            if rejection == "boundary":
                raise carrier.OccupiedCarrierBoundaryError("Injected rejected candidate")
            raise WaveSamplingError("Injected rejected candidate")
        return validate(wave, *args)
    monkeypatch.setattr(carrier, "_validate_wave", reject_small)
    failure = WaveMemoryBudgetError("Unresolved original resource pressure")
    def blocked(*args, **kwargs):
        raise failure
    monkeypatch.setattr(carrier, "_residual", blocked)
    wave = _input()
    original = NumericalSupportLedger(.02, 1e-6)
    with pytest.raises(WaveMemoryBudgetError) as captured:
        carrier.propagate_carrier_potential(_snapshot(), wave, (0., 1e-6),
            energy_ev=200000., kinetic_order=2, support_ledger=original,
            support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry",
            grid_numerics=WaveGridNumerics(maximum_pixels=128))
    assert captured.value is failure
    assert original.events == ()


@pytest.mark.parametrize("failure", [ValueError("Unknown field domain"), InterruptedError("Cancelled projection")])
def test_unknown_domain_and_cancellation_are_not_optional_projection_failures(failure, monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot
    coverage = carrier._coverage
    def fail_projected(snapshot, wave, *args):
        if min(wave.amplitude.shape) < 128:
            raise failure
        return coverage(snapshot, wave, *args)
    monkeypatch.setattr(carrier, "_coverage", fail_projected)
    from temsim.physics.wave_grid import WaveMemoryBudgetError
    def blocked(*args, **kwargs):
        raise WaveMemoryBudgetError("Controlled resource-pressure trigger")
    monkeypatch.setattr(carrier, "_residual", blocked)
    ledger = NumericalSupportLedger(.02, 1e-6)
    with pytest.raises(type(failure), match=str(failure)):
        carrier.propagate_carrier_potential(_snapshot(), _input(), (0., 1e-6),
            energy_ev=200000., kinetic_order=2, support_ledger=ledger,
            support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry")
    assert ledger.events == ()


def _failed_canonical_trial(candidate):
    """Deliberately unresolved private output for the retry contract tests."""
    count = candidate.amplitude.shape[1]
    phase = np.exp(2j*np.pi*(int(.45*count)/count)*(np.arange(count)-count//2))
    return replace(candidate, amplitude=candidate.amplitude*phase[None, :])


def test_canonical_output_failure_refines_original_entrance_without_publishing_failed_field(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, _wave
    inputs, failed = [], []
    def canonical(candidate, *args, **kwargs):
        inputs.append(candidate)
        if len(inputs) == 1:
            trial = _failed_canonical_trial(candidate)
            failed.append(trial)
            return trial
        return candidate
    monkeypatch.setattr(carrier, "propagate_plane_wave", canonical)
    output, record = carrier.propagate_carrier_potential(_snapshot(),
        _wave(n=128, width=8e-7), (0., 1e-9), energy_ev=200000., kinetic_order=2)
    assert len(inputs) == 2
    assert inputs[1].amplitude.shape[0] > inputs[0].amplitude.shape[0]
    expected = refine_plane_wave(inputs[0], inputs[1].amplitude.shape)
    np.testing.assert_array_equal(inputs[1].amplitude, expected.amplitude)
    assert output is not failed[0]
    assert output.amplitude.shape == inputs[1].amplitude.shape
    assert any("Nyquist guard" in row.get("reason", "") for row in record["canonical_sampling"])
    # The deliberate high-frequency modulation must not survive the retry.
    carrier._check_envelope_sampling(output.amplitude, 1e-10)


def test_canonical_retry_chain_commits_one_projection_and_never_projects_failed_outputs(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.wave_grid import WaveMemoryBudgetError
    from test_potential_wave_carrier import _snapshot
    inputs, failed_outputs, projection_inputs, accepted = [], [], [], []
    original_resample = NumericalSupportLedger.resample_spectrum_pressure
    def resample(ledger, candidate, **kwargs):
        # Isolate the canonical retry chain: other independently authorized
        # stage boundaries are outside this failure-control experiment.
        if not 0 < len(inputs) < 4:
            return candidate, ledger
        projection_inputs.append(candidate)
        result, pending = original_resample(ledger, candidate, **kwargs)
        if result is not candidate:
            accepted.append((result, pending))
        return result, pending
    def canonical(candidate, *args, **kwargs):
        inputs.append(candidate)
        if len(inputs) == 1:
            raise WaveMemoryBudgetError("Controlled canonical resource-pressure trigger")
        if len(inputs) <= 3:
            trial = _failed_canonical_trial(candidate)
            failed_outputs.append(trial)
            return trial
        return candidate
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", resample)
    monkeypatch.setattr(NumericalSupportLedger, "compact", lambda ledger, candidate, **kw: (candidate, ledger))
    monkeypatch.setattr(carrier, "propagate_plane_wave", canonical)
    original = NumericalSupportLedger(.02, 1e-6)
    output, record = carrier.propagate_carrier_potential(_snapshot(), _input(), (0., 1e-9),
        energy_ev=200000., kinetic_order=2, support_ledger=original,
        support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry",
        grid_numerics=WaveGridNumerics(maximum_pixels=256))
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert len(inputs) == 4
    assert len(projection_inputs) == len(accepted) == len(ledger.events) == 1
    assert projection_inputs[0] is inputs[0]
    assert all(candidate is not failed for candidate in projection_inputs for failed in failed_outputs)
    assert inputs[1] is accepted[0][0]
    # Subsequent trials only embed the already admitted entrance: no
    # shrink/refine loop reissues a projection or resets its spent allowance.
    for before, after in zip(inputs[1:-1], inputs[2:]):
        assert after.amplitude.size > before.amplitude.size
        expected = refine_plane_wave(before, after.amplitude.shape)
        np.testing.assert_array_equal(after.amplitude, expected.amplitude)
    assert ledger.to_dict() == accepted[0][1].to_dict()
    assert ledger.used_probability > 0. and ledger.used_l2 > 0.
    assert ledger.maximum_l2 == original.maximum_l2
    assert ledger.maximum_probability == original.maximum_probability
    assert original.events == ()
    assert all(output is not failed for failed in failed_outputs)
    carrier._check_envelope_sampling(output.amplitude, 1e-10)
