"""Numerical support projection never hides tiny tails or rewrites phase."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave_support import compact_controlled_support, NumericalSupportLedger


def _wave(shape=(128, 128), sigma=4.):
    y, x = np.indices(shape, dtype=float)
    x -= shape[1]//2
    y -= shape[0]//2
    amplitude = np.exp(-(x*x+y*y)/(4*sigma*sigma))*(1+.1j*x/sigma)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.array([[2e-8, 1e-8], [-.3e-8, 3e-8]]),
        np.array([2e-5, -3e-5]), np.array([[1e6, 1e-8], [0., -3e6]]), np.array([.01, -.03]))


def _compact(wave, l2=1e-7, probability=1e-12):
    return compact_controlled_support(wave, wavelength_m=2e-12,
        absolute_l2_budget=l2, absolute_probability_budget=probability)


@pytest.mark.parametrize("shape", [(128, 128), (127, 129), (128, 129), (127, 128)])
def test_projection_preserves_exact_coordinates_and_all_phase_carriers(shape):
    wave = _wave(shape)
    output, receipt = _compact(wave)
    assert receipt and output.amplitude.size < wave.amplitude.size
    (y0, y1), (x0, x1) = receipt["crop_indices_yx"]
    np.testing.assert_array_equal(output.coordinates_m(), wave.coordinates_m()[:, y0:y1, x0:x1])
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(output, key), getattr(wave, key))
    (a, b), (c, d) = receipt["core_indices_yx"]
    np.testing.assert_array_equal(output.amplitude[a-y0:b-y0+1, c-x0:d-x0+1], wave.amplitude[a:b+1, c:d+1])
    embedded = np.zeros(shape, complex)
    embedded[y0:y1, x0:x1] = output.amplitude
    actual = np.linalg.norm(embedded-wave.amplitude)
    assert actual == pytest.approx(receipt["absolute_l2_error"], rel=2e-9)
    assert actual <= 1e-7
    assert receipt["absolute_probability_loss"] <= 1e-12
    assert receipt["origin_shift_m"] == [0., 0.]


@pytest.mark.parametrize("position", [(64, 120), (64, 4), (120, 64), (4, 64)])
def test_sub_roundoff_tail_cannot_exceed_its_probability_budget(position):
    wave = _wave()
    amplitude = np.zeros((128, 128), complex)
    amplitude[64, 64], amplitude[position] = 1., 1e-11
    wave = replace(wave, amplitude=amplitude)
    output, receipt = _compact(wave, l2=1e-10, probability=1e-25)
    assert np.count_nonzero(output.amplitude) == 2
    assert receipt is None or receipt["absolute_probability_loss"] == 0.


def test_permitted_sub_roundoff_tail_is_recorded_directly_not_total_subtraction():
    wave = _wave()
    amplitude = np.zeros((128, 128), complex)
    amplitude[64, 64], amplitude[64, 120] = 1., 1e-11
    output, receipt = _compact(replace(wave, amplitude=amplitude), l2=1e-10, probability=1e-20)
    assert np.count_nonzero(output.amplitude) == 1
    assert output.probability == receipt["input_probability"] == 1.
    assert receipt["absolute_probability_loss"] == pytest.approx(1e-22, rel=2e-15)
    assert receipt["absolute_l2_error"] == pytest.approx(1e-11, rel=2e-15)


def test_underflowing_tail_and_budget_do_not_create_false_zero_error_certificate():
    wave = _wave()
    amplitude = np.zeros((128, 128), complex)
    amplitude[64, 64], amplitude[64, 120] = 1., 1e-170
    candidate = replace(wave, amplitude=amplitude)
    output, receipt = _compact(candidate)
    assert output is candidate and receipt is None
    output, receipt = _compact(wave, l2=1e-200)
    assert output is wave and receipt is None


def _ledger():
    wave = _wave()
    output, ledger = NumericalSupportLedger(1e-5, 1e-10).compact(
        wave, wavelength_m=2e-12, z_m=-.2, mode_weight=.25)
    assert ledger.events
    return wave, output, ledger


def test_ledger_roundtrip_freezes_history_and_scales_absolute_mode_weight():
    wave, output, ledger = _ledger()
    event = ledger.events[0]
    assert ledger.used_l2 == .5*event["absolute_l2_error"]
    assert ledger.used_probability == .25*event["absolute_probability_loss"]
    restored = NumericalSupportLedger.from_dict(ledger.to_dict())
    assert restored == ledger
    with pytest.raises(TypeError):
        ledger.events[0]["mode_weight"] = 100.
    for _ in range(4):
        _, restored = restored.compact(wave, wavelength_m=2e-12, z_m=.3, mode_weight=.25)
    assert len(restored.events) == 5
    assert restored.used_l2 == pytest.approx(5*ledger.used_l2, rel=1e-15)
    assert restored.used_l2 <= restored.maximum_l2


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(l2_per_reference=0.),
    lambda row: row.update(probability_per_reference=0.),
    lambda row: row.update(absolute_l2_error=1.),
    lambda row: row.update(mode_weight=1.),
    lambda row: row.update(event_index=2),
    lambda row: row.update(z_m=float("nan")),
    lambda row: row.update(output_probability=0.),
    lambda row: row.update(l2_per_reference=False),
    lambda row: row.update(from_shape=[1, 1]),
    lambda row: row.update(crop_indices_yx=[[0, 32], [0, 32]]),
    lambda row: row.update(extra=True),
    lambda row: row.pop("error_scope"),
])
def test_restored_ledger_rejects_forged_receipts(mutation):
    _, _, ledger = _ledger()
    data = copy.deepcopy(ledger.to_dict())
    mutation(data["events"][0])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(data)


def test_independent_free_complex_propagation_converges_with_tighter_support():
    # This finite-distance check is deliberately independent of the carrier
    # implementation. The initial projection receipt is NOT a bound on using
    # a different periodic domain; that second error is measured here.
    wave = replace(_wave((256, 256), sigma=7.), curvature_m1=None, tilt_rad=None,
        basis_m=np.eye(2)*2e-8, origin_m=np.zeros(2))
    wavelength, distance = 2e-12, .0002
    def propagate(amplitude):
        fy, fx = [np.fft.fftfreq(n, d=2e-8) for n in amplitude.shape]
        phase = np.exp(-1j*np.pi*wavelength*distance*(fy[:, None]**2+fx[None, :]**2))
        return np.fft.ifft2(np.fft.fft2(amplitude)*phase)
    baseline = propagate(wave.amplitude)
    errors = []
    for budget in (1e-4, 1e-6, 1e-8):
        candidate, receipt = _compact(wave, l2=budget, probability=budget**2)
        assert receipt
        (y0, y1), (x0, x1) = receipt["crop_indices_yx"]
        embedded = np.zeros_like(baseline)
        embedded[y0:y1, x0:x1] = propagate(candidate.amplitude)
        error = np.linalg.norm(embedded-baseline)
        errors.append(error)
        assert error < budget
    assert errors[1] < errors[0]/40 and errors[2] < errors[1]/40


def test_carrier_sampling_retry_records_support_without_mutating_original_field():
    from test_potential_wave_carrier import _wave as carrier_wave, _snapshot, _pure_hex
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    from temsim.physics.wave_grid import WaveGridNumerics
    wave = carrier_wave(n=128, pitch=1e-7, width=3e-7)
    original = wave.amplitude.copy()
    output, evidence = propagate_carrier_potential(_snapshot(_pure_hex(5e7)), wave, (0., 1e-5),
        energy_ev=200000., scalar_splitting=True, support_ledger=NumericalSupportLedger(1e-5, 1e-9),
        support_mode_weight=.25, grid_numerics=WaveGridNumerics(maximum_pixels=512))
    ledger = NumericalSupportLedger.from_dict(evidence["support_ledger"])
    assert ledger.events and evidence["new_support_event_count"] == len(ledger.events)
    assert all(event["mode_weight"] == .25 for event in ledger.events)
    assert all(not event["physical_absorption"] for event in ledger.events)
    assert ledger.used_l2 <= 1e-5 and ledger.used_probability <= 1e-9
    assert output.probability <= wave.probability+2e-13
    np.testing.assert_array_equal(wave.amplitude, original)


def test_explicit_step_start_support_removes_redundant_padding_with_independent_complex_check():
    from test_potential_wave_carrier import _wave as carrier_wave, _snapshot
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    wave = carrier_wave(n=128, pitch=1e-7, width=3e-7)
    arguments = dict(energy_ev=200000., kinetic_order=2)
    expected, _ = propagate_carrier_potential(_snapshot(), wave, (0., 1e-5), **arguments)
    actual, record = propagate_carrier_potential(_snapshot(), wave, (0., 1e-5), **arguments,
        support_ledger=NumericalSupportLedger(1e-5, 1e-9), support_projection_policy="step_start_and_retry")
    assert record["new_support_event_count"] > 0
    assert actual.amplitude.size < expected.amplitude.size
    embedded = np.zeros_like(expected.amplitude)
    y0 = expected.amplitude.shape[0]//2-actual.amplitude.shape[0]//2
    x0 = expected.amplitude.shape[1]//2-actual.amplitude.shape[1]//2
    embedded[y0:y0+actual.amplitude.shape[0], x0:x0+actual.amplitude.shape[1]] = actual.amplitude
    np.testing.assert_allclose(actual.basis_m, expected.basis_m, rtol=1e-14)
    assert np.linalg.norm(embedded-expected.amplitude) < 1e-7


def test_checkpoint_support_history_survives_storage_and_cannot_be_reset(tmp_path):
    from test_potential_wave_carrier import _wave as carrier_wave, _snapshot, _pure_hex
    from test_potential_wave_observation import _mode
    from temsim.physics.potential_wave_observation import observe_potential_checkpoint, PotentialAperturePlane
    from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
    from temsim.physics.wave_flux import BeamState
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_grid import WaveGridNumerics
    mode = replace(_mode(), plane=carrier_wave(n=128, pitch=1e-7, width=3e-7), energy_kev=200.)
    entrance = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1., 1e-12, {"scope": "controlled"})
    snapshot = _snapshot(_pure_hex(5e7))
    options = dict(maximum_step_m=1e-5, scalar_splitting=True,
                   grid_numerics=WaveGridNumerics(maximum_pixels=512))
    initial_ledger = NumericalSupportLedger(1e-5, 1e-9)
    middle, = observe_potential_checkpoint(snapshot, entrance, (.00101,), support_ledger=initial_ledger, **options)
    stored = NumericalSupportLedger.from_dict(middle.record["support_ledger"])
    assert stored.events and stored.events[0]["mode_weight"] == mode.weight_per_reference_electron
    assert middle.record["modes"][0]["aperture_loss"] == 0.
    assert middle.record["modes"][0]["numerical_support_loss_per_reference"] == stored.used_probability
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 16*1024**2)
    key = store.key("support-check", middle.digest)
    store.put(key, middle)
    loaded = store.get(key)
    assert loaded.digest == store.get(key).digest
    assert loaded.record["support_ledger"] == middle.record["support_ledger"]
    aperture = PotentialAperturePlane("physical-stop", .00101, 4e-7)
    resumed, = observe_potential_checkpoint(snapshot, loaded, (.00101,), apertures=(aperture,), **options)
    assert resumed.record["support_ledger"] == middle.record["support_ledger"]
    assert resumed.record["modes"][0]["aperture_loss"] > .1
    assert resumed.record["modes"][0]["numerical_support_loss_per_reference"] == 0.
    account = resumed.record["modes"][0]
    assert account["input_mode_weight"] == pytest.approx(account["output_mode_weight"]
        +account["aperture_loss_per_reference"]+account["numerical_support_loss_per_reference"], abs=2e-14)
    assert stored.used_probability > 0.
    # Continue from the saved unit-shape/absolute-weight factorization without
    # granting a new budget. A direct two-stage call retains the same history.
    continuation, = observe_potential_checkpoint(snapshot, loaded, (.00102,), **options)
    _, direct = observe_potential_checkpoint(snapshot, entrance, (.00101, .00102),
        support_ledger=initial_ledger, **options)
    resumed_ledger = NumericalSupportLedger.from_dict(continuation.record["support_ledger"])
    assert resumed_ledger.events[:len(stored.events)] == stored.events
    first, second = continuation.beam.modes[0], direct.beam.modes[0]
    np.testing.assert_allclose(first.plane.amplitude*np.sqrt(first.weight_per_reference_electron),
        second.plane.amplitude*np.sqrt(second.weight_per_reference_electron), rtol=3e-10, atol=3e-13)
    with pytest.raises(ValueError, match="cannot be reset"):
        observe_potential_checkpoint(snapshot, loaded, (.00102,), support_ledger=initial_ledger, **options)
    with pytest.raises(ValueError, match="cannot be reset"):
        observe_potential_checkpoint(snapshot, loaded, (.00102,), support_ledger=None, **options)
    multiple = replace(entrance, beam=BeamState((mode, replace(mode, mode_id="controlled:second")), TIP_REFERENCE))
    with pytest.raises(ValueError, match="one mode"):
        observe_potential_checkpoint(snapshot, multiple, (.00101,), support_ledger=initial_ledger, **options)


def _edge_wave(shape=(64, 67), tail=1e-5):
    amplitude = np.zeros(shape, complex)
    amplitude[shape[0]//2, shape[1]//2] = .8+.6j
    amplitude[0, 7] = tail*(.6+.8j)
    amplitude[1, 8] = tail*(.8-.6j)
    amplitude[-2, -2] = tail*(-.6+.8j)
    amplitude[3, 4] = complex(-0., -0.)
    return replace(_wave(shape), amplitude=amplitude)


@pytest.mark.parametrize("shape", [(64, 67), (65, 66), (64, 64)])
def test_same_grid_boundary_taper_preserves_the_interior_carrier_and_direct_costs(shape):
    wave = _edge_wave(shape)
    before = wave.amplitude.tobytes()
    ledger = NumericalSupportLedger(.01, 1e-6)
    output, updated = ledger.taper_boundary(wave, z_m=-.25, mode_weight=.25)
    assert len(updated.events) == 1 and ledger.events == ()
    event = updated.events[0]
    assert event["kind"] == "numerical_boundary_taper"
    # Independent exact two-cell weights: outer sample 0, next sample 1/2,
    # every inner sample 1. The corner transition has both axis factors.
    expected = wave.amplitude.copy()
    expected[[0, -1], :] = 0.
    expected[:, [0, -1]] = 0.
    expected[[1, -2], :] *= .5
    expected[:, [1, -2]] *= .5
    np.testing.assert_array_equal(output.amplitude, expected)
    assert output.amplitude[2:-2, 2:-2].tobytes() == wave.amplitude[2:-2, 2:-2].tobytes()
    assert wave.amplitude.tobytes() == before
    assert output.amplitude.shape == wave.amplitude.shape
    for field in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        assert getattr(output, field).tobytes() == getattr(wave, field).tobytes()
    expected_loss = 1e-10*(1.+.75+.9375)
    expected_error = np.linalg.norm(expected-wave.amplitude)
    assert event["absolute_probability_loss"] == pytest.approx(expected_loss, rel=3e-16)
    assert event["absolute_l2_error"] == pytest.approx(expected_error, rel=3e-16)
    assert updated.used_probability == .25*event["absolute_probability_loss"]
    assert updated.used_l2 == .5*event["absolute_l2_error"]
    assert output.probability < wave.probability  # No normalization after loss.
    assert event["physical_absorption"] is False and event["probability_repair"] is False
    assert event["transition_cells"] == 2
    assert NumericalSupportLedger.from_dict(updated.to_dict()) == updated


def test_boundary_taper_records_sub_roundoff_loss_without_total_norm_subtraction():
    wave = _edge_wave(tail=1e-11)
    result, ledger = NumericalSupportLedger(1e-7, 1e-15).taper_boundary(wave, z_m=0.)
    assert result.probability == wave.probability == 1.
    assert ledger.used_probability == pytest.approx(2.6875e-22, rel=3e-16)
    assert ledger.used_l2 > 1e-11
    assert result.amplitude[0, 7] == 0.


@pytest.mark.parametrize("budgets,weight", [((0., 1.), 1.), ((1., 0.), 1.), ((1., 1.), 0.),
    ((1e-7, 1.), .25), ((1., 1e-12), .25)])
def test_boundary_taper_never_spends_disabled_or_excessive_individual_budgets(budgets, weight):
    wave = _edge_wave()
    before = wave.amplitude.tobytes()
    ledger = NumericalSupportLedger(*budgets)
    output, after = ledger.taper_boundary(wave, z_m=.2, mode_weight=weight)
    assert output is wave and after is ledger
    assert wave.amplitude.tobytes() == before and not ledger.events


def test_boundary_taper_rejects_unrepresentable_probability_cost():
    wave = _edge_wave(tail=1e-170)
    ledger = NumericalSupportLedger(1e-5, 1e-10)
    output, updated = ledger.taper_boundary(wave, z_m=0.)
    assert output is wave and updated is ledger


def test_boundary_taper_cumulative_remaining_budget_is_not_renewed():
    wave = _edge_wave()
    _, measured = NumericalSupportLedger(1., 1.).taper_boundary(wave, z_m=0.)
    loss, error = measured.used_probability, measured.used_l2
    ledger = NumericalSupportLedger(149.5*error, 149.5*loss)
    for index in range(149):
        output, updated = ledger.taper_boundary(wave, z_m=index*.001)
        assert output is not wave
        assert updated.events[:-1] == ledger.events
        ledger = updated
    output, updated = ledger.taper_boundary(wave, z_m=.15)
    assert output is wave and updated is ledger
    assert ledger.used_probability <= ledger.maximum_probability
    assert ledger.used_l2 <= ledger.maximum_l2
    assert ledger.maximum_l2-ledger.used_l2 < error


def test_boundary_taper_upgrades_only_its_own_event_and_preserves_legacy_history():
    _, _, original = _ledger()
    legacy = original.to_dict()
    assert legacy["schema"] == "controlled-numerical-support-ledger-v1"
    assert NumericalSupportLedger.from_dict(legacy).to_dict() == legacy
    _, updated = original.taper_boundary(_edge_wave(tail=1e-8), z_m=.3, mode_weight=.25)
    assert updated.events[:-1] == original.events
    assert updated.events[-1]["kind"] == "numerical_boundary_taper"
    encoded = updated.to_dict()
    assert encoded["schema"] == "controlled-numerical-support-ledger-v2"
    assert NumericalSupportLedger.from_dict(encoded) == updated
    with pytest.raises(ValueError, match="schema"):
        NumericalSupportLedger.from_dict({**encoded, "schema": legacy["schema"]})
    with pytest.raises(ValueError, match="schema"):
        NumericalSupportLedger.from_dict({**legacy, "schema": encoded["schema"]})


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(transition_cells=4),
    lambda row: row.update(to_shape=[32, 32]),
    lambda row: row.update(core_indices_yx=[[3, 60], [3, 63]]),
    lambda row: row.update(origin_shift_m=[1e-20, 0.]),
    lambda row: row.update(kind="numerical_support_projection"),
    lambda row: row.update(probability_repair=True),
    lambda row: row.update(l2_per_reference=0.),
    lambda row: row.update(probability_per_reference=0.),
    lambda row: row.update(error_scope="total propagated error"),
    lambda row: row.update(window="unqualified adjustable taper"),
])
def test_boundary_taper_receipt_rejects_changed_window_or_scope(mutation):
    _, ledger = NumericalSupportLedger(.01, 1e-6).taper_boundary(_edge_wave(), z_m=.2)
    record = ledger.to_dict()
    mutation(record["events"][0])
    with pytest.raises(ValueError):
        NumericalSupportLedger.from_dict(record)
