"""A global kinetic phase bound cannot be reset by cuts, retries or storage."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.potential_wave_carrier import propagate_carrier_potential
from temsim.physics.potential_wave_observation import observe_potential_checkpoint, PotentialAperturePlane
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_flux import BeamState
from test_potential_wave_carrier import _wave, _snapshot
from test_potential_wave_observation import _mode


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_every_supported_kinetic_order_retains_its_original_budget_and_identity(order):
    original = KineticPhaseLedger(.01).require("fixed-field", 300000., order, .45)
    stored = original.append(.45, .46, (.001, .002))
    restored = KineticPhaseLedger.from_dict(stored.to_dict())
    assert restored.binding["kinetic_order"] == order
    assert restored.maximum_phase_rad == .01
    assert restored.used_phase_rad == .003
    assert restored.require("fixed-field", 300000., order, .46) is restored
    for changed in (value for value in (2, 4, 6, 8) if value != order):
        with pytest.raises(ValueError, match="operator order changed"):
            restored.require("fixed-field", 300000., changed, .46)


@pytest.mark.parametrize("order", [True, 4., np.int64(4), "4", 3, 10])
def test_existing_bound_phase_ledger_also_requires_a_strict_supported_integer_order(order):
    ledger = KineticPhaseLedger(.01).require("fixed-field", 300000., 4, .45)
    with pytest.raises(ValueError, match="kinetic_order"):
        ledger.require("fixed-field", 300000., order, .45)


def test_ledger_roundtrip_is_immutable_strict_and_contiguous():
    ledger = KineticPhaseLedger(.01).require("native", 300000., 4, .6)
    first = ledger.append(.6, .7, (.001, .002))
    second = first.append(.7, .8, (.003,))
    assert ledger.used_phase_rad == 0.
    assert second.used_phase_rad == math.fsum((.001, .002, .003))
    assert KineticPhaseLedger.from_dict(second.to_dict()) == second
    with pytest.raises(TypeError):
        second.events[0]["upper_z_m"] = .9
    with pytest.raises(ValueError, match="cumulative"):
        second.append(.8, .9, (.005,))
    with pytest.raises(ValueError, match="same observation"):
        second.append(.9, 1., (.001,))
    for arguments in (("different", 300000., 4, .8), ("native", 200000., 4, .8),
                      ("native", 300000., 2, .8), ("native", 300000., 4, .81)):
        with pytest.raises(ValueError):
            second.require(*arguments)
    corrupted = second.to_dict()
    corrupted["events"][0]["accepted_bounds_rad"][0] = 0.
    with pytest.raises(ValueError, match="digest"):
        KineticPhaseLedger.from_dict(corrupted)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1.])
def test_invalid_phase_budget_and_spend_are_rejected(value):
    with pytest.raises(ValueError):
        KineticPhaseLedger(value)
    with pytest.raises(ValueError):
        KineticPhaseLedger(.01).require("native", 300000., 4, 0.).append(0., 1., (value,))


def test_cumulative_allocation_accepts_local_peak_without_changing_operator_or_total_bound():
    wave, snapshot = _wave(n=32, width=2e-7, tilt=(.001, 0.)), _snapshot()
    nodes = (0., 1e-6, 1e-4)
    expected, evidence = propagate_carrier_potential(snapshot, wave, nodes, energy_ev=200000.)
    bounds = [row["kinetic_remainder_phase_bound"] for row in evidence["residual_steps"]]
    limit = 1.1*math.fsum(bounds)
    with pytest.raises(ValueError, match="kinetic truncation"):
        propagate_carrier_potential(snapshot, wave, nodes, energy_ev=200000., kinetic_remainder_phase_limit=limit)
    actual, record = propagate_carrier_potential(snapshot, wave, nodes, energy_ev=200000.,
                                                kinetic_phase_ledger=KineticPhaseLedger(limit))
    for key in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    ledger = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    assert ledger.used_phase_rad == math.fsum(bounds) < limit
    assert len(ledger.events[0]["accepted_bounds_rad"]) == 4


def test_failed_sampling_candidate_does_not_spend_phase_budget(monkeypatch):
    import temsim.physics.potential_wave_carrier as module
    from temsim.physics.wave_grid import WaveSamplingError
    original, calls = module._residual, []
    def attempt(*args, **kwargs):
        output, record = original(*args, **kwargs)
        calls.append(record["kinetic_remainder_phase_bound"])
        if len(calls) == 1:
            raise WaveSamplingError("Deliberate resolved retry", 1.1)
        return output, record
    monkeypatch.setattr(module, "_residual", attempt)
    _, record = module.propagate_carrier_potential(_snapshot(), _wave(width=2e-7), (0., 1e-5),
        energy_ev=200000., kinetic_phase_ledger=KineticPhaseLedger(.01))
    ledger = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    assert len(calls) == 3
    assert ledger.used_phase_rad == math.fsum(calls[1:])


def test_checkpoint_phase_budget_survives_storage_cuts_and_rejects_reset(tmp_path):
    snapshot = _snapshot()
    entrance = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), 1000., 1e-12, {})
    options = dict(maximum_step_m=1e-6)
    initial = KineticPhaseLedger(.01)
    middle, = observe_potential_checkpoint(snapshot, entrance, (1.000002,), kinetic_phase_ledger=initial, **options)
    stored = KineticPhaseLedger.from_dict(middle.record["kinetic_phase_ledger"])
    assert stored.used_phase_rad > 0.
    store = ExecutedWaveStore(tmp_path, snapshot.numerical_digest, 16*1024**2)
    key = store.key("phase-budget", middle.digest)
    store.put(key, middle)
    resumed, = observe_potential_checkpoint(snapshot, store.get(key), (1.000004,), **options)
    final = KineticPhaseLedger.from_dict(resumed.record["kinetic_phase_ledger"])
    assert final.events[:len(stored.events)] == stored.events
    assert stored.used_phase_rad < final.used_phase_rad < .01
    for replacement in (initial, None, KineticPhaseLedger(.02)):
        with pytest.raises(ValueError, match="cannot be reset"):
            observe_potential_checkpoint(snapshot, middle, (1.000004,), kinetic_phase_ledger=replacement, **options)
    modes = replace(entrance, beam=BeamState((_mode(), replace(_mode(), mode_id="second")), TIP_REFERENCE))
    with pytest.raises(ValueError, match="one mode"):
        observe_potential_checkpoint(snapshot, modes, (1.000002,), kinetic_phase_ledger=initial, **options)


def test_fully_absorbed_mode_advances_zero_cost_phase_history():
    snapshot = _snapshot()
    entrance = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), 1000., 1e-12, {})
    aperture = PotentialAperturePlane("blocking", 1., 1e-8, (1., 0.))
    first, second = observe_potential_checkpoint(snapshot, entrance, (1.000001, 1.000002),
        kinetic_phase_ledger=KineticPhaseLedger(.01), apertures=(aperture,))
    assert first.beam.total_weight == second.beam.total_weight == 0.
    ledger = KineticPhaseLedger.from_dict(second.record["kinetic_phase_ledger"])
    assert ledger.used_phase_rad == 0.
    result, = observe_potential_checkpoint(snapshot, second, (1.000003,))
    assert result.beam.total_weight == 0.
    assert KineticPhaseLedger.from_dict(result.record["kinetic_phase_ledger"]).events[-1]["upper_z_m"] == 1.000003


def test_low_level_distinct_float_planes_reject_but_verified_checkpoint_mm_restores_exact_metres():
    z = .7
    rounded = z*1000.*.001
    assert rounded != z
    ledger = KineticPhaseLedger(.01).require("native", 300000., 4, z-.001).append(z-.001, z, (1e-5,))
    with pytest.raises(ValueError, match="different observation"):
        ledger.require("native", 300000., 4, rounded)
    with pytest.raises(ValueError, match="same observation"):
        ledger.append(rounded, z+.001, (1e-5,))
    snapshot = _snapshot()
    bound = replace(ledger, binding={**dict(ledger.binding), "snapshot_digest": snapshot.numerical_digest})
    checkpoint = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), z*1000., 1e-12,
        {"snapshot_digest": snapshot.numerical_digest, "kinetic_phase_ledger": bound.to_dict()})
    result, = observe_potential_checkpoint(snapshot, checkpoint, (z+1e-6,), maximum_step_m=1e-6)
    events = KineticPhaseLedger.from_dict(result.record["kinetic_phase_ledger"]).events
    assert events[-1]["lower_z_m"] == z
    with pytest.raises(ValueError, match="Checkpoint plane"):
        observe_potential_checkpoint(snapshot, replace(checkpoint, plane_z_mm=np.nextafter(z*1000., np.inf)),
                                      (z+1e-6,), maximum_step_m=1e-6)


def test_explicit_material_native_receipt_cannot_be_bypassed_by_canonical_mm_identity():
    from temsim.physics.potential_material_phase_budget import capture_material_kinetic, stamp_material_kinetic_plane
    snapshot = _snapshot()
    native_mm = 1599.2
    exact_m = native_mm*.001
    assert exact_m*1000. != native_mm
    ledger = KineticPhaseLedger(.01).require(snapshot.numerical_digest, 300000., 4, .6).append(.6, exact_m, (.001,))
    checkpoint = TipGunCheckpoint(BeamState((_mode(),), TIP_REFERENCE), exact_m*1000., 1e-12,
        {"snapshot_digest": snapshot.numerical_digest, "kinetic_phase_ledger": ledger.to_dict()})
    context = stamp_material_kinetic_plane(capture_material_kinetic(checkpoint, snapshot.numerical_digest), native_mm)
    checkpoint = replace(checkpoint, plane_z_mm=native_mm,
        record={**context, "snapshot_digest": snapshot.numerical_digest})
    output, = observe_potential_checkpoint(snapshot, checkpoint, (exact_m+1e-6,), maximum_step_m=1e-6)
    assert output.record["material_kinetic_history"] == checkpoint.record["material_kinetic_history"]
    with pytest.raises(ValueError, match="exact kinetic ledger plane"):
        observe_potential_checkpoint(snapshot, replace(checkpoint, plane_z_mm=exact_m*1000.),
            (exact_m+1e-6,), maximum_step_m=1e-6)
    with pytest.raises(ValueError):
        observe_potential_checkpoint(snapshot, replace(checkpoint, plane_z_mm=exact_m*1000.,
            record={**dict(checkpoint.record), "material_kinetic_native_plane": None}),
            (exact_m+1e-6,), maximum_step_m=1e-6)
