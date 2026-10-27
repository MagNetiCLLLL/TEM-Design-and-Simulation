"""A conditional ensemble inherits one nonspendable support prefix."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.physics.potential_support_lineage import (
    capture_inherited_support, validate_inherited_support, common_support_summary,
)
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.potential_wave_observation import observe_potential_checkpoint
from temsim.physics.tip_gun_wave import TipGunCheckpoint, TIP_REFERENCE
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from test_potential_wave_support import _wave
from test_potential_wave_carrier import _snapshot


def _checkpoint():
    plane, ledger = NumericalSupportLedger(1e-5, 1e-10).compact(
        _wave(), wavelength_m=2e-12, z_m=.2, mode_weight=.25)
    # The test receipt came from precisely this original 0.25-weight mode.
    norm = plane.probability
    mode = WaveMode(replace(plane, amplitude=plane.amplitude/np.sqrt(norm)),
        .25*norm, TIP_REFERENCE, "parent-with-projected-tail", 300.)
    snapshot = _snapshot()
    checkpoint = TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 200., 1e-12,
        {"snapshot_digest": snapshot.numerical_digest, "support_ledger": ledger.to_dict()})
    return checkpoint, snapshot, ledger


def test_shared_prefix_preserves_receipts_once_across_multiple_conditional_histories_and_store(tmp_path):
    original, snapshot, ledger = _checkpoint()
    prefix = capture_inherited_support(original, snapshot.numerical_digest)
    children = tuple(replace(original.beam.modes[0], mode_id=f"conditional:{index}",
        weight_per_reference_electron=original.beam.total_weight/3, energy_kev=300.-index*.1)
        for index in range(3))
    branched = TipGunCheckpoint(BeamState(children, TIP_REFERENCE), 200., 1e-12,
        {"snapshot_digest": snapshot.numerical_digest, "inherited_potential_support": prefix,
         "support_prefix_digest": prefix["digest"]})
    summary = common_support_summary(branched.record)
    assert summary["accounting_multiplicity"] == 1
    assert summary["probability_removed_per_reference"] == ledger.used_probability > 0
    assert summary["prefix_l2_per_reference"] == ledger.used_l2
    assert not summary["remaining_budget_reissued"]
    assert "no conditional" in summary["scope"]
    store = ExecutedWaveStore(tmp_path, "shared-support-prefix", 1<<26)
    restored = store.put(store.key("branches"), branched)
    assert common_support_summary(restored.record) == summary
    assert restored.beam.total_weight == original.beam.total_weight
    with pytest.raises(TypeError):
        prefix["ledger"]["maximum_l2"] = 1.


def test_signed_global_entrance_retains_legal_negative_z_projection_history():
    original, snapshot, _ = _checkpoint()
    _, ledger = NumericalSupportLedger(1e-5, 1e-10).compact(
        _wave(), wavelength_m=2e-12, z_m=-.2, mode_weight=.25)
    # The lineage contract is independent of the existing Tip checkpoint's
    # positive tip-relative entrance restriction.
    signed = SimpleNamespace(record={"snapshot_digest": snapshot.numerical_digest,
        "support_ledger": ledger.to_dict()}, beam=original.beam, plane_z_mm=-200.,
        digest="controlled-signed-global-coordinate")
    prefix = capture_inherited_support(signed, snapshot.numerical_digest)
    record = {"snapshot_digest": snapshot.numerical_digest, "inherited_potential_support": prefix}
    assert validate_inherited_support(record)["entrance_z_mm"] == -200.
    assert common_support_summary(record)["probability_removed_per_reference"] == ledger.used_probability


@pytest.mark.parametrize("entrance_z_mm", (-201., np.nextafter(-200., -np.inf)))
def test_future_projection_event_cannot_be_inherited_at_an_earlier_entrance(entrance_z_mm):
    original, snapshot, _ = _checkpoint()
    _, ledger = NumericalSupportLedger(1e-5, 1e-10).compact(
        _wave(), wavelength_m=2e-12, z_m=-.2, mode_weight=.25)
    signed = SimpleNamespace(record={"snapshot_digest": snapshot.numerical_digest,
        "support_ledger": ledger.to_dict()}, beam=original.beam, plane_z_mm=entrance_z_mm,
        digest="controlled-invalid-time-order")
    with pytest.raises(ValueError, match="after the declared entrance"):
        capture_inherited_support(signed, snapshot.numerical_digest)


@pytest.mark.parametrize("mutation", ("ledger", "identity", "field", "prefix"))
def test_tampered_or_mismatched_support_prefix_is_rejected(mutation):
    original, snapshot, _ = _checkpoint()
    prefix = thaw_json(capture_inherited_support(original, snapshot.numerical_digest))
    record = {"snapshot_digest": snapshot.numerical_digest, "inherited_potential_support": prefix,
              "support_prefix_digest": prefix["digest"]}
    if mutation == "ledger":
        prefix["ledger"]["events"][0]["probability_per_reference"] = 0.
    elif mutation == "identity":
        prefix["entrance_checkpoint_digest"] = "other"
    elif mutation == "field":
        record["snapshot_digest"] = "other"
    else:
        record["support_prefix_digest"] = "other"
    with pytest.raises(ValueError):
        validate_inherited_support(record)


@pytest.mark.parametrize("options", ({"support_ledger": NumericalSupportLedger(1e-5, 1e-10)},
    {"support_mode_weight": 1.}, {"support_projection_policy": "sampling_retry"}))
def test_observer_cannot_allocate_a_new_projection_budget_after_material_branching(options):
    original, snapshot, _ = _checkpoint()
    prefix = capture_inherited_support(original, snapshot.numerical_digest)
    checkpoint = replace(original, record={"snapshot_digest": snapshot.numerical_digest,
        "inherited_potential_support": prefix, "support_prefix_digest": prefix["digest"]})
    with pytest.raises(ValueError, match="projection is disabled"):
        observe_potential_checkpoint(snapshot, checkpoint, (.200001,), **options)
