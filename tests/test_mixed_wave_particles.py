"""Exact particle-source factory wiring with 1024 mathematical columns."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.physics import mixed_wave_particles as factory
from temsim.physics.core import PropagationCheckpoints


FIELDS = ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s")


def incident_signature(state):
    return json_digest(("mathematical-incident", state.control, state.sample.z_mm,
                        state.sample.thickness_nm))


class Snapshot:
    def __init__(self, state):
        self._state = deepcopy(state)
        self.digest = incident_signature(state)

    def restore(self):
        return deepcopy(self._state)


@pytest.fixture
def mathematical_result(monkeypatch):
    """No gun emitter, numerical field integration, or CUDA calculation runs."""
    import temsim.calculation_cache as calculation_cache
    monkeypatch.setattr(factory, "capture_instrument_snapshot", Snapshot)
    monkeypatch.setattr(calculation_cache, "calculation_signatures",
        lambda state: {"incident": incident_signature(state)})
    count = 1024
    state = SimpleNamespace(control=1., sample=SimpleNamespace(z_mm=1600., thickness_nm=10.),
                            vacuum_map=SimpleNamespace(enabled=False))
    target = factory.specimen_entrance_z_mm(state)
    rng = np.random.default_rng(731)
    emitted_ids = np.arange(count, dtype=np.int64)*7+19
    emitted_weights = np.arange(1., count+1)
    emitted_weights /= emitted_weights.sum()
    order = rng.permutation(count)
    ids, weights = emitted_ids[order], emitted_weights[order]
    phase = {name: rng.normal(size=(3, count)) for name in FIELDS}
    phase["x_m"] *= 1e-10
    phase["y_m"] *= 1e-10
    phase["tx_rad"] *= .01
    phase["ty_rad"] *= .01
    phase["kinetic_energy_ev"] = 300000.+phase["kinetic_energy_ev"]*.1
    phase["flight_time_s"] = 8e-9+abs(phase["flight_time_s"])*1e-12
    checkpoint = PropagationCheckpoints(np.array([target-.4, target, target+.2]),
        phase["x_m"], phase["tx_rad"], phase["y_m"], phase["ty_rad"],
        phase["flight_time_s"], phase["kinetic_energy_ev"])
    blocked = np.full(count, np.nan)
    blocked[:3] = (target-.1, target, target+.1)
    alive = np.ones(count, bool)
    alive[:5] = False
    branch = SimpleNamespace(ray_weight=weights, source_ray_id=ids, blocked_z=blocked,
        blocked_key=["upstream", "surface", "later", "unknown", "unknown"]+[""]*(count-5), alive=alive,
        # Deliberately unusable plotting histories must never supply the lift.
        z=np.array([target-.4, target+.2]), x=np.full((2, count), np.nan), y=np.full((2, count), np.nan))
    gun = SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=emitted_ids, weight=emitted_weights),
        emission_reference={"ray_id": emitted_ids.copy()}, emitted_current_a=1.25e-5)
    result = SimpleNamespace(state_snapshot=state,
        simulation=SimpleNamespace(incident=branch, incident_checkpoints=checkpoint, gun_trace=gun),
        signatures={"incident": incident_signature(state)})
    return result


def expected_keep(result):
    branch = result.simulation.incident
    target = factory.specimen_entrance_z_mm(result.state_snapshot)
    return ((branch.alive & ~np.isfinite(branch.blocked_z)) | (branch.blocked_z > target)) & (branch.ray_weight > 0)


def test_exact_solver_row_keeps_full_precision_identity_current_and_future_losses(mathematical_result, monkeypatch):
    monkeypatch.setattr(factory, "_continue", lambda *args: pytest.fail("Exact row was recomputed"))
    result = mathematical_result
    source = factory.particle_source_from_result(result)
    keep = expected_keep(result)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(source, name), getattr(result.simulation.incident_checkpoints, name)[1, keep])
        assert getattr(source, name).dtype == np.float64
    np.testing.assert_array_equal(source.ray_id, result.simulation.incident.source_ray_id[keep])
    np.testing.assert_array_equal(source.weight, result.simulation.incident.ray_weight[keep])
    assert not keep[0] and not keep[1] and keep[2] and not keep[3]
    assert source.original_positive_count == 1024
    assert source.reference_current_a == 1.25e-5
    assert source.reference_current_a*source.source_fraction < source.reference_current_a
    assert source.input_digest == Snapshot(result.state_snapshot).digest
    assert source.provenance["interpolated"] is False
    assert source.provenance["source_regenerated"] is False


def test_missing_plane_uses_mocked_exact_continuation_never_display_interpolation(mathematical_result, monkeypatch):
    result = mathematical_result
    checkpoints = result.simulation.incident_checkpoints
    result.simulation.incident_checkpoints = PropagationCheckpoints(checkpoints.z_mm[[0, 2]],
        *(getattr(checkpoints, name)[[0, 2]] for name in
          ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")))
    target = factory.specimen_entrance_z_mm(result.state_snapshot)
    calls = []
    keep = expected_keep(result)
    keep[5] = False  # A newly executed aperture loss is authoritative.
    phase = {name: getattr(checkpoints, name)[[0, 1]].copy() for name in FIELDS}
    phase["x_m"][1] += 1.234567890123e-10
    continued = PropagationCheckpoints(np.array([checkpoints.z_mm[0], target]),
        *(phase[name] for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")))

    def execute(state, original, row, observed, weights, active, stops, labels, cancelled):
        calls.append(row)
        assert row == 0 and observed == target
        np.testing.assert_array_equal(weights, result.simulation.incident.ray_weight)
        assert not active[3] and active[0] and active[1] and active[2]
        assert np.isnan(stops[:3]).all() and labels[:3] == ["", "", ""]
        selection = np.flatnonzero(active)
        selected_checkpoint = PropagationCheckpoints(continued.z_mm,
            *(getattr(continued, name)[:, selection] for name in
              ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")))
        return selected_checkpoint, keep[selection], selection

    monkeypatch.setattr(factory, "_continue", execute)
    source = factory.particle_source_from_result(result)
    assert calls == [0]
    np.testing.assert_array_equal(source.x_m, phase["x_m"][1, keep])
    np.testing.assert_array_equal(source.ray_id, result.simulation.incident.source_ray_id[keep])
    np.testing.assert_array_equal(source.weight, result.simulation.incident.ray_weight[keep])
    assert "continuation" in source.provenance["handoff"] and not source.provenance["interpolated"]


@pytest.mark.parametrize("failure", ["weights", "ids", "reference", "minimum", "precision", "clock"])
def test_factory_rejects_mismatched_source_ledger_and_incomplete_precision(mathematical_result, failure):
    result = mathematical_result
    branch = result.simulation.incident
    gun = result.simulation.gun_trace
    if failure == "weights":
        branch.ray_weight = branch.ray_weight*.5
    elif failure == "ids":
        branch.source_ray_id = branch.source_ray_id+1
    elif failure == "reference":
        gun.emission_reference["ray_id"] = gun.emission_reference["ray_id"][::-1]
    elif failure == "minimum":
        gun.exit_bundle.weight = gun.exit_bundle.weight.copy()
        gun.exit_bundle.weight[-25:] = 0.
    else:
        original = result.simulation.incident_checkpoints
        values = {name: getattr(original, name) for name in FIELDS}
        values["z_mm"] = original.z_mm
        values["x_m" if failure == "precision" else "flight_time_s"] = (
            original.x_m.astype(np.float32) if failure == "precision" else None)
        result.simulation.incident_checkpoints = SimpleNamespace(**values)
    with pytest.raises(ValueError):
        factory.particle_source_from_result(result)


def test_changed_snapshot_cannot_relabel_old_executed_phase_as_new_optics(mathematical_result):
    result = mathematical_result
    result.state_snapshot.control = 2.
    with pytest.raises(ValueError, match="captured|signature|incident|inputs"):
        factory.particle_source_from_result(result)


def test_cancelled_handoff_does_not_read_or_publish_source(mathematical_result):
    with pytest.raises(InterruptedError, match="cancelled"):
        factory.particle_source_from_result(mathematical_result, cancelled=lambda: True)


def test_native_continuation_consumes_stop_receipt_without_expanding_internal_state(mathematical_result, monkeypatch):
    from temsim.physics import core, instrument_magnetic, recording_clipping, unified_field_particle
    from temsim.physics import aperture_clipping, column_wall

    @dataclass
    class Plan:
        save_index: object
        checkpoint_index: object

    result = mathematical_result
    state = deepcopy(result.state_snapshot)
    state.simulation_mode = "unified"
    state.apertures = state.recording_planes = ()
    checkpoints = result.simulation.incident_checkpoints
    target = factory.specimen_entrance_z_mm(state)
    branch = result.simulation.incident
    alive = np.ones(len(branch.ray_weight), bool)
    alive[[0, 3, 4]] = False
    selected = np.flatnonzero(alive)
    stops = np.full(len(alive), np.nan)
    keys = [""]*len(alive)
    internal = SimpleNamespace(active=np.ones((1, len(selected)), bool))
    checkpoint = SimpleNamespace(unified_state=internal)
    calls = []
    monkeypatch.setattr(instrument_magnetic, "active_column_events", lambda state: ())
    monkeypatch.setattr(instrument_magnetic, "events_overlapping_interval", lambda *args: ())

    def plan(*args, **kwargs):
        assert kwargs["checkpoint_z_mm"] == (target,)
        return Plan(np.array([0, 1]), np.array([1]))

    def execute(state, plan, x, tx, y, ty, **kwargs):
        np.testing.assert_array_equal(x, checkpoints.x_m[0, selected])
        np.testing.assert_array_equal(kwargs["initial_kinetic_energy_ev"], checkpoints.kinetic_energy_ev[0, selected])
        np.testing.assert_array_equal(kwargs["initial_time_s"], checkpoints.flight_time_s[0, selected])
        assert kwargs["include_initial_plane_kicks"] is False
        z = np.array([checkpoints.z_mm[0], target])
        return z, np.stack((x, x)), np.stack((tx, tx)), np.stack((y, y)), np.stack((ty, ty)), None, checkpoint

    def native(z, live, blocked, labels):
        calls.append("native")
        live[1] = False
        blocked[1], labels[1] = target-.01, "native-posed-wall"
        return live, blocked, labels

    def recording(state, z, x, y, live, blocked, labels):
        calls.append("recording")
        assert not live[1] and labels[1] == "native-posed-wall"
        live[2] = False
        return live, blocked, labels

    monkeypatch.setattr(core, "build_propagation_plan", plan)
    monkeypatch.setattr(core, "execute_propagation_plan", execute)
    monkeypatch.setattr(unified_field_particle, "merge_unified_stops", native)
    monkeypatch.setattr(recording_clipping, "clip_recording_planes", recording)
    monkeypatch.setattr(aperture_clipping, "clip_segment", lambda *args: pytest.fail("Native loss repeated through legacy aperture clipping"))
    monkeypatch.setattr(column_wall, "clip_column_wall", lambda *args: pytest.fail("Native loss repeated through legacy wall clipping"))
    actual, live, indices = factory._continue(state, checkpoints, 0, target,
        branch.ray_weight, alive, stops, keys, lambda: False)
    assert actual is checkpoint and actual.unified_state is internal
    assert actual.unified_state.active.shape == (1, len(selected))
    np.testing.assert_array_equal(indices, selected)
    assert calls == ["native", "recording"] and not live[1] and not live[2]


def test_canonical_factory_samples_actual_xyz_and_electron_charge_units(mathematical_result, monkeypatch):
    from scipy.constants import e, h
    from temsim.physics import unified_field_runtime
    result = mathematical_result
    state = deepcopy(result.state_snapshot)
    state.simulation_mode = "unified"
    target = factory.specimen_entrance_z_mm(state)
    checkpoint = result.simulation.incident_checkpoints
    phase = {name: getattr(checkpoint, name)[1] for name in FIELDS}
    vector = np.tile([1e-9, -2e-9, 3e-9], (1024, 1))
    seen = []

    def sample(points):
        seen.append(points.copy())
        return {"A_t_m": vector}

    snapshot = SimpleNamespace(sample=sample, numerical_digest="c"*64)
    runtime = SimpleNamespace(snapshot=snapshot)
    monkeypatch.setattr(unified_field_runtime, "get_unified_field_runtime", lambda current: runtime)
    shift, provenance = factory._canonical_frequency_shift(state, phase, target)
    np.testing.assert_array_equal(seen[0], np.column_stack((phase["x_m"], phase["y_m"], np.full(1024, target*.001))))
    np.testing.assert_array_equal(shift, (-e/h*1e-10)*vector[:, :2])
    assert provenance["canonical_snapshot_digest"] == "c"*64
    assert provenance["canonical_shift_units"] == "cycles per angstrom"


def test_factory_transfers_canonical_shift_without_replacing_mechanical_slopes(mathematical_result, monkeypatch):
    result = mathematical_result
    keep = expected_keep(result)
    shift = np.tile([-.0001, .0002], (int(keep.sum()), 1))

    def canonical(state, phase, target):
        np.testing.assert_array_equal(phase["tx_rad"], result.simulation.incident_checkpoints.tx_rad[1, keep])
        return shift, {"canonical_snapshot_digest": "d"*64}

    monkeypatch.setattr(factory, "_canonical_frequency_shift", canonical)
    source = factory.particle_source_from_result(result)
    np.testing.assert_array_equal(source.canonical_frequency_shift_inv_angstrom, shift)
    np.testing.assert_array_equal(source.tx_rad, result.simulation.incident_checkpoints.tx_rad[1, keep])
    assert source.provenance["canonical_snapshot_digest"] == "d"*64
