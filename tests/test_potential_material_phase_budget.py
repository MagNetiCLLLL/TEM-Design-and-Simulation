"""Independent energy-history accounting and contractive-material admission."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import json_digest, thaw_json
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.potential_material_phase_budget import (
    capture_material_kinetic, prepare_material_kinetic, validate_material_kinetic_record,
    require_contractive_collision, material_kinetic_summary,
)


def _entrance():
    ledger = KineticPhaseLedger(.01).require("snapshot", 300000., 4, 0.).append(0., .001, (.001, .002))
    mode = SimpleNamespace(mode_id="source", energy_kev=300., scattering_history=())
    checkpoint = SimpleNamespace(record={"kinetic_phase_ledger": ledger.to_dict()},
        beam=SimpleNamespace(modes=(mode,)), plane_z_mm=1., digest="input-checkpoint")
    return checkpoint, mode, ledger


def _collision(before, loss, z):
    return {"kind": "real_plasmon", "input_energy_kev": before,
        "output_energy_kev": before-loss*.001, "energy_loss_ev": loss, "z_mm": z}


def _advance(record, end, costs):
    ledger = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
    ledger = ledger.append(ledger.last_z_m, end, costs)
    return {**record, "kinetic_phase_ledger": ledger.to_dict()}


def test_material_context_retains_operator_policy_and_rejects_a_silent_split_change(monkeypatch):
    from dataclasses import replace
    from temsim.physics.potential_wave_policy import potential_wave_operator_policy
    from temsim.physics.potential_material_phase_budget import kinetic_context
    from temsim.physics.potential_specimen_transport import PotentialMaterialTransport
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_potential_specimen_transport import _fixture, _snapshot
    state, checkpoint = _fixture()
    snapshot = _snapshot(state)
    policy = potential_wave_operator_policy({"scalar_splitting": True})
    record = {**thaw_json(checkpoint.record), "snapshot_digest": snapshot.numerical_digest,
              "potential_operator_policy": policy}
    checkpoint = replace(checkpoint, record=record)
    copied = kinetic_context(checkpoint.record)
    assert copied["potential_operator_policy"] == thaw_json(policy)
    copied["potential_operator_policy"]["policy"]["quartic_scalar_splitting"] = True
    assert checkpoint.record["potential_operator_policy"]["policy"]["quartic_scalar_splitting"] is False
    transport = PotentialMaterialTransport(snapshot,
        (checkpoint.plane_z_mm, checkpoint.plane_z_mm+.001), "operator-policy-regression",
        operator_options={"scalar_splitting": True, "quartic_scalar_splitting": True})
    monkeypatch.setattr("temsim.physics.potential_specimen_transport.observe_potential_mode",
        lambda *_a, **_k: pytest.fail("Changed numerical policy must be rejected before propagation"))
    with pytest.raises(ValueError, match="operator policy changed"):
        transport(state, checkpoint, checkpoint.plane_z_mm+.001,
            maximum_step_mm=.001, grid_numerics=WaveGridNumerics(), cancelled=lambda: False)


def test_static_acceleration_and_real_collisions_preserve_both_original_budgets():
    from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory
    from test_electric_wave_reference import _electric
    snapshot = _electric(axial_rise=.001, length=.01)
    start, entrance, collision = 0., .001, .002
    reference = ElectrostaticWaveReference.capture(snapshot, start, 300000.)
    electric = ElectricWaveHistory(reference).append(start, entrance, 1e-8)
    kinetic = KineticPhaseLedger(.01).require(snapshot.numerical_digest, 300000., 4, start).append(start, entrance, (.001,))
    mode = SimpleNamespace(mode_id="source", energy_kev=float(reference.axis_energy(snapshot, entrance))*.001,
                           scattering_history=())
    cp = SimpleNamespace(record={"kinetic_phase_ledger": kinetic.to_dict(), "electric_wave_history": electric.to_dict()},
        beam=SimpleNamespace(modes=(mode,)), plane_z_mm=entrance*1000., digest="actual-electric-prefix")
    prefix = capture_material_kinetic(cp, snapshot.numerical_digest, snapshot=snapshot)
    assert thaw_json(prefix["material_kinetic_history"]["prefix"]["electric_history"]) == electric.to_dict()
    # A changed local K from the same field is not a material collision and
    # must not close/restart the original energy-bound kinetic ledger.
    context = _advance(prefix, collision, (.002,))
    electric = electric.append(entrance, collision, 2e-8)
    context["electric_wave_history"] = electric.to_dict()
    mode.energy_kev = float(reference.axis_energy(snapshot, collision))*.001
    context, active = prepare_material_kinetic(context, mode, collision*1000., snapshot.numerical_digest, 4, snapshot=snapshot)
    assert active.binding["energy_ev"] == 300000.
    assert active.used_phase_rad == .003
    assert context["material_kinetic_history"]["completed_energy_segments"] == ()
    summaries = []
    for name, loss in (("a", 15.), ("b", 30.)):
        event = _collision(mode.energy_kev, loss, collision*1000.)
        branch = SimpleNamespace(mode_id="source/"+name, energy_kev=event["output_energy_kev"], scattering_history=(event,))
        changed, active = prepare_material_kinetic(context, branch, collision*1000., snapshot.numerical_digest, 4, snapshot=snapshot)
        current_electric = ElectricWaveHistory.from_dict(changed["electric_wave_history"])
        assert current_electric.reference.entrance_z_m == collision
        assert current_electric.reference.entrance_energy_ev == branch.energy_kev*1000.
        assert current_electric.maximum_forward_error == (electric.maximum_forward_error-electric.used_forward_error)/(1+electric.used_forward_error)
        assert active.maximum_phase_rad == .007 and active.used_phase_rad == 0.
        assert len(changed["material_kinetic_history"]["completed_energy_segments"]) == 1
        # Continued acceleration after the collision uses the NEW conserved
        # energy reference, without applying the collision loss a second time.
        stop = .003
        changed = _advance(changed, stop, (.0005,))
        current_electric = current_electric.append(collision, stop, 1e-8)
        changed["electric_wave_history"] = current_electric.to_dict()
        branch.energy_kev = float(current_electric.reference.axis_energy(snapshot, stop))*.001
        changed, again = prepare_material_kinetic(changed, branch, stop*1000., snapshot.numerical_digest, 4, snapshot=snapshot)
        assert again.used_phase_rad == .0005
        summaries.append(material_kinetic_summary(changed))
        reset = deepcopy(thaw_json(changed))
        reset["electric_wave_history"] = ElectricWaveHistory(current_electric.reference).to_dict()
        with pytest.raises(ValueError, match="histories disagree|allowance"):
            validate_material_kinetic_record(reset)
    assert summaries[0]["common_prefix_digest"] == summaries[1]["common_prefix_digest"]
    assert all(summary["history_used_phase_rad"] == .0035 for summary in summaries)
    assert all(summary["electric_forward"]["maximum_error"] == 1e-6 for summary in summaries)
    assert prefix["kinetic_phase_ledger"] == kinetic.to_dict()


def test_electric_collision_requires_actual_local_energy_and_rejects_dynamic_material_history():
    from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory
    from temsim.physics.retarded_wave import initial_dynamic_state
    from test_electric_wave_reference import _electric
    snapshot = _electric(axial_rise=.001)
    reference = ElectrostaticWaveReference.capture(snapshot, 0., 300000.)
    electric = ElectricWaveHistory(reference).append(0., 1e-4, 1e-8)
    ledger = KineticPhaseLedger(.01).require(snapshot.numerical_digest, 300000., 4, 0.).append(0., 1e-4, (.001,))
    mode = SimpleNamespace(mode_id="source", energy_kev=float(reference.axis_energy(snapshot, 1e-4))*.001, scattering_history=())
    cp = SimpleNamespace(record={"kinetic_phase_ledger": ledger.to_dict(), "electric_wave_history": electric.to_dict()},
        beam=SimpleNamespace(modes=(mode,)), plane_z_mm=.1, digest="source")
    context = capture_material_kinetic(cp, snapshot.numerical_digest, snapshot=snapshot)
    event = _collision(300., 15., .1)  # Incorrectly uses the launch energy.
    child = SimpleNamespace(mode_id="source/a", energy_kev=event["output_energy_kev"], scattering_history=(event,))
    with pytest.raises(ValueError, match="actual material collision"):
        prepare_material_kinetic(context, child, .1, snapshot.numerical_digest, 4, snapshot=snapshot)
    dynamic = ElectricWaveHistory(reference, dynamic_state=initial_dynamic_state(.2, 1e-8)).append(
        0., 1e-4, 1e-8, dynamic_state=initial_dynamic_state(.2, 1e-8))
    cp.record["electric_wave_history"] = dynamic.to_dict()
    with pytest.raises(ValueError, match="time-energy scattering"):
        capture_material_kinetic(cp, snapshot.numerical_digest, snapshot=snapshot)


def test_branches_inherit_one_exact_prefix_and_only_remaining_budget_after_real_energy_changes():
    cp, mode, original = _entrance()
    prefix = capture_material_kinetic(cp, "snapshot")
    children = []
    for name, cost in (("a", .001), ("b", .002)):
        child = SimpleNamespace(mode_id="source/"+name, energy_kev=300., scattering_history=())
        context, active = prepare_material_kinetic(prefix, child, 1., "snapshot", 4)
        assert active.to_dict() == original.to_dict()
        context = _advance(context, .002, (cost,))
        event = _collision(300., 100., 2.)
        child.energy_kev, child.scattering_history = 299.9, (event,)
        context, active = prepare_material_kinetic(context, child, 2., "snapshot", 4)
        assert active.maximum_phase_rad == .01-(.003+cost)
        assert active.binding["energy_ev"] == 299900. and active.last_z_m == .002
        assert active.used_phase_rad == 0.
        context = _advance(context, .003, (.0005,))
        summary = material_kinetic_summary(context)
        assert summary["common_prefix_used_phase_rad"] == .003
        assert summary["history_used_phase_rad"] == pytest.approx(.003+cost+.0005)
        assert summary["maximum_phase_rad"] == .01
        children.append(context)
    assert children[0]["material_kinetic_history"]["prefix"] == children[1]["material_kinetic_history"]["prefix"]
    assert prefix["kinetic_phase_ledger"] == original.to_dict()
    assert len(prefix["material_kinetic_history"]["completed_energy_segments"]) == 0


@pytest.mark.parametrize("case", ("no_collision", "wrong_loss", "old_collision", "changed_record", "different_mode"))
def test_arbitrary_energy_or_history_rewrites_cannot_allocate_a_new_budget(case):
    cp, mode, _ = _entrance()
    context = _advance(capture_material_kinetic(cp, "snapshot"), .002, (.001,))
    event = _collision(300., 100., 2.)
    mode.energy_kev = 299.9
    mode.scattering_history = () if case == "no_collision" else (event,)
    if case == "wrong_loss":
        mode.scattering_history = ({**event, "energy_loss_ev": 200.},)
    elif case == "different_mode":
        mode.mode_id = "unrelated"
    elif case in ("old_collision", "changed_record"):
        context, _ = prepare_material_kinetic(context, mode, 2., "snapshot", 4)
        if case == "old_collision":
            mode.energy_kev = 299.8
        else:
            mode.scattering_history = ({**event, "energy_loss_ev": 101.},)
    with pytest.raises(ValueError, match="collision|descend"):
        prepare_material_kinetic(context, mode, 2., "snapshot", 4)


@pytest.mark.parametrize("case", ("reset", "duplicate_segment", "prefix_event", "orphan", "orphan_native", "option_reset"))
def test_reset_duplicate_prefix_and_changed_completed_receipts_are_rejected(case):
    cp, mode, _ = _entrance()
    context = _advance(capture_material_kinetic(cp, "snapshot"), .002, (.001,))
    mode.energy_kev, mode.scattering_history = 299.9, (_collision(300., 100., 2.),)
    context, _ = prepare_material_kinetic(context, mode, 2., "snapshot", 4)
    changed = deepcopy(thaw_json(context))
    options = None
    if case == "reset":
        changed["kinetic_phase_ledger"] = KineticPhaseLedger(.01).require("snapshot", 299900., 4, .002).to_dict()
    elif case == "duplicate_segment":
        history = changed["material_kinetic_history"]
        history["completed_energy_segments"].append(deepcopy(history["completed_energy_segments"][0]))
        history["digest"] = json_digest({k: v for k, v in history.items() if k != "digest"})
    elif case == "prefix_event":
        changed["material_kinetic_history"]["prefix"]["ledger"]["events"][0]["accepted_bounds_rad"][0] = 0.
    elif case == "orphan":
        changed.pop("material_kinetic_history")
    elif case == "orphan_native":
        changed.pop("material_kinetic_history")
        changed.pop("material_kinetic_prefix_digest")
        changed["material_kinetic_native_plane"] = {"native_z_mm": 2., "executed_z_m": .002}
    else:
        options = {"kinetic_phase_ledger": KineticPhaseLedger(.01)}
    with pytest.raises(ValueError):
        validate_material_kinetic_record(changed, options)


def test_only_constant_spatial_conditioning_is_admitted_without_a_new_error_contract():
    cp, _, _ = _entrance()
    context = capture_material_kinetic(cp, "snapshot")
    distribution = SimpleNamespace(absorbed_probability=0., channels=(
        SimpleNamespace(key="real_zero_loss", probability=.7), SimpleNamespace(key="real_plasmon", probability=.3)))
    for inside in (np.ones((4, 4), bool), np.zeros((4, 4), bool)):
        require_contractive_collision(context, inside, distribution)
    mixed = np.eye(4, dtype=bool)
    with pytest.raises(ValueError, match="spatially varying conditional normalization"):
        require_contractive_collision(context, mixed, distribution)
    require_contractive_collision({}, mixed, distribution)  # Historical route is unchanged.
    identity = SimpleNamespace(absorbed_probability=0., channels=(SimpleNamespace(key="real_zero_loss", probability=1.),))
    require_contractive_collision(context, mixed, identity)


def test_energy_segment_planes_remain_exact_and_reject_a_neighbouring_native_plane():
    cp, mode, _ = _entrance()
    context = capture_material_kinetic(cp, "snapshot")
    with pytest.raises(ValueError, match="exact kinetic ledger plane"):
        prepare_material_kinetic(context, mode, np.nextafter(1., np.inf), "snapshot", 4)


def test_explicit_native_material_face_restores_only_its_exact_receipt():
    from temsim.physics.potential_material_phase_budget import (
        stamp_material_kinetic_plane, exact_material_kinetic_plane,
    )
    cp, mode, _ = _entrance()
    native_mm = 1599.2
    exact_m = native_mm*.001
    assert exact_m*1000. != native_mm
    context = _advance(capture_material_kinetic(cp, "snapshot"), exact_m, (.001,))
    with pytest.raises(ValueError, match="exact kinetic ledger plane"):
        exact_material_kinetic_plane(context, native_mm)
    stamped = stamp_material_kinetic_plane(context, native_mm)
    assert exact_material_kinetic_plane(stamped, native_mm) == exact_m
    cp.record, cp.plane_z_mm = stamped, native_mm
    assert capture_material_kinetic(cp, "snapshot") == thaw_json(stamped)
    for neighbouring in (np.nextafter(native_mm, -np.inf), np.nextafter(native_mm, np.inf)):
        with pytest.raises(ValueError, match="exact kinetic ledger plane"):
            exact_material_kinetic_plane(stamped, neighbouring)
        cp.plane_z_mm = neighbouring
        with pytest.raises(ValueError, match="exact kinetic ledger plane"):
            capture_material_kinetic(cp, "snapshot")
    mode.energy_kev = 299.9
    mode.scattering_history = (_collision(300., 100., native_mm),)
    updated, active = prepare_material_kinetic(stamped, mode, native_mm, "snapshot", 4)
    assert active.last_z_m == exact_m
    assert active.maximum_phase_rad == .01-.004
    assert exact_material_kinetic_plane(updated, native_mm) == exact_m
