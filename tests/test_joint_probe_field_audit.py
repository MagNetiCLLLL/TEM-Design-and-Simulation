"""Structural audit contracts using synthetic plans; no physical propagation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_joint_probe_fields.py"
_SPEC = importlib.util.spec_from_file_location("joint_probe_field_audit_tests", _PATH)
audit_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_module)


@pytest.fixture
def fixture(monkeypatch):
    from temsim.physics.lens_field_provider import CoordinateRegistration
    from temsim.physics.posed_wave_aperture import PosedWaveAperture

    aperture = NS(key="condenser_aperture_2", installed=True, enabled=True, inserted=True,
        diameter_mm=.1, radius_mm=.05)
    hp = NS(key="probe_hp1_hexapole", enabled=True, z_mm=6., strength_m3=2.6e6,
        maximum_strength_m3=1e9, orientation_rad=.1, effective_length_mm=2.,
        mechanical_length_mm=3., mechanical_clear_bore_diameter_mm=1.)
    lens = NS(key="objective_lens", enabled=True, z_mm=10., percent=60., polarity=1., cs_mm=123.)
    state = NS(unified=True, apertures=[aperture], sample=NS(upper_surface_z_mm=10., wave_enabled=False),
        electron_gun=NS(source_representation="classical_particles", exit_plane_z_mm=1.,
            nominal_exit_energy_ev=300000., emitted_current_a=1e-8,
            emitter=NS(ray_count=193, energy_spread_fwhm_ev=.3, virtual_source_fwhm_nm=1.)),
        beam_voltage_kv=300., column_current_limit_percent=100.,
        lenses=[lens], corrector_elements=[hp], stigmators=[])

    class Snapshot:
        def __init__(self, inputs):
            self.state = deepcopy(inputs)
            from temsim.immutable_json import json_digest
            self.digest = json_digest(json.loads(json.dumps(vars(inputs), default=vars)))

        def restore(self):
            return deepcopy(self.state)

    class Runtime(NS):
        def require_state(self, inputs):
            if Snapshot(inputs).digest != self.source_identity:
                raise ValueError("Stale state")

    class Plan(NS):
        pass

    rows = tuple(NS(source_id=key, owner_ids=(key,), kind=kind, potential=NS(digest=key+"-digest"))
        for key, kind in (("objective_lens", "magnetic"), ("probe_hp1_hexapole", "magnetic"),
                          ("instrument_electric", "electric")))
    piece = PosedWaveAperture("condenser_aperture_2", 3., 4., .05, CoordinateRegistration())
    runtime = Runtime(source_identity=Snapshot(state).digest, digest="runtime", metadata={
        "empirical_aberrations_applied": False, "equivalent_image_maps_applied": False,
        "round_radial_order": 3}, snapshot=NS(contributions=rows, numerical_digest="fields", scene=None),
        geometry=NS(digest="geometry", aperture_inventory=({"key": aperture.key, "installed": True,
            "enabled": True, "inserted": True},), boundaries=(NS(host_key="c2-host", pieces=(piece,)),)))
    contract = {"mode": "unified", "interval_mm": [1., 10.], "unified_capture": {
        "runtime": "runtime", "field": "fields", "geometry": "geometry"}, "interactions": {"records": [
            {"interaction_id": row.source_id, "representation": "field", "effect_family": "electromagnetic_potential",
             "source_digest": row.potential.digest, "covered_effects": [row.owner_ids[0]+":"+row.kind]} for row in rows]}}
    plan = Plan(unified_runtime=runtime, solver_signature="unified-potential-rk4-v1:runtime", signature="plan",
        z_mm=np.array([1., 5., 10.]), physics_contract=contract, electric_field=None, electric_field_identity=None,
        **{name: np.zeros(3) for name in audit_module._EXTRA_ARRAYS},
        **{name: () for name in audit_module._EXTRA_LISTS})
    monkeypatch.setattr(audit_module, "capture_instrument_snapshot", Snapshot)
    monkeypatch.setattr(audit_module, "UnifiedFieldRuntime", Runtime)
    monkeypatch.setattr(audit_module, "AxialPropagationPlan", Plan)
    monkeypatch.setattr(audit_module, "is_unified", lambda item: item.unified)
    monkeypatch.setattr(audit_module, "validate_plan_contract", lambda value: None)
    monkeypatch.setattr(audit_module, "_canonical_owner", lambda scene, owner: owner)
    monkeypatch.setattr(audit_module, "unified_source_selection", lambda _: ((lens.key,), (hp.key,)))
    monkeypatch.setattr(audit_module, "active_column_events", lambda _: ())
    monkeypatch.setattr(audit_module, "build_propagation_plan", lambda *a: pytest.fail("Unexpected plan construction"))
    return state, plan


def test_actual_actions_not_inactive_legacy_coefficients(fixture):
    state, plan = fixture
    original = deepcopy(state)
    report = audit_module.audit(state, plan)
    assert report["schema"] == "temsim-joint-probe-field-audit-v1"
    assert report["passed"] is True and report["status"] == "PASS"
    assert report["c2_aperture"]["diameter_mm"] == .1
    assert report["c2_aperture"]["actual_boundary_radius_mm"] == .05
    assert report["controls"]["corrector_elements"][0]["strength_m3"] == 2.6e6
    assert state.lenses[0].cs_mm == 123.
    assert report["field_ownership"]["natural_field_aberrations_retained"] is True
    assert report["source"]["ray_count"] == 193
    assert report["execution_model"]["particle_propagation_performed_by_audit"] is False
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert state == original


def test_missing_plan_builds_only_on_restored_inputs(fixture, monkeypatch):
    state, plan = fixture
    calls = []
    def build(private, start, end, events):
        assert private is not state and private == state
        calls.append((start, end, events))
        return plan
    monkeypatch.setattr(audit_module, "build_propagation_plan", build)
    report = audit_module.audit(state)
    assert calls == [(1., 10., ())]
    assert report["execution_model"]["audit_method"] == "private_production_plan_construction"


@pytest.mark.parametrize("name,value", [("diameter_mm", .09), ("radius_mm", .04),
    ("enabled", False), ("installed", False), ("inserted", False)])
def test_bad_input_aperture_rejected(fixture, name, value):
    state, plan = fixture
    setattr(state.apertures[0], name, value)
    with pytest.raises(ValueError, match="C2"):
        audit_module.audit(state, plan)


@pytest.mark.parametrize("case", ["legacy", "coherent_source", "coherent_sample"])
def test_wrong_execution_model_rejected(fixture, case):
    state, plan = fixture
    if case == "legacy":
        state.unified = False
    elif case == "coherent_source":
        state.electron_gun.source_representation = "coherent_wave"
    else:
        state.sample.wave_enabled = True
    with pytest.raises(ValueError):
        audit_module.audit(state, plan)


@pytest.mark.parametrize("name", audit_module._EXTRA_ARRAYS)
def test_actual_additional_array_rejected(fixture, name):
    state, plan = fixture
    getattr(plan, name)[1] = 1e-30
    with pytest.raises(ValueError, match="Additional"):
        audit_module.audit(state, plan)


@pytest.mark.parametrize("name", audit_module._EXTRA_LISTS)
def test_actual_additional_list_rejected(fixture, name):
    state, plan = fixture
    setattr(plan, name, ("unexpected",))
    with pytest.raises(ValueError, match="Additional"):
        audit_module.audit(state, plan)


@pytest.mark.parametrize("case", ["missing_flag", "active_empirical", "extra_electric", "stale_state",
    "wrong_surface", "bad_radius", "missing_aperture", "missing_owner", "duplicate_owner", "empirical_interaction"])
def test_missing_or_contradictory_runtime_evidence_fails_closed(fixture, case):
    state, plan = fixture
    runtime = plan.unified_runtime
    if case == "missing_flag":
        del runtime.metadata["empirical_aberrations_applied"]
    elif case == "active_empirical":
        runtime.metadata["empirical_aberrations_applied"] = True
    elif case == "extra_electric":
        plan.electric_field = object()
    elif case == "stale_state":
        state.lenses[0].percent = 61.
    elif case == "wrong_surface":
        plan.z_mm[-1] -= .01
    elif case == "bad_radius":
        from dataclasses import replace
        host = runtime.geometry.boundaries[0]
        host.pieces = (replace(host.pieces[0], radius_mm=.04),)
    elif case == "missing_aperture":
        runtime.geometry.boundaries = ()
    elif case == "missing_owner":
        runtime.snapshot.contributions = runtime.snapshot.contributions[1:]
        plan.physics_contract["interactions"]["records"] = plan.physics_contract["interactions"]["records"][1:]
    elif case == "duplicate_owner":
        runtime.snapshot.contributions[1].owner_ids = ("objective_lens",)
        plan.physics_contract["interactions"]["records"][1]["covered_effects"] = ["objective_lens:magnetic"]
    else:
        plan.physics_contract["interactions"]["records"][0]["effect_family"] = "Cs"
    with pytest.raises(ValueError):
        audit_module.audit(state, plan)


def test_private_construction_cannot_silently_retune(fixture, monkeypatch):
    state, plan = fixture
    def build(private, *args):
        private.lenses[0].percent += 1.
        return plan
    monkeypatch.setattr(audit_module, "build_propagation_plan", build)
    with pytest.raises(ValueError, match="Stale state"):
        audit_module.audit(state)
    assert state.lenses[0].percent == 60.


@pytest.mark.parametrize("missing_second", [False, True])
def test_logical_deflector_requires_every_actual_coil(fixture, monkeypatch, missing_second):
    state, plan = fixture
    state.deflectors = [NS(key="ac_deflector", enabled=True,
        upper_z_mm=7., lower_z_mm=8., upper_x_mrad=0., upper_y_mrad=0., lower_x_mrad=0., lower_y_mrad=0.)]
    runtime = plan.unified_runtime
    runtime.source_identity = audit_module.capture_instrument_snapshot(state).digest
    monkeypatch.setattr(audit_module, "unified_source_selection", lambda _: (
        ("objective_lens",), ("probe_hp1_hexapole", "ac_deflector")))
    for index in range(1 if missing_second else 2):
        key = f"ac_deflector:{index}"
        row = NS(source_id=key, owner_ids=(key,), kind="magnetic", potential=NS(digest=key+"-digest"))
        runtime.snapshot.contributions += (row,)
        plan.physics_contract["interactions"]["records"].append({"interaction_id": key,
            "representation": "field", "effect_family": "electromagnetic_potential",
            "source_digest": row.potential.digest, "covered_effects": [key+":magnetic"]})
    if missing_second:
        with pytest.raises(ValueError, match="ac_deflector:1"):
            audit_module.audit(state, plan)
    else:
        report = audit_module.audit(state, plan)
        assert report["field_ownership"]["required_magnetic_owners"]["ac_deflector"] == ["ac_deflector:0", "ac_deflector:1"]


def test_shared_host_body_bore_is_not_an_extra_optical_plate(fixture):
    from temsim.physics.posed_wave_hardware import PosedWaveBore
    state, plan = fixture
    host = plan.unified_runtime.geometry.boundaries[0]
    plate = host.pieces[0]
    body = PosedWaveBore(plate.key, 2., 5., 2.5, plate.registration)
    host.pieces = (body, plate)
    assert audit_module.audit(state, plan)["c2_aperture"]["actual_boundary_radius_mm"] == .05
    host.pieces = (body,)
    with pytest.raises(ValueError, match="finite C2 aperture"):
        audit_module.audit(state, plan)
