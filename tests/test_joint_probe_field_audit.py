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


def _set_expected_aperture(fixture, diameter_mm):
    from dataclasses import replace
    state, plan = fixture
    state.apertures[0].diameter_mm = diameter_mm
    state.apertures[0].radius_mm = diameter_mm*.5
    runtime = plan.unified_runtime
    host = runtime.geometry.boundaries[0]
    host.pieces = (replace(host.pieces[0], radius_mm=diameter_mm*.5),)
    runtime.source_identity = audit_module.capture_instrument_snapshot(state).digest
    return state, plan


def test_explicit_70um_audit_checks_state_native_plate_and_full_ownership(fixture):
    """Synthetic plan and native mask operations; no finite-source transport."""
    state, plan = _set_expected_aperture(fixture, .070)
    before = deepcopy(state)
    report = audit_module.audit(state, plan, expected_c2_diameter_mm=.070)
    assert report["passed"]
    assert report["c2_aperture"]["expected_diameter_mm"] == .070
    assert report["c2_aperture"]["actual_boundary_radius_mm"] == .035
    assert report["identities"]["geometry_digest"] == plan.unified_runtime.geometry.digest
    assert report["plan_actions"]["interaction_count"] == 3
    plate = plan.unified_runtime.geometry.boundaries[0].pieces[0]
    np.testing.assert_array_equal(plate.transmission_mask(np.array([.034, .036]), np.zeros(2), 3.5),
                                  [True, False])
    assert state == before
    with pytest.raises(ValueError, match="C2 diameter"):
        audit_module.audit(state, plan)  # No inference from the current opening.


@pytest.mark.parametrize("damage", ["state_diameter", "state_radius", "captured_plate", "stale_plan", "missing_owner"])
def test_explicit_diameter_does_not_weaken_geometry_state_or_plan_checks(fixture, damage):
    from dataclasses import replace
    state, plan = _set_expected_aperture(fixture, .070)
    runtime = plan.unified_runtime
    if damage == "state_diameter":
        state.apertures[0].diameter_mm = .100
    elif damage == "state_radius":
        state.apertures[0].radius_mm = .05
    elif damage == "captured_plate":
        host = runtime.geometry.boundaries[0]
        host.pieces = (replace(host.pieces[0], radius_mm=.05),)
    elif damage == "stale_plan":
        runtime.source_identity = "old-100um-state"
    else:
        runtime.snapshot.contributions = runtime.snapshot.contributions[1:]
        plan.physics_contract["interactions"]["records"] = plan.physics_contract["interactions"]["records"][1:]
    with pytest.raises(ValueError):
        audit_module.audit(state, plan, expected_c2_diameter_mm=.070)


@pytest.mark.parametrize("expected", [0., -.07, np.nan, np.inf, True, None])
def test_expected_diameter_must_be_explicit_positive_and_finite(fixture, expected):
    state, plan = fixture
    with pytest.raises(ValueError, match="Expected C2 diameter"):
        audit_module.audit(state, plan, expected_c2_diameter_mm=expected)


def test_real_two_condenser_70um_opening_matches_native_contact_and_component_mask(monkeypatch):
    """Capture mechanics and test array masks without emitting or transporting a source."""
    from dataclasses import replace
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    from temsim.optics.electron_gun import source
    from temsim.physics.aperture_clipping import _transmits
    from temsim.physics.posed_wave_aperture import PosedWaveAperture
    from temsim.physics.potential_geometry import capture_potential_geometry

    def forbidden(*args, **kwargs):
        pytest.fail("Static geometry validation must not emit or transport particles")

    monkeypatch.setattr(ColdFieldEmitter, "emit", forbidden)
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), column="C2", recording="No Energy Filter"))
    physical = {part.key for part in state._resolved_assembly.parts}
    assert {"condenser_lens_1", "condenser_lens_2"} <= physical
    assert "condenser_lens_3" not in physical
    aperture = state.condenser_aperture_2
    aperture.diameter_um = 70.
    assert aperture.diameter_mm == .070 and aperture.radius_mm == .035
    start = float(state.electron_gun.exit_plane_z_mm)
    stop = float(state.sample.upper_surface_z_mm)
    geometry = capture_potential_geometry(state, start*1e-3, stop*1e-3)
    report = audit_module._c2_geometry(aperture, NS(geometry=geometry), start, stop,
                                      expected_c2_diameter_mm=.070)
    assert report["actual_boundary_radius_mm"] == .035
    plate, = (piece for host in geometry.boundaries for piece in host.pieces
              if piece.key == aperture.key and type(piece) is PosedWaveAperture)
    assert plate.local_end_z_mm-plate.local_start_z_mm == pytest.approx(aperture.plate_thickness_mm)
    x, y = np.array([.034, .036]), np.zeros(2)
    expected = np.array([True, False])
    np.testing.assert_array_equal(_transmits(aperture, x, y), expected)
    np.testing.assert_array_equal(plate.transmission_mask(x, y, aperture.z_mm), expected)
    before = np.column_stack((x, y, np.full(2, plate.local_start_z_mm-.1)))
    after = np.column_stack((x, y, np.full(2, plate.local_end_z_mm+.1)))
    contacts = plate.first_contact_fraction(before, after)
    assert np.isinf(contacts[0]) and 0. < contacts[1] < 1.
    with pytest.raises(ValueError, match="operating opening"):
        audit_module._c2_geometry(aperture, NS(geometry=geometry), start, stop)
