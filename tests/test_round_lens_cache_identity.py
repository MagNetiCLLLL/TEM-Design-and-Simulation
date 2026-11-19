"""Round-source cache invalidation with configuration and mocked operators only."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from temsim.calculation_cache import calculation_signatures, incident_field_dependencies, matching_products
from temsim.optics.aberrations import _aberration_cache_signature, effective_aberration_comparison
from temsim.round_lens_models import set_gaussian_geometry_coupling, set_round_lens_field_model


@pytest.fixture(autouse=True)
def forbid_physical_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cache identity checks must not execute an emitter, Gun, column or wave")
    from temsim.optics.electron_gun.emitter import ColdFieldEmitter
    monkeypatch.setattr(ColdFieldEmitter, "emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)
    monkeypatch.setattr("temsim.physics.core.propagate", forbidden)
    monkeypatch.setattr("temsim.physics.simulation.run", forbidden)


@pytest.fixture(scope="module")
def payload():
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    state = default_state()
    switch_mode(state, "unified")
    return state.to_dict()


@pytest.fixture
def state(payload):
    from temsim.optics.model import State
    return State.from_dict(deepcopy(payload))


def _edit_part(state, key, **changes):
    parts = tuple(replace(part, data={**part.data, **changes}) if part.key == key else part
                  for part in state._resolved_assembly.parts)
    state._resolved_assembly = replace(state._resolved_assembly, parts=parts)


def _edit_source(state, change, key="diffraction_lens"):
    if change == "excitation":
        next(lens for lens in state.lenses if lens.key == key).percent += 1.
    elif change in {"ampere_turns", "gaussian_reference"}:
        part = next(part for part in state._resolved_assembly.parts if part.key == key)
        field = "air_core_reference_ampere_turns" if change == "ampere_turns" else "gaussian_reference_coil_length_mm"
        _edit_part(state, key, **{field: float(part.data[field])*1.1})
    elif change == "pose":
        _edit_part(state, key, offset_x_mm=.01, rotation_y_mrad=.1)
    else:
        from temsim.magnetic_circuits import optical_owner
        parts = {part.key: part for part in state._resolved_assembly.parts}
        coil = next(part for part in parts.values()
                    if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
                    and optical_owner(part, parts) == key)
        if change == "coil_axial":
            changed = replace(coil, end_z_mm=coil.end_z_mm+.125)
            state._resolved_assembly = replace(state._resolved_assembly,
                parts=tuple(changed if part.key == coil.key else part for part in parts.values()))
        else:
            if change == "wire_area":
                _edit_part(state, coil.key, winding_wire_cross_section_mm2=.125)
            else:
                field = "mechanical_inner_diameter_mm" if change == "coil_inner" else "mechanical_outer_diameter_mm"
                _edit_part(state, coil.key, **{field: float(coil.data[field])+.125})


@pytest.mark.parametrize("key", ["diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"])
def test_every_post_specimen_copper_tail_invalidates_incident_and_scattering(state, key):
    set_round_lens_field_model(state, "copper_coil")
    before = calculation_signatures(state)
    assert key in {row["key"] for row in incident_field_dependencies(state)["rows"]}
    _edit_source(state, "excitation", key)
    after = calculation_signatures(state)
    reused = matching_products(before, after)
    assert not {"incident", "elastic", "eds", "sample_region"} & reused


@pytest.mark.parametrize("change", ["ampere_turns", "coil_inner", "coil_outer", "coil_axial", "pose", "wire_area"])
def test_copper_winding_metadata_and_pose_invalidate_stage_and_aberration_caches(state, change):
    set_round_lens_field_model(state, "copper_coil")
    before = calculation_signatures(state)
    aberration = _aberration_cache_signature(state, "image")
    _edit_source(state, change)
    after = calculation_signatures(state)
    assert not {"incident", "elastic", "eds", "sample_region"} & matching_products(before, after)
    assert _aberration_cache_signature(state, "image") != aberration


@pytest.mark.parametrize("change", ["coil_axial", "gaussian_reference", "excitation"])
def test_linked_gaussian_source_keeps_downstream_geometry_in_incident_identity(state, change):
    set_gaussian_geometry_coupling(state, "axial_length")
    before = calculation_signatures(state)
    aberration = _aberration_cache_signature(state, "image")
    _edit_source(state, change)
    assert calculation_signatures(state)["incident"] != before["incident"]
    assert _aberration_cache_signature(state, "image") != aberration


def test_fixed_gaussian_retains_historical_independent_downstream_control_reuse(state):
    before = calculation_signatures(state)
    _edit_source(state, "ampere_turns")
    assert calculation_signatures(state)["incident"] == before["incident"]


def test_source_selection_clears_derived_caches_and_is_in_signature_without_setter(state):
    before = _aberration_cache_signature(state, "image")
    state._effective_aberration_cache = {"image": ("old", object())}
    state._multiplane_transfer_cache = ("old", {1.: object()})
    set_round_lens_field_model(state, "copper_coil")
    assert not hasattr(state, "_effective_aberration_cache")
    assert not hasattr(state, "_multiplane_transfer_cache")
    assert _aberration_cache_signature(state, "image") != before
    state.round_lens_field_model = "gaussian"
    assert _aberration_cache_signature(state, "image") == before
    state.gaussian_geometry_coupling = "axial_length"
    assert _aberration_cache_signature(state, "image") != before


def test_field_derived_fit_is_recomputed_after_source_selection_and_winding_edit(state, monkeypatch):
    state.image_aberrations = {"mode": "field_derived"}
    calls = []
    def fit(*_args):
        result = object()
        calls.append(result)
        return result
    monkeypatch.setattr("temsim.optics.field_aberrations.derive_field_aberrations", fit)
    first = effective_aberration_comparison(state, "image")
    assert effective_aberration_comparison(state, "image") is first
    set_round_lens_field_model(state, "copper_coil")
    second = effective_aberration_comparison(state, "image")
    _edit_source(state, "ampere_turns")
    third = effective_aberration_comparison(state, "image")
    assert len(calls) == 3 and second is not first and third is not second


def test_canonical_transfer_cache_recomputes_after_winding_edit(state, monkeypatch):
    from temsim.physics.multiplane_wave import _canonical_map_and_offset
    calls = []
    def transfer(*_args):
        calls.append(None)
        return NS(matrix=np.eye(4), position_offset_m=(0., 0.), angle_offset_rad=(0., 0.))
    monkeypatch.setattr("temsim.optics.direct_alignment.diffraction_transfer", transfer)
    monkeypatch.setattr("temsim.physics.camera_wave._post_sample_kick_events", lambda *_args: ())
    monkeypatch.setattr("temsim.physics.core.fields", lambda *_args: (np.zeros(1),))
    set_round_lens_field_model(state, "copper_coil")
    plane = state.sample.z_mm+1.
    first = _canonical_map_and_offset(state, plane)
    assert _canonical_map_and_offset(state, plane) is first
    _edit_source(state, "ampere_turns")
    second = _canonical_map_and_offset(state, plane)
    assert len(calls) == 2 and second is not first


def _saved_incident(state, signatures):
    z = np.array([state.sample.z_mm], dtype=np.float64)
    phase = np.zeros((1, 1), dtype=np.float64)
    cp = NS(z_mm=z, **{name: phase.copy() for name in
        ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")})
    incident = NS(z=z, x=phase.copy())
    return NS(signatures=signatures, simulation=NS(incident=incident,
        incident_checkpoints=cp, incident_plan=object(), gun_trace=object()))


def test_gui_worker_workflow_gate_rejects_old_incident_after_downstream_copper_edit(state):
    from temsim.simulation_workflow import require_incident_result
    set_round_lens_field_model(state, "copper_coil")
    previous = _saved_incident(state, calculation_signatures(state))
    assert require_incident_result(state, previous, calculation_signatures(state)) is previous.simulation
    _edit_source(state, "ampere_turns")
    with pytest.raises(ValueError, match="upstream source, optics or numerical settings"):
        require_incident_result(state, previous, calculation_signatures(state))


def test_gui_page_worker_reports_stale_copper_incident_without_replacing_completed_result(state, monkeypatch, qapp):
    from temsim.gui import calculation_controller as gui
    from temsim.simulation_workflow import require_incident_result
    set_round_lens_field_model(state, "copper_coil")
    previous = _saved_incident(state, calculation_signatures(state))
    retained = dict(previous.signatures)
    _edit_source(state, "ampere_turns")
    def page_entry(captured, *, existing_result, workflow, **kwargs):
        assert existing_result is previous and workflow == "stem"
        return require_incident_result(captured, existing_result, calculation_signatures(captured))
    monkeypatch.setattr(gui, "calculate", page_entry)
    monkeypatch.setattr(gui, "assert_external_input_inventory_unchanged", lambda *_args: None)
    monkeypatch.setattr(gui, "classical_backend_preflight", lambda *_args: ())
    worker = gui.CalculationWorker(1, "High accuracy", state, existing_result=previous,
        external_inputs=(), workflow="stem")
    errors, results = [], []
    worker.signals.error.connect(lambda *_args: errors.append(_args[-1]))
    worker.signals.result.connect(lambda *_args: results.append(_args))
    worker.run()
    assert len(errors) == 1 and "Run Ray Diagram first" in errors[0]
    assert not results and previous.signatures == retained
