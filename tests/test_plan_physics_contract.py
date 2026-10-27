"""Existing column execution keeps one immutable owner per physical action.

These are provenance/integration regressions. The coarse full-column fixtures
are not an independent electron-optical convergence or instrument benchmark.
"""
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.optics.equivalent_image_lenses import IMAGE_LENS_KEYS
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.instrument_magnetic import active_column_events, events_overlapping_interval
from temsim.physics.interaction_registry import InteractionRegistry
from temsim.physics.physical_scene import PhysicalSceneSnapshot
from temsim.physics.plan_contract import validate_plan_contract


VARIANTS = ("C2", "C3", "C3_ImageCorrector", "C3_ProbeCorrector",
            "C3_ProbeCorrector_ImageCorrector", "C3_BCOR", "C3_ProbeCorrector_BCOR")


@pytest.fixture(scope="module")
def catalog():
    return AssemblyCatalog()


def _assembled(catalog, variant):
    state = default_state()
    option = next(option for option in catalog.columns
                  if option.file == f"column/{variant}.toml")
    catalog.apply(state, AssemblySelection("FEG", option.name, "No Energy Filter"))
    state.step_mm = 2.
    state.history_step_mm = 10.
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.chromatic_aberration_enabled = True
    state.projector_mode = "Image"
    state.equivalent_image_lenses_enabled = False
    return state


def _plan(state, lower, upper):
    events = events_overlapping_interval(state, active_column_events(state), lower, upper)
    return build_propagation_plan(state, lower, upper, events,
                                  checkpoint_z_mm=(lower, upper))


def _records(plan):
    validate_plan_contract(plan.physics_contract)
    return InteractionRegistry.from_dict(plan.physics_contract["interactions"]).records


@pytest.fixture(scope="module", params=VARIANTS)
def column_plans(catalog, request):
    state = _assembled(catalog, request.param)
    state.image_diffraction_deflector.upper_x_mrad = .03
    state.descan_deflector.enabled = True
    state.descan_deflector.kick_x_mrad = -.01
    lower = min(lens.z_mm for lens in state.lenses if lens.enabled)-10.
    upper = float(state.fluorescent_screen.z_mm)
    full = _plan(state, lower, upper)
    post = _plan(state, float(state.sample.z_mm), upper)
    state.equivalent_image_lenses_enabled = True
    reduced = _plan(state, float(state.sample.z_mm), upper)
    state.equivalent_image_lenses_enabled = False
    return state, full, post, reduced


def test_seven_columns_share_one_descan_physical_host(column_plans):
    state, full, post, reduced = column_plans
    host, channel = state.image_diffraction_deflector, state.descan_deflector
    assert channel._physical_host is host
    for plan in (full, post, reduced):
        contract = plan.physics_contract
        assert contract["mode"] == "legacy_compatible"
        assert tuple(contract["interval_mm"]) == (plan.z_mm[0], plan.z_mm[-1])
        scene = PhysicalSceneSnapshot.from_dict(contract["scene"])
        assert scene.owner_for(channel.key) == scene.owner_for(host.key) == host.key
        records = _records(plan)
        coils = [row for row in records if row.owner_id == host.key and row.effect_family == "dipole"]
        assert len(coils) == 2
        assert len({row.interaction_id for row in coils}) == 2
        assert all({host.key, channel.key} <= set(row.logical_drive_ids) for row in coils)
        assert not any(row.owner_id == channel.key for row in records)
        assert not any(row.effect_family == "slope_kick" for row in records)


def test_seven_columns_replace_image_fields_instead_of_adding_a_second_action(column_plans):
    state, full, post, reduced = column_plans
    expected = {lens.key for lens in state.lenses
                if lens.enabled and lens.key in IMAGE_LENS_KEYS}
    native_rows = [row for row in _records(post) if row.effect_family == "round"]
    reduced_rows = [row for row in _records(reduced) if row.effect_family == "round"]
    for key in expected:
        effect = f"{key}:round"
        original = [row for row in native_rows if effect in row.covered_effects]
        replacement = [row for row in reduced_rows if effect in row.covered_effects]
        assert len(original) == len(replacement) == 1
        assert original[0].representation == "field"
        assert replacement[0].representation == "reduced"
    assert not np.any(full.thin_power_m1)
    assert not np.any(post.thin_power_m1)
    assert np.any(reduced.thin_power_m1)
    # Reduction replaces linear focusing only; declared residuals have their
    # own coverage rather than pretending they were inferred from the field.
    for plan in (full, post, reduced):
        for row in _records(plan):
            if row.representation == "residual":
                assert row.residual_provenance
                assert not any(effect.endswith(":round") for effect in row.covered_effects)


def test_seven_column_contracts_are_recursively_immutable(column_plans):
    _, full, _, _ = column_plans
    contract = full.physics_contract
    before = thaw_json(contract)
    with pytest.raises(TypeError):
        contract["mode"] = "changed"
    with pytest.raises(TypeError):
        contract["scene"]["payload"]["ownership"]["descan_deflector"]["owner_key"] = "changed"
    with pytest.raises(TypeError):
        contract["interactions"]["records"][0]["owner_id"] = "changed"
    detached = thaw_json(contract)
    detached["scene"]["payload"]["ownership"]["descan_deflector"]["owner_key"] = "changed"
    assert thaw_json(contract) == before


def test_seven_columns_use_the_same_contract_in_the_wave_plan(column_plans):
    from temsim.physics.column_wave import _prepare_column
    state, _, _, _ = column_plans
    lens = next(lens for lens in state.lenses if lens.key == "condenser_lens_1")
    lower, upper = lens.z_mm-1., lens.z_mm+1.
    rays = _plan(state, lower, upper)
    wave, _, _, _ = _prepare_column(state, lower, upper, 2.)
    assert wave.physics_contract["scene"] == rays.physics_contract["scene"]
    # Wave boundary events can change numerical sampling. Compare the physical
    # actions and their ownership, not source hashes of different sample grids.
    def actions(plan):
        return {(row.owner_id, row.effect_family, row.representation,
                 row.interval_mm, row.covered_effects, row.logical_drive_ids)
                for row in _records(plan)}
    assert actions(wave) == actions(rays)


def test_repeated_identified_coil_cannot_also_become_an_external_thin_kick(catalog):
    state = _assembled(catalog, "C3")
    state.image_diffraction_deflector.upper_x_mrad = .2
    host = state.image_diffraction_deflector
    lower, upper = host.upper_z_mm-1., host.lower_z_mm+1.
    events = events_overlapping_interval(state, active_column_events(state), lower, upper)
    event = next(row for row in events if row[0] == host.upper_z_mm and (row[1] or row[2]))
    with pytest.raises(ValueError, match="Duplicate physical action"):
        build_propagation_plan(state, lower, upper, (*events, event))


def test_parameter_and_host_geometry_edits_leave_executed_snapshot_detached(catalog):
    state = _assembled(catalog, "C3")
    host = state.image_diffraction_deflector
    lower, upper = host.upper_z_mm-3., host.lower_z_mm+3.
    before = _plan(state, lower, upper)
    record = thaw_json(before.physics_contract)
    host.upper_x_mrad += .02
    driven = _plan(state, lower, upper)
    assert driven.physics_contract["scene"]["digest"] != before.physics_contract["scene"]["digest"]
    host.optical_center_z_mm += .125
    moved = _plan(state, lower, upper)
    assert moved.physics_contract["scene"]["geometry_digest"] != driven.physics_contract["scene"]["geometry_digest"]
    assert thaw_json(before.physics_contract) == record
    validate_plan_contract(before.physics_contract)


def test_ownership_metadata_does_not_change_numerical_execution(catalog):
    state = _assembled(catalog, "C3_ProbeCorrector_BCOR")
    lens = next(lens for lens in state.lenses if lens.key == "condenser_lens_2")
    lower, upper = lens.z_mm-2., lens.z_mm+2.
    captured = _plan(state, lower, upper)
    legacy = replace(captured, physics_contract=None)
    assert legacy.physics_contract is None
    x, y = np.array((1., -2., 3.))*1e-9, np.array((-2., 1., 3.))*1e-9
    outputs = [execute_propagation_plan(state, plan, x, x*.01, y, y*.01,
        initial_kinetic_energy_ev=np.array((299700., 300000., 300300.)),
        initial_time_s=np.zeros(3), return_flight_times=True)[-1] for plan in (captured, legacy)]
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        np.testing.assert_array_equal(getattr(outputs[0], name), getattr(outputs[1], name))


def _bind_objective_map(state, tmp_path, *, offset_m=0.):
    from temsim.physics.lens_field_provider import (
        CoordinateRegistration, bind_imported_lens_field_map, load_magnetic_field_map,
    )
    from test_lens_field_provider import _write_axisymmetric_map
    lens = state.objective_lens
    path = tmp_path / "objective_field.npz"
    binding = _write_axisymmetric_map(path, state, lens)
    field_map = load_magnetic_field_map(path, geometry_binding=binding,
        provenance_kind="fem", reference_excitation_percent=100.)
    if offset_m:
        field_map = replace(field_map,
            registration=CoordinateRegistration(origin_global_m=(0., 0., offset_m)))
    bind_imported_lens_field_map(state, lens.key, field_map, native_provider=lens)
    return lens


def test_reduced_mapped_objective_does_not_claim_an_unexecuted_native_cs(catalog, tmp_path):
    state = _assembled(catalog, "C3")
    state.equivalent_image_lenses_enabled = True
    lens = state.objective_lens
    lens.cs_mm = 2.
    _bind_objective_map(state, tmp_path)
    plan = _plan(state, state.sample.z_mm, state.fluorescent_screen.z_mm)
    assert not any(field.lens_key == lens.key for field in plan.mapped_fields)
    assert np.any(plan.thin_power_m1)
    index = int(np.argmin(abs(plan.z_mm-lens.z_mm)))
    assert plan.cs_kick_m3[index] == 0.
    owned = [row for row in _records(plan) if row.owner_id == lens.key]
    assert any(row.effect_family == "round" and row.representation == "reduced" for row in owned)
    assert not any(row.effect_family in {"Cs", "Cc"} for row in owned)


def test_mapped_field_outside_interval_does_not_reappear_as_native_round_or_cs(catalog, tmp_path):
    state = _assembled(catalog, "C3")
    state.objective_lens.cs_mm = 2.
    lens = _bind_objective_map(state, tmp_path, offset_m=.1)
    plan = _plan(state, lens.z_mm-1., lens.z_mm+1.)
    assert not any(field.lens_key == lens.key for field in plan.mapped_fields)
    assert not np.any(plan.cs_kick_m3)
    assert not any(row.owner_id == lens.key and row.effect_family in {"round", "Cs", "Cc"}
                   for row in _records(plan))


def test_chromatic_event_at_rounded_endpoint_is_present_in_execution_and_contract():
    from temsim.physics.posed_aberrations import kick_indices
    state = default_state()
    state.chromatic_aberration_enabled = True
    state.equivalent_image_lenses_enabled = False
    endpoint = float(state.objective_lens.z_mm)
    plan = _plan(state, endpoint-1., endpoint)
    kick, = [kick for kick in plan.intrinsic_aberration_kicks
             if kick.lens_key == state.objective_lens.key
             and kick.chromatic_reference_energy_ev is not None]
    assert kick.z_mm != endpoint  # The fixture reproduces the mm/m ULP boundary.
    assert kick in kick_indices((kick,), plan.z_mm)[len(plan.z_mm)-1]
    row, = [row for row in _records(plan) if row.owner_id == kick.lens_key and row.effect_family == "Cc"]
    assert row.interval_mm == (endpoint, endpoint)


def test_real_joint_nonlinear_plan_captures_total_once_and_rejects_member_reapplication():
    from test_nonlinear_magnetostatics import _shared
    from temsim.physics.interaction_registry import InteractionOwnershipError
    from temsim.physics.lens_field_provider import active_vector_providers

    # The existing synthetic two-channel, three-pole FEM fixture exercises the
    # production compiler and its pre-freeze metadata, not invented field maps.
    state = _shared()
    state.beam_voltage_kv = 300.
    state.step_mm = 2.
    state.stigmators = []
    state.equivalent_image_lenses_enabled = False
    state.chromatic_aberration_enabled = False
    plan = build_propagation_plan(state, -20., 20., include_spherical_aberration=False)
    providers = {provider.lens_key: provider for provider in active_vector_providers(state)}
    assert providers["a"].model_status == "joint_nonlinear_field"
    assert providers["b"].model_status == "included_in_joint_nonlinear_field"
    assert np.any(providers["a"].field_map.components_t)
    assert not np.any(providers["b"].field_map.components_t)
    fields = {field.lens_key: field for field in plan.mapped_fields}
    assert set(fields) == {"a", "b"}
    points = np.array(((0., 0., -.005), (0., 0., .005)))
    assert np.any(fields["a"].field_at_global_positions_t(points))
    np.testing.assert_array_equal(fields["b"].field_at_global_positions_t(points), np.zeros((2, 3)))
    assert not np.any(plan.magnetic_t)  # No native axial fallback duplicates the maps.
    assert not np.any(plan.cs_kick_m3)

    records = _records(plan)
    total, = [row for row in records if row.executes]
    member, = [row for row in records if not row.executes]
    assert total.owner_id == "a" and total.representation == "field"
    assert total.logical_drive_ids == ("a", "b")
    assert total.circuit_id == "joint_nonlinear"
    assert total.source_digest == fields["a"].fingerprint
    assert set(total.covered_effects) >= {
        "a:round", "a:Cs", "a:Cc", "b:round", "b:Cs", "b:Cc",
    }
    assert member.owner_id == "b" and member.representation == "observation"
    assert member.effect_family == "joint_field_member"
    assert member.source_digest == fields["b"].fingerprint
    registry = InteractionRegistry.from_dict(plan.physics_contract["interactions"])
    assert registry.executing_action_ids == (total.action_id,)
    assert InteractionRegistry.from_dict(registry.to_dict()).digest == registry.digest

    # A member cannot be applied again as an independent field, including at
    # the total field's terminal plane. Diagnostic observations remain valid.
    for interval in (total.interval_mm, (total.interval_mm[1], total.interval_mm[1])):
        duplicate = replace(member, interaction_id="injected_member_field",
            action_id="injected_member_action", effect_family="round", representation="field",
            interval_mm=interval, covered_effects=("b:round",))
        with pytest.raises(InteractionOwnershipError, match="b:round.*executed twice"):
            InteractionRegistry((*records, duplicate))
