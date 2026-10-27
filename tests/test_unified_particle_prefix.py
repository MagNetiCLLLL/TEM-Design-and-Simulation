"""Safe executed-prefix reuse retains real particles, clocks and field work."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.calculation_manifest import solver_source_identity
from temsim.physics.core import build_propagation_plan, execute_propagation_plan, propagation_plan_common_prefix_nodes
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.potential_snapshot import PotentialContribution
from temsim.physics.unified_spatial_dependencies import potential_is_zero_on_box, spatial_runtime_identity
from temsim.physics.unified_particle_restart import validate_unified_checkpoint
from test_unified_field_particle import controlled
from test_unified_field_runtime import parked_state
from test_particle_sections import fixture


def compact(strength=1e-5, start=451.):
    field = FrozenPosedMultipole("coil", "coil", CoordinateRegistration(),
        ((1, 0, strength), (0, 1, -strength)), "uniform", (start+452.)*.0005,
        (452.-start)*.001, (start, 452.), .005, 1e-22)
    return PotentialContribution("coil", ("coil",), AnalyticPotentialDomain(AnalyticMagneticPotential(field)), "magnetic")


def plan_for(state, runtime, monkeypatch):
    import temsim.physics.unified_field_particle as module
    monkeypatch.setattr(module, "get_unified_field_runtime", lambda _: runtime)
    return build_propagation_plan(state, 450., 452., maximum_step_mm=.1,
        checkpoint_z_mm=(450., 450.5, 451., 451.5, 452.))


def execute(state, plan, *, initial=None, previous=None, row=None):
    if previous is None:
        initial = np.array([[1e-5, 0., .002, -.001], [2e-5, 0., 0., 0.]]) if initial is None else initial
        phase = (initial[:, 0], initial[:, 2], initial[:, 1], initial[:, 3])
        energy, times, start, resume = np.array([299999.8, 300000.2]), np.array([2e-9, 2.1e-9]), 0, None
    else:
        phase = tuple(getattr(previous, key)[row] for key in ("x_m", "tx_rad", "y_m", "ty_rad"))
        energy, times = previous.kinetic_energy_ev[row], previous.flight_time_s[row]
        start = int(np.flatnonzero(plan.z_mm == previous.z_mm[row])[0])
        resume = (previous, row)
    energies = []
    output = execute_propagation_plan(state, plan, *phase, start_index=start,
        initial_time_s=times, initial_kinetic_energy_ev=energy,
        return_flight_times=True, energy_output=energies, unified_resume=resume)
    return output, energies[0]


def test_compact_edit_reuses_only_executed_upstream_and_matches_fresh(monkeypatch):
    state = parked_state()
    old_plan = plan_for(state, controlled(state, (compact(),)), monkeypatch)
    before, _ = execute(state, old_plan)
    current = plan_for(state, controlled(state, (compact(3e-5),)), monkeypatch)
    cp = before[-1]
    common = propagation_plan_common_prefix_nodes(old_plan, current, checkpoints=cp,
        implementation_digest=solver_source_identity())
    assert current.z_mm[common-1] == 450.5
    full, full_energy = execute(state, current)
    partial, partial_energy = execute(state, current, previous=cp, row=1)
    index = np.flatnonzero(full[0] == partial[0][0])[0]
    for expected, actual in zip(full[1:6], partial[1:6]):
        np.testing.assert_array_equal(actual, expected[index:])
    np.testing.assert_array_equal(partial_energy, full_energy[index:])
    assert not np.array_equal(before[1][-1], full[1][-1])
    with pytest.raises(ValueError, match="upstream fields or geometry"):
        execute(state, current, previous=cp, row=3)
    assert propagation_plan_common_prefix_nodes(old_plan, current) == 0


@pytest.mark.parametrize("geometry_change", ("diameter", "position"))
def test_aperture_edit_preserves_past_losses_and_revives_future_losses(monkeypatch, geometry_change):
    from temsim.physics.unified_field_particle import merge_unified_stops
    state = parked_state()
    upstream = PosedWaveAperture("upstream", 450.2, 450.3, .015, CoordinateRegistration())
    downstream = PosedWaveAperture("downstream", 451.2, 451.3, .005, CoordinateRegistration())
    old = plan_for(state, controlled(state, pieces=(upstream, downstream)), monkeypatch)
    initial = np.array([[1e-5, 0., 0., 0.], [2e-5, 0., 0., 0.]])
    before, _ = execute(state, old, initial=initial)
    changed = (replace(downstream, radius_mm=.025) if geometry_change == "diameter" else
               replace(downstream, registration=CoordinateRegistration((1e-5, 0., 0.))))
    current = plan_for(state, controlled(state, pieces=(upstream, changed)), monkeypatch)
    cp = before[-1]
    row = 2
    assert cp.unified_state.active[row].tolist() == [True, False]
    full, energies = execute(state, current, initial=initial)
    partial, resumed_energies = execute(state, current, previous=cp, row=row)
    index = np.flatnonzero(full[0] == partial[0][0])[0]
    for expected, actual in zip(full[1:6], partial[1:6]):
        np.testing.assert_array_equal(actual, expected[index:])
    np.testing.assert_array_equal(resumed_energies, energies[index:])
    actual = merge_unified_stops(partial[0], np.ones(2, bool), np.full(2, np.nan), ["", ""])
    assert actual[0].tolist() == [True, False]
    assert actual[2] == ["", "upstream"]
    assert actual[1][1] == pytest.approx(450.2)


def test_dynamic_and_electric_reference_resume_is_bitwise_equivalent(monkeypatch):
    from test_dynamic_potential import coil
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    state = parked_state(); state.simulation_time_s = .21e-9
    radial, axial = np.array([0., .02]), np.array([.449, .453])
    potential = np.broadcast_to(200000.+2e6*(axial-.45), (2, 2)).copy()
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, radial, axial, potential))
    runtime = controlled(state, (PotentialContribution("electric", ("electric",), electric, "electric"),))
    runtime = replace(runtime, snapshot=DynamicPotentialSnapshot(runtime.snapshot, (coil(),)),
        metadata={"emission_time_s": state.simulation_time_s})
    plan = plan_for(state, runtime, monkeypatch)
    full, energies = execute(state, plan)
    cp = full[-1]
    assert np.any(cp.unified_state.induction_work_ev[2] != 0.)
    partial, resumed_energies = execute(state, plan, previous=cp, row=2)
    index = np.flatnonzero(full[0] == 451.)[0]
    for expected, actual in zip(full[1:6], partial[1:6]):
        np.testing.assert_array_equal(actual, expected[index:])
    np.testing.assert_array_equal(resumed_energies, energies[index:])
    changed = replace(cp, unified_state=replace(cp.unified_state, emission_time_s=0.))
    with pytest.raises(ValueError, match="emission epoch"):
        execute(state, plan, previous=changed, row=2)


def test_gaussian_tail_never_uses_native_support_as_zero():
    bounds = np.array([[-1e-5, -2e-5, .450], [1e-5, 2e-5, .451]])
    near = HighOrderRoundLensPotential(FrozenAnalyticField("C2", ((.04, .65, .01125),),
        CoordinateRegistration(), (600., 700.), .005), radial_order=3, axial_domain="gaussian_all_z")
    far = HighOrderRoundLensPotential(FrozenAnalyticField("downstream", ((.04, 1.1, .0081),),
        CoordinateRegistration(), (1060., 1140.), .005), radial_order=3, axial_domain="gaussian_all_z")
    assert not potential_is_zero_on_box(near, bounds)
    assert potential_is_zero_on_box(far, bounds)
    assert np.any(near.sample(np.array([[1e-5, 0., .45]]))["B_t"] != 0.)
    np.testing.assert_array_equal(far.sample(np.array([[1e-5, 0., .45]]))["B_t"], 0.)


def test_checkpoint_archive_keeps_complete_restart_state(monkeypatch, tmp_path):
    from temsim.artifact_store import ArtifactStore
    from temsim.calculation_manifest import capture_calculation_manifest
    state = parked_state()
    plan = plan_for(state, controlled(state, (compact(),)), monkeypatch)
    full, energies = execute(state, plan)
    store = ArtifactStore(tmp_path, quota_bytes=10_000_000)
    manifest = capture_calculation_manifest(state)
    store.put_propagation_checkpoints(manifest, full[-1])
    restored = store.get_propagation_checkpoints(manifest)
    retained = validate_unified_checkpoint(restored, plan)
    assert retained.spatial_dependency_identities == full[-1].unified_state.spatial_dependency_identities
    partial, resumed_energies = execute(state, plan, previous=restored, row=2)
    index = np.flatnonzero(full[0] == 451.)[0]
    np.testing.assert_array_equal(partial[1], full[1][index:])
    np.testing.assert_array_equal(resumed_energies, energies[index:])
    bad = retained.swept_bounds_m.copy(); bad[-1, 1, 2] = .45
    with pytest.raises(ValueError, match="bounds"):
        validate_unified_checkpoint(replace(restored, unified_state=replace(retained, swept_bounds_m=bad)))


@pytest.mark.parametrize("section", (False, True))
def test_public_particle_workflows_retain_identity_and_reuse_proved_prefix(fixture, monkeypatch, section):
    """The bounded synthetic gun fixture exercises real downstream integration."""
    from temsim.physics.simulation import run
    import temsim.physics.simulation as simulation_module
    import temsim.physics.particle_sections as section_module
    import temsim.physics.unified_field_particle as particle_module
    state, gun, calls = fixture
    state.simulation_mode = "unified"
    state.sample.z_mm = 451.5
    setting = [1e-5]
    monkeypatch.setattr(particle_module, "get_unified_field_runtime",
        lambda current: replace(controlled(current, (compact(setting[0]),)),
            metadata={"source_contract": "Synthetic gun fixture; real bounded column integration"}))
    monkeypatch.setattr(section_module, "section_limits", lambda current: (450., 452.))
    monkeypatch.setattr(simulation_module, "_column_checkpoint_planes",
        lambda start, stop, count: np.arange(start, stop+.01, .5))
    options = {"optical_only": True, "observation_stop_z_mm": 451.5} if section else {}
    first = run(state, resolved_layout=object(), **options)
    setting[0] = 3e-5
    second = run(state, resolved_layout=object(), existing_simulation=first, **options)
    cache = second.metrics["column_segment_cache"]
    assert cache["hit"] and cache["resume_z_mm"] == 450.5
    fresh = run(state, resolved_layout=object(), **options)
    for key in ("x", "y", "tx", "ty", "flight_time_s", "kinetic_energy_ev", "alive", "blocked_z"):
        np.testing.assert_array_equal(getattr(second.incident, key), getattr(fresh.incident, key))
    np.testing.assert_array_equal(second.incident.source_ray_id, gun.exit_bundle.ray_id)
    np.testing.assert_array_equal(second.incident.ray_weight, gun.exit_bundle.weight)
    validate_unified_checkpoint(second.incident_checkpoints, second.incident_plan)
    if not section:
        from temsim.physics.unified_particle_restart import STATE_ARRAY_FIELDS
        cp = second.incident_checkpoints
        internal = sum(getattr(cp.unified_state, name).nbytes for name in STATE_ARRAY_FIELDS)
        assert cache["checkpoint_internal_state_bytes"] == internal
        assert cache["checkpoint_retained_bytes"] == internal + sum(getattr(cp, name).nbytes for name in
            ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"))
    same = run(state, resolved_layout=object(), existing_simulation=second, **options)
    assert same.metrics["column_segment_cache"]["resume_z_mm"] == 451.5
    np.testing.assert_array_equal(same.incident.x, fresh.incident.x)
    # Altered source sample identity/weights are part of the gate, even when
    # the nominal reference energy and every optical parameter are unchanged.
    state.electron_gun.emitter.ray_count += 1
    changed_gun = replace(gun, exit_bundle=replace(gun.exit_bundle, weight=gun.exit_bundle.weight*.5))
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", lambda _: changed_gun)
    incompatible = run(state, resolved_layout=object(), existing_simulation=second, **options)
    assert not incompatible.metrics["column_segment_cache"]["hit"]


def test_optical_reference_downstream_reuses_without_changing_original_mesh(fixture, monkeypatch):
    from temsim.physics.simulation import run
    from temsim.physics.core import propagate
    import temsim.physics.simulation as simulation_module
    import temsim.physics.particle_sections as section_module
    import temsim.physics.unified_field_particle as particle_module
    state, gun, _ = fixture
    state.simulation_mode = "unified"
    state.sample.z_mm, state.step_mm = 450.5, .3
    setting = [1e-5]
    monkeypatch.setattr(particle_module, "get_unified_field_runtime",
        lambda current: replace(controlled(current, (compact(setting[0], start=451.5),)),
            metadata={"source_contract": "Synthetic gun fixture; real bounded column integration"}))
    monkeypatch.setattr(section_module, "section_limits", lambda current: (450., 452.))
    monkeypatch.setattr(simulation_module, "_column_checkpoint_planes",
        lambda start, stop, count: np.arange(start, stop+.01, .5))
    first = run(state, optical_only=True, resolved_layout=object())
    incident = first.incident
    reference = propagate(state, state.sample.z_mm, 452., incident.x[-1], incident.tx[-1],
        incident.y[-1], incident.ty[-1], energy_offset_ev=gun.exit_bundle.energy_offset_ev,
        include_initial_plane_kicks=False, defer_nonfinite_until_clipping=True,
        initial_time_s=incident.flight_time_s[-1], return_flight_times=True,
        initial_kinetic_energy_ev=incident.kinetic_energy_ev[-1])
    old = first.branches["000"]
    for key, expected in zip(("z", "x", "tx", "y", "ty", "flight_time_s"), reference):
        np.testing.assert_array_equal(getattr(old, key), expected)
    setting[0] = 3e-5
    updated = run(state, optical_only=True, resolved_layout=object(), existing_simulation=first)
    cache = updated.metrics["column_segment_cache"]
    assert cache["resume_z_mm"] == state.sample.z_mm
    assert cache["downstream"]["hit"]
    assert state.sample.z_mm < cache["downstream"]["resume_z_mm"] < 451.5
    fresh = run(state, optical_only=True, resolved_layout=object())
    for key in ("z", "x", "tx", "y", "ty", "flight_time_s", "kinetic_energy_ev", "alive", "blocked_z"):
        np.testing.assert_array_equal(getattr(updated.branches["000"], key), getattr(fresh.branches["000"], key))
    assert updated.completed_post_sections[0].initial_dependency_signature != ""
    from types import SimpleNamespace
    from temsim.physics.completed_particle_section import capture_completed_particle_section
    from temsim.particle_section_io import _pack_record_graph, _unpack_record_graph
    result = SimpleNamespace(simulation=updated, state_snapshot=state, workflow="rays",
        specimen_interactions=None, specimen_exit=None, wave_imaging=None)
    assert capture_completed_particle_section(result)
    assert len(updated.section_checkpoint.segments) == 2
    assert updated.metrics["section_resumable_through_z_mm"] == 452.
    arrays = {}
    graph = _pack_record_graph(updated, arrays)
    restored = _unpack_record_graph(graph, arrays, maximum_unpacked_bytes=20_000_000)
    warmed = run(state, optical_only=True, resolved_layout=object(), existing_simulation=restored)
    assert warmed.metrics["column_segment_cache"]["downstream"]["resume_z_mm"] == 452.
    np.testing.assert_array_equal(warmed.branches["000"].x, fresh.branches["000"].x)


def test_completed_optical_reference_keeps_already_absorbed_particles_absent(fixture, monkeypatch):
    from temsim.physics.simulation import run
    import temsim.physics.particle_sections as section_module
    import temsim.physics.unified_field_particle as particle_module
    state, gun, _ = fixture
    state.simulation_mode = "unified"
    state.sample.z_mm = 451.
    aperture = PosedWaveAperture("entrance", 450.2, 450.3, .00005, CoordinateRegistration())
    monkeypatch.setattr(particle_module, "get_unified_field_runtime",
        lambda current: replace(controlled(current, pieces=(aperture,)),
            metadata={"source_contract": "Synthetic gun fixture; real bounded column integration"}))
    monkeypatch.setattr(section_module, "section_limits", lambda current: (450., 452.))
    first = run(state, optical_only=True, resolved_layout=object())
    assert first.incident.alive.tolist() == [True, False, False]
    assert np.isfinite(first.incident.x[-1]).all()
    assert np.isnan(first.incident.kinetic_energy_ev[-1, 1:]).all()
    post = first.completed_post_sections[0]
    assert post.checkpoints.unified_state.active[-1].tolist() == [True, False, False]
    repeated = run(state, optical_only=True, resolved_layout=object(), existing_simulation=first)
    np.testing.assert_array_equal(repeated.branches["000"].alive, first.incident.alive)
    np.testing.assert_array_equal(repeated.branches["000"].blocked_z, first.incident.blocked_z)
    assert repeated.branches["000"].blocked_key == first.incident.blocked_key


def test_spatial_zero_dynamic_drive_retains_global_flyback_guard(monkeypatch):
    from test_dynamic_potential import coil
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    state = parked_state()
    source = coil(raster=True)
    base = source.base.base
    source = replace(source, base=replace(source.base,
        base=replace(base, field=replace(base.field, native_support_mm=(1450., 1452.)))))
    ordinary = controlled(state)
    old = replace(ordinary, snapshot=DynamicPotentialSnapshot(ordinary.snapshot, (source,)))
    changed = replace(source, drives=(replace(source.drives[0], period_s=2e-7),))
    new = replace(ordinary, snapshot=DynamicPotentialSnapshot(ordinary.snapshot, (changed,)))
    bounds = np.array([[0., 0., .450], [1e-5, 1e-5, .451]])
    assert potential_is_zero_on_box(source, bounds)
    assert spatial_runtime_identity(old, bounds) != spatial_runtime_identity(new, bounds)


def test_inactive_input_reference_is_preserved_without_energy_reinterpretation(monkeypatch):
    state = parked_state()
    plan = plan_for(state, controlled(state), monkeypatch)
    values = np.array([0., np.nan])
    result = execute_propagation_plan(state, plan, values, values, values, values,
        initial_kinetic_energy_ev=np.array([300000., np.nan]), initial_time_s=np.array([2e-9, np.nan]),
        defer_nonfinite_until_clipping=True, return_flight_times=True)
    # Unused references are distinct from published physical kinetic energy,
    # whose existing contract still admits only positive values or NaN.
    retained = replace(result[-1].unified_state, reference_energy_ev=np.array([300000., 0.]))
    assert not retained.active[:, 1].any()
    assert retained.reference_energy_ev[1] == 0.
    validate_unified_checkpoint(replace(result[-1], unified_state=retained))
